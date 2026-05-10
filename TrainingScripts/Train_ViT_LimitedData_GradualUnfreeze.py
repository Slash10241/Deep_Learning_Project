"""
Train_ViT_LimitedData_GradualUnfreeze.py
=========================================
Limited-data study — Gradual Unfreeze (3 transformer block groups).

Outer loop : data fraction  (10%, 1%)   [100% already run]
Middle loop : L2 regime      (wd_standard=1e-4  |  wd_strong=1e-2)
Inner loop  : augmentation   (all 12 experiments from experiments.json)

Checkpoint directory is split per fraction:
  ../Checkpoints/vit_limited_data_10pct_gradual/
  ../Checkpoints/vit_limited_data_01pct_gradual/

Master CSV per fraction:
  CHECKPOINT_BASE/all_experiment_results_gradual_data_frac<pct>.csv

Each run also writes its own:
  CHECKPOINT_DIR/experiment_results.csv
  CHECKPOINT_DIR/training_curves.png
  CHECKPOINT_DIR/checkpoint.pt
"""

import sys
import json
import os
import csv
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

sys.path.insert(0, "../Models")
sys.path.insert(0, "../DataLoader")
sys.path.insert(0, "../Training")

from DataLoader     import (BatchAugmenter, CreateDataset,
                             get_train_transform, get_eval_transform,
                             _parse_annotation_file, _pet_collate)
from ViTFinetune    import ViTGradualUnfreeze
from TrainingEngine import TrainingEngine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if device.type == "cuda":
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# FIXED CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT    = "../../../Dataset/"
BATCH_SIZE      = 64
IMAGE_SIZE      = 224
VAL_SPLIT       = 0.2
NUM_WORKERS     = 0
SEED            = 42
LR_PATIENCE     = 3
LR_FACTOR       = 0.1
LR_MIN          = 1e-6
ES_PATIENCE     = 6
LABEL_MODE      = "breed"
MODEL_NAME      = "vit_tiny_patch16_224"
TRAINING_TYPE   = "LimitedData_Gradual_ViT_Tiny"
CHECKPOINT_ROOT = "../Checkpoints/"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
NUM_CLASSES     = 37 if LABEL_MODE == "breed" else 2

# Fixed: 3 block groups
NUM_BLOCKS   = 12
HEAD_EPOCHS  = 20
BLOCK_EPOCHS = 15
HEAD_LR      = 1e-3
BLOCK_LR     = 1e-4

DATA_FRACTIONS = [ 0.25, 0.1]

L2_REGIMES = {
    "wd_standard": 1e-4
}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
def build_phase_schedule(num_blocks, head_epochs, block_epochs, head_lr, block_lr):
    schedule = [{"epochs": head_epochs, "lr": head_lr}]
    for i in range(num_blocks):
        phase_lr = block_lr if i < (num_blocks // 2 + 1) else block_lr * 0.5
        schedule.append({"epochs": block_epochs, "lr": phase_lr})
    return schedule


PHASE_SCHEDULE = build_phase_schedule(NUM_BLOCKS, HEAD_EPOCHS, BLOCK_EPOCHS, HEAD_LR, BLOCK_LR)
LAST_PHASE_IDX = len(PHASE_SCHEDULE) - 1


# ─────────────────────────────────────────────────────────────────────────────
# LOAD AUGMENTATION EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────
with open("experiments_final_augs.json", "r") as f:
    exp_config = json.load(f)

FIXED_AUGS = exp_config["fixed"]
for key in ("crop_scale", "crop_ratio"):
    if key in FIXED_AUGS:
        FIXED_AUGS[key] = tuple(FIXED_AUGS[key])

EXPERIMENTS = []
for exp in exp_config["experiments"]:
    augs = {**FIXED_AUGS}
    overrides = exp.pop("override_fixed", {})
    augs.update(overrides)
    augs.update(exp)
    for key in ("crop_scale", "crop_ratio", "erasing_scale"):
        if key in augs and isinstance(augs[key], list):
            augs[key] = tuple(augs[key])
    EXPERIMENTS.append(augs)

total_runs = len(DATA_FRACTIONS) * len(L2_REGIMES) * len(EXPERIMENTS)
print(f"\nLoaded {len(EXPERIMENTS)} augmentation experiments.")
print(f"Data fractions : {DATA_FRACTIONS}")
print(f"L2 regimes     : {list(L2_REGIMES.keys())}")
print(f"Total runs     : {total_runs}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class SmartCELoss(nn.Module):
    def __init__(self, weight=None, smoothing_hard=0.1):
        super().__init__()
        self.weight         = weight
        self.smoothing_hard = smoothing_hard

    def forward(self, logits, labels):
        smoothing = 0.0 if labels.ndim > 1 else self.smoothing_hard
        return F.cross_entropy(
            logits, labels,
            weight=self.weight,
            label_smoothing=smoothing,
        )
        
class LabelSelector:
    def __init__(self, loader, mode):
        self.loader = loader
        self.mode   = mode

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, (y1, y2) in self.loader:
            yield x, (y2 if self.mode == "breed" else y1)


class EarlyStopping:
    def __init__(self, patience: int = 5, save_path: str = "best_model.pt", delta: float = 1e-4):
        self.patience  = patience
        self.save_path = save_path
        self.delta     = delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False

    def reset_counter(self):
        self.counter   = 0
        self.triggered = False
        print("  Early-stop counter reset for new phase.")

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            self.triggered = False
            torch.save(model.state_dict(), self.save_path)
            print(f"  ✔ New best val_loss={val_loss:.4f} — checkpoint saved.")
        else:
            self.counter += 1
            print(f"  Early-stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
                print("  ✖ Early stopping triggered.")


def build_limited_loaders(dataset_root, fraction, augs, image_size,
                          batch_size, val_split, num_workers, seed):
    from pathlib import Path
    import random

    root         = Path(dataset_root)
    images_dir   = root / "images"
    trainval_txt = root / "annotations" / "trainval.txt"
    test_txt     = root / "annotations" / "test.txt"

    trainval_records = _parse_annotation_file(str(trainval_txt))
    test_records     = _parse_annotation_file(str(test_txt))

    rng = random.Random(seed)
    rng.shuffle(trainval_records)
    n_val         = int(len(trainval_records) * val_split)
    train_records = trainval_records[: len(trainval_records) - n_val]
    val_records   = trainval_records[len(trainval_records) - n_val :]

    full_train_dataset = CreateDataset(
        train_records, images_dir,
        transform=get_train_transform(augs, image_size), one_hot=False,
    )

    labels   = np.array([r["breed"] for r in train_records])
    indices  = np.arange(len(train_records))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
    subset_idx, _ = next(splitter.split(indices, labels))
    train_subset  = Subset(full_train_dataset, subset_idx)
    print(f"  Stratified subset: {len(train_subset)}/{len(full_train_dataset)} "
          f"samples ({fraction*100:.0f}%)")

    val_dataset  = CreateDataset(val_records,  images_dir, transform=get_eval_transform(image_size), one_hot=False)
    test_dataset = CreateDataset(test_records, images_dir, transform=get_eval_transform(image_size), one_hot=False)

    train_loader = DataLoader(train_subset,  batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=_pet_collate, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, collate_fn=_pet_collate)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, collate_fn=_pet_collate)
    return train_loader, val_loader, test_loader


def save_training_curves(history, checkpoint_dir, title, phase_boundaries):
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes  = plt.subplots(1, 3, figsize=(16, 4))

    for ax, tk, vk, label in [
        (axes[0], "train_loss", "val_loss", "Loss"),
        (axes[1], "train_acc",  "val_acc",  "Accuracy"),
    ]:
        ax.plot(epochs_ran, history[tk], label="Train")
        ax.plot(epochs_ran, history[vk], label="Val")
        for b in phase_boundaries:
            ax.axvline(x=b, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(label); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True)

    axes[2].plot(epochs_ran, history["val_f1"], label="Val Macro F1", color="green")
    for b in phase_boundaries:
        axes[2].axvline(x=b, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[2].set_title("Validation Macro F1"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True)

    plt.suptitle(f"ViT LimitedData Gradual — {title}", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")


def save_run_csv(results: dict, checkpoint_dir: str):
    csv_path = os.path.join(checkpoint_dir, "experiment_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"  Run CSV saved to {csv_path}")


def save_master_csv(results: dict, checkpoint_base: str, frac_pct: str):
    csv_path    = os.path.join(checkpoint_base,
                               f"all_experiment_results_gradual_data_frac{frac_pct}.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"  Master CSV appended: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP  outer: fraction → L2 → augmentation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    run_counter = 0

    for fraction in DATA_FRACTIONS:
        frac_pct        = f"{int(fraction * 100):02d}pct"
        CHECKPOINT_BASE = os.path.join(CHECKPOINT_ROOT,
                                       f"{TRAINING_TYPE}_{frac_pct}_gradual/")
        os.makedirs(CHECKPOINT_BASE, exist_ok=True)

        print("\n" + "█" * 70)
        print(f"DATA FRACTION : {fraction*100:.0f}%  →  {CHECKPOINT_BASE}")
        print("█" * 70)

        for wd_name, weight_decay in L2_REGIMES.items():

            print("\n" + "▓" * 70)
            print(f"  L2 REGIME : {wd_name}  (weight_decay={weight_decay})")
            print("▓" * 70)

            for AUGMENTATIONS in EXPERIMENTS:
                run_counter += 1
                aug_name    = AUGMENTATIONS["name"]
                run_label   = f"{wd_name}_{aug_name}"

                print("\n" + "=" * 70)
                print(f"RUN {run_counter}/{total_runs}  |  "
                      f"data={fraction*100:.0f}%  L2={wd_name}  aug={aug_name}")
                print("=" * 70)

                CHECKPOINT_DIR = os.path.join(
                    CHECKPOINT_BASE,
                    f"{MODEL_NAME}_{TRAINING_TYPE}_{LABEL_MODE}_{run_label}"
                )
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)

                train_loader, val_loader, test_loader = build_limited_loaders(
                    dataset_root = DATASET_ROOT,
                    fraction     = fraction,
                    augs         = AUGMENTATIONS,
                    image_size   = IMAGE_SIZE,
                    batch_size   = BATCH_SIZE,
                    val_split    = VAL_SPLIT,
                    num_workers  = NUM_WORKERS,
                    seed         = SEED,
                )

                train_selector = LabelSelector(train_loader, LABEL_MODE)
                val_selector   = LabelSelector(val_loader,   LABEL_MODE)
                test_selector  = LabelSelector(test_loader,  LABEL_MODE)

                model           = ViTGradualUnfreeze(num_classes=NUM_CLASSES, model_name=MODEL_NAME)
                loss_fn = SmartCELoss(smoothing_hard=0.1)
                batch_augmenter = BatchAugmenter(augs=AUGMENTATIONS, num_classes=NUM_CLASSES)

                early_stopping  = EarlyStopping(
                    patience  = ES_PATIENCE,
                    save_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pt"),
                )

                history = {
                    "train_loss": [], "train_acc": [],
                    "val_loss":   [], "val_acc":   [], "val_f1": [],
                }

                phase_boundaries = []
                total_blocks_vit = model.total_block_count
                global_epoch     = 0
                terminate_run    = False
                train_start      = time.time()

                for phase_idx, phase in enumerate(PHASE_SCHEDULE):
                    if terminate_run:
                        break

                    is_last_phase = (phase_idx == LAST_PHASE_IDX)

                    if phase_idx > 0:
                        block_to_unfreeze = total_blocks_vit - phase_idx
                        if block_to_unfreeze >= 0:
                            model.unfreeze_next_block(block_to_unfreeze)
                            phase_boundaries.append(global_epoch)
                            print(f"\n  ▶ Unfroze block {block_to_unfreeze} "
                                  f"(phase {phase_idx}/{LAST_PHASE_IDX})")

                    optimizer = torch.optim.Adam(
                        model.trainable_params(),
                        lr=phase["lr"], weight_decay=weight_decay,
                    )
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=LR_FACTOR,
                        patience=LR_PATIENCE, min_lr=LR_MIN, verbose=True,
                    )

                    engine = TrainingEngine(
                        network=model, data_iterator=train_selector,
                        loss_fn=loss_fn, opt=optimizer,
                        compute_device=device, scheduler=None,
                        batch_aug=batch_augmenter,
                    )

                    early_stopping.reset_counter()

                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"\n{'─'*70}")
                    print(f"Phase {phase_idx}/{LAST_PHASE_IDX} | "
                          f"LR={phase['lr']:.0e} | WD={weight_decay:.0e} | "
                          f"Unfrozen={model.unfrozen_block_count} | "
                          f"Trainable={trainable:,} | "
                          f"{'FINAL' if is_last_phase else 'intermediate'}")
                    print(f"{'─'*70}")

                    for phase_epoch in range(1, phase["epochs"] + 1):
                        global_epoch += 1
                        print(f"\n── Phase {phase_idx} | "
                              f"Epoch {phase_epoch}/{phase['epochs']} "
                              f"(global {global_epoch}) ──")

                        train_stats = engine.train_one_epoch(epoch_num=global_epoch, print_freq=50)
                        val_stats   = engine.evaluate(val_selector, phase_label="Validation")

                        history["train_loss"].append(train_stats["epoch_loss"])
                        history["train_acc"].append(train_stats["epoch_acc"])
                        history["val_loss"].append(val_stats["epoch_loss"])
                        history["val_acc"].append(val_stats["epoch_acc"])
                        history["val_f1"].append(val_stats["macro_f1"])

                        scheduler.step(val_stats["epoch_loss"])
                        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

                        early_stopping(val_stats["epoch_loss"], model)

                        if early_stopping.triggered:
                            if is_last_phase:
                                print(f"\n  ✖ Early stopping on final phase "
                                      f"(global epoch {global_epoch}). Terminating.")
                                terminate_run = True
                            else:
                                print(f"\n  ⏭  Early stopping on intermediate phase {phase_idx}. "
                                      f"Advancing to next block.")
                            break

                train_time_sec = time.time() - train_start

                print("\nLoading best checkpoint for test evaluation ...")
                model.load_state_dict(torch.load(
                    os.path.join(CHECKPOINT_DIR, "checkpoint.pt"), map_location=device,
                ))
                model.to(device)
                test_stats = engine.evaluate(test_selector, phase_label="Test")

                print("\n── Final Test Results ──────────────────────────────")
                print(f"  Fraction   : {fraction*100:.0f}%  |  L2: {wd_name}")
                print(f"  Loss       : {test_stats['epoch_loss']:.4f}")
                print(f"  Accuracy   : {test_stats['epoch_acc']*100:.2f}%")
                print(f"  Macro F1   : {test_stats['macro_f1']*100:.2f}%")
                print(f"  Train time : {train_time_sec/60:.1f} min")

                row = {
                    "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "experiment":       aug_name,
                    "training_type":    TRAINING_TYPE,
                    "data_fraction":    fraction,
                    "n_train_samples":  len(train_loader.dataset),
                    "l2_regime":        wd_name,
                    "weight_decay":     weight_decay,
                    "num_blocks":       NUM_BLOCKS,
                    "phases_completed": len(phase_boundaries) + 1,
                    "blocks_unfrozen":  model.unfrozen_block_count,
                    "model":            MODEL_NAME,
                    "label_mode":       LABEL_MODE,
                    "num_classes":      NUM_CLASSES,
                    "batch_size":       BATCH_SIZE,
                    "epochs_ran":       global_epoch,
                    "best_val_loss":    round(early_stopping.best_loss, 4),
                    "loss":             round(test_stats["epoch_loss"], 4),
                    "accuracy":         round(test_stats["epoch_acc"] * 100, 2),
                    "macro_f1":         round(test_stats["macro_f1"] * 100, 2),
                    "macro_p":          round(test_stats["macro_precision"] * 100, 2),
                    "macro_r":          round(test_stats["macro_recall"] * 100, 2),
                    "train_time_sec":   round(train_time_sec, 1),
                    "train_time_min":   round(train_time_sec / 60, 2),
                    "augmentations":    str(AUGMENTATIONS),
                    "phase_schedule":   str(PHASE_SCHEDULE),
                }

                save_training_curves(
                    history, CHECKPOINT_DIR,
                    title=f"{fraction*100:.0f}% | {wd_name} | {aug_name}",
                    phase_boundaries=phase_boundaries,
                )
                save_run_csv(row, CHECKPOINT_DIR)
                save_master_csv(row, CHECKPOINT_BASE, frac_pct)

    print("\n" + "=" * 70)
    print(f"All {total_runs} runs complete.")
