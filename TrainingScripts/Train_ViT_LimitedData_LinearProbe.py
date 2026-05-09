"""
Train_ViT_LimitedData_LinearProbe.py
======================================
Limited-data study — Linear Probe (head-only training).

Outer loop : data fraction  (10%, 1%)   [100% already run]
Inner loop : augmentation   (all 12 experiments from experiments.json)

Within each run, two L2 weight-decay values are tested:
  • wd_standard : 1e-4  (same as baseline)
  • wd_strong   : 1e-2  (stronger regularisation for limited data)

Stratified subsets preserve class proportions at every fraction via
StratifiedShuffleSplit on breed labels. Val and test sets are always
the full unmodified splits for fair cross-fraction comparison.

Outputs per run
---------------
  CHECKPOINT_DIR/checkpoint.pt            best model weights
  CHECKPOINT_DIR/training_curves.png      loss / acc / F1 curves
  CHECKPOINT_DIR/experiment_results.csv   single-run summary
  CHECKPOINT_BASE/all_experiment_results_linear_<frac_pct>.csv
                                          appended across all runs
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

sys.path.insert(0, "../Models")
sys.path.insert(0, "../DataLoader")
sys.path.insert(0, "../Training")

from DataLoader     import (BatchAugmenter, CreateDataset,
                             get_train_transform, get_eval_transform,
                             _parse_annotation_file, _pet_collate)
from ViTFinetune    import ViTLinearProbe
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
NUM_EPOCHS      = 50
IMAGE_SIZE      = 224
VAL_SPLIT       = 0.2
NUM_WORKERS     = 0
SEED            = 42
LEARNING_RATE   = 1e-3
LR_PATIENCE     = 3
LR_FACTOR       = 0.1
LR_MIN          = 1e-6
ES_PATIENCE     = 6
LABEL_MODE      = "breed"
MODEL_NAME      = "vit_base_patch16_224"
TRAINING_TYPE   = "LimitedData_Linear"
CHECKPOINT_ROOT = "../Checkpoints/"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
NUM_CLASSES     = 37 if LABEL_MODE == "breed" else 2

# 100% already run — only 10% and 1% here
# DATA_FRACTIONS = [0.5, 0.25, 0.1]
DATA_FRACTIONS = [0.25]
L2_REGIMES = {
    "wd_standard": 1e-4,
    "wd_strong":   1e-2,
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD AUGMENTATION EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────
with open("experiments_final.json", "r") as f:
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
        self.stop      = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"  ✔ New best val_loss={val_loss:.4f} — checkpoint saved.")
        else:
            self.counter += 1
            print(f"  Early-stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                print("  ✖ Early stopping triggered.")


def build_limited_loaders(dataset_root, fraction, augs, image_size,
                          batch_size, val_split, num_workers, seed):
    """Stratified subset of train split; full val and test."""
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
        transform=get_train_transform(augs, image_size),
        one_hot=False,
    )

    labels   = np.array([r["breed"] for r in train_records])
    indices  = np.arange(len(train_records))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
    subset_idx, _ = next(splitter.split(indices, labels))
    train_subset  = Subset(full_train_dataset, subset_idx)
    print(f"  Stratified subset: {len(train_subset)}/{len(full_train_dataset)} "
          f"training samples ({fraction*100:.0f}%)")

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


def save_training_curves(history, checkpoint_dir, title):
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes  = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs_ran, history["train_loss"], label="Train")
    axes[0].plot(epochs_ran, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs_ran, history["train_acc"], label="Train")
    axes[1].plot(epochs_ran, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs_ran, history["val_f1"], label="Val Macro F1", color="green")
    axes[2].set_title("Validation Macro F1"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True)

    plt.suptitle(f"ViT LimitedData Linear — {title}", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")


def save_run_csv(results: dict, checkpoint_dir: str):
    """Save a single-run summary CSV alongside the checkpoint."""
    csv_path = os.path.join(checkpoint_dir, "experiment_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"  Run CSV saved to {csv_path}")


def save_master_csv(results: dict, checkpoint_base: str, frac_pct: str):
    """Append to the per-fraction master CSV."""
    csv_path    = os.path.join(checkpoint_base,
                               f"all_experiment_results_linear_data_frac{frac_pct}.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"  Master CSV appended: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP  outer: fraction → L2 regime → augmentation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    run_counter = 0

    for fraction in DATA_FRACTIONS:
        frac_pct       = f"{int(fraction * 100):02d}pct"   # "10pct" / "01pct"
        CHECKPOINT_BASE = os.path.join(CHECKPOINT_ROOT,
                                       f"vit_limited_data_{frac_pct}_linear/")
        os.makedirs(CHECKPOINT_BASE, exist_ok=True)

        print("\n" + "█" * 70)
        print(f"DATA FRACTION : {fraction*100:.0f}%  ({frac_pct})  →  {CHECKPOINT_BASE}")
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

                model     = ViTLinearProbe(num_classes=NUM_CLASSES, model_name=MODEL_NAME)
                optimizer = torch.optim.Adam(
                    model.trainable_params(),
                    lr=LEARNING_RATE, weight_decay=weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=LR_FACTOR,
                    patience=LR_PATIENCE, min_lr=LR_MIN, verbose=True,
                )
                loss_fn         = nn.CrossEntropyLoss(label_smoothing=0.1)
                batch_augmenter = BatchAugmenter(augs=AUGMENTATIONS, num_classes=NUM_CLASSES)

                engine = TrainingEngine(
                    network=model, data_iterator=train_selector,
                    loss_fn=loss_fn, opt=optimizer,
                    compute_device=device, scheduler=None,
                    batch_aug=batch_augmenter,
                )

                early_stopping = EarlyStopping(
                    patience  = ES_PATIENCE,
                    save_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pt"),
                )

                history = {"train_loss": [], "train_acc": [],
                           "val_loss":   [], "val_acc":   [], "val_f1": []}

                train_start = time.time()

                for epoch in range(1, NUM_EPOCHS + 1):
                    print(f"\n── Epoch {epoch}/{NUM_EPOCHS} ──")
                    train_stats = engine.train_one_epoch(epoch_num=epoch, print_freq=50)
                    val_stats   = engine.evaluate(val_selector, phase_label="Validation")

                    history["train_loss"].append(train_stats["epoch_loss"])
                    history["train_acc"].append(train_stats["epoch_acc"])
                    history["val_loss"].append(val_stats["epoch_loss"])
                    history["val_acc"].append(val_stats["epoch_acc"])
                    history["val_f1"].append(val_stats["macro_f1"])

                    scheduler.step(val_stats["epoch_loss"])
                    print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

                    early_stopping(val_stats["epoch_loss"], model)
                    if early_stopping.stop:
                        print(f"\nTraining stopped early at epoch {epoch}.")
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
                    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "experiment":      aug_name,
                    "training_type":   TRAINING_TYPE,
                    "data_fraction":   fraction,
                    "n_train_samples": len(train_loader.dataset),
                    "l2_regime":       wd_name,
                    "weight_decay":    weight_decay,
                    "model":           MODEL_NAME,
                    "label_mode":      LABEL_MODE,
                    "num_classes":     NUM_CLASSES,
                    "batch_size":      BATCH_SIZE,
                    "epochs_ran":      len(history["train_loss"]),
                    "best_val_loss":   round(early_stopping.best_loss, 4),
                    "loss":            round(test_stats["epoch_loss"], 4),
                    "accuracy":        round(test_stats["epoch_acc"] * 100, 2),
                    "macro_f1":        round(test_stats["macro_f1"] * 100, 2),
                    "macro_p":         round(test_stats["macro_precision"] * 100, 2),
                    "macro_r":         round(test_stats["macro_recall"] * 100, 2),
                    "train_time_sec":  round(train_time_sec, 1),
                    "train_time_min":  round(train_time_sec / 60, 2),
                    "augmentations":   str(AUGMENTATIONS),
                }

                save_training_curves(
                    history, CHECKPOINT_DIR,
                    title=f"{fraction*100:.0f}% | {wd_name} | {aug_name}",
                )
                save_run_csv(row, CHECKPOINT_DIR)
                save_master_csv(row, CHECKPOINT_BASE, frac_pct)

    print("\n" + "=" * 70)
    print(f"All {total_runs} runs complete.")
