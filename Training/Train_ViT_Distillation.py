"""
Train_ViT_Distillation.py
=========================
Knowledge distillation: ViT-Base (teacher) → ViT-Tiny (student).

Loop structure
--------------
  Outermost : student experiment grid — 4 combinations of:
                  pretrained_student : True | False
                  gradual_unfreeze   : True | False
  Middle    : strategy (baseline | weighted_ce | oversampling)
  Inner     : augmentation (all experiments from experiments_final_augs.json)

Each of the 4 student experiments writes to its own CHECKPOINT_BASE
subdirectory and appends to its own master CSV, so results are kept
fully separate and comparable.

Data pipeline flags (shared across all 4 experiments)
------------------------------------------------------
  USE_CLASS_IMBALANCE — if True, cats are undersampled by CAT_FRACTION
  USE_STRATIFIED_DATA — if True, only DATA_FRACTION of training data is used,
                        sampled with StratifiedShuffleSplit (class-proportional)
Both can be combined. When USE_CLASS_IMBALANCE=False, strategies auto-reduce
to ["baseline"].

CSV/metric saving mirrors Train_ViT_ClassImbalance_GradualUnfreeze.py:
  CHECKPOINT_DIR/checkpoint_student.pt   — best student state_dict
  CHECKPOINT_DIR/experiment_results.csv  — single-row run summary
  CHECKPOINT_DIR/per_class_metrics.csv   — one row per class
  CHECKPOINT_DIR/training_curves.png     — loss / acc / F1 curves
  CHECKPOINT_BASE/all_experiment_results_distillation.csv  — master CSV per experiment
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

from DataLoader          import (BatchAugmenter, CreateDataset,
                                  get_train_transform, get_eval_transform,
                                  _parse_annotation_file, _pet_collate)
from ClassImbalance      import make_weighted_sampler
from DataPipeline        import create_cat_imbalanced_subset
from ViTDistillation     import TeacherViT, StudentViTTiny, LossWeights
from DistillationEngine  import DistillationEngine

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
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT    = "../../../Dataset/"
BATCH_SIZE      = 64
IMAGE_SIZE      = 224
VAL_SPLIT       = 0.2
NUM_WORKERS     = 0
SEED            = 42
WEIGHT_DECAY    = 1e-4
LR_PATIENCE     = 3
LR_FACTOR       = 0.1
LR_MIN          = 1e-6
ES_PATIENCE     = 6
LABEL_MODE      = "breed"

TEACHER_MODEL_NAME = "vit_base_patch16_224"
STUDENT_MODEL_NAME = "vit_tiny_patch16_224"
TRAINING_TYPE      = "Distillation_GradualUnfreeze"
TEACHER_CKPT       = "../Checkpoints/Baseline_Startegy/vit_gradual_unfreeze/vit_base_patch16_224_Gradual_Unfreeze_Multiple_breed_blocks3_erasing_jitter_mixup_cutmix_low_prob/checkpoint.pt"

CHECKPOINT_ROOT = "../Checkpoints/Distillation"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

NUM_CLASSES = 37 if LABEL_MODE == "breed" else 2

# ── Data pipeline flags ───────────────────────────────────────────────────────
USE_CLASS_IMBALANCE = False
CAT_FRACTION        = 0.3    # ignored when USE_CLASS_IMBALANCE=False

USE_STRATIFIED_DATA = False
DATA_FRACTION       = 0.5    # ignored when USE_STRATIFIED_DATA=False

# ── Strategies ────────────────────────────────────────────────────────────────
# weighted_ce and oversampling only make sense with imbalanced data.
# When USE_CLASS_IMBALANCE=False they are automatically dropped.
_ALL_STRATEGIES = ["baseline", "weighted_ce", "oversampling"]
STRATEGIES = _ALL_STRATEGIES if USE_CLASS_IMBALANCE else ["baseline"]

# ── Student experiment grid ───────────────────────────────────────────────────
# All 4 combinations are run sequentially, each writing to its own
# checkpoint subdirectory and its own master CSV.
STUDENT_EXPERIMENTS = {
    "pretrained_gradual": {"pretrained": True,  "gradual_unfreeze": True},
    "pretrained_full":    {"pretrained": True,  "gradual_unfreeze": False},
    "scratch_gradual":    {"pretrained": False, "gradual_unfreeze": True},
    "scratch_full":       {"pretrained": False, "gradual_unfreeze": False},
}

# ── Distillation loss weights ─────────────────────────────────────────────────
LOSS_WEIGHTS = LossWeights(
    w_ce=0.3, w_kl=0.5, w_mse=0.1, w_feat=0.1, temperature=4.0,
)

# ── Gradual unfreeze schedule ─────────────────────────────────────────────────
NUM_BLOCKS   = 12
HEAD_EPOCHS  = 20
BLOCK_EPOCHS = 10
FINAL_EPOCHS = 30
HEAD_LR      = 1e-3
BLOCK_LR     = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# PHASE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
def build_phase_schedule(num_blocks, head_epochs, block_epochs, final_epochs,
                          head_lr, block_lr):
    """
    Phase 0              : head only         (head_epochs, head_lr)
    Phases 1 .. N-1      : one block each    (block_epochs, decaying lr)
    Phase N (last)       : all blocks live   (final_epochs, block_lr * 0.5)
    """
    schedule = [{"epochs": head_epochs, "lr": head_lr}]
    for i in range(num_blocks - 1):
        phase_lr = block_lr if i < (num_blocks // 2 + 1) else block_lr * 0.5
        schedule.append({"epochs": block_epochs, "lr": phase_lr})
    schedule.append({"epochs": final_epochs, "lr": block_lr * 0.5})
    return schedule


PHASE_SCHEDULE = build_phase_schedule(
    NUM_BLOCKS, HEAD_EPOCHS, BLOCK_EPOCHS, FINAL_EPOCHS, HEAD_LR, BLOCK_LR,
)
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
    augs      = {**FIXED_AUGS}
    overrides = exp.pop("override_fixed", {})
    augs.update(overrides)
    augs.update(exp)
    for key in ("crop_scale", "crop_ratio", "erasing_scale"):
        if key in augs and isinstance(augs[key], list):
            augs[key] = tuple(augs[key])
    EXPERIMENTS.append(augs)

total_runs = len(STUDENT_EXPERIMENTS) * len(STRATEGIES) * len(EXPERIMENTS)
print(f"\nLoaded {len(EXPERIMENTS)} augmentation experiments.")
print(f"Student experiments : {list(STUDENT_EXPERIMENTS.keys())}")
print(f"Strategies          : {STRATEGIES}")
print(f"USE_CLASS_IMBALANCE : {USE_CLASS_IMBALANCE}  (cat_fraction={CAT_FRACTION})")
print(f"USE_STRATIFIED_DATA : {USE_STRATIFIED_DATA}  (data_fraction={DATA_FRACTION})")
print(f"Total runs          : {total_runs}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
class LabelSelector:
    """Unwraps the (species, breed) label tuple and yields the requested mode."""
    def __init__(self, loader, mode):
        self.loader = loader
        self.mode   = mode

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, (y1, y2) in self.loader:
            yield x, (y2 if self.mode == "breed" else y1)


class EarlyStopping:
    def __init__(self, patience: int = 5, save_path: str = "best_model.pt",
                 delta: float = 1e-4):
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


def compute_class_weights(train_subset, num_classes, device):
    """Inverse-frequency class weights from a Subset's breed labels."""
    labels = []
    for i in range(len(train_subset)):
        _, (_, breed) = train_subset.dataset[train_subset.indices[i]]
        labels.append(breed.item() if hasattr(breed, "item") else int(breed))
    labels        = np.array(labels)
    class_counts  = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_counts  = np.where(class_counts == 0, 1.0, class_counts)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def build_loaders(dataset_root, augs, image_size, batch_size,
                  val_split, num_workers, seed, strategy,
                  use_class_imbalance, cat_fraction,
                  use_stratified_data, data_fraction):
    """
    Builds train/val/test DataLoaders.
    Step 1 (optional): cat undersampling (USE_CLASS_IMBALANCE).
    Step 2 (optional): stratified subsampling (USE_STRATIFIED_DATA).
    Both steps are composable.

    Returns train_loader, val_loader, test_loader, train_subset.
    """
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

    # ── Step 1: optional class imbalance ──────────────────────────────────────
    if use_class_imbalance:
        train_subset = create_cat_imbalanced_subset(
            full_train_dataset, cat_fraction=cat_fraction, seed=seed,
        )
        n_cat = sum(1 for i in train_subset.indices
                    if full_train_dataset[i][1][0].item() == 0)
        n_dog = len(train_subset) - n_cat
        print(f"  [Imbalance] {len(train_subset)} samples "
              f"(cat={n_cat}, dog={n_dog})")
    else:
        train_subset = Subset(full_train_dataset, list(range(len(full_train_dataset))))

    # ── Step 2: optional stratified subsampling ───────────────────────────────
    if use_stratified_data:
        breed_labels = np.array([
            full_train_dataset[train_subset.indices[i]][1][1].item()
            if hasattr(full_train_dataset[train_subset.indices[i]][1][1], "item")
            else int(full_train_dataset[train_subset.indices[i]][1][1])
            for i in range(len(train_subset))
        ])
        local_indices       = np.arange(len(train_subset))
        splitter            = StratifiedShuffleSplit(
            n_splits=1, train_size=data_fraction, random_state=seed,
        )
        subset_local_idx, _ = next(splitter.split(local_indices, breed_labels))
        global_indices      = [train_subset.indices[i] for i in subset_local_idx]
        train_subset        = Subset(full_train_dataset, global_indices)
        print(f"  [Stratified] {len(train_subset)} samples "
              f"({data_fraction*100:.0f}%, class-proportional)")

    print(f"  Final train set: {len(train_subset)} samples")

    # ── Build loaders ─────────────────────────────────────────────────────────
    if strategy == "oversampling" and use_class_imbalance:
        sampler      = make_weighted_sampler(train_subset)
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            collate_fn=_pet_collate, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            collate_fn=_pet_collate, drop_last=True,
        )

    val_dataset  = CreateDataset(val_records,  images_dir,
                                  transform=get_eval_transform(image_size), one_hot=False)
    test_dataset = CreateDataset(test_records, images_dir,
                                  transform=get_eval_transform(image_size), one_hot=False)

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=_pet_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=_pet_collate)

    return train_loader, val_loader, test_loader, train_subset


def evaluate_per_class(student, loader_selector, device, num_classes):
    """Full per-class evaluation. student.eval() → plain logits (no tuple)."""
    from sklearn.metrics import precision_recall_fscore_support
    student.eval()
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for x, y in loader_selector:
            x      = x.to(device, memory_format=torch.channels_last, non_blocking=True)
            logits = student(x)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_targets.append(y.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0,
    )
    return {
        "accuracy":        float((y_pred == y_true).mean()),
        "macro_f1":        float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall":    float(macro_r),
        "per_class": {
            "precision": per_p.tolist(), "recall": per_r.tolist(),
            "f1": per_f1.tolist(), "support": per_support.tolist(),
        },
    }


def flatten_per_class(per_class, num_classes):
    flat = {}
    for c in range(num_classes):
        flat[f"p_c{c}"]   = round(per_class["precision"][c], 4)
        flat[f"r_c{c}"]   = round(per_class["recall"][c], 4)
        flat[f"f1_c{c}"]  = round(per_class["f1"][c], 4)
        flat[f"sup_c{c}"] = int(per_class["support"][c])
    return flat


def save_per_class_csv(per_class: dict, checkpoint_dir: str, run_meta: dict):
    csv_path   = os.path.join(checkpoint_dir, "per_class_metrics.csv")
    fieldnames = list(run_meta.keys()) + ["class_idx", "precision", "recall", "f1", "support"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls_idx in range(len(per_class["f1"])):
            writer.writerow({
                **run_meta,
                "class_idx": cls_idx,
                "precision": round(per_class["precision"][cls_idx], 4),
                "recall":    round(per_class["recall"][cls_idx], 4),
                "f1":        round(per_class["f1"][cls_idx], 4),
                "support":   int(per_class["support"][cls_idx]),
            })
    print(f"  Per-class CSV saved to {csv_path}")


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

    plt.suptitle(f"ViT Distillation Gradual — {title}", fontsize=11)
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


def save_master_csv(results: dict, checkpoint_base: str):
    csv_path    = os.path.join(checkpoint_base, "all_experiment_results_distillation.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"  Master CSV appended: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP   outermost: student experiment → strategy → augmentation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    run_counter = 0

    for exp_label, student_flags in STUDENT_EXPERIMENTS.items():

        pretrained_student = student_flags["pretrained"]
        gradual_unfreeze   = student_flags["gradual_unfreeze"]

        # Each student experiment gets its own checkpoint dir and master CSV
        CHECKPOINT_BASE = os.path.join(
            CHECKPOINT_ROOT, f"vit_tiny_distillation_{exp_label}/",
        )
        os.makedirs(CHECKPOINT_BASE, exist_ok=True)

        # training_type encodes the student flags for dir/CSV naming
        training_type = (
            f"Distillation_"
            f"{'pretrained' if pretrained_student else 'scratch'}_"
            f"{'gradual' if gradual_unfreeze else 'full'}"
        )

        print("\n" + "▓" * 70)
        print(f"STUDENT EXPERIMENT : {exp_label.upper()}")
        print(f"  pretrained={pretrained_student}  gradual_unfreeze={gradual_unfreeze}")
        print(f"  checkpoint base → {CHECKPOINT_BASE}")
        print("▓" * 70)

        for strategy in STRATEGIES:

            print("\n" + "█" * 70)
            print(f"STRATEGY : {strategy.upper()}")
            print("█" * 70)

            for AUGMENTATIONS in EXPERIMENTS:
                run_counter += 1
                aug_name  = AUGMENTATIONS["name"]
                run_label = f"{strategy}_{aug_name}"

                print("\n" + "=" * 70)
                print(f"RUN {run_counter}/{total_runs}  |  "
                      f"exp={exp_label}  strategy={strategy}  aug={aug_name}")
                print("=" * 70)

                CHECKPOINT_DIR = os.path.join(
                    CHECKPOINT_BASE,
                    f"{STUDENT_MODEL_NAME}_{training_type}_{LABEL_MODE}_{run_label}",
                )
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)

                # ── Data ──────────────────────────────────────────────────────
                train_loader, val_loader, test_loader, train_subset = build_loaders(
                    dataset_root        = DATASET_ROOT,
                    augs                = AUGMENTATIONS,
                    image_size          = IMAGE_SIZE,
                    batch_size          = BATCH_SIZE,
                    val_split           = VAL_SPLIT,
                    num_workers         = NUM_WORKERS,
                    seed                = SEED,
                    strategy            = strategy,
                    use_class_imbalance = USE_CLASS_IMBALANCE,
                    cat_fraction        = CAT_FRACTION,
                    use_stratified_data = USE_STRATIFIED_DATA,
                    data_fraction       = DATA_FRACTION,
                )

                train_selector = LabelSelector(train_loader, LABEL_MODE)
                val_selector   = LabelSelector(val_loader,   LABEL_MODE)
                test_selector  = LabelSelector(test_loader,  LABEL_MODE)

                # ── Models ────────────────────────────────────────────────────
                teacher = TeacherViT(
                    checkpoint_path = TEACHER_CKPT,
                    num_classes     = NUM_CLASSES,
                )
                student = StudentViTTiny(
                    num_classes      = NUM_CLASSES,
                    pretrained       = pretrained_student,
                    gradual_unfreeze = gradual_unfreeze,
                )

                batch_augmenter = BatchAugmenter(
                    augs=AUGMENTATIONS, num_classes=NUM_CLASSES,
                )

                early_stopping = EarlyStopping(
                    patience  = ES_PATIENCE,
                    save_path = os.path.join(CHECKPOINT_DIR, "checkpoint_student.pt"),
                )

                history = {
                    "train_loss": [], "train_acc": [],
                    "val_loss":   [], "val_acc":   [], "val_f1": [],
                }

                phase_boundaries = []
                total_blocks_vit = student.total_block_count
                global_epoch     = 0
                terminate_run    = False
                train_start      = time.time()

                # ── Phase loop ────────────────────────────────────────────────
                for phase_idx, phase in enumerate(PHASE_SCHEDULE):
                    if terminate_run:
                        break

                    is_last_phase = (phase_idx == LAST_PHASE_IDX)

                    if phase_idx > 0:
                        block_to_unfreeze = total_blocks_vit - phase_idx
                        if block_to_unfreeze >= 0:
                            student.unfreeze_next_block(block_to_unfreeze)
                            phase_boundaries.append(global_epoch)
                            print(f"\n  ▶ Unfroze student block {block_to_unfreeze} "
                                  f"(phase {phase_idx}/{LAST_PHASE_IDX})")

                    # Build loss weights — wire in class weights for weighted_ce
                    if strategy == "weighted_ce" and USE_CLASS_IMBALANCE:
                        class_weights = compute_class_weights(
                            train_subset, NUM_CLASSES, device,
                        )
                        print(f"  Weighted CE: min={class_weights.min():.4f}  "
                              f"max={class_weights.max():.4f}")
                        from dataclasses import replace as dc_replace
                        run_loss_weights = dc_replace(
                            LOSS_WEIGHTS, ce_class_weights=class_weights,
                        )
                    else:
                        run_loss_weights = LOSS_WEIGHTS

                    # Optimizer: student params only — teacher fully frozen
                    optimizer = torch.optim.Adam(
                        student.trainable_params(),
                        lr=phase["lr"], weight_decay=WEIGHT_DECAY,
                    )
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=LR_FACTOR,
                        patience=LR_PATIENCE, min_lr=LR_MIN, verbose=True,
                    )

                    engine = DistillationEngine(
                        teacher        = teacher,
                        student        = student,
                        data_iterator  = train_selector,
                        loss_weights   = run_loss_weights,
                        opt            = optimizer,
                        compute_device = device,
                        scheduler      = None,
                        batch_aug      = batch_augmenter,
                    )

                    early_stopping.reset_counter()

                    trainable = sum(
                        p.numel() for p in student.parameters() if p.requires_grad
                    )
                    print(f"\n{'─'*70}")
                    print(f"Phase {phase_idx}/{LAST_PHASE_IDX} | "
                          f"LR={phase['lr']:.0e} | Strategy={strategy} | "
                          f"Student unfrozen={student.unfrozen_block_count} | "
                          f"Student trainable={trainable:,} | "
                          f"{'FINAL' if is_last_phase else 'intermediate'}")
                    print(f"{'─'*70}")

                    for phase_epoch in range(1, phase["epochs"] + 1):
                        global_epoch += 1
                        print(f"\n── Phase {phase_idx} | "
                              f"Epoch {phase_epoch}/{phase['epochs']} "
                              f"(global {global_epoch}) ──")

                        train_stats = engine.train_one_epoch(
                            epoch_num=global_epoch, print_freq=50,
                        )
                        val_stats = engine.evaluate(
                            val_selector, phase_label="Validation",
                        )

                        history["train_loss"].append(train_stats["epoch_loss"])
                        history["train_acc"].append(train_stats["epoch_acc"])
                        history["val_loss"].append(val_stats["epoch_loss"])
                        history["val_acc"].append(val_stats["epoch_acc"])
                        history["val_f1"].append(val_stats["macro_f1"])

                        scheduler.step(val_stats["epoch_loss"])
                        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

                        early_stopping(val_stats["epoch_loss"], student)

                        if early_stopping.triggered:
                            if is_last_phase:
                                print(f"\n  ✖ Early stopping on final phase "
                                      f"(global epoch {global_epoch}). Terminating.")
                                terminate_run = True
                            else:
                                print(f"\n  ⏭  Early stopping on intermediate "
                                      f"phase {phase_idx}. Advancing to next block.")
                            break

                train_time_sec = time.time() - train_start

                # ── Test evaluation ───────────────────────────────────────────
                print("\nLoading best student checkpoint for test evaluation ...")
                student.load_state_dict(torch.load(
                    os.path.join(CHECKPOINT_DIR, "checkpoint_student.pt"),
                    map_location=device,
                ))
                student.to(device)
                test_results = evaluate_per_class(
                    student, test_selector, device, NUM_CLASSES,
                )

                print("\n── Final Test Results ──────────────────────────────")
                print(f"  Exp        : {exp_label}  |  Strategy: {strategy}  "
                      f"|  Aug: {aug_name}")
                print(f"  Accuracy   : {test_results['accuracy']*100:.2f}%")
                print(f"  Macro F1   : {test_results['macro_f1']*100:.2f}%")
                print(f"  Train time : {train_time_sec/60:.1f} min")

                per_f1  = test_results["per_class"]["f1"]
                bottom5 = sorted(range(NUM_CLASSES), key=lambda c: per_f1[c])[:5]
                print("  Bottom-5 classes by F1:")
                for cls in bottom5:
                    print(f"    class {cls:2d} → F1={per_f1[cls]:.3f}  "
                          f"support={int(test_results['per_class']['support'][cls])}")

                # ── CSV saving ────────────────────────────────────────────────
                run_meta = {
                    "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "experiment":  aug_name,
                    "strategy":    strategy,
                    "student_exp": exp_label,
                    "model":       STUDENT_MODEL_NAME,
                }
                per_class_flat = flatten_per_class(
                    test_results["per_class"], NUM_CLASSES,
                )

                row = {
                    "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "experiment":          aug_name,
                    "training_type":       training_type,
                    "student_exp":         exp_label,
                    "strategy":            strategy,
                    "teacher_model":       TEACHER_MODEL_NAME,
                    "student_model":       STUDENT_MODEL_NAME,
                    "pretrained_student":  pretrained_student,
                    "gradual_unfreeze":    gradual_unfreeze,
                    "use_class_imbalance": USE_CLASS_IMBALANCE,
                    "cat_fraction":        CAT_FRACTION if USE_CLASS_IMBALANCE else 1.0,
                    "use_stratified_data": USE_STRATIFIED_DATA,
                    "data_fraction":       DATA_FRACTION if USE_STRATIFIED_DATA else 1.0,
                    "n_train_samples":     len(train_subset),
                    "loss_w_ce":           LOSS_WEIGHTS.w_ce,
                    "loss_w_kl":           LOSS_WEIGHTS.w_kl,
                    "loss_w_mse":          LOSS_WEIGHTS.w_mse,
                    "loss_w_feat":         LOSS_WEIGHTS.w_feat,
                    "temperature":         LOSS_WEIGHTS.temperature,
                    "num_blocks":          NUM_BLOCKS,
                    "phases_completed":    len(phase_boundaries) + 1,
                    "blocks_unfrozen":     student.unfrozen_block_count,
                    "label_mode":          LABEL_MODE,
                    "num_classes":         NUM_CLASSES,
                    "batch_size":          BATCH_SIZE,
                    "epochs_ran":          global_epoch,
                    "best_val_loss":       round(early_stopping.best_loss, 4),
                    "accuracy":            round(test_results["accuracy"] * 100, 2),
                    "macro_f1":            round(test_results["macro_f1"] * 100, 2),
                    "macro_p":             round(test_results["macro_precision"] * 100, 2),
                    "macro_r":             round(test_results["macro_recall"] * 100, 2),
                    "train_time_sec":      round(train_time_sec, 1),
                    "train_time_min":      round(train_time_sec / 60, 2),
                    **per_class_flat,
                    "augmentations":       str(AUGMENTATIONS),
                    "phase_schedule":      str(PHASE_SCHEDULE),
                }

                save_training_curves(
                    history, CHECKPOINT_DIR,
                    title=f"{exp_label} | {strategy} | {aug_name}",
                    phase_boundaries=phase_boundaries,
                )
                save_per_class_csv(test_results["per_class"], CHECKPOINT_DIR, run_meta)
                save_run_csv(row, CHECKPOINT_DIR)
                save_master_csv(row, CHECKPOINT_BASE)

    print("\n" + "=" * 70)
    print(f"All {total_runs} runs complete.")
    print(f"Results per experiment in: {CHECKPOINT_ROOT}/")