import sys
import json
import torch
import torch.nn as nn
import os
import csv
from datetime import datetime

sys.path.insert(0, "../Models")
sys.path.insert(0, "../DataLoader")

from DataLoader     import build_dataloaders, BatchAugmenter
from ViTFinetune    import ViTLinearProbe
from TrainingEngine import TrainingEngine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
WEIGHT_DECAY    = 1e-4
LR_PATIENCE     = 3
LR_FACTOR       = 0.1
LR_MIN          = 1e-6
ES_PATIENCE     = 6
LABEL_MODE      = "breed"
MODEL_NAME      = "vit_base_patch16_224"
TRAINING_TYPE   = "Linear_Probe"
CHECKPOINT_ROOT = "../Checkpoints/"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
CHECKPOINT_BASE = CHECKPOINT_ROOT+"vit_linear_probe/"
os.makedirs(CHECKPOINT_BASE, exist_ok=True)
NUM_CLASSES     = 37 if LABEL_MODE == "breed" else 2


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EXPERIMENTS FROM JSON
# ─────────────────────────────────────────────────────────────────────────────
with open("experiments.json", "r") as f:
    exp_config = json.load(f)

FIXED_AUGS  = exp_config["fixed"]

# tuples are serialised as lists in JSON — convert crop_scale/crop_ratio back
for key in ("crop_scale", "crop_ratio"):
    if key in FIXED_AUGS:
        FIXED_AUGS[key] = tuple(FIXED_AUGS[key])

EXPERIMENTS = []
for exp in exp_config["experiments"]:
    # start with fixed augs
    augs = {**FIXED_AUGS}

    # apply any fixed overrides first
    overrides = exp.pop("override_fixed", {})
    augs.update(overrides)

    # then merge experiment-specific keys
    augs.update(exp)

    # convert lists back to tuples
    for key in ("crop_scale", "crop_ratio", "erasing_scale"):
        if key in augs and isinstance(augs[key], list):
            augs[key] = tuple(augs[key])

    EXPERIMENTS.append(augs)

print(f"\nLoaded {len(EXPERIMENTS)} experiments from experiments.json")


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
            print(f"  ✔ New best val_loss={val_loss:.4f} — checkpoint saved to '{self.save_path}'")
        else:
            self.counter += 1
            print(f"  Early-stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                print("  ✖ Early stopping triggered.")


def save_training_curves(history, checkpoint_dir, exp_name):
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

    plt.suptitle(f"ViT Linear Probe — {exp_name}", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")


def save_results_csv(results: dict, checkpoint_base: str):
    csv_path    = os.path.join(checkpoint_base, "all_experiment_results.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"  Results appended to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────


for exp_idx, AUGMENTATIONS in enumerate(EXPERIMENTS):
    exp_name = AUGMENTATIONS["name"]

    print("\n" + "=" * 70)
    print(f"EXPERIMENT {exp_idx + 1}/{len(EXPERIMENTS)} — {exp_name}")
    print("=" * 70)

    # ── checkpoint dir ────────────────────────────────────────────────────────
    CHECKPOINT_DIR = os.path.join(
        CHECKPOINT_BASE,
        f"{MODEL_NAME}_{TRAINING_TYPE}_{LABEL_MODE}_{exp_name}"
    )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Checkpoint dir : {os.path.abspath(CHECKPOINT_DIR)}")

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root = DATASET_ROOT,
        val_split    = VAL_SPLIT,
        batch_size   = BATCH_SIZE,
        one_hot      = False,
        image_size   = IMAGE_SIZE,
        augs         = AUGMENTATIONS,
        num_workers  = NUM_WORKERS,
        seed         = SEED,
    )

    train_selector = LabelSelector(train_loader, LABEL_MODE)
    val_selector   = LabelSelector(val_loader,   LABEL_MODE)
    test_selector  = LabelSelector(test_loader,  LABEL_MODE)

    # ── fresh model for every experiment ─────────────────────────────────────
    model = ViTLinearProbe(num_classes=NUM_CLASSES, model_name=MODEL_NAME)

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.trainable_params(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        factor    = LR_FACTOR,
        patience  = LR_PATIENCE,
        min_lr    = LR_MIN,
        verbose   = True,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── engine ────────────────────────────────────────────────────────────────
    batch_augmenter = BatchAugmenter(augs=AUGMENTATIONS, num_classes=NUM_CLASSES)
    engine = TrainingEngine(
        network        = model,
        data_iterator  = train_selector,
        loss_fn        = loss_fn,
        opt            = optimizer,
        compute_device = device,
        scheduler      = None,
        batch_aug      = batch_augmenter,
    )

    early_stopping = EarlyStopping(
        patience  = ES_PATIENCE,
        save_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pt"),
    )

    # ── training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   [], "val_f1": []}

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

    # ── test evaluation ───────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation ...")
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, "checkpoint.pt"),
        map_location=device,
    ))
    model.to(device)
    test_stats = engine.evaluate(test_selector, phase_label="Test")

    print("\n── Final Test Results ──────────────────────────────")
    print(f"  Loss      : {test_stats['epoch_loss']:.4f}")
    print(f"  Accuracy  : {test_stats['epoch_acc']*100:.2f}%")
    print(f"  Macro F1  : {test_stats['macro_f1']*100:.2f}%")
    print(f"  Macro P   : {test_stats['macro_precision']*100:.2f}%")
    print(f"  Macro R   : {test_stats['macro_recall']*100:.2f}%")

    # ── save curves & results ─────────────────────────────────────────────────
    save_training_curves(history, CHECKPOINT_DIR, exp_name)
    save_results_csv({
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment":    exp_name,
        "model":         MODEL_NAME,
        "label_mode":    LABEL_MODE,
        "num_classes":   NUM_CLASSES,
        "batch_size":    BATCH_SIZE,
        "epochs_ran":    len(history["train_loss"]),
        "best_val_loss": round(early_stopping.best_loss, 4),
        "loss":          round(test_stats["epoch_loss"], 4),
        "accuracy":      round(test_stats["epoch_acc"] * 100, 2),
        "macro_f1":      round(test_stats["macro_f1"] * 100, 2),
        "macro_p":       round(test_stats["macro_precision"] * 100, 2),
        "macro_r":       round(test_stats["macro_recall"] * 100, 2),
        "augmentations": str(AUGMENTATIONS),
    }, CHECKPOINT_BASE)

print("\n" + "=" * 70)
print(f"All {len(EXPERIMENTS)} experiments complete.")
print(f"Results CSV : {os.path.join(CHECKPOINT_BASE, TRAINING_TYPE+'_all_experiment_results.csv')}")
