import sys
import json
import os
import csv
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "../Models")
sys.path.insert(0, "../DataLoader")

from DataLoader     import (BatchAugmenter, CreateDataset,
                             get_train_transform, get_eval_transform,
                             _parse_annotation_file, _pet_collate)
from ClassImbalance import make_weighted_sampler
from DataPipeline   import create_cat_imbalanced_subset
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
WEIGHT_DECAY    = 1e-4
LR_PATIENCE     = 3
LR_FACTOR       = 0.1
LR_MIN          = 1e-6
ES_PATIENCE     = 6
LABEL_MODE      = "breed"
MODEL_NAME      = "vit_base_patch16_224"
TRAINING_TYPE   = "ClassImbalance_Linear"
CHECKPOINT_ROOT = "../Checkpoints/"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
CHECKPOINT_BASE = os.path.join(CHECKPOINT_ROOT, "vit_class_imbalance_linear/")
os.makedirs(CHECKPOINT_BASE, exist_ok=True)
NUM_CLASSES     = 37 if LABEL_MODE == "breed" else 2
CAT_FRACTION    = 0.2
STRATEGIES      = ["baseline", "weighted_ce", "oversampling"]


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

total_runs = len(STRATEGIES) * len(EXPERIMENTS)
print(f"\nLoaded {len(EXPERIMENTS)} augmentation experiments.")
print(f"Strategies : {STRATEGIES}")
print(f"Cat fraction retained : {CAT_FRACTION*100:.0f}%")
print(f"Total runs : {total_runs}")


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


def compute_class_weights(imbalanced_subset, num_classes, device):
    """Inverse-frequency weights over breed labels in the imbalanced subset."""
    labels = []
    for i in range(len(imbalanced_subset)):
        _, (_, breed) = imbalanced_subset.dataset[imbalanced_subset.indices[i]]
        labels.append(breed.item() if hasattr(breed, "item") else int(breed))
    labels        = np.array(labels)
    class_counts  = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_counts  = np.where(class_counts == 0, 1.0, class_counts)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def build_imbalanced_loaders(dataset_root, augs, image_size, batch_size,
                              val_split, num_workers, seed, strategy):
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

    imbalanced_train = create_cat_imbalanced_subset(
        full_train_dataset, cat_fraction=CAT_FRACTION, seed=seed,
    )
    n_cat = sum(1 for i in imbalanced_train.indices
                if full_train_dataset[i][1][0].item() == 0)
    n_dog = len(imbalanced_train) - n_cat
    print(f"  Imbalanced train: {len(imbalanced_train)} samples  "
          f"(cat={n_cat}, dog={n_dog})")

    if strategy == "oversampling":
        sampler      = make_weighted_sampler(imbalanced_train)
        train_loader = DataLoader(imbalanced_train, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=True, collate_fn=_pet_collate, drop_last=True)
    else:
        train_loader = DataLoader(imbalanced_train, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True, collate_fn=_pet_collate, drop_last=True)

    val_dataset  = CreateDataset(val_records,  images_dir, transform=get_eval_transform(image_size), one_hot=False)
    test_dataset = CreateDataset(test_records, images_dir, transform=get_eval_transform(image_size), one_hot=False)

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=_pet_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=_pet_collate)
    return train_loader, val_loader, test_loader, imbalanced_train


def evaluate_per_class(model, loader_selector, device, num_classes):
    from sklearn.metrics import precision_recall_fscore_support
    model.eval()
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for x, y in loader_selector:
            x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
            logits = model(x)
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
    """
    Flattens nested per-class dict into wide-format columns for a single CSV row.
    Produces: p_c0..p_c36, r_c0..r_c36, f1_c0..f1_c36, sup_c0..sup_c36
    """
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
            writer.writerow({**run_meta,
                             "class_idx": cls_idx,
                             "precision": round(per_class["precision"][cls_idx], 4),
                             "recall":    round(per_class["recall"][cls_idx], 4),
                             "f1":        round(per_class["f1"][cls_idx], 4),
                             "support":   int(per_class["support"][cls_idx])})
    print(f"  Per-class CSV saved to {csv_path}")


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

    plt.suptitle(f"ViT ClassImbalance Linear — {title}", fontsize=11)
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
    csv_path    = os.path.join(checkpoint_base,
                               "all_experiment_results_linear_class_imbalance.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"  Master CSV appended: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP  outer: strategy → augmentation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    run_counter = 0

    for strategy in STRATEGIES:

        print("\n" + "█" * 70)
        print(f"STRATEGY : {strategy.upper()}")
        print("█" * 70)

        for AUGMENTATIONS in EXPERIMENTS:
            run_counter += 1
            aug_name    = AUGMENTATIONS["name"]
            run_label   = f"{strategy}_{aug_name}"

            print("\n" + "=" * 70)
            print(f"RUN {run_counter}/{total_runs}  |  strategy={strategy}  aug={aug_name}")
            print("=" * 70)

            CHECKPOINT_DIR = os.path.join(
                CHECKPOINT_BASE,
                f"{MODEL_NAME}_{TRAINING_TYPE}_{LABEL_MODE}_{run_label}"
            )
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            train_loader, val_loader, test_loader, imbalanced_train = \
                build_imbalanced_loaders(
                    dataset_root=DATASET_ROOT, augs=AUGMENTATIONS,
                    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                    val_split=VAL_SPLIT, num_workers=NUM_WORKERS,
                    seed=SEED, strategy=strategy,
                )

            train_selector = LabelSelector(train_loader, LABEL_MODE)
            val_selector   = LabelSelector(val_loader,   LABEL_MODE)
            test_selector  = LabelSelector(test_loader,  LABEL_MODE)

            if strategy == "weighted_ce":
                class_weights = compute_class_weights(imbalanced_train, NUM_CLASSES, device)
                loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
                print(f"  Weighted CE: min={class_weights.min():.4f}  max={class_weights.max():.4f}")
            else:
                loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

            model     = ViTLinearProbe(num_classes=NUM_CLASSES, model_name=MODEL_NAME)
            optimizer = torch.optim.Adam(
                model.trainable_params(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=LR_FACTOR,
                patience=LR_PATIENCE, min_lr=LR_MIN, verbose=True,
            )
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
            test_results = evaluate_per_class(model, test_selector, device, NUM_CLASSES)

            print("\n── Final Test Results ──────────────────────────────")
            print(f"  Strategy   : {strategy}  |  Aug: {aug_name}")
            print(f"  Accuracy   : {test_results['accuracy']*100:.2f}%")
            print(f"  Macro F1   : {test_results['macro_f1']*100:.2f}%")
            print(f"  Train time : {train_time_sec/60:.1f} min")

            per_f1  = test_results["per_class"]["f1"]
            bottom5 = sorted(range(NUM_CLASSES), key=lambda c: per_f1[c])[:5]
            print("  Bottom-5 classes by F1:")
            for cls in bottom5:
                print(f"    class {cls:2d} → F1={per_f1[cls]:.3f}  "
                      f"support={int(test_results['per_class']['support'][cls])}")

            run_meta = {
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "experiment": aug_name,
                "strategy":   strategy,
                "model":      MODEL_NAME,
            }
            per_class_flat = flatten_per_class(test_results["per_class"], NUM_CLASSES)

            row = {
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "experiment":       aug_name,
                "training_type":    TRAINING_TYPE,
                "strategy":         strategy,
                "cat_fraction":     CAT_FRACTION,
                "model":            MODEL_NAME,
                "label_mode":       LABEL_MODE,
                "num_classes":      NUM_CLASSES,
                "batch_size":       BATCH_SIZE,
                "epochs_ran":       len(history["train_loss"]),
                "best_val_loss":    round(early_stopping.best_loss, 4),
                "accuracy":         round(test_results["accuracy"] * 100, 2),
                "macro_f1":         round(test_results["macro_f1"] * 100, 2),
                "macro_p":          round(test_results["macro_precision"] * 100, 2),
                "macro_r":          round(test_results["macro_recall"] * 100, 2),
                "train_time_sec":   round(train_time_sec, 1),
                "train_time_min":   round(train_time_sec / 60, 2),
                **per_class_flat,
                "augmentations":    str(AUGMENTATIONS),
            }

            save_training_curves(history, CHECKPOINT_DIR, title=f"{strategy} | {aug_name}")
            save_per_class_csv(test_results["per_class"], CHECKPOINT_DIR, run_meta)
            save_run_csv(row, CHECKPOINT_DIR)
            save_master_csv(row, CHECKPOINT_BASE)

    print("\n" + "=" * 70)
    print(f"All {total_runs} runs complete.")
    print(f"Results CSV : {os.path.join(CHECKPOINT_BASE, 'all_experiment_results_linear_class_imbalance.csv')}")
