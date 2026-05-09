import sys
import json
import torch
import torch.nn as nn
import os
import csv
from datetime import datetime

sys.path.insert(0, "../Models")
sys.path.insert(0, "../DataLoader")
sys.path.insert(0, "../Training")

from DataLoader     import build_dataloaders, BatchAugmenter
from ViTFinetune    import ViTGradualUnfreeze
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
MODEL_NAME      = "vit_base_patch16_224"
TRAINING_TYPE   = "Gradual_Unfreeze_Multiple"
CHECKPOINT_ROOT = "../Checkpoints/"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
CHECKPOINT_BASE = CHECKPOINT_ROOT + "vit_gradual_unfreeze_multiple/"
os.makedirs(CHECKPOINT_BASE, exist_ok=True)
NUM_CLASSES     = 37 if LABEL_MODE == "breed" else 2

# ── block-count experiment configs ────────────────────────────────────────────
BLOCK_EXPERIMENTS = [
    {"num_blocks": 1, "head_epochs": 20, "block_epochs": 20, "head_lr": 1e-3, "block_lr": 1e-4},
    {"num_blocks": 2, "head_epochs": 20, "block_epochs": 20, "head_lr": 1e-3, "block_lr": 1e-4},
    {"num_blocks": 3, "head_epochs": 20, "block_epochs": 20, "head_lr": 1e-3, "block_lr": 1e-4},
    {"num_blocks": 5, "head_epochs": 20, "block_epochs": 20, "head_lr": 1e-3, "block_lr": 1e-4},
    {"num_blocks": 6, "head_epochs": 20, "block_epochs": 20, "head_lr": 1e-3, "block_lr": 1e-4},
]


def build_phase_schedule(num_blocks, head_epochs, block_epochs, head_lr, block_lr):
    """
    Phase 0  : head only.
    Phase 1‥N: one new block unfrozen per phase.
    LR is halved for the deeper half of block phases.
    """
    schedule = [{"epochs": head_epochs, "lr": head_lr}]
    for i in range(num_blocks):
        phase_lr = block_lr if i < (num_blocks // 2 + 1) else block_lr * 0.5
        schedule.append({"epochs": block_epochs, "lr": phase_lr})
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# LOAD AUGMENTATION EXPERIMENTS FROM JSON
# ─────────────────────────────────────────────────────────────────────────────
with open("experiments.json", "r") as f:
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

print(f"\nLoaded {len(EXPERIMENTS)} augmentation experiments from experiments.json")
print(f"Block-count experiments: {[b['num_blocks'] for b in BLOCK_EXPERIMENTS]}")
print(f"Total runs: {len(BLOCK_EXPERIMENTS) * len(EXPERIMENTS)}")


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
        self.triggered = False   # True when patience exhausted this phase

    def reset_counter(self):
        """Reset patience counter for a new phase.
        best_loss is intentionally preserved — the global best checkpoint
        is never overwritten by a worse model after phase transition."""
        self.counter   = 0
        self.triggered = False
        print("  Early-stop counter reset for new phase.")

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            self.triggered = False
            torch.save(model.state_dict(), self.save_path)
            print(f" New best val_loss={val_loss:.4f} — checkpoint saved to '{self.save_path}'")
        else:
            self.counter += 1
            print(f"  Early-stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
                print(" Early stopping triggered.")


def save_training_curves(history, checkpoint_dir, title, phase_boundaries):
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes  = plt.subplots(1, 3, figsize=(16, 4))

    for ax, train_key, val_key, label in [
        (axes[0], "train_loss", "val_loss", "Loss"),
        (axes[1], "train_acc",  "val_acc",  "Accuracy"),
    ]:
        ax.plot(epochs_ran, history[train_key], label="Train")
        ax.plot(epochs_ran, history[val_key],   label="Val")
        for boundary in phase_boundaries:
            ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(label); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True)

    axes[2].plot(epochs_ran, history["val_f1"], label="Val Macro F1", color="green")
    for boundary in phase_boundaries:
        axes[2].axvline(x=boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[2].set_title("Validation Macro F1"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True)

    plt.suptitle(f"ViT Gradual Unfreeze — {title}", fontsize=13)
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
# MAIN LOOP  —  outer: block-count  |  inner: augmentation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    total_runs  = len(BLOCK_EXPERIMENTS) * len(EXPERIMENTS)
    run_counter = 0

    for blk_cfg in BLOCK_EXPERIMENTS:
        num_blocks     = blk_cfg["num_blocks"]
        PHASE_SCHEDULE = build_phase_schedule(
            num_blocks   = num_blocks,
            head_epochs  = blk_cfg["head_epochs"],
            block_epochs = blk_cfg["block_epochs"],
            head_lr      = blk_cfg["head_lr"],
            block_lr     = blk_cfg["block_lr"],
        )
        last_phase_idx = len(PHASE_SCHEDULE) - 1   # index of the final phase

        print("\n" + "█" * 70)
        print(f"BLOCK EXPERIMENT — unfreeze {num_blocks} block(s)")
        print(f"Phase schedule   — {len(PHASE_SCHEDULE)} phases, "
              f"total budget ≈ {sum(p['epochs'] for p in PHASE_SCHEDULE)} epochs (before early stopping)")
        print("█" * 70)

        for aug_idx, AUGMENTATIONS in enumerate(EXPERIMENTS):
            run_counter += 1
            aug_name     = AUGMENTATIONS["name"]
            run_label    = f"blocks{num_blocks}_{aug_name}"

            print("\n" + "=" * 70)
            print(f"RUN {run_counter}/{total_runs}  —  {num_blocks} block(s)  |  aug: {aug_name}")
            print("=" * 70)

            # ── checkpoint dir ────────────────────────────────────────────────
            CHECKPOINT_DIR = os.path.join(
                CHECKPOINT_BASE,
                f"{MODEL_NAME}_{TRAINING_TYPE}_{LABEL_MODE}_{run_label}"
            )
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            print(f"Checkpoint dir : {os.path.abspath(CHECKPOINT_DIR)}")

            # ── dataloaders ───────────────────────────────────────────────────
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

            # ── fresh model ───────────────────────────────────────────────────
            model           = ViTGradualUnfreeze(num_classes=NUM_CLASSES, model_name=MODEL_NAME)
            loss_fn         = nn.CrossEntropyLoss(label_smoothing=0.1)
            batch_augmenter = BatchAugmenter(augs=AUGMENTATIONS, num_classes=NUM_CLASSES)

            early_stopping  = EarlyStopping(
                patience  = ES_PATIENCE,
                save_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pt"),
            )

            history = {
                "train_loss": [], "train_acc": [],
                "val_loss":   [], "val_acc":   [], "val_f1": [],
            }

            phase_boundaries  = []
            total_blocks      = model.total_block_count   # 12 for ViT-B/16
            global_epoch      = 0
            terminate_run     = False  # set True only when ES fires on the last phase

            # ── phase loop ────────────────────────────────────────────────────
            for phase_idx, phase in enumerate(PHASE_SCHEDULE):
                if terminate_run:
                    break

                is_last_phase = (phase_idx == last_phase_idx)

                # Phase 0 = head only; phases 1..N each unfreeze the next block
                if phase_idx > 0:
                    block_to_unfreeze = total_blocks - phase_idx   # 11, 10, 9 ...
                    if block_to_unfreeze >= 0:
                        model.unfreeze_next_block(block_to_unfreeze)
                        phase_boundaries.append(global_epoch)
                        print(f"\n  ▶ Unfroze block {block_to_unfreeze} "
                              f"(phase {phase_idx}/{last_phase_idx})")

                # Re-instantiate optimizer with current trainable params + new LR
                optimizer = torch.optim.Adam(
                    model.trainable_params(),
                    lr           = phase["lr"],
                    weight_decay = WEIGHT_DECAY,
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode     = "min",
                    factor   = LR_FACTOR,
                    patience = LR_PATIENCE,
                    min_lr   = LR_MIN,
                    verbose  = True,
                )

                engine = TrainingEngine(
                    network        = model,
                    data_iterator  = train_selector,
                    loss_fn        = loss_fn,
                    opt            = optimizer,
                    compute_device = device,
                    scheduler      = None,
                    batch_aug      = batch_augmenter,
                )

                # Reset per-phase patience counter on every phase transition
                early_stopping.reset_counter()

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"\n{'─'*70}")
                print(f"Phase {phase_idx}/{last_phase_idx} | LR={phase['lr']:.0e} | "
                      f"Epochs={phase['epochs']} | "
                      f"Unfrozen blocks={model.unfrozen_block_count} | "
                      f"Trainable params={trainable:,} | "
                      f"{'FINAL PHASE' if is_last_phase else 'intermediate'}")
                print(f"{'─'*70}")

                for phase_epoch in range(1, phase["epochs"] + 1):
                    global_epoch += 1
                    print(f"\n── Phase {phase_idx} | Epoch {phase_epoch}/{phase['epochs']} "
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
                            # No more blocks to unfreeze — stop the whole run
                            print(f"\n  ✖ Early stopping on final phase "
                                  f"(global epoch {global_epoch}). Terminating run.")
                            terminate_run = True
                        else:
                            # More blocks remain — skip to the next phase
                            print(f"\n  ⏭  Early stopping on intermediate phase {phase_idx} "
                                  f"(global epoch {global_epoch}). "
                                  f"Advancing to next block unfreeze.")
                        break   # exit epoch loop; outer phase loop handles what comes next

            # ── test evaluation ───────────────────────────────────────────────
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

            save_training_curves(
                history, CHECKPOINT_DIR,
                title=f"{num_blocks} block(s) | {aug_name}",
                phase_boundaries=phase_boundaries,
            )

            save_results_csv({
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "experiment":       aug_name,
                "num_blocks":       num_blocks,
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
                "augmentations":    str(AUGMENTATIONS),
                "phase_schedule":   str(PHASE_SCHEDULE),
            }, CHECKPOINT_BASE)

    print("\n" + "=" * 70)
    print(f"All {total_runs} runs complete.")
    print(f"Results CSV : {os.path.join(CHECKPOINT_BASE, TRAINING_TYPE+'_all_experiment_results.csv')}")
