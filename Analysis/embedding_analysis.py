"""
embedding_analysis.py
=====================
Analyses and compares ViT-Base (teacher) and ViT-Tiny (student) embeddings
on the Oxford-IIIT Pet test set.

Outputs (all written to OUTPUT_DIR)
-------------------------------------
1. umap_overview.png
   UMAP of ALL test embeddings for both models, coloured by class.
   Teacher and student shown on separate subplots for a clean comparison.

2. umap_correctness.png
   UMAP coloured by prediction outcome:
     • Teacher ✓ Student ✓   (both correct)
     • Teacher ✓ Student ✗   (teacher recovers, student fails — the key group)
     • Teacher ✗ Student ✓   (student-only correct)
     • Both ✗

3. correct_gallery_{class_idx}.png  (one per class)
   3 example images correctly classified by BOTH teacher and student,
   with their top-3 softmax predictions shown.

4. student_wrong_gallery_{class_idx}.png  (one per class, when examples exist)
   3 example images where student is WRONG but teacher is RIGHT,
   with teacher and student top-3 predictions shown side by side.

5. embedding_stats.csv
   Per-class summary: n_correct_both, n_teacher_only, n_student_only, n_both_wrong.

Usage
-----
Edit the CONFIG section below then run:
    python embedding_analysis.py
"""

import os
import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import umap
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "../Models")
sys.path.insert(0, "../DataLoader")

from DataLoader    import (CreateDataset, get_eval_transform,
                            _parse_annotation_file, _pet_collate)
from ViTDistillation import TeacherViT, StudentViTTiny

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT   = "../../../Dataset/"
TEACHER_CKPT   = "../Checkpoints/Best_Teacher/vit_base_patch16_224_Gradual_Unfreeze_Multiple_breed_blocks3_erasing_jitter_mixup_cutmix_low_prob/checkpoint.pt"
STUDENT_CKPT   = "../Checkpoints/Best_Student/vit_tiny_patch16_224_Distillation_pretrained_gradual_breed_baseline_erasing_jitter_mixup_cutmix_mod_prob_Data_frac_1.0/checkpoint_student.pt"
OUTPUT_DIR     = "../Analysis/embedding_analysis/"
NUM_CLASSES    = 37
IMAGE_SIZE     = 224
BATCH_SIZE     = 64
NUM_WORKERS    = 0
N_EXAMPLES     = 3       # examples to show per class per gallery type
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.1
UMAP_RANDOM_STATE = 42

# Optional: path to a breed name mapping JSON {"0": "Abyssinian", ...}
# If None, class indices are used as labels.
BREED_NAMES_PATH = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# BREED NAMES
# ─────────────────────────────────────────────────────────────────────────────
if BREED_NAMES_PATH and os.path.exists(BREED_NAMES_PATH):
    with open(BREED_NAMES_PATH) as f:
        _bn = json.load(f)
    BREED_NAMES = {int(k): v for k, v in _bn.items()}
else:
    BREED_NAMES = {i: f"Class {i}" for i in range(NUM_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# HOOK — extracts CLS token from the final norm layer
# ─────────────────────────────────────────────────────────────────────────────
class CLSHook:
    """Attaches to backbone.norm and captures the CLS token output."""
    def __init__(self, module):
        self._feat = None
        self._handle = module.register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        # out: (B, num_tokens, embed_dim) after norm
        self._feat = out[:, 0, :].detach().cpu()

    @property
    def feat(self):
        return self._feat

    def clear(self):
        self._feat = None

    def remove(self):
        self._handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading teacher ViT-Base ...")
teacher = TeacherViT(checkpoint_path=TEACHER_CKPT, num_classes=NUM_CLASSES)
teacher.to(device)
teacher.eval()
# Teacher hook on backbone.norm (final CLS after norm)
teacher_hook = CLSHook(teacher.backbone.norm)

print("Loading student ViT-Tiny ...")
student = StudentViTTiny(num_classes=NUM_CLASSES, pretrained=True, gradual_unfreeze=True)
student.load_state_dict(
    torch.load(STUDENT_CKPT, map_location=device, weights_only=True)
)
for p in student.parameters():
    p.requires_grad = False
    
student.to(device)
student.eval()
# Student hook on backbone.norm
student_hook = CLSHook(student.backbone.norm)


# ─────────────────────────────────────────────────────────────────────────────
# TEST DATASET (no augmentation, no LabelSelector — we need raw images too)
# ─────────────────────────────────────────────────────────────────────────────
root       = Path(DATASET_ROOT)
images_dir = root / "images"
test_txt   = root / "annotations" / "test.txt"

test_records = _parse_annotation_file(str(test_txt))

# Dataset with eval transform — returns (image_tensor, (species, breed))
test_dataset = CreateDataset(
    test_records, images_dir,
    transform=get_eval_transform(IMAGE_SIZE), one_hot=False,
)

# Raw image dataset for gallery display (no transform — PIL images)
class RawImageDataset(torch.utils.data.Dataset):
    def __init__(self, records, images_dir):
        self.records    = records
        self.images_dir = images_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        idx = int(idx) 
        fname = self.records[idx]["image_name"]    
        img_path = Path(self.images_dir) / f"{fname}.jpg"
        if not img_path.exists():
            matches = list(Path(self.images_dir).glob(f"{fname}.*"))
            if not matches:
                raise FileNotFoundError(f"No image found for {fname}")
            img_path = matches[0]
        return Image.open(img_path).convert("RGB")

        
raw_dataset = RawImageDataset(test_records, images_dir)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=_pet_collate,
)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PASS — collect embeddings, logits, labels
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning inference on test set ...")

teacher_embeds = []
student_embeds = []
teacher_probs_all = []
student_probs_all = []
teacher_preds_all = []
student_preds_all = []
true_labels_all   = []

with torch.inference_mode():
    for x, (species, breed) in test_loader:
        x     = x.to(device, memory_format=torch.channels_last, non_blocking=True)
        breed = breed.to(device)

        # ── Teacher ──────────────────────────────────────────────────────────
        teacher_hook.clear()
        t_logits = teacher.backbone(x)          # (B, C) — plain forward
        t_cls    = teacher_hook.feat            # (B, 768)
        t_probs  = F.softmax(t_logits, dim=-1).cpu()
        t_preds  = t_logits.argmax(dim=-1).cpu()

        # ── Student (eval mode — plain logits) ───────────────────────────────
        student_hook.clear()
        s_logits = student(x)                   # (B, C)
        s_cls    = student_hook.feat            # (B, 192)
        s_probs  = F.softmax(s_logits, dim=-1).cpu()
        s_preds  = s_logits.argmax(dim=-1).cpu()

        teacher_embeds.append(t_cls)
        student_embeds.append(s_cls)
        teacher_probs_all.append(t_probs)
        student_probs_all.append(s_probs)
        teacher_preds_all.append(t_preds)
        student_preds_all.append(s_preds)
        true_labels_all.append(breed.cpu())

teacher_hook.remove()
student_hook.remove()

teacher_embeds = torch.cat(teacher_embeds).numpy()    # (N, 768)
student_embeds = torch.cat(student_embeds).numpy()    # (N, 192)
teacher_probs  = torch.cat(teacher_probs_all).numpy() # (N, 37)
student_probs  = torch.cat(student_probs_all).numpy()
teacher_preds  = torch.cat(teacher_preds_all).numpy() # (N,)
student_preds  = torch.cat(student_preds_all).numpy()
true_labels    = torch.cat(true_labels_all).numpy()   # (N,)

N = len(true_labels)
teacher_correct = (teacher_preds == true_labels)  # (N,) bool
student_correct = (student_preds  == true_labels)

# Outcome categories
both_correct         = teacher_correct & student_correct
teacher_only_correct = teacher_correct & ~student_correct
student_only_correct = ~teacher_correct & student_correct
both_wrong           = ~teacher_correct & ~student_correct

print(f"  Test samples      : {N}")
print(f"  Teacher accuracy  : {teacher_correct.mean()*100:.2f}%")
print(f"  Student accuracy  : {student_correct.mean()*100:.2f}%")
print(f"  Both correct      : {both_correct.sum()} ({both_correct.mean()*100:.1f}%)")
print(f"  Teacher only ✓    : {teacher_only_correct.sum()} ({teacher_only_correct.mean()*100:.1f}%)")
print(f"  Student only ✓    : {student_only_correct.sum()} ({student_only_correct.mean()*100:.1f}%)")
print(f"  Both wrong        : {both_wrong.sum()} ({both_wrong.mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# UMAP — fit separately on each model's own embedding space
# The two spaces are not aligned to each other — orientation is arbitrary.
# Compare cluster structure within each plot, not positions across plots.
# ─────────────────────────────────────────────────────────────────────────────
print("\nFitting UMAP on teacher embeddings ...")
teacher_scaled = StandardScaler().fit_transform(teacher_embeds)
reducer_t = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=2,
    random_state=UMAP_RANDOM_STATE,
    metric="cosine",
)
teacher_2d = reducer_t.fit_transform(teacher_scaled)   # (N, 2)

print("Fitting UMAP on student embeddings ...")
student_scaled = StandardScaler().fit_transform(student_embeds)
reducer_s = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=2,
    random_state=UMAP_RANDOM_STATE,
    metric="cosine",
)
student_2d = reducer_s.fit_transform(student_scaled)   # (N, 2)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — UMAP overview: coloured by class, teacher vs student side by side
# ─────────────────────────────────────────────────────────────────────────────
print("Saving umap_overview.png ...")

cmap = plt.get_cmap("tab20", NUM_CLASSES)
colors_all = [cmap(i / NUM_CLASSES) for i in range(NUM_CLASSES)]

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.patch.set_facecolor("#F7FBFD")

for ax, pts, name in [
    (axes[0], teacher_2d, "Teacher — ViT-Base (86M)"),
    (axes[1], student_2d, "Student — ViT-Tiny  (5.7M, distilled)"),
]:
    ax.set_facecolor("#F7FBFD")
    for cls in range(NUM_CLASSES):
        mask = (true_labels == cls)
        ax.scatter(
            pts[mask, 0], pts[mask, 1],
            c=[colors_all[cls]], s=14, alpha=0.65, linewidths=0,
            label=BREED_NAMES[cls],
        )
    ax.set_title(name, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_aspect("equal", "datalim")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

# Shared legend on the right
handles = [
    mpatches.Patch(color=colors_all[i], label=BREED_NAMES[i])
    for i in range(NUM_CLASSES)
]
fig.legend(
    handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.5),
    fontsize=7, ncol=1, framealpha=0.8, title="Breed", title_fontsize=9,
)
plt.suptitle("UMAP of CLS embeddings — Teacher vs Student (cosine, separate projections)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "umap_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ umap_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — UMAP coloured by correctness outcome
# ─────────────────────────────────────────────────────────────────────────────
print("Saving umap_correctness.png ...")

outcome_colors = {
    "Both correct":       "#065A82",
    "Teacher ✓ Student ✗": "#D85A30",
    "Student ✓ Teacher ✗": "#0F766E",
    "Both wrong":         "#94A3B8",
}
outcome_alpha = {
    "Both correct":       0.4,
    "Teacher ✓ Student ✗": 0.9,
    "Student ✓ Teacher ✗": 0.9,
    "Both wrong":         0.35,
}
outcome_size = {
    "Both correct":       10,
    "Teacher ✓ Student ✗": 30,
    "Student ✓ Teacher ✗": 30,
    "Both wrong":         10,
}

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.patch.set_facecolor("#F7FBFD")

for ax, pts, name in [
    (axes[0], teacher_2d, "Teacher — ViT-Base"),
    (axes[1], student_2d, "Student — ViT-Tiny (distilled)"),
]:
    ax.set_facecolor("#F7FBFD")
    outcomes = [
        ("Both correct",        both_correct),
        ("Both wrong",          both_wrong),
        ("Student ✓ Teacher ✗", student_only_correct),
        ("Teacher ✓ Student ✗", teacher_only_correct),   # last = on top
    ]
    for label, mask in outcomes:
        ax.scatter(
            pts[mask, 0], pts[mask, 1],
            c=outcome_colors[label],
            s=outcome_size[label],
            alpha=outcome_alpha[label],
            linewidths=0,
            label=label,
            zorder=3 if "✗" in label else 1,
        )
    ax.set_title(name, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_aspect("equal", "datalim")
    ax.grid(True, linewidth=0.3, alpha=0.4)

handles = [
    mpatches.Patch(color=outcome_colors[k], label=f"{k}  ({m.sum()})")
    for k, m in [
        ("Both correct",        both_correct),
        ("Teacher ✓ Student ✗", teacher_only_correct),
        ("Student ✓ Teacher ✗", student_only_correct),
        ("Both wrong",          both_wrong),
    ]
]
fig.legend(
    handles=handles, loc="lower center", ncol=4,
    bbox_to_anchor=(0.5, -0.04), fontsize=11, framealpha=0.9,
)
plt.suptitle("UMAP coloured by prediction outcome — Teacher vs Student",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "umap_correctness.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ umap_correctness.png")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — format top-k predictions as a string
# ─────────────────────────────────────────────────────────────────────────────
def top_k_str(probs, k=3):
    top = np.argsort(probs)[::-1][:k]
    return "\n".join(
        f"{BREED_NAMES[i]}: {probs[i]*100:.1f}%"
        for i in top
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — save a gallery of images with prediction annotations
# ─────────────────────────────────────────────────────────────────────────────
def save_gallery(indices, class_idx, gallery_type, n=N_EXAMPLES):
    """
    gallery_type: "correct"  — both teacher and student correct
                  "teacher_only" — teacher correct, student wrong
    """
    indices = indices[:n]
    if len(indices) == 0:
        return

    ncols = len(indices)
    nrows = 1
    fig_w = ncols * 3.5
    fig_h = 5.5 if gallery_type == "teacher_only" else 4.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#F7FBFD")
    if ncols == 1:
        axes = [axes]

    breed_name = BREED_NAMES[class_idx]

    if gallery_type == "correct":
        title = f"Both correct — {breed_name}  (class {class_idx})"
    else:
        title = f"Teacher ✓ Student ✗ — {breed_name}  (class {class_idx})"

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    for col, idx in enumerate(indices):
        ax = axes[col]
        img = raw_dataset[idx]
        ax.imshow(img)
        ax.axis("off")

        t_top = top_k_str(teacher_probs[idx])
        s_top = top_k_str(student_probs[idx])

        if gallery_type == "correct":
            label = (
                f"Teacher ✓\n{t_top}\n\n"
                f"Student ✓\n{s_top}"
            )
            color = "#065A82"
        else:
            s_pred_name = BREED_NAMES[student_preds[idx]]
            label = (
                f"Teacher ✓\n{t_top}\n\n"
                f"Student ✗ (pred: {s_pred_name})\n{s_top}"
            )
            color = "#D85A30"

        ax.set_xlabel(label, fontsize=7, color=color,
                      labelpad=4, ha="center", va="top",
                      wrap=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor(color)

    plt.tight_layout()

    if gallery_type == "correct":
        fname = f"correct_gallery_{class_idx:02d}.png"
    else:
        fname = f"teacher_only_gallery_{class_idx:02d}.png"

    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE GALLERIES PER CLASS
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating per-class galleries ...")

stats_rows = []

for cls in range(NUM_CLASSES):
    cls_mask = (true_labels == cls)

    n_both    = (both_correct & cls_mask).sum()
    n_t_only  = (teacher_only_correct & cls_mask).sum()
    n_s_only  = (student_only_correct & cls_mask).sum()
    n_neither = (both_wrong & cls_mask).sum()

    stats_rows.append({
        "class_idx":         cls,
        "breed_name":        BREED_NAMES[cls],
        "n_both_correct":    int(n_both),
        "n_teacher_only":    int(n_t_only),
        "n_student_only":    int(n_s_only),
        "n_both_wrong":      int(n_neither),
        "total":             int(cls_mask.sum()),
    })

#     # ── 3 correct by both ─────────────────────────────────────────────────────
#     correct_idx = np.where(both_correct & cls_mask)[0]
#     if len(correct_idx) > 0:
#         save_gallery(correct_idx, cls, gallery_type="correct")

#     # ── 3 teacher-only correct ────────────────────────────────────────────────
#     t_only_idx = np.where(teacher_only_correct & cls_mask)[0]
#     if len(t_only_idx) > 0:
#         save_gallery(t_only_idx, cls, gallery_type="teacher_only")

#     if (cls + 1) % 10 == 0:
#         print(f"  Class {cls+1}/{NUM_CLASSES} done")

# print(f"  ✔ All galleries saved")


# ─────────────────────────────────────────────────────────────────────────────
# STATS CSV
# ─────────────────────────────────────────────────────────────────────────────
stats_path = os.path.join(OUTPUT_DIR, "embedding_stats.csv")
with open(stats_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=stats_rows[0].keys())
    writer.writeheader()
    writer.writerows(stats_rows)
print(f"\n✔ embedding_stats.csv saved to {stats_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BONUS PLOT — embedding distance analysis
# Per-class: mean cosine distance between teacher and student embeddings
# (after normalisation) — a proxy for how different their internal
# representations are per breed
# ─────────────────────────────────────────────────────────────────────────────
print("Saving embedding_distance_per_class.png ...")

teacher_norm = teacher_embeds / (np.linalg.norm(teacher_embeds, axis=1, keepdims=True) + 1e-8)
student_norm = student_embeds / (np.linalg.norm(student_embeds, axis=1, keepdims=True) + 1e-8)

# Cosine distance = 1 - cosine similarity, but dims differ (768 vs 192)
# so we compare within each model via intra-class compactness instead
teacher_compactness = []
student_compactness = []

for cls in range(NUM_CLASSES):
    mask = (true_labels == cls)
    if mask.sum() < 2:
        teacher_compactness.append(0)
        student_compactness.append(0)
        continue
    t_cls_vecs = teacher_norm[mask]
    s_cls_vecs = student_norm[mask]
    # Mean pairwise cosine similarity = proxy for cluster compactness
    t_sim = (t_cls_vecs @ t_cls_vecs.T).mean()
    s_sim = (s_cls_vecs @ s_cls_vecs.T).mean()
    teacher_compactness.append(float(t_sim))
    student_compactness.append(float(s_sim))

x = np.arange(NUM_CLASSES)
width = 0.38

fig, ax = plt.subplots(figsize=(18, 5))
fig.patch.set_facecolor("#F7FBFD")
ax.set_facecolor("#F7FBFD")

bars_t = ax.bar(x - width/2, teacher_compactness, width, label="Teacher (ViT-Base)",
                color="#065A82", alpha=0.85)
bars_s = ax.bar(x + width/2, student_compactness, width, label="Student (ViT-Tiny)",
                color="#028090", alpha=0.85)

ax.set_xlabel("Class index", fontsize=12)
ax.set_ylabel("Mean intra-class cosine similarity", fontsize=12)
ax.set_title("Embedding compactness per class — higher = tighter cluster in representation space",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=8)
ax.legend(fontsize=11)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.set_ylim(0, 1.05)

# Annotate classes where student compactness < teacher by > 0.1
for i in range(NUM_CLASSES):
    gap = teacher_compactness[i] - student_compactness[i]
    if gap > 0.1:
        ax.annotate("↓", xy=(i + width/2, student_compactness[i] + 0.01),
                    ha="center", va="bottom", fontsize=8, color="#D85A30")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "embedding_distance_per_class.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ embedding_distance_per_class.png")

print(f"\n{'='*60}")
print(f"All outputs saved to: {OUTPUT_DIR}")
print(f"  • umap_overview.png")
print(f"  • umap_correctness.png")
print(f"  • embedding_distance_per_class.png")
print(f"  • correct_gallery_XX.png   (one per class, up to {N_EXAMPLES} examples)")
print(f"  • teacher_only_gallery_XX.png (where student fails, teacher succeeds)")
print(f"  • embedding_stats.csv")
