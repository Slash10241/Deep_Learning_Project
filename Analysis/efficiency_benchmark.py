"""
efficiency_benchmark.py
=======================
Measures and compares ViT-Base (teacher) and ViT-Tiny (student) on:
  1. Parameter count (total + trainable)
  2. FLOPs (via fvcore)
  3. Inference latency (CPU and CUDA if available)
  4. Memory footprint (model size on disk + peak GPU memory)
  5. Throughput (images per second)

Outputs
-------
    efficiency_report.csv   — all metrics in one row per model
    efficiency_plots.png    — bar charts for the key metrics
    latency_distribution.png — latency distribution per model (boxplot)
"""

import os
import sys
import csv
import time
import gc
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "../Models")
sys.path.insert(0, "../Models")
from ViTDistillation import TeacherViT, StudentViTTiny

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TEACHER_CKPT   = "../Checkpoints/Extension/Best_Teacher/vit_base_patch16_224_Gradual_Unfreeze_Multiple_breed_blocks3_erasing_jitter_mixup_cutmix_low_prob/checkpoint.pt"
STUDENT_CKPT   = "../Checkpoints/Extension/Best_Student/vit_tiny_patch16_224_Distillation_pretrained_gradual_breed_baseline_erasing_jitter_mixup_cutmix_mod_prob_Data_frac_1.0/checkpoint_student.pt"

OUTPUT_DIR  = "../Analysis/efficiency/"
NUM_CLASSES = 37
IMAGE_SIZE  = 224
BATCH_SIZES = [1, 8, 32, 64]   # latency measured at these batch sizes

# Warmup + timing iterations
N_WARMUP = 50
N_TIMING = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS — freeze everything, eval mode
# ─────────────────────────────────────────────────────────────────────────────
def load_models():
    print("\nLoading teacher ViT-Base ...")
    teacher = TeacherViT(checkpoint_path=TEACHER_CKPT, num_classes=NUM_CLASSES)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    print("Loading student ViT-Tiny ...")
    student = StudentViTTiny(num_classes=NUM_CLASSES, pretrained=True, gradual_unfreeze=True)
    student.load_state_dict(
        torch.load(STUDENT_CKPT, map_location="cpu", weights_only=True)
    )
    for p in student.parameters():
        p.requires_grad = False
    student.eval()

    return teacher, student


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMETER COUNT
# ─────────────────────────────────────────────────────────────────────────────
def count_params(model, name):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{name}] Total params    : {total:,}  ({total/1e6:.2f}M)")
    print(f"  [{name}] Trainable params: {trainable:,}  ({trainable/1e6:.2f}M)")
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# 2. FLOPs
# ─────────────────────────────────────────────────────────────────────────────
def count_flops(model, name, input_size=(1, 3, 224, 224)):
    dummy = torch.randn(*input_size)
    flops_val = None

    try:
        from fvcore.nn import FlopCountAnalysis
        flop_counter = FlopCountAnalysis(model, dummy)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        flops_val = flop_counter.total()
        source = "fvcore"
    except ImportError:
        pass

    print(f"  [{name}] FLOPs: {flops_val:,}  ({flops_val/1e9:.2f}G)  [{source}]")
    return flops_val


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL SIZE ON DISK
# ─────────────────────────────────────────────────────────────────────────────
    

def model_disk_size_mb(model, name):
    tmp_path = tempfile.mktemp(suffix=".pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / 1e6
    os.remove(tmp_path)
    print(f"  [{name}] Disk size: {size_mb:.1f} MB")
    return size_mb


# ─────────────────────────────────────────────────────────────────────────────
# 4. LATENCY — returns dict of batch_size → stats
# ─────────────────────────────────────────────────────────────────────────────
def measure_latency(model, name, batch_sizes, device):
    model = model.to(device)
    results = {}

    for bs in batch_sizes:
        dummy = torch.randn(bs, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        if device.type == "cuda":
            dummy = dummy.to(memory_format=torch.channels_last)

        # Warmup
        with torch.inference_mode():
            for _ in range(N_WARMUP):
                _ = model(dummy) if "Student" not in type(model).__name__ else model.backbone(dummy)

        # Timing
        latencies_ms = []

        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                for _ in range(N_TIMING):
                    starter.record()
                    if isinstance(model, TeacherViT):
                        _ = model.backbone(dummy)
                    else:
                        _ = model(dummy)
                    ender.record()
                    torch.cuda.synchronize()
                    latencies_ms.append(starter.elapsed_time(ender))
        else:
            with torch.inference_mode():
                for _ in range(N_TIMING):
                    t0 = time.perf_counter()
                    if isinstance(model, TeacherViT):
                        _ = model.backbone(dummy)
                    else:
                        _ = model(dummy)
                    latencies_ms.append((time.perf_counter() - t0) * 1000)

        lat = np.array(latencies_ms)
        mean_ms  = float(lat.mean())
        std_ms   = float(lat.std())
        p50_ms   = float(np.percentile(lat, 50))
        p95_ms   = float(np.percentile(lat, 95))
        p99_ms   = float(np.percentile(lat, 99))
        throughput = bs / (mean_ms / 1000)   # images per second

        results[bs] = {
            "mean_ms":    round(mean_ms, 3),
            "std_ms":     round(std_ms, 3),
            "p50_ms":     round(p50_ms, 3),
            "p95_ms":     round(p95_ms, 3),
            "p99_ms":     round(p99_ms, 3),
            "throughput": round(throughput, 1),
            "all_ms":     lat,
        }
        print(f"  [{name}] BS={bs:3d}: mean={mean_ms:.2f}ms  p50={p50_ms:.2f}ms  "
              f"p95={p95_ms:.2f}ms  throughput={throughput:.0f} img/s")

    model.cpu()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. PEAK GPU MEMORY
# ─────────────────────────────────────────────────────────────────────────────
def measure_peak_memory_mb(model, name, batch_size=1):
    if device.type != "cuda":
        print(f"  [{name}] GPU memory: N/A (CPU only)")
        return None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = model.to(device)
    dummy = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.inference_mode():
        if isinstance(model, TeacherViT):
            _ = model.backbone(dummy)
        else:
            _ = model(dummy)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    model.cpu()
    torch.cuda.empty_cache()
    print(f"  [{name}] Peak GPU memory (BS=1): {peak_mb:.1f} MB")
    return round(peak_mb, 1)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def save_efficiency_plots(t_stats, s_stats, t_latency, s_latency):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#F7FBFD")
    fig.suptitle("Teacher (ViT-Base) vs Student (ViT-Tiny) — Efficiency Comparison",
                 fontsize=14, fontweight="bold")

    colors = {"teacher": "#065A82", "student": "#028090"}
    labels = ["Teacher\n(ViT-Base)", "Student\n(ViT-Tiny)"]

    # 1. Parameter count (M)
    ax = axes[0, 0]
    ax.set_facecolor("#F7FBFD")
    vals = [t_stats["total_params"]/1e6, s_stats["total_params"]/1e6]
    bars = ax.bar(labels, vals, color=[colors["teacher"], colors["student"]], width=0.45, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}M", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Parameter Count (M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Parameters (M)")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_ylim(0, max(vals)*1.25)
    ratio = vals[0]/vals[1]
    ax.text(0.5, 0.92, f"{ratio:.1f}× compression", transform=ax.transAxes,
            ha="center", fontsize=10, color="#C2410C", style="italic")

    # 2. FLOPs (G)
    ax = axes[0, 1]
    ax.set_facecolor("#F7FBFD")
    t_f = t_stats.get("flops")
    s_f = s_stats.get("flops")
    if t_f and s_f:
        vals = [t_f/1e9, s_f/1e9]
        bars = ax.bar(labels, vals, color=[colors["teacher"], colors["student"]], width=0.45, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f"{v:.2f}G", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title("FLOPs (G) — batch size 1", fontsize=12, fontweight="bold")
        ax.set_ylabel("GFLOPs")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_ylim(0, max(vals)*1.25)
        ratio = vals[0]/vals[1]
        ax.text(0.5, 0.92, f"{ratio:.1f}× fewer FLOPs", transform=ax.transAxes,
                ha="center", fontsize=10, color="#C2410C", style="italic")
    else:
        ax.text(0.5, 0.5, "FLOPs N/A\nInstall fvcore", transform=ax.transAxes,
                ha="center", fontsize=12, color="gray")
        ax.set_title("FLOPs (G)", fontsize=12, fontweight="bold")

    # 3. Model disk size (MB)
    ax = axes[0, 2]
    ax.set_facecolor("#F7FBFD")
    vals = [t_stats["disk_mb"], s_stats["disk_mb"]]
    bars = ax.bar(labels, vals, color=[colors["teacher"], colors["student"]], width=0.45, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{v:.0f}MB", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Model Size on Disk (MB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("MB")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_ylim(0, max(vals)*1.25)

    # 4. Mean latency vs batch size (BS=1)
    ax = axes[1, 0]
    ax.set_facecolor("#F7FBFD")
    bs_list = sorted(t_latency.keys())
    t_means = [t_latency[bs]["mean_ms"] for bs in bs_list]
    s_means = [s_latency[bs]["mean_ms"] for bs in bs_list]
    ax.plot(bs_list, t_means, "o-", color=colors["teacher"], label="Teacher", linewidth=2, markersize=7)
    ax.plot(bs_list, s_means, "o-", color=colors["student"], label="Student", linewidth=2, markersize=7)
    ax.set_title("Mean Latency vs Batch Size (ms)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=10)
    ax.grid(linewidth=0.4, alpha=0.5)

    # 5. Throughput vs batch size
    ax = axes[1, 1]
    ax.set_facecolor("#F7FBFD")
    t_tput = [t_latency[bs]["throughput"] for bs in bs_list]
    s_tput = [s_latency[bs]["throughput"] for bs in bs_list]
    ax.plot(bs_list, t_tput, "s-", color=colors["teacher"], label="Teacher", linewidth=2, markersize=7)
    ax.plot(bs_list, s_tput, "s-", color=colors["student"], label="Student", linewidth=2, markersize=7)
    ax.set_title("Throughput vs Batch Size (img/s)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Images / second")
    ax.legend(fontsize=10)
    ax.grid(linewidth=0.4, alpha=0.5)

    # 6. Latency distribution (BS=1 boxplot)
    ax = axes[1, 2]
    ax.set_facecolor("#F7FBFD")
    bs1 = BATCH_SIZES[0]
    t_all = t_latency[bs1]["all_ms"]
    s_all = s_latency[bs1]["all_ms"]
    bp = ax.boxplot([t_all, s_all], labels=labels, patch_artist=True,
                    medianprops={"color":"white","linewidth":2},
                    whiskerprops={"linewidth":1.5},
                    capprops={"linewidth":1.5})
    bp["boxes"][0].set_facecolor(colors["teacher"])
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(colors["student"])
    bp["boxes"][1].set_alpha(0.75)
    ax.set_title(f"Latency Distribution — BS={bs1} (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "efficiency_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✔ efficiency_plots.png saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV REPORT
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(t_stats, s_stats, t_latency, s_latency):
    rows = []
    for name, stats, latency in [
        ("ViT-Base (Teacher)", t_stats, t_latency),
        ("ViT-Tiny (Student)", s_stats, s_latency),
    ]:
        row = {
            "model":          name,
            "total_params":   stats["total_params"],
            "total_params_M": round(stats["total_params"]/1e6, 2),
            "flops":          stats.get("flops", "N/A"),
            "flops_G":        round(stats["flops"]/1e9, 3) if stats.get("flops") else "N/A",
            "disk_mb":        stats["disk_mb"],
            "peak_gpu_mb":    stats.get("peak_gpu_mb", "N/A"),
        }
        for bs in sorted(latency.keys()):
            d = latency[bs]
            row[f"bs{bs}_mean_ms"]    = d["mean_ms"]
            row[f"bs{bs}_p50_ms"]     = d["p50_ms"]
            row[f"bs{bs}_p95_ms"]     = d["p95_ms"]
            row[f"bs{bs}_p99_ms"]     = d["p99_ms"]
            row[f"bs{bs}_throughput"] = d["throughput"]
        rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, "efficiency_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✔ efficiency_report.csv saved to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(t_stats, s_stats, t_latency, s_latency):
    print("\n" + "="*70)
    print("EFFICIENCY SUMMARY")
    print("="*70)
    print(f"{'Metric':<35} {'Teacher (ViT-Base)':>18} {'Student (ViT-Tiny)':>18} {'Ratio':>8}")
    print("-"*70)

    def row(label, t_val, s_val, fmt=".2f", unit="", ratio_inv=False):
        if t_val is None or s_val is None:
            print(f"  {label:<33} {'N/A':>18} {'N/A':>18} {'':>8}")
            return
        ratio = t_val/s_val if not ratio_inv else s_val/t_val
        print(f"  {label:<33} {t_val:{fmt}}{unit:>4} {s_val:{fmt}}{unit:>4} {ratio:>7.1f}×")

    row("Parameters (M)",       t_stats["total_params"]/1e6, s_stats["total_params"]/1e6)
    row("FLOPs (G)",            t_stats.get("flops",0)/1e9 if t_stats.get("flops") else None,
                                s_stats.get("flops",0)/1e9 if s_stats.get("flops") else None)
    row("Disk size (MB)",       t_stats["disk_mb"],  s_stats["disk_mb"])
    if t_stats.get("peak_gpu_mb") and s_stats.get("peak_gpu_mb"):
        row("Peak GPU memory (MB)", t_stats["peak_gpu_mb"], s_stats["peak_gpu_mb"])
    print("-"*70)

    for bs in sorted(t_latency.keys()):
        row(f"Latency BS={bs} mean (ms)",
            t_latency[bs]["mean_ms"], s_latency[bs]["mean_ms"], fmt=".2f")
        row(f"Throughput BS={bs} (img/s)",
            t_latency[bs]["throughput"], s_latency[bs]["throughput"],
            fmt=".0f", ratio_inv=True)

    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    teacher, student = load_models()

    print("\n── Parameter counts ─────────────────────────────────────────────")
    t_total, t_trainable = count_params(teacher, "Teacher")
    s_total, s_trainable = count_params(student,  "Student")

    print("\n── FLOPs ────────────────────────────────────────────────────────")
    # Teacher: use backbone only (same as inference path)
    t_flops = count_flops(teacher.backbone, "Teacher")
    s_flops = count_flops(student,          "Student")

    print("\n── Disk size ────────────────────────────────────────────────────")
    t_disk = model_disk_size_mb(teacher, "Teacher")
    s_disk = model_disk_size_mb(student, "Student")

    print("\n── Peak GPU memory (BS=1) ───────────────────────────────────────")
    t_gpu = measure_peak_memory_mb(teacher, "Teacher", batch_size=1)
    s_gpu = measure_peak_memory_mb(student, "Student", batch_size=1)

    t_stats = {
        "total_params": t_total,
        "flops":        t_flops,
        "disk_mb":      t_disk,
        "peak_gpu_mb":  t_gpu,
    }
    s_stats = {
        "total_params": s_total,
        "flops":        s_flops,
        "disk_mb":      s_disk,
        "peak_gpu_mb":  s_gpu,
    }

    print("\n── Inference latency ────────────────────────────────────────────")
    print(f"  Device: {device}  |  Warmup: {N_WARMUP}  |  Timing: {N_TIMING}")
    print("  Teacher:")
    t_latency = measure_latency(teacher, "Teacher", BATCH_SIZES, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("  Student:")
    s_latency = measure_latency(student, "Student", BATCH_SIZES, device)

    print_summary(t_stats, s_stats, t_latency, s_latency)
    save_csv(t_stats, s_stats, t_latency, s_latency)
    save_efficiency_plots(t_stats, s_stats, t_latency, s_latency)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
