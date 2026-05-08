"""
DistillationEngine.py
=====================
Drop-in replacement for TrainingEngine from TrainingEngine.py.
Handles the distillation-specific forward (tuple unpacking, teacher call,
composite loss) while keeping identical interface and behaviour otherwise.

Key differences vs TrainingEngine
----------------------------------
  • Runs teacher.forward() each step (fully frozen, inside no_grad)
  • Unpacks (logits, feat_proj) from student train-mode forward
  • Computes distillation_loss() instead of a plain loss_fn call
  • Feat loss gating: w_feat is forced to 0.0 until the student's hook
    block is in _unfrozen_blocks, preventing the projector from learning
    a stale mapping before block 5 is free to change
  • evaluate() uses CE-only loss (student in eval mode → plain logits,
    no teacher needed), consistent with LR scheduler and early stopping

Optimizer note
--------------
  Teacher is fully frozen (backbone + projector both frozen).
  Optimizer needs only student.trainable_params().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace
from typing import Dict, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, "../Models")
from ViTDistillation import TeacherViT, StudentViTTiny, LossWeights


# ─────────────────────────────────────────────────────────────────────────────
# DISTILLATION LOSS
# ─────────────────────────────────────────────────────────────────────────────
def distillation_loss(
    student_logits: torch.Tensor,   # (B, C)        — raw student logits
    student_feat:   torch.Tensor,   # (B, PROJ_DIM) — student projected CLS
    teacher_logits: torch.Tensor,   # (B, C)        — raw teacher logits
    teacher_feat:   torch.Tensor,   # (B, PROJ_DIM) — teacher projected CLS
    labels:         torch.Tensor,   # (B,)          — hard integer labels
    weights:        LossWeights = None,
) -> tuple[torch.Tensor, dict]:
    """
    Composite distillation loss.

    L_CE
        CrossEntropy(student_logits, labels, label_smoothing=0.1)
        Optional class weights via weights.ce_class_weights for weighted_ce strategy.

    L_KL
        KLDiv( log_softmax(student/T) || softmax(teacher/T) ) * T²
        Soft-label distillation. T² restores gradient magnitude (Hinton 2015).

    L_MSE  (on logits)
        MSE(student_logits, teacher_logits.detach())
        Complements KL by matching absolute logit magnitudes.

    L_feat  (cosine similarity loss on projected intermediate CLS)
        (1 - cosine_similarity(student_feat, teacher_feat.detach())).mean()
        Bounded in [0, 2]. Skipped entirely (no compute) when w_feat=0.0,
        which DistillationEngine enforces before the hook block is unfrozen.

    Returns
    -------
    total_loss : scalar tensor
    components : dict of individual loss scalars for logging
    """
    if weights is None:
        weights = LossWeights()

    T = weights.temperature

    # ── L_CE ──────────────────────────────────────────────────────────────────
    l_ce = F.cross_entropy(
        student_logits, labels,
        weight=weights.ce_class_weights,
        label_smoothing=0.1,
    )

    # ── L_KL ──────────────────────────────────────────────────────────────────
    student_log_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft     = F.softmax(teacher_logits     / T, dim=-1)
    l_kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T ** 2)

    # ── L_MSE on logits ───────────────────────────────────────────────────────
    l_mse = F.mse_loss(student_logits, teacher_logits.detach())

    # ── L_feat — cosine similarity loss ──────────────────────────────────────
    # Skipped when w_feat=0.0 (hook block not yet unfrozen) to avoid wasting
    # compute and to prevent the projector training on frozen representations.
    if weights.w_feat > 0.0:
        cos_sim = F.cosine_similarity(student_feat, teacher_feat.detach(), dim=-1)
        l_feat  = (1 - cos_sim).mean()   # bounded [0, 2], 0 = perfectly aligned
    else:
        l_feat = torch.tensor(0.0, device=student_logits.device)

    # ── Total ─────────────────────────────────────────────────────────────────
    total = (
        weights.w_ce   * l_ce  +
        weights.w_kl   * l_kl  +
        weights.w_mse  * l_mse +
        weights.w_feat * l_feat
    )

    components = {
        "loss_total": total.item(),
        "loss_ce":    l_ce.item(),
        "loss_kl":    l_kl.item(),
        "loss_mse":   l_mse.item(),
        "loss_feat":  l_feat.item(),
    }

    return total, components


# ─────────────────────────────────────────────────────────────────────────────
# DISTILLATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class DistillationEngine:
    """
    Training + evaluation engine for knowledge distillation.

    Mirrors TrainingEngine exactly:
        • AMP + GradScaler
        • Scheduler overflow guard (skip step on grad spike)
        • tqdm progress bar + print_freq step logging
        • evaluate() returns same dict (epoch_loss, epoch_acc, macro_f1, ...)

    Args
    ----
    teacher        : TeacherViT — fully frozen (backbone + JL projector)
    student        : StudentViTTiny
    data_iterator  : LabelSelector-wrapped DataLoader
    loss_weights   : LossWeights dataclass
    opt            : optimizer over student.trainable_params() only
    compute_device : torch.device
    scheduler      : optional step-based LR scheduler
                     (ReduceLROnPlateau is stepped manually outside the engine)
    batch_aug      : optional BatchAugmenter for CutMix / MixUp
    """

    def __init__(
        self,
        teacher:        TeacherViT,
        student:        StudentViTTiny,
        data_iterator,
        loss_weights:   LossWeights,
        opt:            torch.optim.Optimizer,
        compute_device: torch.device,
        scheduler:      Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        batch_aug:      Optional[object] = None,
    ) -> None:
        self.teacher        = teacher.to(compute_device)
        self.student        = student.to(compute_device)
        self.data_iterator  = data_iterator
        self.loss_weights   = loss_weights
        self.opt            = opt
        self.compute_device = compute_device
        self.scheduler      = scheduler
        self.batch_aug      = batch_aug

        amp_device_str = 'cuda' if compute_device.type == 'cuda' else 'cpu'
        self.grad_scaler = torch.amp.GradScaler(
            device=amp_device_str,
            enabled=compute_device.type == 'cuda',
        )

        self.total_loss:    float = 0.0
        self.correct_preds: int   = 0
        self.total_samples: int   = 0

    # ── helpers ───────────────────────────────────────────────────────────────
    def _clear_trackers(self) -> None:
        self.total_loss    = 0.0
        self.correct_preds = 0
        self.total_samples = 0

    @torch.inference_mode()
    def _accumulate_stats(
        self,
        batch_loss:     torch.Tensor,
        student_logits: torch.Tensor,
        y_batch:        torch.Tensor,
    ) -> None:
        b_size = y_batch.size(0)
        self.total_loss += batch_loss.item() * b_size
        class_preds = student_logits.argmax(dim=-1)
        y_true = y_batch.argmax(dim=-1) if y_batch.ndim > 1 else y_batch
        self.correct_preds += torch.eq(class_preds, y_true).sum().item()
        self.total_samples += b_size

    def _calculate_epoch_stats(self) -> Dict[str, float]:
        denom = max(self.total_samples, 1)
        return {
            "epoch_loss": self.total_loss / denom,
            "epoch_acc":  self.correct_preds / denom,
        }

    def _active_loss_weights(self) -> LossWeights:
        """
        Returns loss weights with w_feat forced to 0.0 if the student's hook
        block has not yet been unfrozen.  Once unfrozen, full w_feat is used.
        """
        if (self.loss_weights.w_feat > 0.0
                and self.student.hook_block not in self.student._unfrozen_blocks
                and self.student.gradual_unfreeze):
            return replace(self.loss_weights, w_feat=0.0)
        return self.loss_weights

    # ── training pass ─────────────────────────────────────────────────────────
    def train_one_epoch(
        self,
        epoch_num:  int,
        print_freq: Optional[int] = 50,
    ) -> Dict[str, float]:
        """Single distillation training pass. Matches TrainingEngine signature."""
        self.student.train()   # activates projector in student.forward()
        self.teacher.train()   # TeacherViT.train() override keeps it in eval
        self._clear_trackers()

        amp_device_type = 'cuda' if self.compute_device.type == 'cuda' else 'cpu'
        active_weights  = self._active_loss_weights()

        feat_status = (
            f"w_feat={active_weights.w_feat}"
            if active_weights.w_feat > 0.0
            else "w_feat=0.0 (hook block frozen)"
        )
        tqdm.write(f"  [Distill] {feat_status}")

        pbar = tqdm(
            self.data_iterator,
            desc=f"[Epoch {epoch_num:03d}] Distill",
            unit="batch",
            dynamic_ncols=True,
            colour="green",
        )

        for step_idx, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(self.compute_device, non_blocking=True)
            y_batch = y_batch.to(self.compute_device, non_blocking=True)

            if self.batch_aug is not None:
                x_batch, y_batch = self.batch_aug(x_batch, y_batch)

            self.opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=amp_device_type,
                                enabled=self.compute_device.type == 'cuda'):
                # Teacher: fully frozen, always returns (logits, feat_proj)
                teacher_logits, teacher_feat = self.teacher(x_batch)

                # Student train mode: returns (logits, feat_proj)
                student_logits, student_feat = self.student(x_batch)

                hard_labels = (
                    y_batch.argmax(dim=-1) if y_batch.ndim > 1 else y_batch
                )

                batch_loss, loss_components = distillation_loss(
                    student_logits, student_feat,
                    teacher_logits, teacher_feat,
                    hard_labels,
                    weights=active_weights,
                )

            self.grad_scaler.scale(batch_loss).backward()
            scale_before = self.grad_scaler.get_scale()
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            scale_after = self.grad_scaler.get_scale()

            if self.scheduler is not None:
                if scale_before <= scale_after:
                    self.scheduler.step()
                else:
                    tqdm.write("Warning: Gradient overflow — scheduler step skipped.")

            self._accumulate_stats(batch_loss, student_logits, y_batch)

            current_stats = self._calculate_epoch_stats()
            pbar.set_postfix(
                loss=f"{current_stats['epoch_loss']:.4f}",
                acc=f"{current_stats['epoch_acc']*100:.2f}%",
                ce=f"{loss_components['loss_ce']:.3f}",
                kl=f"{loss_components['loss_kl']:.3f}",
                feat=f"{loss_components['loss_feat']:.3f}",
            )

            if print_freq and (step_idx + 1) % print_freq == 0:
                tqdm.write(
                    f"  [Epoch {epoch_num:03d}] "
                    f"Step {step_idx + 1:04d}/{len(pbar):04d} | "
                    f"loss={current_stats['epoch_loss']:.4f} | "
                    f"acc={current_stats['epoch_acc']:.4f} | "
                    f"ce={loss_components['loss_ce']:.3f} | "
                    f"kl={loss_components['loss_kl']:.3f} | "
                    f"feat={loss_components['loss_feat']:.3f}"
                )

        final_stats = self._calculate_epoch_stats()
        tqdm.write(
            f"[Epoch {epoch_num:03d}] COMPLETED | "
            f"avg_loss={final_stats['epoch_loss']:.4f} | "
            f"avg_acc={final_stats['epoch_acc']:.4f}"
        )
        return final_stats

    # ── evaluation pass ───────────────────────────────────────────────────────
    @torch.inference_mode()
    def evaluate(
        self,
        inference_loader,
        phase_label: str = "Validation",
    ) -> Dict[str, float]:
        """
        Gradient-free evaluation. Matches TrainingEngine.evaluate() exactly.

        student.eval() → plain logits, no tuple, no teacher needed.
        Eval loss is CE-only, consistent with LR scheduler and early stopping.
        """
        from sklearn.metrics import precision_recall_fscore_support

        self.student.eval()
        self.teacher.eval()
        self._clear_trackers()

        hardware_backend = 'cuda' if self.compute_device.type == 'cuda' else 'cpu'
        is_cuda_active   = hardware_backend == 'cuda'
        amp_context      = torch.autocast(device_type=hardware_backend, enabled=is_cuda_active)

        all_preds:   list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        pbar = tqdm(
            inference_loader,
            desc=f"[{phase_label.capitalize()}]      ",
            unit="batch",
            dynamic_ncols=True,
            colour="blue",
        )

        for features, targets in pbar:
            features = features.to(
                device=self.compute_device,
                memory_format=torch.channels_last,
                non_blocking=True,
            )
            targets = targets.to(device=self.compute_device, non_blocking=True)

            with amp_context:
                student_logits = self.student(features)   # plain logits in eval
                hard_labels    = (
                    targets.argmax(dim=-1) if targets.ndim > 1 else targets
                )
                step_loss = F.cross_entropy(
                    student_logits, hard_labels, label_smoothing=0.1,
                )

            self._accumulate_stats(step_loss, student_logits, targets)

            batch_preds = student_logits.argmax(dim=-1).detach().cpu()
            batch_targets = (
                targets.argmax(dim=-1).detach().cpu()
                if targets.ndim > 1
                else targets.detach().cpu()
            )
            all_preds.append(batch_preds)
            all_targets.append(batch_targets)

            current_stats = self._calculate_epoch_stats()
            pbar.set_postfix(
                loss=f"{current_stats['epoch_loss']:.4f}",
                acc=f"{current_stats['epoch_acc']*100:.2f}%",
            )

        phase_results = self._calculate_epoch_stats()

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0,
        )

        phase_results["macro_precision"] = float(macro_precision)
        phase_results["macro_recall"]    = float(macro_recall)
        phase_results["macro_f1"]        = float(macro_f1)

        tqdm.write(
            f"[{phase_label.upper()}] metrics | "
            f"Mean Loss: {phase_results['epoch_loss']:.4f} | "
            f"Accuracy: {phase_results['epoch_acc'] * 100:.2f}% | "
            f"Macro F1: {phase_results['macro_f1'] * 100:.2f}% | "
            f"Macro P:  {phase_results['macro_precision'] * 100:.2f}% | "
            f"Macro R:  {phase_results['macro_recall'] * 100:.2f}%"
        )

        return phase_results