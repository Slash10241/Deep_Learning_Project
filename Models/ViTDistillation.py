"""
ViTDistillation.py
==================
Wraps ViT-Base (teacher) and ViT-Tiny (student) for intermediate-layer
knowledge distillation with a shared-space projector design.

  Use DistillationEngine (DistillationEngine.py) for the training loop.
  Optimizer needs only student.trainable_params() — teacher is fully frozen.

StudentViTTiny flags
--------------------
  pretrained      (default True)  — ImageNet warm start vs random init
  gradual_unfreeze (default True) — freeze backbone, unfreeze block by block
                                    vs full backbone trainable from start
"""

import torch
import torch.nn as nn
import timm
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TEACHER_MODEL = "vit_base_patch16_224"
STUDENT_MODEL = "vit_tiny_patch16_224"

TEACHER_DIM = 768   # ViT-Base embed dim
STUDENT_DIM = 192   # ViT-Tiny embed dim
PROJ_DIM    = 256   # shared projection dim

TEACHER_HOOK_BLOCK = 5
STUDENT_HOOK_BLOCK = 5


# ─────────────────────────────────────────────────────────────────────────────
# LOSS WEIGHTS CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LossWeights:
    """
    Controls the contribution of each distillation loss component.

        w_ce   = 0.3  — hard label CE (supports optional class weights)
        w_kl   = 0.5  — soft-label KL distillation
        w_mse  = 0.1  — raw logit magnitude matching
        w_feat = 0.1  — cosine similarity on projected intermediate CLS
                        (automatically zeroed by DistillationEngine until
                        the hook block is unfrozen)
        temperature   — KL softening temperature; does NOT affect CE or MSE

    ce_class_weights: optional (C,) tensor for weighted CE when using the
                      weighted_ce strategy with class imbalance. None = uniform.
    """
    w_ce: float = 0.3
    w_kl: float = 0.5
    w_mse: float = 0.1
    w_feat: float = 0.1
    temperature: float = 4.0
    ce_class_weights: Optional[torch.Tensor] = field(default=None, compare=False)


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTOR — student-side trainable bottleneck FC
# ─────────────────────────────────────────────────────────────────────────────
class FeatureProjector(nn.Module):
    """
    Maps student CLS token (192) to PROJ_DIM (256).
    Architecture: Linear → GELU → LayerNorm → Linear
    Bottleneck = max(in_dim, out_dim).
    Training-only — dropped at inference via StudentViTTiny's training-aware forward.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        bottleneck_dim = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# HOOK MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class BlockHook:
    """
    Forward hook on a transformer block.
    Stores the CLS token (index 0) from block output (B, num_tokens, embed_dim).
    """

    def __init__(self, block: nn.Module):
        self._feat: torch.Tensor | None = None
        self._handle = block.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        self._feat = output[:, 0, :]

    @property
    def cls_token(self) -> torch.Tensor:
        if self._feat is None:
            raise RuntimeError("Hook has not fired yet — run a forward pass first.")
        return self._feat

    def clear(self):
        self._feat = None

    def remove(self):
        self._handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# TEACHER — frozen ViT-Base with frozen random JL projector
# ─────────────────────────────────────────────────────────────────────────────
class TeacherViT(nn.Module):
    """
    Frozen ViT-Base teacher.

    Backbone: fully frozen, always eval.
    Projector: frozen random Linear (768 → PROJ_DIM, no bias, JL scaling).

    """

    def __init__(
        self,
        checkpoint_path: str,
        model: str = TEACHER_MODEL,
        num_classes: int = 37,
        hook_block: int = TEACHER_HOOK_BLOCK,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        backbone = timm.create_model(model, pretrained=False, num_classes=0)
        embed_dim = backbone.num_features   # 768

        self.backbone = backbone
        self.backbone.head = nn.Linear(embed_dim, num_classes)

        # ── Frozen random projector (JL) ──────────────────────────────────────
        proj = nn.Linear(TEACHER_DIM, PROJ_DIM, bias=False)
        nn.init.normal_(proj.weight, mean=0.0, std=1.0 / PROJ_DIM ** 0.5)
        for p in proj.parameters():
            p.requires_grad = False
        self.projector = proj

        # ── Load checkpoint ───────────────────────────────────────────────────
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.items()}

        missing, unexpected = self.load_state_dict(state, strict=False)
        missing_real    = [k for k in missing if not k.startswith("projector.")]
        unexpected_real = [k for k in unexpected if not k.startswith("projector.")]
        assert not missing_real, f"[Teacher] Missing backbone keys: {missing_real}"
        assert not unexpected_real, f"[Teacher] Unexpected keys: {unexpected_real}"
        print(f"  [Teacher] Checkpoint loaded from {checkpoint_path}")

        # ── Freeze backbone ───────────────────────────────────────────────────
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ── Hook ─────────────────────────────────────────────────────────────
        self._hook      = BlockHook(self.backbone.blocks[hook_block])
        self.hook_block = hook_block

        self.eval()

    def forward(self, x: torch.Tensor):
        """
        Returns:
            logits    : (B, num_classes)
            feat_proj : (B, PROJ_DIM) — fixed random projection of teacher CLS
        """
        self._hook.clear()
        with torch.no_grad():
            logits = self.backbone(x)
            feat_raw = self._hook.cls_token       # (B, 768)
            feat_proj = self.projector(feat_raw)   # (B, PROJ_DIM)
        return logits, feat_proj

    def train(self, mode: bool = True):
        """Always stays in eval — backbone and projector are both fully frozen."""
        return super().train(False)


# ─────────────────────────────────────────────────────────────────────────────
# STUDENT — ViT-Tiny with gradual unfreeze + training-aware projector
# ─────────────────────────────────────────────────────────────────────────────
class StudentViTTiny(nn.Module):
    """
    ViT-Tiny student for knowledge distillation.

    Args
    ----
    num_classes      : output classes (37 for Oxford-IIIT Pets)
    hook_block       : block to hook for feat distillation (default 5)
    pretrained       : True = ImageNet warm start (recommended)
                       False = random init (ablation)
    gradual_unfreeze : True  = backbone frozen at init, unfreeze block by block
                       False = full backbone trainable from the start;
                               unfreeze_next_block() becomes a no-op
    """

    def __init__(
        self,
        num_classes:      int  = 37,
        hook_block:       int  = STUDENT_HOOK_BLOCK,
        pretrained:       bool = True,
        gradual_unfreeze: bool = True,
    ):
        super().__init__()

        self.gradual_unfreeze = gradual_unfreeze

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            STUDENT_MODEL, pretrained=pretrained, num_classes=0,
        )
        embed_dim = self.backbone.num_features   # 192
        self.backbone.head = nn.Linear(embed_dim, num_classes)

        # ── Projector: 192 → PROJ_DIM (training only, always trainable) ───────
        self.projector = FeatureProjector(in_dim=STUDENT_DIM, out_dim=PROJ_DIM)

        # ── Freezing ──────────────────────────────────────────────────────────
        if gradual_unfreeze:
            # Freeze entire backbone
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze norm + head — matches ViTGradualUnfreeze behaviour
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
            for p in self.backbone.head.parameters():
                p.requires_grad = True
            # projector: requires_grad=True by default
        else:
            # Full unfreeze — entire backbone trainable from the start
            for p in self.backbone.parameters():
                p.requires_grad = True

        # ── Hook ─────────────────────────────────────────────────────────────
        self._hook      = BlockHook(self.backbone.blocks[hook_block])
        self.hook_block = hook_block
        self._unfrozen_blocks: list[int] = []

        mode_str = (
            f"gradual_unfreeze=True | pretrained={pretrained}"
            if gradual_unfreeze
            else f"gradual_unfreeze=False (full) | pretrained={pretrained}"
        )
        print(f"  [Student ViT-Tiny] {mode_str}")
        self._print_param_summary()

    def forward(self, x: torch.Tensor):
        """
        Train → (logits, feat_proj)   feat_proj: (B, PROJ_DIM)
        Eval  → logits                projector skipped entirely
        """
        self._hook.clear()
        logits = self.backbone(x)   # (B, num_classes)

        if self.training:
            feat_raw  = self._hook.cls_token       # (B, 192)
            feat_proj = self.projector(feat_raw)   # (B, PROJ_DIM)
            return logits, feat_proj

        return logits

    def unfreeze_next_block(self, block_idx: int) -> None:
        """
        Unfreezes transformer block at block_idx.
        No-op when gradual_unfreeze=False.
        Rebuild the optimizer after calling this to include new parameters.
        """
        if not self.gradual_unfreeze:
            print(f"  ⚠ [Student] unfreeze_next_block({block_idx}) — "
                  f"gradual_unfreeze=False, skipping.")
            return
        blocks = list(self.backbone.blocks)
        if block_idx < 0 or block_idx >= len(blocks):
            print(f"  ⚠ Block {block_idx} out of range — skipping.")
            return
        if block_idx in self._unfrozen_blocks:
            print(f"  ⚠ Block {block_idx} already unfrozen — skipping.")
            return
        for p in blocks[block_idx].parameters():
            p.requires_grad = True
        self._unfrozen_blocks.append(block_idx)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  ✔ [Student] Unfroze block {block_idx}. "
              f"Trainable: {trainable:,}  "
              f"({len(self._unfrozen_blocks)}/12 blocks unfrozen)")

    def unfreeze_norm(self) -> None:
        """API compatibility — norm is already unfrozen at __init__."""
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    @property
    def total_block_count(self) -> int:
        return len(list(self.backbone.blocks))   # 12

    @property
    def unfrozen_block_count(self) -> int:
        return len(self._unfrozen_blocks)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _print_param_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  [Student ViT-Tiny] Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)")