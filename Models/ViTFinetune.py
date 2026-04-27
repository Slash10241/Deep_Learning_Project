import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


class ViTLinearProbe(nn.Module):
    """
    Linear Probing.

    Freezes the entire ViT backbone and trains only the classification head.

    Args:
        num_classes:  Number of output classes (37 for Oxford-IIIT Pets).
        model_name:   Any timm ViT variant string.
    """

    def __init__(
        self,
        num_classes: int = 37,
        model_name: str = "vit_base_patch16_224",
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=True)

        # freeze the entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # replace classification head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # raw logits

    def trainable_params(self):
        """Returns only the trainable parameters (head only)."""
        return filter(lambda p: p.requires_grad, self.parameters())


class ViTSimultaneousUnfreeze(nn.Module):
    """
    Simultaneous Last-l-Layer Unfreezing.

    Freezes all layers except the last `unfreeze_last_l` transformer blocks,
    the final LayerNorm, and the classification head. All unfrozen layers
    are trained together from the start.

    Args:
        num_classes:        Number of output classes.
        unfreeze_last_l:    How many transformer blocks (counting from the
                            output end) to unfreeze alongside the head.
        model_name:         Any timm ViT variant string.
    """

    def __init__(
        self,
        num_classes: int = 37,
        unfreeze_last_l: int = 4,
        model_name: str = "vit_base_patch16_224",
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=True)
        self.unfreeze_last_l = unfreeze_last_l

        # freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # unfreeze the last l transformer blocks
        total_blocks = len(self.backbone.blocks)
        if unfreeze_last_l > total_blocks:
            raise ValueError(
                f"unfreeze_last_l ({unfreeze_last_l}) exceeds total "
                f"number of blocks ({total_blocks})."
            )

        for block in self.backbone.blocks[total_blocks - unfreeze_last_l:]:
            for param in block.parameters():
                param.requires_grad = True

        # unfreeze final LayerNorm
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # replace and unfreeze classification head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # raw logits

    def trainable_params(self):
        """Returns only the trainable parameters (last l blocks + head)."""
        return filter(lambda p: p.requires_grad, self.parameters())

    def frozen_block_count(self) -> int:
        """Returns the number of frozen transformer blocks."""
        return len(self.backbone.blocks) - self.unfreeze_last_l


class ViTGradualUnfreeze(nn.Module):
    """
    Starts with only the classification head trainable, then exposes one
    transformer block at a time (from the output end toward the input),
    allowing each block to adapt before the next is unfrozen. The optimizer
    must be re-instantiated after each call to `unfreeze_next_block` so the
    new parameters are included in the update.

    Args:
        num_classes:  Number of output classes.
        model_name:   Any timm ViT variant string.

    Example training loop::

        model = ViTGradualUnfreeze(num_classes=37)
        engine = TrainingEngine(network=model, ...)

        # phase 0: head only
        for epoch in range(5):
            engine.train_one_epoch(epoch)

        # gradually unfreeze from last block backward
        total_blocks = len(model.backbone.blocks)
        for block_idx in range(total_blocks - 1, -1, -1):
            model.unfreeze_next_block(block_idx)
            engine.opt = torch.optim.Adam(model.trainable_params(), lr=1e-4)
            for epoch in range(3):
                engine.train_one_epoch(epoch)
    """

    def __init__(
        self,
        num_classes: int = 37,
        model_name: str = "vit_base_patch16_224",
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=True)
        self._unfrozen_block_indices: list[int] = []

        # freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # unfreeze final LayerNorm
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # replace and unfreeze classification head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

    def unfreeze_next_block(self, block_idx: int) -> None:
        """
        Unfreezes transformer block at position `block_idx`.

        After calling this, re-instantiate the optimizer so the newly
        unfrozen parameters are included in the update step.

        Args:
            block_idx: Index into self.backbone.blocks to unfreeze.
        """
        total_blocks = len(self.backbone.blocks)
        if block_idx < 0 or block_idx >= total_blocks:
            raise ValueError(
                f"block_idx {block_idx} is out of range "
                f"[0, {total_blocks - 1}]."
            )

        block = self.backbone.blocks[block_idx]
        for param in block.parameters():
            param.requires_grad = True

        self._unfrozen_block_indices.append(block_idx)
        print(
            f"Unfrozen block {block_idx}/{total_blocks - 1} | "
            f"Total unfrozen blocks: {len(self._unfrozen_block_indices)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # raw logits

    def trainable_params(self):
        """Returns only the trainable parameters for use with an optimizer."""
        return filter(lambda p: p.requires_grad, self.parameters())

    @property
    def unfrozen_block_count(self) -> int:
        return len(self._unfrozen_block_indices)

    @property
    def total_block_count(self) -> int:
        return len(self.backbone.blocks)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Linear Probe ---")
    lp_model = ViTLinearProbe(num_classes=37)
    trainable = sum(p.numel() for p in lp_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in lp_model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    print("\n--- Simultaneous Unfreeze (last 4 blocks) ---")
    sim_model = ViTSimultaneousUnfreeze(num_classes=37, unfreeze_last_l=4)
    trainable = sum(p.numel() for p in sim_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in sim_model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    print(f"Frozen blocks: {sim_model.frozen_block_count()}")

    print("\n--- Gradual Unfreeze (head only at init) ---")
    grad_model = ViTGradualUnfreeze(num_classes=37)
    trainable = sum(p.numel() for p in grad_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in grad_model.parameters())
    print(f"Trainable params at init: {trainable:,} / {total:,}")

    grad_model.unfreeze_next_block(11)
    grad_model.unfreeze_next_block(10)
    trainable = sum(p.numel() for p in grad_model.parameters() if p.requires_grad)
    print(f"Trainable params after unfreezing blocks 11+10: {trainable:,} / {total:,}")