import torch
import torch.nn as nn
import timm


class ResNetGradualUnfreeze(nn.Module):
    """
    Gradual unfreezing for ResNet architectures.

    Starts with only the classifier trainable, then progressively unfreezes
    residual stages from deepest to shallowest:

        layer4 -> layer3 -> layer2 -> layer1

    This mirrors the ViT gradual unfreezing strategy while adapting to
    ResNet's stage-based hierarchy.

    Recommended model for similar parameter count to ViT-Base (~86M):
        - resnet152 (~60M)
        - resnet200d (~64M)
        - resnetrs152 (~83M)

    Example:
        model = ResNetGradualUnfreeze(
            num_classes=37,
            model_name="resnetrs152"
        )

        # Head only
        for epoch in range(5):
            train(...)

        # Gradually unfreeze deeper stages
        for stage in [4, 3, 2, 1]:
            model.unfreeze_next_stage(stage)

            optimizer = torch.optim.AdamW(
                model.trainable_params(),
                lr=1e-4
            )

            for epoch in range(3):
                train(...)
    """

    def __init__(
        self,
        num_classes: int = 37,
        model_name: str = "resnetrs152",
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
        )

        self._unfrozen_stages = []

        # Freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace classifier
        if hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        elif hasattr(self.backbone, "classifier"):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(
                in_features,
                num_classes
            )

        else:
            raise ValueError(
                "Unknown classifier structure in backbone."
            )

    def unfreeze_next_block(self, block_idx: int) -> None:
        """
        Unfreezes a ResNet stage.

        Stage mapping:
            1 -> layer1
            2 -> layer2
            3 -> layer3
            4 -> layer4

        Recommended order:
            4 -> 3 -> 2 -> 1

        Args:
            block_idx: ResNet stage index.
        """

        valid_stages = [1, 2, 3, 4]

        if block_idx not in valid_stages:
            raise ValueError(
                f"block_idx must be one of {valid_stages}"
            )

        layer_name = f"layer{block_idx}"

        if not hasattr(self.backbone, layer_name):
            raise ValueError(
                f"{self.backbone.__class__.__name__} "
                f"does not contain {layer_name}"
            )

        layer = getattr(self.backbone, layer_name)

        for param in layer.parameters():
            param.requires_grad = True

        if block_idx not in self._unfrozen_stages:
            self._unfrozen_stages.append(block_idx)

        print(
            f"Unfroze {layer_name} | "
            f"Total unfrozen stages: "
            f"{len(self._unfrozen_stages)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trainable_params(self):
        """
        Returns only trainable parameters.
        """
        return filter(
            lambda p: p.requires_grad,
            self.parameters()
        )

    @property
    def unfrozen_block_count(self) -> int:
        return len(self._unfrozen_stages)

    @property
    def total_block_count(self) -> int:
        return 4