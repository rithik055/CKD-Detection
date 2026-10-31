import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetCKD(nn.Module):
    """
    EfficientNet-B0 backbone for 4-class CKD stage classification.

    - Uses ImageNet pretraining.
    - Accepts 1-channel input (we repeat to 3 channels inside forward).
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Replace classifier head: original is (Dropout, Linear)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your dataset gives 1-channel images: (N, 1, H, W)
        # EfficientNet expects 3-channel: repeat along channel dim
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.backbone(x)
        return logits
