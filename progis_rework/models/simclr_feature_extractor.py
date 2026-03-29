"""
SimCLR ResNet50 feature extractor pretrained on TCGA-BRCA histology.

Model: hf-hub:1aurent/resnet50.tcga_brca_simclr
       Self-supervised SimCLR, trained on 2M patches from TCGA Breast Cancer.

Provides per-pixel features [B, proj_channels, H, W] for ProGIS prototype
matching. The encoder is fully frozen — Stage 1 contrastive training is not
needed because SimCLR already provides good histology representations.

Uses ResNet50 stage 2 (out_indices=[2]) for better spatial resolution:
  - stage 2: H/8, 512 channels  → 64×64 for 512×512 input  ← used
  - stage 4: H/32, 2048 channels → 16×16 for 512×512 input  (too coarse)

Only the small 1×1 projection head (512 → proj_channels) is a new layer.

Usage:
    extractor = SimCLRFeatureExtractor(proj_channels=32)
    features = extractor(images)   # images: [B, 3, H, W], float32, 0-255
    # features: [B, 32, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRFeatureExtractor(nn.Module):
    """
    Frozen ResNet50-SimCLR encoder + small trainable projection head.

    Architecture:
        ResNet50 encoder (frozen, SimCLR pretrained on TCGA-BRCA)
        → [B, 512, H/8, W/8]   (stage 2, better spatial resolution)
        → Conv2d(512, proj_channels, 1)  [trainable projection]
        → [B, proj_channels, H/8, W/8]
        → bilinear upsample to input (H, W)
        → [B, proj_channels, H, W]

    Input:  [B, 3, H, W], float32, pixel values 0-255
    Output: [B, proj_channels, H, W]
    """

    # ImageNet normalization used by the SimCLR model
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    # Number of output channels from the selected ResNet50 stage
    _STAGE_CHANNELS = 512   # out_indices=[2] → layer2 output

    def __init__(self, proj_channels: int = 32):
        super().__init__()
        self.encoder = self._build_encoder()

        # 1×1 projection: 512 → proj_channels
        self.proj = nn.Conv2d(self._STAGE_CHANNELS, proj_channels, kernel_size=1, bias=False)

        # Normalization buffers — move automatically with .to(device)
        self.register_buffer('mean', torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(self._STD ).view(1, 3, 1, 1))

    @staticmethod
    def _build_encoder():
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for SimCLRFeatureExtractor.\n"
                "Install with:  pip install 'timm>=0.9.0'"
            )
        encoder = timm.create_model(
            "hf-hub:1aurent/resnet50.tcga_brca_simclr",
            pretrained=True,
            features_only=True,
            out_indices=[2],   # stage 2 → [B, 512, H/8, W/8]  (64×64 for 512 input)
        )
        # Freeze encoder: no gradient, no update
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W], float32, pixel values 0-255
        returns: [B, proj_channels, H, W]
        """
        H, W = x.shape[-2], x.shape[-1]

        # Normalise pixel values 0-255 → ImageNet distribution
        x_norm = x / 255.0
        x_norm = (x_norm - self.mean) / self.std

        # Frozen encoder forward (no gradient through encoder)
        with torch.no_grad():
            feat = self.encoder(x_norm)[0]   # [B, 512, H/8, W/8]

        # Project to low-dim space (gradient flows here)
        feat = self.proj(feat)               # [B, proj_channels, H/8, W/8]

        # Upsample back to input resolution
        feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
        return feat                          # [B, proj_channels, H, W]
