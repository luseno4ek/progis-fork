"""
Feature extractor backbones for ProGIS prototype matching.

Backbones produce per-pixel feature maps [B, proj_channels, H, W] from a
full-resolution input image. They are used ONLY in the inference prototype
initialisation step — NOT during training (training uses only segment_part).

Available backbones
-------------------
  'efficientunet'  EfficientUNet-B0 encoder, ImageNet-pretrained, frozen.
                   Output channels: 32 (hardcoded by the EfficientUNet-B0 arch).

  'simclr'         ResNet50 pretrained with SimCLR on 2M TCGA-BRCA patches.
                   Frozen encoder + small trainable projection head.
                   Output channels: proj_channels (default 32).
                   Requires: pip install 'timm>=0.9.0'

Usage
-----
    from progis_rework.models.backbones import build_backbone

    backbone = build_backbone('efficientunet')
    backbone = build_backbone('simclr', proj_channels=32)
    backbone = build_backbone('simclr', proj_channels=32, proj_ckpt='/path/proj.pth')

Note on imports
---------------
EfficientUNet and SimCLR live in the original `models/` directory.
This module adds that directory to sys.path automatically (same-repo setup).
When moving to a standalone package, those files should be vendored here.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# ── Path bootstrap: make original models/ importable ─────────────────────────
_MODELS_DIR = Path(__file__).parents[2] / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))


# ── Abstract base ─────────────────────────────────────────────────────────────

class FeatureExtractorBase(ABC, nn.Module):
    """
    Common interface for all ProGIS feature extractors.

    Subclasses must implement forward() with the signature below.
    The encoder is expected to be frozen; only a small projection head
    (if any) is trainable.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] float32, pixel values 0-255.

        Returns:
            [B, proj_channels, H, W] float32 feature map.
        """
        ...


# ── EfficientUNet-B0 backbone ─────────────────────────────────────────────────

class EfficientUNetBackbone(FeatureExtractorBase):
    """
    EfficientUNet-B0 used as a frozen feature extractor (backbone=True mode).

    This is the Stage-1 encoder from the original ProGIS paper.
    It is pretrained on ImageNet and, optionally, fine-tuned via contrastive
    learning (Stage 1 training in backbone_efficientunet_train.py).

    Output: [B, 32, H, W] — same spatial resolution as the input.

    Args:
        pretrained: load ImageNet weights (default True).
        freeze:     freeze all parameters (default True).
                    Set False only when continuing Stage-1 contrastive training.
    """

    OUT_CHANNELS: int = 32   # fixed by EfficientUNet-B0 architecture

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        from efficientunet import get_efficientunet_b0
        self.net = get_efficientunet_b0(
            out_channels=1, concat_input=True,
            pretrained=pretrained, backbone=True,
        )
        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [B, 32, H, W]


# ── SimCLR ResNet50 backbone ──────────────────────────────────────────────────

class SimCLRBackbone(FeatureExtractorBase):
    """
    Frozen ResNet50 pretrained with SimCLR on TCGA-BRCA histology images
    + a small trainable 1×1 projection head.

    Stage-1 contrastive training is NOT needed — SimCLR already provides
    rich histology representations.

    Args:
        proj_channels: output feature channels after projection (default 32).
        proj_ckpt:     path to a saved projection head state dict.
                       If None or file missing, projection head is randomly init.
        freeze_encoder: freeze the ResNet50 encoder (default True).
    """

    def __init__(
        self,
        proj_channels:  int  = 32,
        proj_ckpt:      Optional[str] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        from progis_rework.models.simclr_feature_extractor import SimCLRFeatureExtractor
        self.extractor = SimCLRFeatureExtractor(proj_channels=proj_channels)

        if not freeze_encoder:
            for p in self.extractor.encoder.parameters():
                p.requires_grad = True

        if proj_ckpt is not None:
            ckpt_path = Path(proj_ckpt)
            if ckpt_path.exists():
                self.extractor.proj.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu")
                )
                print(f"[SimCLRBackbone] Loaded projection head: {proj_ckpt}")
            else:
                print(f"[SimCLRBackbone] proj_ckpt not found, using random init: {proj_ckpt}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)   # [B, proj_channels, H, W]


# ── Petroscope ResNet34 backbone ──────────────────────────────────────────────

class PetroscopeResNetBackbone(FeatureExtractorBase):
    """
    Frozen ResNet34 encoder from petroscope (pretrained on LumenStone S1+S2)
    + a small trainable 1×1 projection head (512 → proj_channels).

    Architecture:
      petroscope ResUNet(backbone="resnet34") → backbone_features (layer0..layer4)
      layer4 output: [B, 512, H/32, W/32]
      bilinear upsample → [B, 512, H, W]
      1×1 conv → [B, proj_channels, H, W]

    The encoder is frozen; only the projection head is trained (during Stage 2).
    Input pixel values are expected in [0, 255] (same as other backbones).

    Args:
        proj_channels: output feature channels (default 32).
        model_name:    key in petroscope MODEL_REGISTRY (default 's1s2_resnet34_x05').
    """

    def __init__(
        self,
        proj_channels: int = 32,
        model_name:    str = "s1s2_resnet34_x05",
    ):
        super().__init__()
        from petroscope.segmentation.models.resunet import ResUNet as PetroResUNet

        petro = PetroResUNet.from_pretrained(model_name, device="cpu")
        self.encoder = petro.model.backbone_features  # nn.ModuleDict: layer0..layer4

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.proj = nn.Conv2d(512, proj_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = x.shape[2], x.shape[3]
        x = x / 255.0                           # petroscope normalises to [0, 1]
        x = self.encoder["layer0"](x)
        x = self.encoder["layer1"](x)
        x = self.encoder["layer2"](x)
        x = self.encoder["layer3"](x)
        x = self.encoder["layer4"](x)           # [B, 512, H/32, W/32]
        x = nn.functional.interpolate(
            x, size=orig_hw, mode="bilinear", align_corners=False,
        )                                        # [B, 512, H, W]
        return self.proj(x)                      # [B, proj_channels, H, W]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_backbone(name: str, **kwargs) -> FeatureExtractorBase:
    """
    Instantiate a feature extractor backbone by name.

    Args:
        name:   'efficientunet', 'simclr', or 'petroscope_resnet34'.
        **kwargs: forwarded to the backbone constructor.

    Returns:
        FeatureExtractorBase instance.

    Example:
        backbone = build_backbone('efficientunet')
        backbone = build_backbone('simclr', proj_channels=32, proj_ckpt='/p.pth')
        backbone = build_backbone('petroscope_resnet34', proj_channels=32)
    """
    registry: dict[str, type[FeatureExtractorBase]] = {
        "efficientunet":       EfficientUNetBackbone,
        "simclr":              SimCLRBackbone,
        "petroscope_resnet34": PetroscopeResNetBackbone,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list(registry)}"
        )
    return registry[name](**kwargs)
