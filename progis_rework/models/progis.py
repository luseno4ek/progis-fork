"""
ProGISModel — backbone-agnostic interactive segmentation model.

Unifies EfficientUNet_proto and SimCLRProto from the original codebase into
a single class. The backbone is injected via FeatureExtractorBase, so swapping
feature extractors requires only changing one argument.

Architecture
------------
  backbone     : frozen feature extractor → [B, C, H, W] feature map
  segment_part : 6-channel EfficientUNet-B0 → [B, 1, H/W, H/W] segmentation

Two-phase forward
-----------------
Training (Stage 2, CU-Training):
    # called per batch in the training loop
    pred1 = model.segment(images, zeros_mask, signal)
    new_signal = process_masks(pred1, gt)[0]
    pred2 = model.segment(images, threshold(pred1), union(new_signal, signal))
    loss  = dice_loss(pred1, gt) + dice_loss(pred2, gt)

Inference (prototype initialisation + iterative correction):
    out = model.forward_prototype(roi_input, roi_signal, full_image, mask_box, threshold)
    # out.prototype_mask   : initial prototype-guided mask
    # out.roi_mask         : ROI segmentation mask (thresholded at 0.95)
    # out.similarity_map   : continuous cosine-similarity map
    # then iterative correction loop calls model.segment() up to 20 times

Usage
-----
    from progis_rework.models.backbones import build_backbone
    from progis_rework.models.progis import ProGISModel

    # Build & train
    model = ProGISModel(build_backbone('efficientunet'))

    # Load trained checkpoint
    model = ProGISModel.from_checkpoint(
        backbone_name='efficientunet',
        roi_ckpt='/path/to/best.pth',
    )
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import FeatureExtractorBase, build_backbone

# ── Path bootstrap (same as backbones.py) ────────────────────────────────────
_MODELS_DIR = Path(__file__).parents[2] / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))


# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class PrototypeOutput:
    """
    Result of ProGISModel.forward_prototype().

    Attributes:
        prototype_mask:  [B, 1, H, W] binary — cosine-similarity map thresholded
                         at the given threshold. Used as initial segmentation.
        roi_mask:        [B, 1, H, W] binary — ROI segmentation output thresholded
                         at 0.95. Used to identify fg pixels for prototype.
        similarity_map:  [B, 1, H, W] float in [0, 1] — continuous normalised
                         cosine-similarity map. Used in the correction loop.
        features:        [B, C, H, W] — backbone feature map (full image).
    """
    prototype_mask:  torch.Tensor
    roi_mask:        torch.Tensor
    similarity_map:  torch.Tensor
    features:        torch.Tensor


# ── Model ─────────────────────────────────────────────────────────────────────

class ProGISModel(nn.Module):
    """
    ProGIS interactive segmentation model.

    Args:
        backbone:      FeatureExtractorBase instance (frozen feature extractor).
        roi_crop_size: spatial size of the ROI crop fed to segment_part (default 256).
        seg_threshold: threshold for binarising the initial ROI segmentation
                       before computing the prototype (default 0.95).
    """

    def __init__(
        self,
        backbone:      Optional[FeatureExtractorBase] = None,
        roi_crop_size: int   = 256,
        seg_threshold: float = 0.95,
    ):
        super().__init__()
        from models.efficientunet import get_efficientunet_b0
        self.backbone      = backbone
        self.segment_part  = get_efficientunet_b0(
            out_channels=1, concat_input=True, pretrained=False, backbone=False
        )
        self.roi_crop_size = roi_crop_size
        self.seg_threshold = seg_threshold

    # ── Training interface ────────────────────────────────────────────────────

    def segment(
        self,
        image_crop: torch.Tensor,   # [B, 3, H, W]
        prev_mask:  torch.Tensor,   # [B, 1, H, W]
        signal:     torch.Tensor,   # [B, 2, H, W]
    ) -> torch.Tensor:
        """
        Single 6-channel UNet forward pass.

        Used both in the Stage-2 training loop (CU-Training) and in the
        iterative correction loop during inference.

        Input channels = RGB(3) + prev_mask(1) + fg_signal(1) + bg_signal(1) = 6.

        Returns:
            [B, 1, H, W] float32 segmentation probability map.
        """
        x = torch.cat([image_crop, prev_mask, signal], dim=1)   # 6 channels
        return self.segment_part(x)

    # ── Inference interface ───────────────────────────────────────────────────

    def forward_prototype(
        self,
        roi_input:  torch.Tensor,   # [B, 3, crop, crop]   — cropped RGB
        roi_signal: torch.Tensor,   # [B, 2, crop, crop]   — cropped guiding signal
        full_image: torch.Tensor,   # [B, 3, H, W]         — full image
        mask_box:   torch.Tensor,   # [B, 1, H, W]         — ROI box mask
        threshold:  float = 0.5,
    ) -> PrototypeOutput:
        """
        Prototype initialisation forward (inference only, backbone is used here).

        Steps:
          1. Initial ROI segmentation → roi_mask (thresholded at seg_threshold).
          2. Backbone feature extraction from the full image → features [B, C, H, W].
          3. Crop features around the ROI box centre (256×256 window).
          4. Compute prototype = mean feature vector at roi_mask foreground pixels.
          5. Cosine similarity between per-pixel features and prototype → similarity_map.
          6. Threshold similarity_map at `threshold` → prototype_mask.

        Args:
            roi_input:  [B, 3, crop, crop] — ROI crop of the RGB image.
            roi_signal: [B, 2, crop, crop] — ROI crop of the guiding signal.
            full_image: [B, 3, H, W]       — full-resolution image.
            mask_box:   [B, 1, H, W]       — binary mask marking the ROI region.
            threshold:  cosine-similarity threshold for binarisation.

        Returns:
            PrototypeOutput with prototype_mask, roi_mask, similarity_map, features.
        """
        B = roi_input.shape[0]
        device = roi_input.device

        # ── 1. Initial ROI segmentation ──────────────────────────────────────
        zero_prev = torch.zeros(B, 1, *roi_input.shape[2:], device=device)
        roi_seg   = self.segment(roi_input, zero_prev, roi_signal)   # [B,1,crop,crop]

        roi_mask = (roi_seg > self.seg_threshold).float()            # binary at 0.95

        # ── 2. Backbone feature extraction ───────────────────────────────────
        if self.backbone is None:
            raise RuntimeError(
                "forward_prototype() requires a backbone. "
                "Pass backbone= to ProGISModel or use from_checkpoint()."
            )
        features = self.backbone(full_image)                         # [B, C, H, W]
        C = features.shape[1]

        # ── 3. Crop features around ROI box centre ───────────────────────────
        crop      = self.roi_crop_size
        box       = mask_box.squeeze(1)                              # [B, H, W]
        x_cropped = features * box.unsqueeze(1)                      # [B, C, H, W]

        cropped_regions: list[torch.Tensor] = []
        for i in range(B):
            feat = x_cropped[i]                                      # [C, H, W]
            nz   = torch.nonzero(box[i], as_tuple=True)
            if len(nz[0]) > 0:
                cy = int(((nz[0].min() + nz[0].max()) // 2).item())
                cx = int(((nz[1].min() + nz[1].max()) // 2).item())
                sy = min(max(cy - crop // 2, 0), feat.shape[1] - crop)
                sx = min(max(cx - crop // 2, 0), feat.shape[2] - crop)
                cropped_regions.append(feat[:, sy:sy+crop, sx:sx+crop])
            else:
                cropped_regions.append(
                    torch.zeros(C, crop, crop, dtype=features.dtype, device=device)
                )

        cropped = torch.stack(cropped_regions)                       # [B, C, crop, crop]

        # ── 4. Prototype = mean feature at foreground pixels ─────────────────
        fg_count   = roi_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)  # [B,1,1,1]
        prototype  = (cropped * roi_mask).sum(dim=(2, 3), keepdim=True) / fg_count
        prototype  = prototype.squeeze(3).squeeze(2)                 # [B, C]

        # ── 5. Cosine similarity map over full image ──────────────────────────
        feat_norm  = F.normalize(features,  dim=1)                   # [B, C, H, W]
        proto_norm = F.normalize(prototype, dim=1)                   # [B, C]
        sim        = torch.einsum("bchw,bc->bhw", feat_norm, proto_norm) ** 2
        sim        = sim.unsqueeze(1)                                 # [B, 1, H, W]

        # ── 6. Normalise and threshold ────────────────────────────────────────
        sim_min  = sim.flatten(1).min(1).values.view(B, 1, 1, 1)
        sim_max  = sim.flatten(1).max(1).values.view(B, 1, 1, 1)
        sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-6)      # [B,1,H,W] in [0,1]

        proto_mask = (sim_norm > threshold).float()

        return PrototypeOutput(
            prototype_mask = proto_mask,
            roi_mask       = roi_mask,
            similarity_map = sim_norm,
            features       = features,
        )

    def forward_prototype_multiclass(
        self,
        roi_inputs:  list[torch.Tensor],   # n_cls × [B, 3, crop, crop]
        roi_signals: list[torch.Tensor],   # n_cls × [B, 2, crop, crop]
        full_image:  torch.Tensor,         # [B, 3, H, W]
        mask_boxes:  list[torch.Tensor],   # n_cls × [B, 1, H, W]
        threshold:   float = 0.5,
    ) -> list[PrototypeOutput]:
        """
        Multi-class prototype navigation without pixel-level class overlap.

        Runs the prototype initialisation for all classes simultaneously,
        then resolves conflicts via argmax: each pixel is assigned to the
        class with the highest normalised similarity, but only if that
        similarity exceeds `threshold`. If no class exceeds the threshold
        the pixel is background in all classes.

        The backbone is called **once** and its features are shared across
        all classes — more efficient than calling forward_prototype() N times.

        Args:
            roi_inputs:  list of [B, 3, crop, crop] — per-class ROI crops.
            roi_signals: list of [B, 2, crop, crop] — per-class guiding signals.
            full_image:  [B, 3, H, W] — full-resolution image (shared).
            mask_boxes:  list of [B, 1, H, W] — per-class ROI box masks.
            threshold:   cosine-similarity threshold for binarisation.

        Returns:
            List of PrototypeOutput, one per class, with non-overlapping masks.
        """
        if self.backbone is None:
            raise RuntimeError(
                "forward_prototype_multiclass() requires a backbone."
            )

        n_cls  = len(roi_inputs)
        B      = full_image.shape[0]
        device = full_image.device
        crop   = self.roi_crop_size

        # ── 1. Shared backbone feature extraction (called once) ───────────────
        features = self.backbone(full_image)                         # [B, C, H, W]
        C = features.shape[1]
        feat_norm = F.normalize(features, dim=1)                     # [B, C, H, W]

        # ── 2-5. Per-class: roi_mask → prototype → similarity map ─────────────
        roi_masks:   list[torch.Tensor] = []
        sim_norms:   list[torch.Tensor] = []   # raw sim_norm per class

        for k in range(n_cls):
            roi_input  = roi_inputs[k]
            roi_signal = roi_signals[k]
            mask_box   = mask_boxes[k]

            # Step 1: initial ROI segmentation
            zero_prev = torch.zeros(B, 1, *roi_input.shape[2:], device=device)
            roi_seg   = self.segment(roi_input, zero_prev, roi_signal)
            roi_mask  = (roi_seg > self.seg_threshold).float()

            # Step 3: crop features around ROI box centre
            box = mask_box.squeeze(1)                                # [B, H, W]
            x_cropped = features * box.unsqueeze(1)                  # [B, C, H, W]

            cropped_regions: list[torch.Tensor] = []
            for i in range(B):
                feat = x_cropped[i]
                nz   = torch.nonzero(box[i], as_tuple=True)
                if len(nz[0]) > 0:
                    cy = int(((nz[0].min() + nz[0].max()) // 2).item())
                    cx = int(((nz[1].min() + nz[1].max()) // 2).item())
                    sy = min(max(cy - crop // 2, 0), feat.shape[1] - crop)
                    sx = min(max(cx - crop // 2, 0), feat.shape[2] - crop)
                    cropped_regions.append(feat[:, sy:sy+crop, sx:sx+crop])
                else:
                    cropped_regions.append(
                        torch.zeros(C, crop, crop, dtype=features.dtype, device=device)
                    )
            cropped = torch.stack(cropped_regions)                   # [B, C, crop, crop]

            # Step 4: prototype = mean feature at foreground pixels
            fg_count  = roi_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            prototype = (cropped * roi_mask).sum(dim=(2, 3), keepdim=True) / fg_count
            prototype = prototype.squeeze(3).squeeze(2)              # [B, C]

            # Step 5: cosine similarity map
            proto_norm = F.normalize(prototype, dim=1)               # [B, C]
            sim = torch.einsum("bchw,bc->bhw", feat_norm, proto_norm) ** 2
            sim = sim.unsqueeze(1)                                   # [B, 1, H, W]

            sim_min = sim.flatten(1).min(1).values.view(B, 1, 1, 1)
            sim_max = sim.flatten(1).max(1).values.view(B, 1, 1, 1)
            sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-6)  # [B, 1, H, W]

            roi_masks.append(roi_mask)
            sim_norms.append(sim_norm)

        # ── 6. Argmax conflict resolution ─────────────────────────────────────
        # sim_stack: [B, n_cls, H, W]
        sim_stack = torch.cat(sim_norms, dim=1)                      # [B, n_cls, H, W]
        best_sim, best_cls = sim_stack.max(dim=1)                    # [B, H, W] each

        outputs: list[PrototypeOutput] = []
        for k in range(n_cls):
            # Pixel belongs to class k iff:
            #   (a) class k has the highest similarity among all classes
            #   (b) that similarity exceeds the threshold
            proto_mask_k = (
                (best_cls == k) & (best_sim > threshold)
            ).float().unsqueeze(1)                                   # [B, 1, H, W]

            outputs.append(PrototypeOutput(
                prototype_mask = proto_mask_k,
                roi_mask       = roi_masks[k],
                similarity_map = sim_norms[k],
                features       = features,
            ))

        return outputs

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        backbone_name: str,
        roi_ckpt:      str,
        backbone_kwargs: Optional[dict] = None,
        **model_kwargs,
    ) -> "ProGISModel":
        """
        Build model and load a trained segment_part checkpoint.

        Args:
            backbone_name:   'efficientunet' or 'simclr'.
            roi_ckpt:        path to segment_part state dict (.pth).
            backbone_kwargs: kwargs forwarded to build_backbone().
            **model_kwargs:  kwargs forwarded to ProGISModel.__init__().

        Returns:
            ProGISModel with loaded weights (eval mode, all params frozen).

        Example:
            model = ProGISModel.from_checkpoint(
                backbone_name='simclr',
                roi_ckpt='/data/best.pth',
                backbone_kwargs={'proj_channels': 32, 'proj_ckpt': '/data/proj.pth'},
            )
        """
        backbone = build_backbone(backbone_name, **(backbone_kwargs or {}))
        model    = cls(backbone, **model_kwargs)

        ckpt_path = Path(roi_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ROI checkpoint not found: {roi_ckpt}")

        model.segment_part.load_state_dict(
            torch.load(ckpt_path, map_location="cpu")
        )
        print(f"[ProGISModel] Loaded segment_part: {roi_ckpt}")

        for p in model.parameters():
            p.requires_grad = False

        return model.eval()
