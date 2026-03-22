"""
ProGIS interactive segmentation REST API (FastAPI).

Provides a stateless /segment endpoint: given an image and user click
coordinates, runs the full ProGIS pipeline and returns the segmentation mask.

Protocol
--------
POST /segment
  Request (JSON):
    {
      "image":      "<base64-encoded PNG or JPEG>",
      "fg_clicks":  [[x1, y1], ...],   // foreground click coordinates
      "bg_clicks":  [[x2, y2], ...],   // background click coordinates (optional)
      "n_iter":     20,                 // correction iterations (optional, default 20)
      "threshold":  0.5                 // prototype threshold (optional, default 0.5)
    }

  Response (JSON):
    {
      "mask":        "<base64-encoded PNG, single-channel 0/255>",
      "mask_shape":  [H, W]
    }

GET /health
  Returns {"status": "ok"}.

Setup
-----
  pip install fastapi uvicorn python-multipart pillow

Start server:
  python -m progis_rework.interactive.api \\
      --roi_ckpt runs/stage2/stage2_best.pth \\
      --backbone efficientunet \\
      --device cpu \\
      --host 0.0.0.0 --port 8000

  # Or with uvicorn directly (for production):
  uvicorn progis_rework.interactive.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import io
from typing import Optional

import numpy as np
import torch

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from progis_rework.data.signal_utils import generate_guiding_signal
from progis_rework.interactive.roi import (
    roi_crop_for_prototype,
    roi_crop_for_correction,
    paste_crop_into_mask,
)
from progis_rework.interactive.signals import process_masks
from progis_rework.models.progis import ProGISModel


# ── Model singleton (loaded once at startup) ──────────────────────────────────

_model: Optional[ProGISModel] = None
_device: torch.device = torch.device("cpu")
_crop_size: int = 256


def get_model() -> ProGISModel:
    if _model is None:
        raise RuntimeError(
            "Model not loaded. Call configure() before starting the server."
        )
    return _model


def configure(
    roi_ckpt:        str,
    backbone_name:   str  = "efficientunet",
    backbone_kwargs: dict | None = None,
    device:          str  = "cpu",
    crop_size:       int  = 256,
) -> None:
    """
    Load model and set global config. Call this before starting the server.

    Args:
        roi_ckpt:        path to segment_part .pth checkpoint.
        backbone_name:   'efficientunet' or 'simclr'.
        backbone_kwargs: extra kwargs for build_backbone().
        device:          torch device string.
        crop_size:       ROI crop size (default 256).
    """
    global _model, _device, _crop_size
    _device    = torch.device(device)
    _crop_size = crop_size
    _model     = ProGISModel.from_checkpoint(
        backbone_name   = backbone_name,
        roi_ckpt        = roi_ckpt,
        backbone_kwargs = backbone_kwargs or {},
    ).to(_device).eval()
    print(f"[API] Model loaded: backbone={backbone_name}  device={device}")


# ── Signal generation from clicks ────────────────────────────────────────────

def _clicks_to_signal(
    fg_clicks: list[list[int]],
    bg_clicks: list[list[int]],
    H: int,
    W: int,
    radius: int = 3,
) -> torch.Tensor:
    """
    Convert user click coordinates to a [2, H, W] guiding signal tensor.

    Each click places a filled disc of given radius on the signal map.
    The fg channel uses generate_guiding_signal to produce a skeleton-like
    signal (consistent with training); bg channel is a raw disc.

    Args:
        fg_clicks:  list of [x, y] foreground click coordinates.
        bg_clicks:  list of [x, y] background click coordinates.
        H, W:       image spatial dimensions.
        radius:     disc radius in pixels (default 3).

    Returns:
        [2, H, W] float32 tensor on CPU.
    """
    fg_map = np.zeros((H, W), dtype=np.float32)
    bg_map = np.zeros((H, W), dtype=np.float32)

    for x, y in fg_clicks:
        y_lo = max(y - radius, 0);  y_hi = min(y + radius + 1, H)
        x_lo = max(x - radius, 0);  x_hi = min(x + radius + 1, W)
        fg_map[y_lo:y_hi, x_lo:x_hi] = 1.0

    for x, y in bg_clicks:
        y_lo = max(y - radius, 0);  y_hi = min(y + radius + 1, H)
        x_lo = max(x - radius, 0);  x_hi = min(x + radius + 1, W)
        bg_map[y_lo:y_hi, x_lo:x_hi] = 1.0

    # Apply skeleton transform to fg (consistent with training distribution)
    if fg_map.sum() > 0:
        fg_signal = generate_guiding_signal(fg_map, seed=0)
    else:
        fg_signal = fg_map

    signal = np.stack([fg_signal, bg_map], axis=0)  # [2, H, W]
    return torch.tensor(signal, dtype=torch.float32)


# ── Image encode / decode ─────────────────────────────────────────────────────

def _decode_image(b64_str: str) -> tuple[torch.Tensor, int, int]:
    """base64 PNG/JPEG → [1, 3, H, W] float32 tensor + (H, W)."""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow required: pip install pillow")

    data = base64.b64decode(b64_str)
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    arr  = np.array(img, dtype=np.float32)       # [H, W, 3]
    H, W = arr.shape[:2]
    tensor = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, H, W]
    return tensor, H, W


def _encode_mask(mask: torch.Tensor) -> str:
    """[1, 1, H, W] or [H, W] binary tensor → base64 PNG (0/255)."""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow required: pip install pillow")

    arr = mask.squeeze().cpu().numpy()
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── FastAPI app ───────────────────────────────────────────────────────────────

if _FASTAPI_AVAILABLE:
    app = FastAPI(
        title="ProGIS Interactive Segmentation API",
        description="Prototype-Guided Interactive Segmentation for Pathological Images",
        version="0.1.0",
    )

    class SegmentRequest(BaseModel):
        image:      str                      # base64-encoded image
        fg_clicks:  list[list[int]] = []     # [[x,y], ...]
        bg_clicks:  list[list[int]] = []
        n_iter:     int   = 20
        threshold:  float = 0.5

    class SegmentResponse(BaseModel):
        mask:       str          # base64-encoded single-channel PNG
        mask_shape: list[int]    # [H, W]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/segment", response_model=SegmentResponse)
    def segment(req: SegmentRequest) -> SegmentResponse:
        """
        Run ProGIS interactive segmentation.

        Returns binary mask as base64-encoded single-channel PNG.
        """
        model = get_model()

        # ── Decode image ──────────────────────────────────────────────────
        try:
            image_t, H, W = _decode_image(req.image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

        image_t = image_t.to(_device)

        if not req.fg_clicks and not req.bg_clicks:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one fg_click or bg_click.",
            )

        # ── Build signal from clicks ──────────────────────────────────────
        signal = _clicks_to_signal(
            req.fg_clicks, req.bg_clicks, H, W
        ).unsqueeze(0).to(_device)  # [1, 2, H, W]

        # ── Dummy GT mask (all zeros — no GT at inference time) ───────────
        dummy_gt = torch.zeros(1, 1, H, W, device=_device)

        # ── Prototype initialisation ──────────────────────────────────────
        with torch.no_grad():
            proto_crop = roi_crop_for_prototype(
                image_t, signal, dummy_gt, _crop_size,
            )
            proto_out = model.forward_prototype(
                roi_input  = proto_crop.roi_images,
                roi_signal = proto_crop.roi_signals,
                full_image = image_t,
                mask_box   = proto_crop.mask_box,
                threshold  = req.threshold,
            )

            # ── Correction loop ───────────────────────────────────────────
            current_mask = proto_out.prototype_mask.clone()
            union_signal = signal.clone()

            for _ in range(req.n_iter):
                err_signal, centers = process_masks(current_mask, dummy_gt)
                union_signal = torch.bitwise_or(
                    err_signal.to(torch.uint8),
                    union_signal.to(torch.uint8),
                ).float()

                crop_batch = roi_crop_for_correction(
                    image_t, current_mask, union_signal, centers, _crop_size,
                )
                crop_pred = model.segment(
                    crop_batch.roi_images,
                    crop_batch.roi_prev_masks,
                    crop_batch.roi_signals,
                )
                paste_crop_into_mask(
                    current_mask, crop_pred, centers, H, W, _crop_size,
                )

        mask_b64 = _encode_mask(current_mask)
        return SegmentResponse(mask=mask_b64, mask_shape=[H, W])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn pillow")
        return

    parser = argparse.ArgumentParser(
        description="Start ProGIS interactive segmentation API server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--roi_ckpt",   required=True)
    parser.add_argument("--backbone",   default="efficientunet",
                        choices=["efficientunet", "simclr"])
    parser.add_argument("--proj_ckpt",  default="",
                        help="SimCLR projection head checkpoint.")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--crop_size",  type=int, default=256)
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8000)
    args = parser.parse_args()

    backbone_kwargs = {}
    if args.backbone == "simclr" and args.proj_ckpt:
        backbone_kwargs["proj_ckpt"] = args.proj_ckpt

    configure(
        roi_ckpt        = args.roi_ckpt,
        backbone_name   = args.backbone,
        backbone_kwargs = backbone_kwargs,
        device          = args.device,
        crop_size       = args.crop_size,
    )

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        return

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
