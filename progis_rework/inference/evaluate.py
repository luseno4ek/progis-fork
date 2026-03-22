"""
ProGIS evaluation: Number-of-Interactions (NoI) metric.

Runs the full inference pipeline on a validation set:
  1. ROI crop around the initial fg signal.
  2. Prototype initialisation via ProGISModel.forward_prototype().
  3. Iterative correction loop (up to n_iter rounds):
       - process_masks → error-region centroid
       - roi_crop_for_correction → crop around error
       - model.segment() → patch prediction
       - paste back into full mask
  4. Record per-sample Dice and mIoU at every interaction count.

The primary reported metric is Dice@N (mean Dice after N interactions),
matching Table 1-2 in the ProGIS paper.

Usage
-----
  python -m progis_rework.inference.evaluate \\
      --patches_dir data/patches \\
      --splits_json data/patches/fold_splits.json \\
      --fold 1 --cls all \\
      --roi_ckpt runs/stage2/stage2_best.pth \\
      --backbone efficientunet \\
      --n_iter 20 --threshold 0.5 \\
      --device cpu --batch_size 4

  # SimCLR backbone:
  python -m progis_rework.inference.evaluate \\
      --roi_ckpt ... --backbone simclr \\
      --proj_ckpt runs/simclr_proj/proj_best.pth
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from progis_rework.data.dataset import RoISegDataset
from progis_rework.interactive.roi import (
    roi_crop_for_prototype,
    roi_crop_for_correction,
    paste_crop_into_mask,
)
from progis_rework.interactive.signals import process_masks
from progis_rework.models.losses import compute_miou_binary, dice_coeff
from progis_rework.models.progis import ProGISModel


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    # Data
    patches_dir: str
    splits_json: str
    fold:        int   = 1
    cls:         str   = "all"
    crop_size:   int   = 256

    # Model
    backbone:    str   = "efficientunet"
    roi_ckpt:    str   = ""
    proj_ckpt:   str   = ""       # SimCLR only

    # Evaluation
    n_iter:      int   = 20       # max correction iterations
    threshold:   float = 0.5     # prototype similarity threshold

    # Hardware
    device:      str   = "cpu"
    batch_size:  int   = 16
    num_workers: int   = 4

    # Output
    results_dir: str   = ""      # if set, saves per-sample pkl files


# ── Core loop ─────────────────────────────────────────────────────────────────

def iterative_correction(
    model:        ProGISModel,
    images:       torch.Tensor,   # [B, 3, H, W]
    proto_mask:   torch.Tensor,   # [B, 1, H, W]  prototype initialisation
    gt_masks:     torch.Tensor,   # [B, 1, H, W]
    init_signal:  torch.Tensor,   # [B, 2, H, W]  original guiding signal
    n_iter:       int   = 20,
    crop_size:    int   = 256,
) -> list[torch.Tensor]:
    """
    Run the iterative correction loop.

    Starting from proto_mask, repeatedly:
      1. Find the largest error region via process_masks.
      2. Crop around it.
      3. Run model.segment() on the crop.
      4. Paste the result back into the full mask.

    The union signal accumulates across iterations (new error signals
    are OR-ed into existing ones), matching the original paper.

    Args:
        model:       ProGISModel (eval mode, no_grad context expected outside).
        images:      [B, 3, H, W] full-resolution image batch.
        proto_mask:  [B, 1, H, W] initial prediction (from forward_prototype).
        gt_masks:    [B, 1, H, W] ground-truth binary masks.
        init_signal: [B, 2, H, W] original guiding signal from the dataset.
        n_iter:      number of correction rounds (default 20).
        crop_size:   ROI crop size (default 256).

    Returns:
        List of n_iter+1 tensors [B, 1, H, W]: pred at each interaction count
        (index 0 = prototype initialisation, index k = after k-th correction).
    """
    B, _, H, W = images.shape
    device = images.device

    current_mask = proto_mask.clone()
    pred_list    = [current_mask.clone()]

    # Initial error signal
    error_signal, centers = process_masks(current_mask, gt_masks)
    union_signal = torch.bitwise_or(
        error_signal.to(torch.uint8),
        init_signal.to(torch.uint8),
    ).float()

    for _ in range(n_iter):
        # Crop around error centroid
        crop_batch = roi_crop_for_correction(
            images, current_mask, union_signal, centers, crop_size,
        )

        # Predict on crop
        crop_pred = model.segment(
            crop_batch.roi_images,
            crop_batch.roi_prev_masks,
            crop_batch.roi_signals,
        )

        # Paste back
        paste_crop_into_mask(
            current_mask, crop_pred, centers, H, W, crop_size,
        )

        pred_list.append(current_mask.clone())

        # Update error signal for next iteration
        error_signal, centers = process_masks(current_mask, gt_masks)
        union_signal = torch.bitwise_or(
            error_signal.to(torch.uint8),
            union_signal.to(torch.uint8),
        ).float()

    return pred_list


def evaluate(model: ProGISModel, val_loader: DataLoader, cfg: EvalConfig) -> dict:
    """
    Run full evaluation loop and return metrics dict.

    Returns:
        {
          'dice_at_k':  [float] * (n_iter+1)  — mean Dice after k interactions,
          'miou_at_k':  [float] * (n_iter+1)  — mean mIoU after k interactions,
        }
    """
    device = torch.device(cfg.device)
    model  = model.to(device).eval()

    n_steps = cfg.n_iter + 1
    dice_sums = [0.0] * n_steps
    miou_lists: list[list[float]] = [[] for _ in range(n_steps)]
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images, masks, signals = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
            B = images.size(0)
            n_samples += B

            # ── Initial prototype crop + prediction ───────────────────────
            proto_crop = roi_crop_for_prototype(
                images, signals, masks, cfg.crop_size,
            )
            proto_out = model.forward_prototype(
                roi_input  = proto_crop.roi_images,
                roi_signal = proto_crop.roi_signals,
                full_image = images,
                mask_box   = proto_crop.mask_box,
                threshold  = cfg.threshold,
            )

            # ── Iterative correction ──────────────────────────────────────
            pred_list = iterative_correction(
                model        = model,
                images       = images,
                proto_mask   = proto_out.prototype_mask,
                gt_masks     = masks,
                init_signal  = signals,
                n_iter       = cfg.n_iter,
                crop_size    = cfg.crop_size,
            )

            # ── Accumulate metrics at every interaction count ─────────────
            for k, pred in enumerate(pred_list):
                dice_sums[k] += dice_coeff(pred, masks).item() * B
                for p, m in zip(pred, masks):
                    miou = compute_miou_binary(p, m)
                    if not np.isnan(miou):
                        miou_lists[k].append(miou)

    dice_at_k = [d / n_samples for d in dice_sums]
    miou_at_k = [float(np.mean(v)) if v else 0.0 for v in miou_lists]

    return {"dice_at_k": dice_at_k, "miou_at_k": miou_at_k}


def print_results(metrics: dict, n_report: list[int] | None = None) -> None:
    """
    Pretty-print Dice@k and mIoU@k.

    Args:
        n_report: interaction counts to report (default [1,5,10,15,20]).
    """
    dice = metrics["dice_at_k"]
    miou = metrics["miou_at_k"]
    if n_report is None:
        n_report = [1, 5, 10, 15, 20]

    print(f"\n{'NoI':>4}  {'Dice':>7}  {'mIoU':>7}")
    print("─" * 24)
    for k in n_report:
        idx = min(k, len(dice) - 1)
        print(f"@{k:2d}   {dice[idx]:.4f}   {miou[idx]:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ProGIS evaluation (NoI metric).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to a YAML config file. Individual flags override YAML values.")
    # Overrides — all optional when --config is provided
    p.add_argument("--patches_dir")
    p.add_argument("--splits_json")
    p.add_argument("--fold",       type=int)
    p.add_argument("--cls")
    p.add_argument("--crop_size",  type=int)
    p.add_argument("--backbone",   choices=["efficientunet", "simclr"])
    p.add_argument("--roi_ckpt",   help="Path to segment_part .pth checkpoint.")
    p.add_argument("--proj_ckpt",  help="SimCLR projection head checkpoint (simclr only).")
    p.add_argument("--n_iter",     type=int)
    p.add_argument("--threshold",  type=float)
    p.add_argument("--device",     help="Torch device, e.g. 'cpu', 'cuda', 'cuda:1'.")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers",type=int)
    return p


def _cfg_from_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    d = raw.get("data",  {})
    m = raw.get("model", {})
    e = raw.get("eval",  {})
    return {
        "patches_dir": d.get("patches_dir"),
        "splits_json": d.get("splits_json"),
        "fold":        d.get("fold",          1),
        "cls":         d.get("cls",           "all"),
        "crop_size":   d.get("crop_size",     256),
        "backbone":    m.get("backbone",      "efficientunet"),
        "roi_ckpt":    m.get("roi_ckpt",      ""),
        "proj_ckpt":   m.get("proj_ckpt",     ""),
        "n_iter":      e.get("n_iter",        20),
        "threshold":   e.get("threshold",     0.5),
        "device":      e.get("device",        "cpu"),
        "batch_size":  e.get("batch_size",    16),
        "num_workers": e.get("num_workers",   4),
    }


def main() -> None:
    args = _build_parser().parse_args()

    defaults = _cfg_from_yaml(args.config) if args.config else {}

    def get(key, cast=None, fallback=None):
        cli_val = getattr(args, key, None)
        val = cli_val if cli_val is not None else defaults.get(key, fallback)
        return cast(val) if (cast and val is not None) else val

    patches_dir = get("patches_dir")
    splits_json = get("splits_json")
    roi_ckpt    = get("roi_ckpt", str, "")
    if not patches_dir or not splits_json:
        _build_parser().error(
            "--patches_dir and --splits_json are required "
            "(pass them directly or via --config)."
        )
    if not roi_ckpt:
        _build_parser().error(
            "--roi_ckpt is required (pass directly or set model.roi_ckpt in the YAML)."
        )

    cfg = EvalConfig(
        patches_dir = patches_dir,
        splits_json = splits_json,
        fold        = get("fold",        int,   1),
        cls         = get("cls",         str,   "all"),
        crop_size   = get("crop_size",   int,   256),
        backbone    = get("backbone",    str,   "efficientunet"),
        roi_ckpt    = roi_ckpt,
        proj_ckpt   = get("proj_ckpt",  str,   ""),
        n_iter      = get("n_iter",      int,   20),
        threshold   = get("threshold",   float, 0.5),
        device      = get("device",      str,   "cpu"),
        batch_size  = get("batch_size",  int,   16),
        num_workers = get("num_workers", int,   4),
    )

    backbone_kwargs = {}
    if cfg.backbone == "simclr" and cfg.proj_ckpt:
        backbone_kwargs["proj_ckpt"] = cfg.proj_ckpt

    model = ProGISModel.from_checkpoint(
        backbone_name    = cfg.backbone,
        roi_ckpt         = cfg.roi_ckpt,
        backbone_kwargs  = backbone_kwargs,
    )

    val_dataset = RoISegDataset(
        cfg.patches_dir, cfg.splits_json,
        fold=cfg.fold, split="val", cls=cfg.cls, crop_size=cfg.crop_size,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )

    metrics = evaluate(model, val_loader, cfg)
    print_results(metrics)


if __name__ == "__main__":
    main()
