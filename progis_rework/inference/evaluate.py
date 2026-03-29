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

from progis_rework.data.dataset import RoISegDataset, ALL_CLASSES
from progis_rework.interactive.roi import (
    roi_crop_for_prototype,
    roi_crop_for_correction,
    paste_crop_into_mask,
)
from progis_rework.interactive.signals import process_masks
from progis_rework.models.losses import compute_dice_binary, compute_miou_binary
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
    dice_lists: list[list[float]] = [[] for _ in range(n_steps)]
    miou_lists: list[list[float]] = [[] for _ in range(n_steps)]

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images, masks, signals = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )

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

            # ── Accumulate per-sample metrics at every interaction count ──
            for k, pred in enumerate(pred_list):
                for p, m in zip(pred, masks):
                    d = compute_dice_binary(p, m)
                    if not np.isnan(d):
                        dice_lists[k].append(d)
                    iou = compute_miou_binary(p, m)
                    if not np.isnan(iou):
                        miou_lists[k].append(iou)

    dice_at_k = [float(np.mean(v)) if v else 0.0 for v in dice_lists]
    miou_at_k = [float(np.mean(v)) if v else 0.0 for v in miou_lists]

    return {"dice_at_k": dice_at_k, "miou_at_k": miou_at_k}


def print_results(
    metrics:  dict,
    n_report: list[int] | None = None,
    cls_name: str | None = None,
) -> None:
    """
    Pretty-print Dice@k and mIoU@k for a single class or macro average.

    Args:
        metrics:  dict with 'dice_at_k' and 'miou_at_k' lists.
        n_report: interaction counts to report (default [1,5,10,15,20]).
        cls_name: optional class label shown as header.
    """
    dice = metrics["dice_at_k"]
    miou = metrics["miou_at_k"]
    if n_report is None:
        n_report = [1, 5, 10, 15, 20]

    header = f"  [{cls_name}]" if cls_name else ""
    print(f"\n{'NoI':>4}  {'Dice':>7}  {'mIoU':>7}{header}")
    print("─" * (24 + len(header)))
    for k in n_report:
        idx = min(k, len(dice) - 1)
        print(f"@{k:2d}   {dice[idx]:.4f}   {miou[idx]:.4f}")


def print_macro_results(
    per_class_metrics: dict[str, dict],
    n_report: list[int] | None = None,
) -> None:
    """
    Print per-class and macro-averaged Dice@k / mIoU@k table.

    Args:
        per_class_metrics: {class_name: {'dice_at_k': [...], 'miou_at_k': [...]}}
        n_report: interaction counts to report (default [1,5,10,15,20]).
    """
    if n_report is None:
        n_report = [1, 5, 10, 15, 20]

    header = f"{'Class':<28}" + "".join(f"Dice@{k:<4}" for k in n_report) \
             + "  " + "".join(f"IoU@{k:<5}" for k in n_report)
    print("\n" + header)
    print("─" * len(header))

    classes = list(per_class_metrics.keys())
    for cls in classes:
        m = per_class_metrics[cls]
        dice_vals = "".join(
            f"{m['dice_at_k'][min(k, len(m['dice_at_k'])-1)]:<9.4f}" for k in n_report
        )
        iou_vals = "".join(
            f"{m['miou_at_k'][min(k, len(m['miou_at_k'])-1)]:<9.4f}" for k in n_report
        )
        print(f"{cls:<28}{dice_vals}  {iou_vals}")

    # Macro row
    print("─" * len(header))
    n_steps = max(len(m["dice_at_k"]) for m in per_class_metrics.values())
    macro_dice = [
        float(np.mean([per_class_metrics[c]["dice_at_k"][min(k, len(per_class_metrics[c]["dice_at_k"])-1)]
                       for c in classes]))
        for k in range(n_steps)
    ]
    macro_iou = [
        float(np.mean([per_class_metrics[c]["miou_at_k"][min(k, len(per_class_metrics[c]["miou_at_k"])-1)]
                       for c in classes]))
        for k in range(n_steps)
    ]
    dice_vals = "".join(f"{macro_dice[min(k, len(macro_dice)-1)]:<9.4f}" for k in n_report)
    iou_vals  = "".join(f"{macro_iou[min(k, len(macro_iou)-1)]:<9.4f}" for k in n_report)
    print(f"{'MACRO':<28}{dice_vals}  {iou_vals}")


# ── Multi-class prototype evaluation ─────────────────────────────────────────

def evaluate_multiclass_proto(
    model:          ProGISModel,
    class_datasets: dict[str, "RoISegDataset"],
    cfg:            "EvalConfig",
) -> dict[str, dict]:
    """
    Evaluate with the multiclass prototype step (no pixel overlaps).

    Proto step: forward_prototype_multiclass() — argmax over per-class
    similarity maps assigns each pixel to at most one class.
    Correction step: independent per class (same as standard evaluate()).

    Args:
        model:          ProGISModel in eval mode.
        class_datasets: {cls: RoISegDataset(full_patch=True)} for every class.
        cfg:            EvalConfig.

    Returns:
        per_class_metrics: {cls: {'dice_at_k': [...], 'miou_at_k': [...]}}
    """
    from collections import defaultdict

    device = torch.device(cfg.device)
    model  = model.to(device).eval()

    patches_dir = Path(cfg.patches_dir)

    # fname → list of classes that have it
    fname_to_classes: dict[str, list[str]] = defaultdict(list)
    for cls, ds in class_datasets.items():
        for fname, _ in ds.items:
            fname_to_classes[fname].append(cls)

    n_steps = cfg.n_iter + 1
    dice_lists: dict[str, list[list[float]]] = {
        cls: [[] for _ in range(n_steps)] for cls in class_datasets
    }
    miou_lists: dict[str, list[list[float]]] = {
        cls: [[] for _ in range(n_steps)] for cls in class_datasets
    }

    with torch.no_grad():
        for fname, classes in tqdm(fname_to_classes.items(),
                                   desc="Evaluating (multiclass proto)"):
            img_npy = np.load(patches_dir / "all" / "image_npy" / fname)
            image_t = (
                torch.tensor(img_npy.transpose(2, 0, 1), dtype=torch.float32)
                .unsqueeze(0).to(device)
            )

            masks_t:   list[torch.Tensor] = []
            signals_t: list[torch.Tensor] = []
            proto_crops = []

            for cls in classes:
                mask_npy   = np.load(patches_dir / cls / "mask_npy"            / fname)
                signal_npy = np.load(patches_dir / cls / "signal_all_line_npy" / fname)
                m_t = torch.tensor(mask_npy,   dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                s_t = torch.tensor(signal_npy, dtype=torch.float32).unsqueeze(0).to(device)
                masks_t.append(m_t)
                signals_t.append(s_t)
                proto_crops.append(
                    roi_crop_for_prototype(image_t, s_t, m_t, cfg.crop_size)
                )

            proto_outputs = model.forward_prototype_multiclass(
                roi_inputs  = [pc.roi_images  for pc in proto_crops],
                roi_signals = [pc.roi_signals for pc in proto_crops],
                full_image  = image_t,
                mask_boxes  = [pc.mask_box    for pc in proto_crops],
                threshold   = cfg.threshold,
            )

            for k, cls in enumerate(classes):
                pred_list = iterative_correction(
                    model       = model,
                    images      = image_t,
                    proto_mask  = proto_outputs[k].prototype_mask,
                    gt_masks    = masks_t[k],
                    init_signal = signals_t[k],
                    n_iter      = cfg.n_iter,
                    crop_size   = cfg.crop_size,
                )
                for step, pred in enumerate(pred_list):
                    for p, m in zip(pred, masks_t[k]):
                        d = compute_dice_binary(p, m)
                        if not np.isnan(d):
                            dice_lists[cls][step].append(d)
                        iou = compute_miou_binary(p, m)
                        if not np.isnan(iou):
                            miou_lists[cls][step].append(iou)

    return {
        cls: {
            "dice_at_k": [float(np.mean(v)) if v else 0.0 for v in dice_lists[cls]],
            "miou_at_k": [float(np.mean(v)) if v else 0.0 for v in miou_lists[cls]],
        }
        for cls in class_datasets
    }


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
    p.add_argument("--multiclass_proto", action="store_true",
                   help="Use argmax over per-class similarity maps for prototype step "
                        "(no pixel overlaps). Corrections are always independent per class.")
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

    if cfg.cls == "all":
        if args.multiclass_proto:
            class_datasets: dict[str, RoISegDataset] = {}
            for cls in ALL_CLASSES:
                ds = RoISegDataset(
                    cfg.patches_dir, cfg.splits_json,
                    fold=cfg.fold, split="val", cls=cls, crop_size=cfg.crop_size,
                    full_patch=True,
                )
                if len(ds) > 0:
                    class_datasets[cls] = ds
            per_class_metrics = evaluate_multiclass_proto(model, class_datasets, cfg)
        else:
            per_class_metrics: dict[str, dict] = {}
            for cls in ALL_CLASSES:
                val_dataset = RoISegDataset(
                    cfg.patches_dir, cfg.splits_json,
                    fold=cfg.fold, split="val", cls=cls, crop_size=cfg.crop_size,
                    full_patch=True,
                )
                if len(val_dataset) == 0:
                    print(f"[{cls}] no val samples, skipping.")
                    continue
                val_loader = DataLoader(
                    val_dataset, batch_size=cfg.batch_size,
                    shuffle=False, num_workers=cfg.num_workers,
                )
                per_class_metrics[cls] = evaluate(model, val_loader, cfg)
        print_macro_results(per_class_metrics)
    else:
        val_dataset = RoISegDataset(
            cfg.patches_dir, cfg.splits_json,
            fold=cfg.fold, split="val", cls=cfg.cls, crop_size=cfg.crop_size,
            full_patch=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size,
            shuffle=False, num_workers=cfg.num_workers,
        )
        metrics = evaluate(model, val_loader, cfg)
        print_results(metrics, cls_name=cfg.cls)


if __name__ == "__main__":
    main()
