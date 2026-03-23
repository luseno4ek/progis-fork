"""
Stage 2 training: P-RoISeg (6-channel EfficientUNet-B0).

CU-Training protocol (two forward passes per backward):
  Pass 1 — cold start: input = cat(image, zeros_mask, pre-computed_signal)
  Pass 2 — correction: input = cat(image, threshold(pred1), union(new_signal, pre-computed_signal))
  Loss  = dice(pred1, gt) + dice(pred2, gt)

Saves:
  checkpoint_dir/
    stage2_best.pth      ← segment_part weights (best val Dice)
    stage2_last.pth      ← segment_part weights (last epoch)

Usage
-----
  python -m progis_rework.training.train_stage2 \\
      --patches_dir data/patches \\
      --splits_json data/patches/fold_splits.json \\
      --fold 1 --cls all --epochs 50 \\
      --checkpoint_dir runs/stage2/fold1

  # CPU (macOS / no-GPU):
  python -m progis_rework.training.train_stage2 \\
      --patches_dir data/patches --splits_json data/patches/fold_splits.json \\
      --fold 1 --cls tumor --epochs 5 \\
      --batch_size 2 --num_workers 0 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from progis_rework.data.dataset import RoISegDataset
from progis_rework.interactive.signals import process_masks
from progis_rework.models.losses import dice_coeff, dice_loss, compute_miou_binary
from progis_rework.models.progis import ProGISModel


# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    patches_dir:     str
    splits_json:     str
    fold:            int   = 1
    cls:             str   = "all"
    crop_size:       int   = 256

    # Hardware
    device:          str   = "cpu"
    batch_size:      int   = 16
    num_workers:     int   = 4

    # Optimiser
    lr:              float = 4e-4
    weight_decay:    float = 0.0
    epochs:          int   = 50

    # Output
    checkpoint_dir:  str   = "runs/stage2"
    tensorboard:     bool  = True

    # Inference signals: use GPU-accelerated process_masks when on CUDA
    use_gpu_signals: bool  = False


# ── Loss ──────────────────────────────────────────────────────────────────────

def cu_loss(
    pred1: torch.Tensor,
    pred2: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CU-Training loss: Dice + BCE for both passes.

    BCE prevents the all-foreground local minimum by providing non-zero
    gradient even when sigmoid is saturated (grad = pred - gt ≠ 0).

    Returns:
        (loss, l1_dice, l2_dice)  — total scalar loss and per-pass Dice components
        for diagnostic logging.
    """
    l1 = dice_loss(pred1, masks)
    l2 = dice_loss(pred2, masks)
    bce1 = F.binary_cross_entropy(pred1, masks)
    bce2 = F.binary_cross_entropy(pred2, masks)
    return l1 + l2 + bce1 + bce2, l1, l2


# ── Training function ─────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    """
    Run Stage 2 P-RoISeg training.

    The backbone is NOT used here — only segment_part is trained.
    The saved checkpoint (stage2_best.pth) is segment_part.state_dict().
    Load it at inference time via ProGISModel.from_checkpoint().
    """
    device = torch.device(cfg.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = Path(cfg.checkpoint_dir) / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = RoISegDataset(
        cfg.patches_dir, cfg.splits_json,
        fold=cfg.fold, split="train", cls=cfg.cls, crop_size=cfg.crop_size,
    )
    val_dataset = RoISegDataset(
        cfg.patches_dir, cfg.splits_json,
        fold=cfg.fold, split="val", cls=cfg.cls, crop_size=cfg.crop_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_dataset)} patches | Val: {len(val_dataset)} patches")

    # ── Model: backbone=None, only segment_part is trained ───────────────────
    model = ProGISModel(backbone=None).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer: SummaryWriter | None = None
    if cfg.tensorboard:
        tb_dir = str(ckpt_dir / "tb_logs")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard: {tb_dir}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_dice = 0.0

    for epoch in range(cfg.epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = train_dice = 0.0
        dbg_l1 = dbg_l2 = dbg_p1fg = dbg_p2fg = dbg_signz = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.epochs} [train]",
            leave=False,
        )
        for images, masks, signals in train_bar:
            images, masks, signals = (
                images.to(device), masks.to(device), signals.to(device)
            )
            B = images.size(0)

            optimizer.zero_grad()

            # Pass 1: cold-start prediction
            pred1 = model.segment(images, torch.zeros_like(masks), signals)

            # Build correction signal from Pass-1 error
            new_signal, _ = process_masks(pred1.detach(), masks)
            union_signal = torch.bitwise_or(
                new_signal.to(torch.uint8), signals.to(torch.uint8)
            ).float()

            # Pass 2: correction prediction
            pred1_bin = (pred1 >= 0.5).float()
            pred2 = model.segment(images, pred1_bin, union_signal)

            loss, l1, l2 = cu_loss(pred1, pred2, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B
            train_dice += dice_coeff((pred2 >= 0.5).float(), masks).item() * B
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            dbg_l1    += l1.item() * B
            dbg_l2    += l2.item() * B
            dbg_p1fg  += (pred1 >= 0.5).float().mean().item() * B
            dbg_p2fg  += (pred2 >= 0.5).float().mean().item() * B
            dbg_signz += union_signal.bool().float().mean().item() * B

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_dice /= n_train

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = val_dice = 0.0
        miou_scores: list[float] = []
        dbg_vl1 = dbg_vl2 = dbg_vp1fg = dbg_vp2fg = dbg_vsignz = 0.0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{cfg.epochs} [val]  ",
                leave=False,
            )
            for images, masks, signals in val_bar:
                images, masks, signals = (
                    images.to(device), masks.to(device), signals.to(device)
                )
                B = images.size(0)

                # Same two-pass evaluation (no grad)
                pred1 = model.segment(images, torch.zeros_like(masks), signals)

                new_signal, _ = process_masks(pred1, masks)
                union_signal = torch.bitwise_or(
                    new_signal.to(torch.uint8), signals.to(torch.uint8)
                ).float()

                pred2 = model.segment(images, (pred1 >= 0.5).float(), union_signal)

                batch_loss, vl1, vl2 = cu_loss(pred1, pred2, masks)
                val_loss += batch_loss.item() * B

                preds_bin = (pred2 >= 0.5).float()
                val_dice += dice_coeff(preds_bin, masks).item() * B

                dbg_vl1   += vl1.item() * B
                dbg_vl2   += vl2.item() * B
                dbg_vp1fg += (pred1 >= 0.5).float().mean().item() * B
                dbg_vp2fg += (pred2 >= 0.5).float().mean().item() * B
                dbg_vsignz += union_signal.bool().float().mean().item() * B

                # Per-sample mIoU
                for pred, mask in zip(preds_bin, masks):
                    miou = compute_miou_binary(pred, mask)
                    if not np.isnan(miou):
                        miou_scores.append(miou)

        n_val = len(val_loader.dataset)
        val_loss /= n_val
        val_dice /= n_val
        mean_miou = float(np.mean(miou_scores)) if miou_scores else 0.0

        # ── Gradient norm (after last train batch) ────────────────────────
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for g in optimizer.param_groups
            for p in g["params"]
            if p.grad is not None
        ) ** 0.5

        # ── Logging ───────────────────────────────────────────────────────
        N_tr = len(train_loader.dataset)
        print(
            f"Epoch {epoch+1:3d}/{cfg.epochs} | "
            f"Train loss={train_loss:.4f} dice={train_dice:.4f} | "
            f"Val loss={val_loss:.4f} dice={val_dice:.4f} mIoU={mean_miou:.4f}"
        )
        print(
            f"  [DBG train] l1={dbg_l1/N_tr:.4f} l2={dbg_l2/N_tr:.4f} "
            f"p1fg={dbg_p1fg/N_tr:.3f} p2fg={dbg_p2fg/N_tr:.3f} "
            f"signal_nz={dbg_signz/N_tr:.4f} grad_norm={grad_norm:.4f}"
        )
        print(
            f"  [DBG val]   l1={dbg_vl1/n_val:.4f} l2={dbg_vl2/n_val:.4f} "
            f"p1fg={dbg_vp1fg/n_val:.3f} p2fg={dbg_vp2fg/n_val:.3f} "
            f"signal_nz={dbg_vsignz/n_val:.4f}"
        )

        if writer is not None:
            writer.add_scalars("loss",      {"train": train_loss, "val": val_loss},  epoch)
            writer.add_scalars("dice",      {"train": train_dice, "val": val_dice},  epoch)
            writer.add_scalar("val/mIoU",   mean_miou, epoch)

        # ── Checkpointing ─────────────────────────────────────────────────
        # Always save last checkpoint
        torch.save(
            model.segment_part.state_dict(),
            ckpt_dir / "stage2_last.pth",
        )

        # Save best checkpoint based on val Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = ckpt_dir / f"stage2_best_dice{val_dice:.4f}_ep{epoch+1}.pth"
            torch.save(model.segment_part.state_dict(), best_path)
            # Symlink for convenience: stage2_best.pth always points to current best
            symlink = ckpt_dir / "stage2_best.pth"
            if symlink.exists() or symlink.is_symlink():
                symlink.unlink()
            symlink.symlink_to(best_path.name)
            print(f"  ✓ New best checkpoint: {best_path.name}")

    if writer is not None:
        writer.close()

    print(f"\nTraining complete. Best val Dice = {best_val_dice:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ProGIS Stage 2 training (P-RoISeg).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to a YAML config file (e.g. progis_rework/configs/server_bcss.yaml). "
                        "Individual flags below override the YAML values.")
    # Overrides — all optional when --config is provided
    p.add_argument("--patches_dir")
    p.add_argument("--splits_json")
    p.add_argument("--fold",           type=int)
    p.add_argument("--cls")
    p.add_argument("--crop_size",      type=int)
    p.add_argument("--device",         help="Torch device string, e.g. 'cpu', 'cuda', 'cuda:1'.")
    p.add_argument("--batch_size",     type=int)
    p.add_argument("--num_workers",    type=int)
    p.add_argument("--lr",             type=float)
    p.add_argument("--weight_decay",   type=float)
    p.add_argument("--epochs",         type=int)
    p.add_argument("--checkpoint_dir")
    p.add_argument("--no_tensorboard", action="store_true", default=False)
    return p


def _cfg_from_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    d = raw.get("data",   {})
    s = raw.get("stage2", {})
    return {
        "patches_dir":    d.get("patches_dir"),
        "splits_json":    d.get("splits_json"),
        "fold":           d.get("fold",           1),
        "cls":            d.get("cls",            "all"),
        "crop_size":      d.get("crop_size",      256),
        "device":         s.get("device",         "cpu"),
        "batch_size":     s.get("batch_size",     16),
        "num_workers":    s.get("num_workers",    4),
        "lr":             s.get("lr",             4e-4),
        "weight_decay":   s.get("weight_decay",   0.0),
        "epochs":         s.get("epochs",         50),
        "checkpoint_dir": s.get("checkpoint_dir", "runs/stage2"),
        "tensorboard":    s.get("tensorboard",    True),
    }


def main() -> None:
    args = _build_parser().parse_args()

    # YAML provides base values; CLI flags override any of them
    defaults = _cfg_from_yaml(args.config) if args.config else {}

    def get(key, cast=None, fallback=None):
        cli_val = getattr(args, key, None)
        val = cli_val if cli_val is not None else defaults.get(key, fallback)
        return cast(val) if (cast and val is not None) else val

    patches_dir = get("patches_dir")
    splits_json = get("splits_json")
    if not patches_dir or not splits_json:
        _build_parser().error(
            "--patches_dir and --splits_json are required "
            "(pass them directly or via --config)."
        )

    cfg = TrainConfig(
        patches_dir     = patches_dir,
        splits_json     = splits_json,
        fold            = get("fold",           int,   1),
        cls             = get("cls",            str,   "all"),
        crop_size       = get("crop_size",      int,   256),
        device          = get("device",         str,   "cpu"),
        batch_size      = get("batch_size",     int,   16),
        num_workers     = get("num_workers",    int,   4),
        lr              = get("lr",             float, 4e-4),
        weight_decay    = get("weight_decay",   float, 0.0),
        epochs          = get("epochs",         int,   50),
        checkpoint_dir  = get("checkpoint_dir", str,   "runs/stage2"),
        tensorboard     = defaults.get("tensorboard", True) and not args.no_tensorboard,
    )
    train(cfg)


if __name__ == "__main__":
    main()
