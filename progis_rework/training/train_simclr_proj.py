"""
SimCLR projection head training.

Trains only the 1×1 Conv2d projection head (512 → proj_channels) of the
frozen SimCLR ResNet50 encoder. The encoder itself is never updated.

Loss: prototype alignment BCE.
  For each patch:
    1. Extract features via SimCLRFeatureExtractor (encoder frozen).
    2. Prototype = mean normalised feature at foreground pixels.
    3. Cosine similarity map = dot(feat_norm, proto_norm).
    4. Map similarity to [0,1] via (sim+1)/2.
    5. BCE( sim_01, binary_mask ).

This directly optimises the representation for prototype matching in
ProGISModel.forward_prototype(), requiring only the existing mask_npy data
(no SLIC superpixels needed).

Saves:
  checkpoint_dir/
    simclr_proj_best.pth   — projection head state dict (best val loss)
    simclr_proj_last.pth   — projection head state dict (last epoch)

Usage
-----
  python -m progis_rework.training.train_simclr_proj \\
      --patches_dir data/patches \\
      --splits_json data/patches/fold_splits.json \\
      --fold 1 --cls all --epochs 10 \\
      --checkpoint_dir runs/simclr_proj

  # CPU (macOS / no-GPU):
  python -m progis_rework.training.train_simclr_proj \\
      --patches_dir data/patches --splits_json data/patches/fold_splits.json \\
      --fold 1 --cls tumor --epochs 5 \\
      --batch_size 2 --num_workers 0 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from progis_rework.data.dataset import RoISegDataset
from progis_rework.models.losses import compute_dice_binary

# SimCLRFeatureExtractor lives in the original models/ directory
_MODELS_DIR = Path(__file__).parents[3] / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    patches_dir:    str
    splits_json:    str
    fold:           int   = 1
    cls:            str   = "all"
    crop_size:      int   = 256

    # Model
    proj_channels:  int   = 32

    # Hardware
    device:         str   = "cpu"
    batch_size:     int   = 16
    num_workers:    int   = 4

    # Optimiser
    lr:             float = 1e-3
    weight_decay:   float = 0.0
    epochs:         int   = 10

    # Output
    checkpoint_dir: str   = "runs/simclr_proj"
    tensorboard:    bool  = True

    # Reproducibility
    seed:           int   = 42


# ── Loss ──────────────────────────────────────────────────────────────────────

def prototype_align_loss(
    features: torch.Tensor,   # [B, C, H, W]
    masks:    torch.Tensor,   # [B, 1, H, W]  binary float
) -> torch.Tensor:
    """
    Prototype alignment BCE loss.

    For each sample, computes a prototype as the mean normalised feature
    at foreground pixels, then measures cosine similarity of every pixel
    to that prototype. BCE drives fg pixels toward high similarity (→1)
    and bg pixels toward low similarity (→0).

    Samples with no foreground are skipped.
    """
    feat_norm = F.normalize(features, dim=1)   # [B, C, H, W]

    fg_count  = masks.sum(dim=(2, 3), keepdim=True).clamp(min=1)   # [B,1,1,1]
    prototype = (feat_norm * masks).sum(dim=(2, 3), keepdim=True) / fg_count  # [B,C,1,1]
    proto_norm = F.normalize(
        prototype.squeeze(-1).squeeze(-1), dim=1
    )                                                                # [B, C]

    # Cosine similarity: [B, 1, H, W]
    sim = torch.einsum("bchw,bc->bhw", feat_norm, proto_norm).unsqueeze(1)

    # Map [-1,1] → [0,1] for BCE
    sim_01 = (sim + 1.0) / 2.0

    # Skip samples with no fg (clamp avoids log(0) in BCE)
    has_fg = (masks.sum(dim=(2, 3)) > 0).float().view(-1, 1, 1, 1)
    loss = F.binary_cross_entropy(sim_01.clamp(1e-6, 1 - 1e-6), masks, reduction="none")
    loss = (loss * has_fg).sum() / has_fg.sum().clamp(min=1)

    return loss


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    # ── Model ────────────────────────────────────────────────────────────────
    from simclr_feature_extractor import SimCLRFeatureExtractor
    extractor = SimCLRFeatureExtractor(proj_channels=cfg.proj_channels).to(device)

    # Optimise only the projection head
    optimizer = optim.Adam(
        extractor.proj.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dataset = RoISegDataset(
        cfg.patches_dir, cfg.splits_json,
        fold=cfg.fold, split="train", cls=cfg.cls, crop_size=cfg.crop_size,
    )
    val_dataset = RoISegDataset(
        cfg.patches_dir, cfg.splits_json,
        fold=cfg.fold, split="val",   cls=cfg.cls, crop_size=cfg.crop_size,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )

    # ── Output dir ───────────────────────────────────────────────────────────
    run_dir = Path(cfg.checkpoint_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if cfg.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        except ImportError:
            print("[train_simclr_proj] tensorboard not available, skipping.")

    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        extractor.train()
        train_losses: list[float] = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]", leave=False):
            images = batch[0].to(device)   # [B, 3, H, W]
            masks  = batch[1].to(device)   # [B, 1, H, W]

            features = extractor(images)   # [B, proj_channels, H, W]
            loss = prototype_align_loss(features, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # ── Val ──────────────────────────────────────────────────────────────
        extractor.eval()
        val_losses:  list[float] = []
        val_dices:   list[float] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]", leave=False):
                images = batch[0].to(device)
                masks  = batch[1].to(device)

                features = extractor(images)
                loss = prototype_align_loss(features, masks)
                val_losses.append(loss.item())

                # Dice: threshold sim_01 at 0.5 (= cosine sim > 0)
                feat_norm  = F.normalize(features, dim=1)
                fg_count   = masks.sum(dim=(2, 3), keepdim=True).clamp(min=1)
                prototype  = (feat_norm * masks).sum(dim=(2, 3), keepdim=True) / fg_count
                proto_norm = F.normalize(prototype.squeeze(-1).squeeze(-1), dim=1)
                sim_01 = (torch.einsum("bchw,bc->bhw", feat_norm, proto_norm).unsqueeze(1) + 1.0) / 2.0

                for p, m in zip(sim_01, masks):
                    d = compute_dice_binary(p, m)
                    if not np.isnan(d):
                        val_dices.append(d)

        mean_val_loss = float(np.mean(val_losses))
        mean_val_dice = float(np.mean(val_dices)) if val_dices else 0.0

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train_loss={mean_train_loss:.4f}  "
            f"val_loss={mean_val_loss:.4f}  "
            f"val_dice={mean_val_dice:.4f}"
        )

        if writer:
            writer.add_scalar("loss/train", mean_train_loss, epoch)
            writer.add_scalar("loss/val",   mean_val_loss,   epoch)
            writer.add_scalar("dice/val",   mean_val_dice,   epoch)

        # Save last
        torch.save(extractor.proj.state_dict(), run_dir / "simclr_proj_last.pth")

        # Save best
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(extractor.proj.state_dict(), run_dir / "simclr_proj_best.pth")
            print(f"  => best saved (val_loss={best_val_loss:.4f})")

    if writer:
        writer.close()

    print(f"\nDone. Checkpoints: {run_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Use with:  --backbone simclr --proj_ckpt {run_dir}/simclr_proj_best.pth")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train SimCLR projection head for ProGIS prototype matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",          default=None,
                   help="Path to a YAML config file. Individual flags override YAML values.")
    p.add_argument("--patches_dir")
    p.add_argument("--splits_json")
    p.add_argument("--fold",            type=int)
    p.add_argument("--cls",             default="all")
    p.add_argument("--crop_size",       type=int)
    p.add_argument("--proj_channels",   type=int)
    p.add_argument("--device")
    p.add_argument("--batch_size",      type=int)
    p.add_argument("--num_workers",     type=int)
    p.add_argument("--lr",              type=float)
    p.add_argument("--weight_decay",    type=float)
    p.add_argument("--epochs",          type=int)
    p.add_argument("--checkpoint_dir")
    p.add_argument("--no_tensorboard",  action="store_true")
    p.add_argument("--seed",            type=int)
    return p


def _cfg_from_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    d  = raw.get("data",         {})
    sp = raw.get("simclr_proj",  {})
    return {
        "patches_dir":   d.get("patches_dir"),
        "splits_json":   d.get("splits_json"),
        "fold":          d.get("fold",           1),
        "cls":           d.get("cls",            "all"),
        "crop_size":     d.get("crop_size",      256),
        "proj_channels": sp.get("proj_channels", 32),
        "device":        sp.get("device",        "cpu"),
        "batch_size":    sp.get("batch_size",    16),
        "num_workers":   sp.get("num_workers",   4),
        "lr":            sp.get("lr",            1e-3),
        "weight_decay":  sp.get("weight_decay",  0.0),
        "epochs":        sp.get("epochs",        10),
        "checkpoint_dir":sp.get("checkpoint_dir","runs/simclr_proj"),
        "tensorboard":   sp.get("tensorboard",   True),
        "seed":          sp.get("seed",          42),
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
    if not patches_dir or not splits_json:
        _build_parser().error(
            "--patches_dir and --splits_json are required "
            "(pass them directly or via --config)."
        )

    tensorboard = get("tensorboard", bool, True)
    if args.no_tensorboard:
        tensorboard = False

    cfg = TrainConfig(
        patches_dir    = patches_dir,
        splits_json    = splits_json,
        fold           = get("fold",           int,   1),
        cls            = get("cls",            str,   "all"),
        crop_size      = get("crop_size",      int,   256),
        proj_channels  = get("proj_channels",  int,   32),
        device         = get("device",         str,   "cpu"),
        batch_size     = get("batch_size",     int,   16),
        num_workers    = get("num_workers",    int,   4),
        lr             = get("lr",             float, 1e-3),
        weight_decay   = get("weight_decay",   float, 0.0),
        epochs         = get("epochs",         int,   10),
        checkpoint_dir = get("checkpoint_dir", str,   "runs/simclr_proj"),
        tensorboard    = tensorboard,
        seed           = get("seed",           int,   42),
    )
    train(cfg)


if __name__ == "__main__":
    main()
