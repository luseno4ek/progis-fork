"""
Train the projection head (Conv2d 512→32) of SimCLRFeatureExtractor
using a simple pixel-level contrastive loss.

What is trained:
  ONLY model.proj — Conv2d(512, 32, kernel_size=1) — ~16K parameters.
  The SimCLR ResNet50 encoder stays fully frozen.

Loss:
  For each image, compute fg/bg prototypes from the binary mask.
  For each foreground pixel: push its features toward fg_proto,
  away from bg_proto (InfoNCE-style with prototypes).

Features are computed at H/8 resolution (64×64 for 512 input) for
efficiency — no need to upsample to full resolution for training.

Usage:
    python train_simclr_proj.py
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simclr_feature_extractor import SimCLRFeatureExtractor

# ── Config ────────────────────────────────────────────────────────────────────
FOLD          = 1
PATCHES_DIR   = "/srv/data1/data_repository/BCSS/patches"
SPLITS_JSON   = "/srv/data1/data_repository/BCSS/patches/fold_splits.json"
CLS           = 'tumor'       # class used for fg/bg definition
EPOCHS        = 20
LR            = 4e-4
BATCH_SIZE    = 32
TEMPERATURE   = 0.3
PROJ_CHANNELS = 32
CKPT_DIR      = f"{PATCHES_DIR}/fold_{FOLD}/simclr_proj"
# ─────────────────────────────────────────────────────────────────────────────


class ImageMaskDataset(Dataset):
    """
    Minimal dataset: loads image + binary mask only. No superpixels needed.

    Paths (same layout as ContrastDataset):
      images: patches_dir / all / image_npy / <fname>
      masks:  patches_dir / <cls> / mask_npy / <fname>

    Only fg patches are included (mask file exists only for fg patches).
    Filtered by fold_splits.json to respect train/val split.
    """

    def __init__(self, patches_dir: str, splits_path: str,
                 fold: int, split: str, cls: str):
        patches_dir = Path(patches_dir)
        with open(splits_path) as f:
            fold_splits = json.load(f)

        valid_stems: set[str] = set(fold_splits[f'fold_{fold}'][split])

        def wsi_stem(fname: str) -> str:
            stem = Path(fname).stem
            return stem.rsplit('_patch', 1)[0] if '_patch' in stem else stem

        mask_dir = patches_dir / cls / 'mask_npy'
        self.filenames = [
            f.name for f in sorted(mask_dir.glob('*.npy'))
            if wsi_stem(f.name) in valid_stems
        ]
        self.image_dir = patches_dir / 'all' / 'image_npy'
        self.mask_dir  = mask_dir

        print(f"ImageMaskDataset | fold={fold} {split} | cls={cls} | "
              f"{len(self.filenames)} patches")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = np.load(self.image_dir / fname)          # [H, W, 3]
        mask  = np.load(self.mask_dir  / fname)          # [H, W]
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask  = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask


def pixel_contrastive_loss(feat_proj: torch.Tensor,
                            mask:     torch.Tensor,
                            temperature: float = 0.3) -> torch.Tensor:
    """
    Simple pixel-level prototype contrastive loss.

    For each image in the batch:
      fg_proto = mean of normalized fg pixel features
      bg_proto = mean of normalized bg pixel features
      loss = mean over fg pixels of  -log(sim_fg / (sim_fg + sim_bg))

    Args:
        feat_proj : [B, C, H_f, W_f] — features at H/8 resolution (NOT upsampled)
        mask      : [B, 1, H, W]     — binary mask (>0 = fg), original resolution
        temperature: softmax temperature

    Returns:
        scalar loss tensor
    """
    B, C, H_f, W_f = feat_proj.shape

    # Downsample mask to feature resolution (H/8)
    mask_ds = F.interpolate(mask.float(), size=(H_f, W_f), mode='nearest')  # [B,1,H_f,W_f]

    feat_norm = F.normalize(feat_proj, dim=1)  # [B, C, H_f, W_f]

    total_loss = feat_proj.new_zeros(1)
    count = 0

    for b in range(B):
        feat = feat_norm[b]       # [C, H_f, W_f]
        m    = mask_ds[b, 0]      # [H_f, W_f]

        fg_mask = m > 0.5
        bg_mask = ~fg_mask

        if fg_mask.sum() == 0 or bg_mask.sum() == 0:
            continue

        fg_feat = feat[:, fg_mask].T   # [N_fg, C]
        bg_feat = feat[:, bg_mask].T   # [N_bg, C]

        # Prototypes (L2-normalized)
        fg_proto = F.normalize(fg_feat.mean(0, keepdim=True), dim=1)  # [1, C]
        bg_proto = F.normalize(bg_feat.mean(0, keepdim=True), dim=1)  # [1, C]

        # Similarity of each fg pixel to each prototype
        sim_to_fg = torch.exp(torch.mm(fg_feat, fg_proto.T) / temperature)  # [N_fg, 1]
        sim_to_bg = torch.exp(torch.mm(fg_feat, bg_proto.T) / temperature)  # [N_fg, 1]

        loss = -torch.log(sim_to_fg / (sim_to_fg + sim_to_bg + 1e-8)).mean()

        total_loss = total_loss + loss
        count += 1

    return total_loss / max(count, 1)


def get_features_at_low_res(model: SimCLRFeatureExtractor,
                              images: torch.Tensor) -> torch.Tensor:
    """
    Extract projected features at H/8 resolution (no upsample).
    Gradient flows through model.proj only (encoder stays frozen).

    Returns: [B, proj_channels, H/8, W/8]
    """
    with torch.no_grad():
        x_norm = images / 255.0
        x_norm = (x_norm - model.mean) / model.std
        feat_raw = model.encoder(x_norm)[0]   # [B, 512, H/8, W/8]
    return model.proj(feat_raw)               # [B, 32,  H/8, W/8]  — grad here


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ImageMaskDataset(PATCHES_DIR, SPLITS_JSON,
                                fold=FOLD, split='train', cls=CLS)
    val_ds   = ImageMaskDataset(PATCHES_DIR, SPLITS_JSON,
                                fold=FOLD, split='val',   cls=CLS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=torch.cuda.is_available())

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SimCLRFeatureExtractor(proj_channels=PROJ_CHANNELS).to(device)
    # Verify only proj is trainable
    trainable = sum(p.numel() for p in model.proj.parameters())
    frozen    = sum(p.numel() for p in model.encoder.parameters())
    print(f"Trainable params (proj): {trainable:,}")
    print(f"Frozen params (encoder): {frozen:,}")

    # Optimizer: only projection head parameters
    optimizer = torch.optim.Adam(model.proj.parameters(), lr=LR)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.proj.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader,
                                     desc=f"Epoch {epoch+1}/{EPOCHS} [train]",
                                     leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                feat_proj = get_features_at_low_res(model, images)

            # Loss in fp32 (exp/log needs precision)
            loss = pixel_contrastive_loss(feat_proj.float(), masks, TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # ── Val ───────────────────────────────────────────────────────────────
        model.proj.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader,
                                         desc=f"Epoch {epoch+1}/{EPOCHS} [val]",
                                         leave=False):
                images = images.to(device)
                masks  = masks.to(device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    feat_proj = get_features_at_low_res(model, images)

                loss = pixel_contrastive_loss(feat_proj.float(), masks, TEMPERATURE)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        # Save only projection head weights (tiny: ~64KB)
        ckpt = f"{CKPT_DIR}/proj_epoch{epoch+1}_val{val_loss:.4f}.pth"
        torch.save(model.proj.state_dict(), ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f"{CKPT_DIR}/proj_best.pth"
            torch.save(model.proj.state_dict(), best_path)
            print(f"  → best saved: {best_path}  (val={val_loss:.4f})")


if __name__ == '__main__':
    train()
