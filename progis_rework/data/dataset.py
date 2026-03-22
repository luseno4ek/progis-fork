"""
Fold-aware Dataset classes for ProGIS training.

Uses fold_splits.json for train/val splitting — no data duplication on disk.

Expected patches_dir layout:
  all/
    image_npy/           ← shared image patches [H,W,3] float32
    slic_{n}/            ← shared SLIC superpixels [H,W] int  (Stage 1 only)
  {class}/
    mask_npy/            ← foreground-only mask patches [H,W] float32
    signal_all_line_npy/ ← guiding signal patches [2,H,W] float32

fold_splits.json format:
  {"fold_1": {"train": ["sample_0030", ...], "val": ["sample_0000", ...]}, ...}
"""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


ALL_CLASSES = [
    "tumor",
    "stroma",
    "inflammatory_infiltration",
    "necrosis",
    "others",
]


def load_fold_splits(splits_path: str | Path) -> dict:
    with open(splits_path) as f:
        return json.load(f)


def get_wsi_stem(patch_filename: str) -> str:
    """
    Extract WSI stem from a patch filename.

    'sample_0042_patch0003.npy' → 'sample_0042'
    'sample_0042.npy'           → 'sample_0042'
    """
    stem = Path(patch_filename).stem
    if "_patch" in stem:
        return stem.rsplit("_patch", 1)[0]
    return stem


# ── Stage 2: P-RoISeg Dataset ─────────────────────────────────────────────────

class RoISegDataset(Dataset):
    """
    Dataset for P-RoISeg training (Stage 2).

    Returns (image, mask, signal) per sample, where the image and signal
    are cropped to a 256×256 window centred on the foreground signal.

    Args:
        patches_dir:  path to patches root directory.
        splits_path:  path to fold_splits.json.
        fold:         fold number (1-based).
        split:        'train' or 'val'.
        cls:          tissue class, e.g. 'tumor', or 'all' for all classes.
        crop_size:    spatial crop size (default 256).
    """

    def __init__(
        self,
        patches_dir: str | Path,
        splits_path: str | Path,
        fold:        int,
        split:       str,
        cls:         str,
        crop_size:   int = 256,
    ):
        self.patches_dir = Path(patches_dir)
        self.crop_size   = crop_size

        fold_splits = load_fold_splits(splits_path)
        valid_stems = set(fold_splits[f"fold_{fold}"][split])

        classes = ALL_CLASSES if cls == "all" else [cls]

        # Each item: (filename, class_name)
        self.items: list[tuple[str, str]] = []
        for c in classes:
            mask_dir = self.patches_dir / c / "mask_npy"
            if not mask_dir.exists():
                continue
            for f in sorted(mask_dir.glob("*.npy")):
                if get_wsi_stem(f.name) in valid_stems:
                    self.items.append((f.name, c))

        print(
            f"RoISegDataset | fold={fold} {split} | cls={cls} | "
            f"{len(self.items)} patches"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fname, cls = self.items[idx]
        crop = self.crop_size

        image  = np.load(self.patches_dir / "all"  / "image_npy"            / fname)  # [H,W,3]
        mask   = np.load(self.patches_dir / cls    / "mask_npy"              / fname)  # [H,W]
        signal = np.load(self.patches_dir / cls    / "signal_all_line_npy"   / fname)  # [2,H,W]

        # Crop centred on the foreground signal
        H, W = image.shape[:2]
        fg_ys, fg_xs = np.where(signal[0] > 0)
        if fg_ys.size > 0:
            cy = int(round(fg_ys.mean()))
            cx = int(round(fg_xs.mean()))
            sy = min(max(cy - crop // 2, 0), H - crop)
            sx = min(max(cx - crop // 2, 0), W - crop)
        else:
            sy = np.random.randint(0, H - crop + 1)
            sx = np.random.randint(0, W - crop + 1)

        image  = image[sy:sy+crop, sx:sx+crop, :]
        mask   = mask[sy:sy+crop, sx:sx+crop]
        signal = signal[:, sy:sy+crop, sx:sx+crop]

        image  = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask   = torch.tensor(mask,   dtype=torch.float32).unsqueeze(0)
        signal = torch.tensor(signal, dtype=torch.float32)

        # Order: (image, mask, signal) — matches training loop convention
        return image, mask, signal


# ── Stage 1: Contrastive Learning Dataset ─────────────────────────────────────

class ContrastDataset(Dataset):
    """
    Dataset for Feature Extractor contrastive learning (Stage 1).

    Returns (image, mask, superpixel) per sample.

    Args:
        patches_dir:  path to patches root directory.
        splits_path:  path to fold_splits.json.
        fold:         fold number (1-based).
        split:        'train' or 'val'.
        cls:          tissue class for contrastive learning (default 'tumor').
        n_segments:   SLIC superpixel count (used to locate slic_{n} dir).
    """

    def __init__(
        self,
        patches_dir: str | Path,
        splits_path: str | Path,
        fold:        int,
        split:       str,
        cls:         str   = "tumor",
        n_segments:  int   = 500,
    ):
        self.patches_dir = Path(patches_dir)
        self.cls         = cls
        self.n_segments  = n_segments

        fold_splits = load_fold_splits(splits_path)
        valid_stems = set(fold_splits[f"fold_{fold}"][split])

        mask_dir = self.patches_dir / cls / "mask_npy"
        self.filenames = [
            f.name for f in sorted(mask_dir.glob("*.npy"))
            if get_wsi_stem(f.name) in valid_stems
        ]

        print(
            f"ContrastDataset | fold={fold} {split} | cls={cls} | "
            f"{len(self.filenames)} fg patches"
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]

        image    = np.load(self.patches_dir / "all"     / "image_npy"              / fname)  # [H,W,3]
        mask     = np.load(self.patches_dir / self.cls  / "mask_npy"               / fname)  # [H,W]
        suppixel = np.load(self.patches_dir / "all"     / f"slic_{self.n_segments}" / fname) # [H,W]

        image    = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask     = torch.tensor(mask,     dtype=torch.float32).unsqueeze(0)
        suppixel = torch.tensor(suppixel, dtype=torch.long).unsqueeze(0)

        return image, mask, suppixel
