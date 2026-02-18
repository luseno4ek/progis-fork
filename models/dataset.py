"""
Fold-aware Dataset classes for ProGIS training.

Использует fold_splits.json для разбиения без дублирования данных на диске.

Структура данных (data/patches/):
  all/
    image_npy/           ← image patches, shared
    slic_500/            ← SLIC superpixels, shared
  tumor/
    mask_npy/            ← class-specific fg patches only
    signal_all_line_npy/
  stroma/  ...

fold_splits.json (data/processed/fold_splits.json):
  {
    "fold_1": {"train": ["sample_0030", ...], "val": ["sample_0000", ...]},
    ...
  }
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


def load_fold_splits(splits_path: str) -> dict:
    with open(splits_path) as f:
        return json.load(f)


def get_wsi_stem(patch_filename: str) -> str:
    """
    'sample_0042_patch0003.npy' → 'sample_0042'
    'sample_0042.npy'           → 'sample_0042'
    """
    stem = Path(patch_filename).stem           # без .npy
    if '_patch' in stem:
        return stem.rsplit('_patch', 1)[0]
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: P-RoISeg Dataset
# Input: image [H,W,3] + mask [H,W] + signal [2,H,W]
# ─────────────────────────────────────────────────────────────────────────────

class RoISegDataset(Dataset):
    """
    Dataset for P-RoISeg training (Stage 2).

    Args:
        patches_dir:  path to data/patches/
        splits_path:  path to fold_splits.json
        fold:         int, 1-5
        split:        'train' or 'val'
        cls:          class name, e.g. 'tumor'
    """

    def __init__(self, patches_dir: str, splits_path: str,
                 fold: int, split: str, cls: str):
        self.patches_dir = Path(patches_dir)
        self.cls         = cls
        self.split       = split

        # Загружаем допустимые WSI-стемы для этого фолда
        fold_splits  = load_fold_splits(splits_path)
        valid_stems  = set(fold_splits[f'fold_{fold}'][split])

        # Список файлов: только те, чей WSI-стем в нужном фолде
        mask_dir = self.patches_dir / cls / 'mask_npy'
        all_files = sorted(mask_dir.glob('*.npy'))
        self.filenames = [
            f.name for f in all_files
            if get_wsi_stem(f.name) in valid_stems
        ]

        print(f"RoISegDataset | fold={fold} {split} | cls={cls} | "
              f"{len(self.filenames)} patches")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        image  = np.load(self.patches_dir / 'all' / 'image_npy' / fname)  # [H,W,3]
        mask   = np.load(self.patches_dir / self.cls / 'mask_npy' / fname) # [H,W]
        signal = np.load(self.patches_dir / self.cls / 'signal_all_line_npy' / fname)  # [2,H,W]

        image  = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask   = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        signal = torch.tensor(signal, dtype=torch.float32)

        # Order matches training loop: images, masks, aux_inputs
        return image, mask, signal


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Feature Extractor Dataset
# Input: image [H,W,3] + mask [H,W] + slic [H,W]
# Only foreground patches (mask files exist only for fg patches)
# ─────────────────────────────────────────────────────────────────────────────

class ContrastDataset(Dataset):
    """
    Dataset for Feature Extractor contrastive learning (Stage 1).

    Args:
        patches_dir:  path to data/patches/
        splits_path:  path to fold_splits.json
        fold:         int, 1-5
        split:        'train' or 'val'
        cls:          class for contrastive learning (default: 'tumor')
        n_segments:   SLIC n_segments (default: 500)
    """

    def __init__(self, patches_dir: str, splits_path: str,
                 fold: int, split: str, cls: str = 'tumor',
                 n_segments: int = 500):
        self.patches_dir = Path(patches_dir)
        self.cls         = cls
        self.n_segments  = n_segments

        fold_splits = load_fold_splits(splits_path)
        valid_stems = set(fold_splits[f'fold_{fold}'][split])

        # Берём только fg-патчи (mask файл существует только для fg)
        mask_dir = self.patches_dir / cls / 'mask_npy'
        all_files = sorted(mask_dir.glob('*.npy'))
        self.filenames = [
            f.name for f in all_files
            if get_wsi_stem(f.name) in valid_stems
        ]

        print(f"ContrastDataset | fold={fold} {split} | cls={cls} | "
              f"{len(self.filenames)} fg patches")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        image    = np.load(self.patches_dir / 'all' / 'image_npy' / fname)  # [H,W,3]
        mask     = np.load(self.patches_dir / self.cls / 'mask_npy' / fname) # [H,W]
        slic_dir = self.patches_dir / 'all' / f'slic_{self.n_segments}'
        suppixel = np.load(slic_dir / fname)                                  # [H,W]

        image    = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask     = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        suppixel = torch.tensor(suppixel, dtype=torch.long).unsqueeze(0)

        return image, mask, suppixel
