"""
ProGISDataProcessor — unified data preparation pipeline.

Replaces three separate scripts:
  - convert_bcss_to_npy.py   (Step 1: PNG WSI → NPY)
  - create_patches.py        (Step 2: NPY WSI → 512×512 patches)

Output layout produced by this processor:
  processed_dir/                   ← Step 1 output
    fold_splits.json
    all/image_npy/                 ← RGB WSI [H,W,3] float32
    {class}/mask_npy/              ← binary WSI mask [H,W] float32
    {class}/signal_all_line_npy/   ← guiding signals [2,H,W] float32

  patches_dir/                     ← Step 2 output
    fold_splits.json               ← copied from processed_dir
    all/image_npy/                 ← image patches [512,512,3]
    {class}/mask_npy/              ← fg-filtered mask patches
    {class}/signal_all_line_npy/   ← fg-filtered signal patches

CLI usage
---------
  # Full pipeline from raw PNG to patches:
  python -m progis_rework.data.processor all \\
      --images_dir data/raw/images \\
      --masks_dir  data/raw/masks \\
      --processed_dir data/processed \\
      --patches_dir   data/patches

  # Step 1 only:
  python -m progis_rework.data.processor convert \\
      --images_dir data/raw/images --masks_dir data/raw/masks \\
      --processed_dir data/processed

  # Step 2 only (processed_dir already exists):
  python -m progis_rework.data.processor patch \\
      --processed_dir data/processed --patches_dir data/patches

Domain customisation
--------------------
Pass a custom label_map dict to override BCSS defaults:
  processor = ProGISDataProcessor(
      ...,
      label_map={'rock': [1,2], 'mineral': [3], 'background': [4,5]},
  )
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from .signal_utils import generate_fg_bg_signals

# Allow PIL to open truncated/incomplete PNG files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Defaults for BCSS ─────────────────────────────────────────────────────────

BCSS_LABEL_MAP: dict[str, list[int]] = {
    # BCSS raw pixel labels → ProGIS class names
    # 0 = outside_roi (ignored), 1-4 = main tissue types, 5-21 = everything else
    "tumor":                     [1],
    "stroma":                    [2],
    "inflammatory_infiltration": [3],
    "necrosis":                  [4],
    "others":                    list(range(5, 22)),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_binary_mask(mc_mask: np.ndarray, label_ids: list[int]) -> np.ndarray:
    """Multi-class integer mask → binary float32 mask for one class."""
    out = np.zeros(mc_mask.shape, dtype=np.float32)
    for lid in label_ids:
        out[mc_mask == lid] = 1.0
    return out


def _resize_to_multiple16(image: np.ndarray, mask: np.ndarray):
    """Pad H and W to multiples of 16 (required by EfficientUNet)."""
    h, w = image.shape[:2]
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    if h == new_h and w == new_w:
        return image, mask
    img_r  = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    mask_r = np.array(
        Image.fromarray(mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
    )
    return img_r, mask_r


def _make_fold_splits(stems: list[str], n_folds: int) -> dict:
    """
    Deterministic k-fold split.

    Args:
        stems:   sorted list of WSI stem names (e.g. ['sample_0000', ...]).
        n_folds: number of folds.

    Returns:
        {'fold_1': {'train': [...], 'val': [...]}, ...}
    """
    n = len(stems)
    fold_size = n // n_folds
    splits = {}
    for k in range(n_folds):
        val_start = k * fold_size
        val_end   = val_start + fold_size if k < n_folds - 1 else n
        val   = stems[val_start:val_end]
        train = stems[:val_start] + stems[val_end:]
        splits[f"fold_{k + 1}"] = {"train": train, "val": val}
    return splits


def _sliding_window_coords(h: int, w: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Top-left (y, x) corners for all valid non-overlapping patch positions."""
    return [
        (y, x)
        for y in range(0, h - patch_size + 1, stride)
        for x in range(0, w - patch_size + 1, stride)
    ]


# ── Main class ────────────────────────────────────────────────────────────────

class ProGISDataProcessor:
    """
    Two-step data preparation pipeline for ProGIS training.

    Args:
        processed_dir:  path where step-1 NPY outputs are written.
        patches_dir:    path where step-2 patch outputs are written.
        label_map:      dict mapping class_name → list of raw pixel label IDs.
                        Defaults to BCSS_LABEL_MAP.
        n_folds:        number of CV folds for fold_splits.json (default 5).
        patch_size:     patch size in pixels (default 512).
        stride:         sliding window stride in pixels (default 256).
        min_fg_ratio:   minimum foreground pixel fraction for a patch to be
                        included in a class directory (default 0.05).
        resize:         whether to pad WSI to multiple-of-16 in step 1 (default True).
    """

    def __init__(
        self,
        processed_dir:  str | Path,
        patches_dir:    str | Path,
        label_map:      Optional[dict[str, list[int]]] = None,
        n_folds:        int   = 5,
        patch_size:     int   = 512,
        stride:         int   = 256,
        min_fg_ratio:   float = 0.05,
        resize:         bool  = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.patches_dir   = Path(patches_dir)
        self.label_map     = label_map if label_map is not None else BCSS_LABEL_MAP
        self.classes       = list(self.label_map.keys())
        self.n_folds       = n_folds
        self.patch_size    = patch_size
        self.stride        = stride
        self.min_fg_ratio  = min_fg_ratio
        self.resize        = resize

    # ── Step 1: PNG WSI → NPY ─────────────────────────────────────────────────

    def convert_wsi_to_npy(
        self,
        images_dir: str | Path,
        masks_dir:  str | Path,
    ) -> None:
        """
        Convert raw PNG WSI images and masks to NPY format.

        For each WSI:
          - Image saved once  → processed_dir/all/image_npy/{stem}.npy  [H,W,3] float32
          - Per-class binary mask → processed_dir/{cls}/mask_npy/{stem}.npy
          - Per-class guiding signals → processed_dir/{cls}/signal_all_line_npy/{stem}.npy

        Skips already-processed WSI (resume-safe).
        Writes fold_splits.json at the end.

        Args:
            images_dir: directory with raw PNG images.
            masks_dir:  directory with raw PNG segmentation masks (same filenames).
        """
        images_dir = Path(images_dir)
        masks_dir  = Path(masks_dir)

        image_files = sorted(
            f for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
            for f in images_dir.glob(ext)
        )
        if not image_files:
            raise FileNotFoundError(f"No image files found in {images_dir}")

        self._print_header(
            f"Step 1: PNG → NPY  |  {self.n_folds}-fold CV  |  {len(self.classes)} classes",
            extra_lines=[
                f"Images : {images_dir}  ({len(image_files)} WSI)",
                f"Masks  : {masks_dir}",
                f"Output : {self.processed_dir}",
                f"Classes: {', '.join(self.classes)}",
            ],
        )

        self._make_processed_dirs()

        stems: list[str] = []
        all_img_dir = self.processed_dir / "all" / "image_npy"

        for wsi_idx, img_file in enumerate(image_files):
            stem     = f"sample_{wsi_idx:04d}"
            filename = f"{stem}.npy"
            img_dst  = all_img_dir / filename

            # Resume: skip already processed WSI
            if img_dst.exists():
                print(f"  [skip] {stem} already processed")
                stems.append(stem)
                continue

            # Load image
            try:
                image = np.array(Image.open(img_file).convert("RGB"))
            except Exception as exc:
                print(f"  [warn] failed to load image {img_file.name}: {exc}")
                continue

            # Load multiclass mask (always look for PNG first, then same extension)
            mask_file = masks_dir / (img_file.stem + ".png")
            if not mask_file.exists():
                mask_file = masks_dir / img_file.name
            if not mask_file.exists():
                print(f"  [warn] mask not found for {img_file.name}, skipping")
                continue
            try:
                mc_mask = np.array(Image.open(mask_file))
            except Exception as exc:
                print(f"  [warn] failed to load mask {mask_file.name}: {exc}")
                continue

            if self.resize:
                image, mc_mask = _resize_to_multiple16(image, mc_mask)

            stems.append(stem)
            np.save(img_dst, image.astype(np.float32))

            class_status: dict[str, str] = {}
            for cls_name, label_ids in self.label_map.items():
                binary_mask = _make_binary_mask(mc_mask, label_ids)

                if binary_mask.sum() == 0:
                    class_status[cls_name] = "–"
                    continue

                signal = generate_fg_bg_signals(binary_mask, seed=wsi_idx * 10)

                np.save(self.processed_dir / cls_name / "mask_npy"            / filename, binary_mask)
                np.save(self.processed_dir / cls_name / "signal_all_line_npy" / filename, signal)
                class_status[cls_name] = "✓"

            status_str = "  ".join(f"{c}:{class_status[c]}" for c in self.classes)
            print(f"  [{wsi_idx+1:3d}/{len(image_files)}] {img_file.name} → {stem}  |  {status_str}")

        # Write fold_splits.json
        fold_splits = _make_fold_splits(stems, self.n_folds)
        splits_path = self.processed_dir / "fold_splits.json"
        with open(splits_path, "w") as f:
            json.dump(fold_splits, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Done! WSI processed: {len(stems)}")
        print(f"  fold_splits.json → {splits_path}")
        for fold_name, split in fold_splits.items():
            print(f"    {fold_name}: train={len(split['train'])}  val={len(split['val'])}")
        print(f"{'='*60}")

    # ── Step 2: NPY WSI → patches ─────────────────────────────────────────────

    def create_patches(self) -> None:
        """
        Cut NPY WSI files into sliding-window patches.

        Reads from processed_dir, writes to patches_dir.

        For each WSI:
          - Image patch saved once (if not already present).
          - Per-class mask + signal patches saved only when fg_ratio >= min_fg_ratio.

        Copies fold_splits.json from processed_dir to patches_dir at the end.
        """
        all_img_dir_src = self.processed_dir / "all" / "image_npy"
        if not all_img_dir_src.exists():
            raise FileNotFoundError(
                f"processed_dir not found: {all_img_dir_src}\n"
                "Run convert_wsi_to_npy() first."
            )

        wsi_files = sorted(all_img_dir_src.glob("*.npy"))
        if not wsi_files:
            raise FileNotFoundError(f"No NPY files found in {all_img_dir_src}")

        self._print_header(
            f"Step 2: NPY → {self.patch_size}×{self.patch_size} patches  (stride={self.stride})",
            extra_lines=[
                f"Input : {self.processed_dir}  ({len(wsi_files)} WSI)",
                f"Output: {self.patches_dir}",
                f"Min fg ratio: {self.min_fg_ratio}",
                f"Classes: {', '.join(self.classes)}",
            ],
        )

        self._make_patch_dirs()

        all_img_dir_dst = self.patches_dir / "all" / "image_npy"
        class_patch_counts = {cls: 0 for cls in self.classes}

        for wsi_file in tqdm(wsi_files, desc="Patching WSI"):
            image = np.load(wsi_file)        # [H, W, 3] float32
            h, w  = image.shape[:2]
            coords = _sliding_window_coords(h, w, self.patch_size, self.stride)

            # Load WSI masks for all classes (None if class absent in this WSI)
            class_masks: dict[str, np.ndarray | None] = {}
            for cls in self.classes:
                mask_file = self.processed_dir / cls / "mask_npy" / wsi_file.name
                class_masks[cls] = np.load(mask_file) if mask_file.exists() else None

            for p_idx, (y, x) in enumerate(coords):
                patch_name = f"{wsi_file.stem}_patch{p_idx:04d}.npy"
                ps = self.patch_size

                # Save image patch once (skip if already exists from previous run)
                img_patch_dst = all_img_dir_dst / patch_name
                if not img_patch_dst.exists():
                    np.save(img_patch_dst, image[y:y+ps, x:x+ps])

                # Per-class: save mask + signal only for foreground patches
                for cls in self.classes:
                    mc = class_masks[cls]
                    if mc is None:
                        continue

                    mask_patch = mc[y:y+ps, x:x+ps]
                    fg_ratio   = mask_patch.sum() / (ps * ps)
                    if fg_ratio < self.min_fg_ratio:
                        continue

                    seed   = class_patch_counts[cls]
                    signal = generate_fg_bg_signals(mask_patch, seed=seed)

                    np.save(self.patches_dir / cls / "mask_npy"            / patch_name, mask_patch)
                    np.save(self.patches_dir / cls / "signal_all_line_npy" / patch_name, signal)
                    class_patch_counts[cls] += 1

        # Copy fold_splits.json
        src_splits = self.processed_dir / "fold_splits.json"
        dst_splits = self.patches_dir   / "fold_splits.json"
        if src_splits.exists():
            shutil.copy2(src_splits, dst_splits)

        n_img = len(list(all_img_dir_dst.glob("*.npy")))
        print(f"\n{'='*60}")
        print(f"  Done!  Image patches: {n_img}")
        for cls in self.classes:
            n = len(list((self.patches_dir / cls / "mask_npy").glob("*.npy")))
            print(f"  {cls:30s}: {n} foreground patches")
        if src_splits.exists():
            print(f"  fold_splits.json → {dst_splits}")
        print(f"{'='*60}")

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run_pipeline(self, images_dir: str | Path, masks_dir: str | Path) -> None:
        """Run step 1 (convert) then step 2 (patch) in sequence."""
        self.convert_wsi_to_npy(images_dir, masks_dir)
        self.create_patches()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_processed_dirs(self) -> None:
        (self.processed_dir / "all" / "image_npy").mkdir(parents=True, exist_ok=True)
        for cls in self.classes:
            (self.processed_dir / cls / "mask_npy").mkdir(parents=True, exist_ok=True)
            (self.processed_dir / cls / "signal_all_line_npy").mkdir(parents=True, exist_ok=True)

    def _make_patch_dirs(self) -> None:
        (self.patches_dir / "all" / "image_npy").mkdir(parents=True, exist_ok=True)
        for cls in self.classes:
            (self.patches_dir / cls / "mask_npy").mkdir(parents=True, exist_ok=True)
            (self.patches_dir / cls / "signal_all_line_npy").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _print_header(title: str, extra_lines: list[str] = []) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print(title)
        print(sep)
        for line in extra_lines:
            print(f"  {line}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="ProGISDataProcessor: prepare data for ProGIS training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    parser.add_argument("--config", default=None,
                       help="Path to a YAML config file (reads data.* section). "
                            "Individual flags below override YAML values.")

    # ── shared args factory ──────────────────────────────────────────────
    def _add_common(p):
        p.add_argument("--processed_dir", default=None,
                       help="Directory for step-1 NPY outputs.")
        p.add_argument("--patches_dir",   default="data/patches",
                       help="Directory for step-2 patch outputs.")
        p.add_argument("--n_folds",       type=int,   default=5)
        p.add_argument("--patch_size",    type=int,   default=512)
        p.add_argument("--stride",        type=int,   default=256)
        p.add_argument("--min_fg_ratio",  type=float, default=0.05)
        p.add_argument("--no_resize",     action="store_true",
                       help="Disable padding WSI to multiple-of-16.")

    # ── convert ──────────────────────────────────────────────────────────
    p_convert = sub.add_parser("convert", help="Step 1: PNG → NPY")
    p_convert.add_argument("--images_dir", default=None)
    p_convert.add_argument("--masks_dir",  default=None)
    _add_common(p_convert)

    # ── patch ────────────────────────────────────────────────────────────
    p_patch = sub.add_parser("patch", help="Step 2: NPY → patches")
    _add_common(p_patch)

    # ── all ──────────────────────────────────────────────────────────────
    p_all = sub.add_parser("all", help="Full pipeline: PNG → NPY → patches")
    p_all.add_argument("--images_dir", default=None)
    p_all.add_argument("--masks_dir",  default=None)
    _add_common(p_all)

    return parser


def _cfg_from_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    d = raw.get("data", {})
    return {
        "images_dir":    d.get("images_dir"),
        "masks_dir":     d.get("masks_dir"),
        "processed_dir": d.get("processed_dir"),
        "patches_dir":   d.get("patches_dir"),
        "n_folds":       d.get("n_folds",       5),
        "patch_size":    d.get("patch_size",    512),
        "stride":        d.get("stride",        256),
        "min_fg_ratio":  d.get("min_fg_ratio",  0.05),
        "label_map":     d.get("label_map"),     # dict str→list[int] or None
    }


def main() -> None:
    args = _build_parser().parse_args()

    # YAML provides defaults; CLI flags override
    cfg = _cfg_from_yaml(args.config) if hasattr(args, "config") and args.config else {}

    def get(key, fallback=None):
        cli_val = getattr(args, key, None)
        return cli_val if cli_val is not None else cfg.get(key, fallback)

    processor = ProGISDataProcessor(
        processed_dir = get("processed_dir", "data/processed"),
        patches_dir   = get("patches_dir",   "data/patches"),
        label_map     = get("label_map"),     # None → BCSS defaults
        n_folds       = get("n_folds",       5),
        patch_size    = get("patch_size",    512),
        stride        = get("stride",        256),
        min_fg_ratio  = get("min_fg_ratio",  0.05),
        resize        = not getattr(args, "no_resize", False),
    )

    images_dir = get("images_dir")
    masks_dir  = get("masks_dir")

    if args.command == "convert":
        processor.convert_wsi_to_npy(images_dir, masks_dir)
    elif args.command == "patch":
        processor.create_patches()
    elif args.command == "all":
        processor.run_pipeline(images_dir, masks_dir)


if __name__ == "__main__":
    main()
