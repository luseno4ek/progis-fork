"""
One-time setup: create a flat patches_dir compatible with RoISegDataset
from the existing fold_1/train|val/tumor directory structure.

Layout created:
  <output_dir>/
    all/
      image_npy/          ← symlinks to train sample_0000 + val sample_0001 images
    tumor/
      mask_npy/           ← symlinks to train sample_0000 + val sample_0001 masks
      signal_all_line_npy/ ← symlinks to train sample_0000 + val sample_0001 signals

  <output_dir>/fold_splits.json:
    { "fold_1": { "train": ["sample_0000"], "val": ["sample_0001"] } }

Usage:
  python -m progis_rework.scripts.setup_local_flat \
      --fold_dir  data/patches/fold_1 \
      --output    data/patches/local_flat \
      --cls       tumor
"""

import argparse
import json
from pathlib import Path


def _symlink_files(src_dir: Path, dst_dir: Path, stem_filter: str) -> int:
    """Create relative symlinks in dst_dir for every *.npy in src_dir matching stem_filter."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob(f"{stem_filter}_*.npy")):
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        count += 1
    return count


def setup(fold_dir: Path, output: Path, cls: str = "tumor") -> None:
    fold_dir = fold_dir.resolve()
    output   = output.resolve()

    train_base = fold_dir / "train" / cls
    val_base   = fold_dir / "val"   / cls

    # Destination dirs
    all_img_dst = output / "all"   / "image_npy"
    mask_dst    = output / cls     / "mask_npy"
    sig_dst     = output / cls     / "signal_all_line_npy"

    # Collect stems present in train and val
    train_stems = sorted({
        p.stem.rsplit("_patch", 1)[0]
        for p in (train_base / "mask_npy").glob("*.npy")
    })
    val_stems = sorted({
        p.stem.rsplit("_patch", 1)[0]
        for p in (val_base / "mask_npy").glob("*.npy")
    } - set(train_stems))   # exclude any overlap with train

    print(f"Train stems: {train_stems}")
    print(f"Val   stems: {val_stems}")

    # Symlink train samples
    n_train = 0
    for stem in train_stems:
        n_train += _symlink_files(train_base / "image_npy",          all_img_dst, stem)
        _symlink_files(train_base / "mask_npy",            mask_dst, stem)
        _symlink_files(train_base / "signal_all_line_npy", sig_dst,  stem)

    # Symlink val samples (from val directory)
    n_val = 0
    for stem in val_stems:
        n_val += _symlink_files(val_base / "image_npy",          all_img_dst, stem)
        _symlink_files(val_base / "mask_npy",            mask_dst, stem)
        _symlink_files(val_base / "signal_all_line_npy", sig_dst,  stem)

    # Write splits JSON
    splits = {
        "fold_1": {
            "train": train_stems,
            "val":   val_stems,
        }
    }
    splits_path = output / "fold_splits.json"
    splits_path.write_text(json.dumps(splits, indent=2))

    print(f"\nFlat patches dir: {output}")
    print(f"  train patches: {n_train}")
    print(f"  val   patches: {n_val}")
    print(f"  splits JSON  : {splits_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Set up flat patches dir for RoISegDataset.")
    p.add_argument("--fold_dir", default="data/patches/fold_1",
                   help="Path to fold_1 directory (contains train/ and val/ subdirs).")
    p.add_argument("--output",   default="data/patches/local_flat",
                   help="Output flat directory.")
    p.add_argument("--cls",      default="tumor",
                   help="Tissue class to link (default: tumor).")
    args = p.parse_args()
    setup(Path(args.fold_dir), Path(args.output), args.cls)


if __name__ == "__main__":
    main()
