"""
Guiding signal generation — single source of truth.

Replaces duplicated generate_guiding_signal / generateGuidingSignal found in:
  - convert_bcss_to_npy.py
  - create_patches.py
  - models/backbone_efficientunet_inference_BCSS.py
  - models/train_roi_efficientunet_BCSS.py

All functions operate on NumPy arrays (domain-agnostic, no PyTorch).
The training/inference loop needs a Tensor wrapper — see interactive/signals.py.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def generate_guiding_signal(binary_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Generate a skeleton-based guiding signal from a binary mask.

    Algorithm:
      1. Compute Euclidean distance transform of the mask.
      2. Pick a random threshold in [mean - std, mean + std] of non-zero distances.
      3. Skeletonize the thresholded region → sparse line signal.

    Args:
        binary_mask: [H, W] float or uint8, values in {0, 1}.
        seed:        random seed for reproducibility (pass different seeds for
                     fg and bg signals of the same patch).

    Returns:
        [H, W] float32 skeleton signal (values 0 or 1).
    """
    np.random.seed(seed)
    bm = (binary_mask > 0.5).astype(np.uint8)

    if bm.sum() == 0:
        return np.zeros_like(bm, dtype=np.float32)

    dist = distance_transform_edt(bm)
    nonzero = dist[dist > 0]

    if len(nonzero) == 0:
        return np.zeros_like(bm, dtype=np.float32)

    mean_d = float(nonzero.mean())
    std_d  = float(nonzero.std())
    thresh = max(0.0, float(np.random.uniform(mean_d - std_d, mean_d + std_d)))

    skel_mask = (dist > thresh).astype(np.uint8)
    if skel_mask.sum() == 0:
        skel_mask = bm

    return skeletonize(skel_mask).astype(np.float32)


def generate_fg_bg_signals(binary_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Generate foreground and background guiding signals for a patch.

    Convenience wrapper: calls generate_guiding_signal twice with different seeds
    so fg/bg signals are independently randomised.

    Args:
        binary_mask: [H, W] binary mask (foreground = 1).
        seed:        base random seed; bg uses seed + 1.

    Returns:
        [2, H, W] float32 — channel 0: fg signal, channel 1: bg signal.
    """
    fg = generate_guiding_signal(binary_mask,       seed=seed)
    bg = generate_guiding_signal(1.0 - binary_mask, seed=seed + 1)
    return np.stack([fg, bg], axis=0)
