"""Spatial grounding metrics from attribution heatmaps + GT regions.

A GT region is a binary mask of shape (H, W) in the SAME coordinate frame as
the (resized) image the heatmap was computed on. For CUB v1 the mask is the
filled bounding box; the same code works for segmentation masks.

Heatmaps are expected pre-normalized to [0,1] with nonnegative entries (see
attribution.gradcam.normalize_cam). We re-normalize defensively.
"""
from __future__ import annotations
import numpy as np


def _norm01(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0, None)
    mn, mx = h.min(), h.max()
    if mx - mn < eps:
        return np.zeros_like(h)
    return (h - mn) / (mx - mn)


def energy_in_region(heatmap: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    """Energy-in-Mask/Box: fraction of heatmap energy inside the GT region.

    EIM = sum(heatmap * mask) / sum(heatmap), in [0, 1].
    """
    h = _norm01(heatmap)
    m = (np.asarray(mask) > 0).astype(np.float64)
    tot = h.sum()
    if tot < eps:
        return 0.0
    return float((h * m).sum() / tot)


def energy_outside_region(heatmap: np.ndarray, mask: np.ndarray) -> float:
    return 1.0 - energy_in_region(heatmap, mask)


def pointing_game(heatmap: np.ndarray, mask: np.ndarray) -> int:
    """1 if the heatmap's argmax pixel lies inside the GT region, else 0."""
    h = _norm01(heatmap)
    idx = np.unravel_index(np.argmax(h), h.shape)
    return int(bool(np.asarray(mask)[idx] > 0))


def cam_iou(heatmap: np.ndarray, mask: np.ndarray, frac_of_max: float = 0.5) -> float:
    """IoU between thresholded heatmap (>= frac_of_max * max) and GT region."""
    h = _norm01(heatmap)
    pred = h >= frac_of_max
    gt = np.asarray(mask) > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def box_area_ratio(mask: np.ndarray) -> float:
    """Fraction of the image covered by the GT region (box area / image area)."""
    m = np.asarray(mask)
    return float((m > 0).sum()) / float(m.size)


def normalized_energy_in_region(eir: float, bar: float, eps: float = 1e-6) -> float:
    """NEIB = Energy-in-Box / BoxAreaRatio. Controls for large boxes.

    NEIB == 1 means energy is distributed exactly proportionally to area
    (chance level); NEIB > 1 means energy is concentrated inside the box more
    than its area would predict. Unbounded above; clip for use as a [0,1]
    grounding score in GCE_NEIB.
    """
    return float(eir / max(bar, eps))


def background_excess_energy(eir: float, bar: float) -> float:
    """BEE = EnergyOutsideBox - BackgroundAreaRatio = (1-EIR) - (1-BAR).

    Positive => the heatmap puts MORE energy on the background than its area
    share, i.e. above-chance background reliance.
    """
    return float((1.0 - eir) - (1.0 - bar))


# ---- attribution sanity baselines (not real explanations) -------------------

def random_heatmap(shape, rng: np.random.Generator) -> np.ndarray:
    return rng.random(shape)


def center_gaussian_heatmap(shape, sigma_frac: float = 0.25) -> np.ndarray:
    """A blob centered on the image -- a content-free positional prior baseline."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sy, sx = sigma_frac * h, sigma_frac * w
    g = np.exp(-(((yy - cy) ** 2) / (2 * sy ** 2) + ((xx - cx) ** 2) / (2 * sx ** 2)))
    return g
