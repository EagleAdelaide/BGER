"""Versioned multi-contour MaxBoxAccV2 with official WSOL geometry.

Reference: clovaai/wsolevaluation, commit
96f4b88679af41f005824661f47e6fe3628af2aa, evaluation.py.
The legacy metrics.maxboxacc is deliberately unchanged. This contract uses
101 thresholds, a caller-declared full image lattice (normally 224x224),
uint8 quantization, OpenCV RETR_TREE contours (including hole contours), strict
binary thresholding, and the reference inclusive-coordinate IoU convention.
GT boxes must already be transformed to the heatmap lattice; independent GT
boxes remain separate. The image succeeds on ANY candidate/GT match; this is
not all-lesion recall. Degenerate maps retain the official [0,0,0,0] sentinel,
which can match a one-pixel GT at the origin; the regression test exposes this.
"""
from __future__ import annotations
import numpy as np
import cv2

METRIC_VERSION = "wsol_multicontour_101_v1"
REFERENCE_COMMIT = "96f4b88679af41f005824661f47e6fe3628af2aa"
THRESHOLDS = np.linspace(0., 1., 101)
DELTAS = (0.3, 0.5, 0.7)


def normalize_scoremap(heatmap):
    """Float64 min-max with exact zero for constant maps; reject nonfinite input."""
    h = np.asarray(heatmap, dtype=np.float64)
    if h.ndim != 2 or h.size == 0 or not np.isfinite(h).all():
        raise ValueError("Expected nonempty finite HxW heatmap")
    span = h.max() - h.min()
    return np.zeros_like(h) if span == 0 else (h - h.min()) / span


def candidate_boxes(scoremap, thresholds=THRESHOLDS):
    """Replicate official compute_bboxes_from_scoremaps(multi_contour_eval=True)."""
    h = np.asarray(scoremap)
    if h.ndim != 2 or not np.isfinite(h).all() or h.min() < 0 or h.max() > 1:
        raise ValueError("Scoremap must be finite HxW in [0,1]")
    height, width = h.shape
    image = (h * 255).astype(np.uint8)
    result = []
    for threshold in thresholds:
        _, mask = cv2.threshold(image, int(float(threshold) * image.max()), 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        boxes = []
        for contour in contours:
            x, y, w, hh = cv2.boundingRect(contour)
            boxes.append((x, y, min(x+w, width-1), min(y+hh, height-1)))
        result.append(np.asarray(boxes or [(0, 0, 0, 0)], dtype=np.int64))
    return result


def multiple_iou(box_a, box_b):
    """Official inclusive (+1) coordinate arithmetic, including sentinel boxes."""
    a, b = np.asarray(box_a, dtype=np.float64).reshape(-1, 4), np.asarray(box_b, dtype=np.float64).reshape(-1, 4)
    for boxes in (a, b):
        if not np.isfinite(boxes).all() or (boxes < 0).any() or (boxes[:, 2:] < boxes[:, :2]).any():
            raise ValueError("Invalid x0,y0,x1,y1 box")
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(0, hi-lo+1).prod(-1)
    union = (a[:, 2:]-a[:, :2]+1).prod(-1)[:, None] + (b[:, 2:]-b[:, :2]+1).prod(-1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def per_image_ious(heatmap, gtboxes, thresholds=THRESHOLDS, normalize=True):
    boxes = np.asarray(gtboxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        raise ValueError("Spatial subset must have annotation-defined nonempty GT boxes")
    h = normalize_scoremap(heatmap) if normalize else np.asarray(heatmap)
    return np.asarray([multiple_iou(candidates, boxes).max() for candidates in candidate_boxes(h, thresholds)], dtype=np.float64)


def summarize_ious(ious, thresholds=THRESHOLDS, deltas=DELTAS, counts=None):
    """Maximize the dataset hit fraction, separately per IoU criterion.

    counts are image multiplicities, enabling threshold re-maximization within
    each bootstrap draw. Never average per-image oracle best-threshold hits.
    """
    a = np.asarray(ious, dtype=np.float64)
    if a.ndim != 2 or not len(a) or a.shape[1] != len(thresholds) or not np.isfinite(a).all():
        raise ValueError("Expected finite nonempty image x threshold IoU array")
    weights = np.ones(len(a)) if counts is None else np.asarray(counts, dtype=np.float64)
    if weights.shape != (len(a),) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid bootstrap image counts")
    output = {"metric_version": METRIC_VERSION, "N": len(a)}
    values = []
    for d in deltas:
        rates = weights @ (a >= d) / weights.sum()
        index = int(rates.argmax())
        output[f"MaxBoxAcc_{round(d*10):02d}"] = float(rates[index])
        output[f"BestThreshold_{round(d*10):02d}"] = float(thresholds[index])
        values.append(rates[index])
    output["MaxBoxAccV2"] = float(np.mean(values))
    return output


def maxboxaccv2(heatmaps, gtboxes, thresholds=THRESHOLDS, deltas=DELTAS,
                return_iou_matrix=False, normalize=True):
    if len(heatmaps) != len(gtboxes):
        raise ValueError("Map/GT length mismatch")
    ious = np.asarray([per_image_ious(h, g, thresholds, normalize) for h, g in zip(heatmaps, gtboxes)])
    output = summarize_ious(ious, thresholds, deltas)
    if return_iou_matrix:
        output["ious"] = ious
    return output


def half_open_boxes_to_inclusive(boxes, height, width):
    """Rasterize half-open continuous boxes, returning clipped inclusive pixels.

    Use only when the source convention is explicitly half-open. Official WSOL
    annotation coordinates are already inclusive and must not use this helper.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.column_stack((np.floor(boxes[:, :2]), np.ceil(boxes[:, 2:])-1))
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width-1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height-1)
    if (out[:, 2:] < out[:, :2]).any():
        raise ValueError("Empty half-open box")
    return out.astype(np.int64)


per_image_threshold_iou = per_image_ious
