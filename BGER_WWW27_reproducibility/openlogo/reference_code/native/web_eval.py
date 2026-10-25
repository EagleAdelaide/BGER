#!/usr/bin/env python3
"""Evaluation library for the BGER WWW27 DeepFashion In-shop experiment (E1).

Implements, per BGER_WWW27_Audit_Design.md Sec. 6:
  - identity-relevance retrieval metrics: Recall@1, Recall@10, mAP;
  - retrieval-score-conditioned Grad-CAM for query AND gallery sides,
    differentiating the deployed cosine score s(q, d*) with the opposite
    embedding held fixed (d* = actual top-1, incorrect retrievals included);
  - validation-frozen single-threshold IoU@.5 localization, with all-zero
    maps counted as failures (never correlation=1 backfill);
  - J@1 = correct_top1 * 1(IoU_q>=.5) * 1(IoU_g>=.5), with marginals;
  - paired bootstrap resampling QUERY PRODUCT IDs (all images of a sampled
    product move together; gallery fixed).

No training here. Pure functions over tensors/arrays so they are unit-testable
on CPU with synthetic data (tests/test_web_eval.py).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- retrieval

def retrieval_metrics(sim: np.ndarray, q_pid: np.ndarray, g_pid: np.ndarray,
                      ks=(1, 10), *, query_ids=None, gallery_ids=None):
    """Candidate A4 ranking guard; all queries must have a non-self relevant item.

    Supply both image-ID vectors for real use. Ties: descending exact score,
    then lexicographically ascending gallery image ID. Different image IDs of
    the same product are not self-matches. No query is silently discarded.
    Missing IDs remain supported for legacy synthetic tests only and flagged.
    """
    sim = np.asarray(sim)
    q_pid, g_pid = np.asarray(q_pid), np.asarray(g_pid)
    if sim.ndim != 2 or min(sim.shape) == 0:
        raise ValueError("Nonempty (Q,G) score matrix required")
    if q_pid.shape != (sim.shape[0],) or g_pid.shape != (sim.shape[1],):
        raise ValueError("Product-ID shapes must match scores")
    if not np.isfinite(sim).all():
        raise ValueError("Nonfinite ranking score; do not exclude failed queries")
    if not ks or any(int(k) != k or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    if (query_ids is None) != (gallery_ids is None):
        raise ValueError("Supply both image-ID vectors, or neither in synthetic tests")
    explicit_ids = query_ids is not None
    if explicit_ids:
        qid = np.asarray(query_ids, dtype=str)
        gid = np.asarray(gallery_ids, dtype=str)
        if qid.shape != q_pid.shape or gid.shape != g_pid.shape:
            raise ValueError("Image-ID shapes must match scores")
        if len(np.unique(qid)) != len(qid) or len(np.unique(gid)) != len(gid):
            raise ValueError("Duplicate image IDs within a split")
    else:
        qid = np.array([f"query_{i:012d}" for i in range(sim.shape[0])])
        gid = np.array([f"gallery_{i:012d}" for i in range(sim.shape[1])])
    by_id = np.argsort(gid, kind="stable")
    top, correct, aps = [], [], []
    hits = {int(k): [] for k in ks}
    for i in range(len(q_pid)):
        valid = by_id[gid[by_id] != qid[i]]
        if len(valid) == 0:
            raise ValueError(f"No non-self gallery candidate for query {qid[i]}")
        order = valid[np.argsort(-sim[i, valid], kind="stable")]
        rel = g_pid[order] == q_pid[i]
        n_rel = int(rel.sum())
        if n_rel == 0:
            raise ValueError(f"No relevant non-self gallery item for query {qid[i]}")
        positions = np.flatnonzero(rel)
        aps.append(float(np.mean(np.arange(1, n_rel + 1) / (positions + 1))))
        top.append(int(order[0])); correct.append(int(rel[0]))
        for k in hits:
            hits[k].append(bool(rel[:k].any()))
    return {**{f"recall@{k}": float(np.mean(v)) for k, v in hits.items()},
            "mAP": float(np.mean(aps)), "per_query_AP": np.asarray(aps),
            "top1_correct": np.asarray(correct, dtype=np.int64),
            "top1_index": np.asarray(top, dtype=np.int64),
            "canonical_image_ids_supplied": explicit_ids}

# ------------------------------------------------------- score-side Grad-CAM

def cosine_score(z_q: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
    return (z_q * z_d).sum() / (z_q.norm() * z_d.norm() + 1e-12)


def retrieval_gradcam(feat: torch.Tensor, pool_fn, z_other: torch.Tensor,
                      relu: bool = True) -> torch.Tensor:
    """Grad-CAM of the deployed score s = cos(pool_fn(feat), z_other) w.r.t.
    `feat` (C,Hh,Ww), with the opposite-side embedding z_other FIXED.
    pool_fn maps feat -> embedding (the model's own pooling: gated for BGER,
    GAP otherwise). Returns the (Hh,Ww) CAM (unnormalized)."""
    feat = feat.detach().clone().requires_grad_(True)
    z = pool_fn(feat)
    s = cosine_score(z, z_other.detach())
    g = torch.autograd.grad(s, feat)[0]              # (C,Hh,Ww)
    alpha = g.mean(dim=(1, 2))                       # channel weights
    cam = torch.einsum("c,chw->hw", alpha, feat.detach())
    if relu:
        cam = cam.clamp_min(0)
    return cam

# ----------------------------------------------------------- localization

def normalize_map(cam: np.ndarray):
    """Per-map [0,1] normalization; returns (map, is_degenerate)."""
    lo, hi = float(cam.min()), float(cam.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(cam), True
    return (cam - lo) / (hi - lo), False


def box_from_map(cam01: np.ndarray, thr: float):
    """Tightest box around the largest connected component of cam01>=thr.
    Returns (x0,y0,x1,y1) in map coords or None if empty."""
    mask = cam01 >= thr
    if not mask.any():
        return None
    # 4-connected component labeling without scipy
    lab = np.zeros(mask.shape, dtype=np.int32); cur = 0
    stack = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] and lab[i, j] == 0:
                cur += 1; stack.append((i, j)); lab[i, j] = cur
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        na, nb = a+da, b+db
                        if 0 <= na < mask.shape[0] and 0 <= nb < mask.shape[1] \
                           and mask[na, nb] and lab[na, nb] == 0:
                            lab[na, nb] = cur; stack.append((na, nb))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    comp = sizes.argmax()
    ys, xs = np.nonzero(lab == comp)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / union if union > 0 else 0.0


def select_threshold(val_cams, val_boxes, grid=np.linspace(0.05, 0.95, 19),
                     iou_at=0.5):
    """Frozen rule: single threshold maximizing dev IoU@.5 hit-rate; ties ->
    LOWEST threshold. Degenerate maps count as misses at every threshold."""
    best_thr, best = None, -1.0
    for t in grid:
        hits = []
        for cam, gt in zip(val_cams, val_boxes):
            cam01, degen = normalize_map(cam)
            hits.append(0.0 if degen else float(iou(box_from_map(cam01, t), gt) >= iou_at))
        m = float(np.mean(hits))
        if m > best + 1e-12:
            best, best_thr = m, float(t)
    return best_thr, best


def localization_hit(cam, gt_box, thr, iou_at=0.5):
    cam01, degen = normalize_map(np.asarray(cam))
    if degen:
        return 0, 1
    return int(iou(box_from_map(cam01, thr), gt_box) >= iou_at), 0

# ---------------------------------------------------------------- J@1 + CI

def joint_j_at_1(top1_correct, hit_q, hit_g):
    top1_correct, hit_q, hit_g = map(np.asarray, (top1_correct, hit_q, hit_g))
    j = top1_correct * hit_q * hit_g
    return {"J@1": float(j.mean()),
            "marginal_top1": float(top1_correct.mean()),
            "marginal_loc_query": float(hit_q.mean()),
            "marginal_loc_gallery": float(hit_g.mean())}


def product_bootstrap(per_query: np.ndarray, q_pid: np.ndarray,
                      n_boot=10000, seed=0, stat=np.mean):
    """Percentile CI resampling PRODUCT IDs; all queries of a sampled product
    enter together. `per_query` may be a difference series for paired
    method contrasts (gallery fixed)."""
    rng = np.random.default_rng(seed)
    pids = np.unique(q_pid)
    groups = {p: np.flatnonzero(q_pid == p) for p in pids}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([groups[p] for p in take])
        stats[b] = stat(per_query[idx])
    return {"point": float(stat(per_query)),
            "ci_lo": float(np.percentile(stats, 2.5)),
            "ci_hi": float(np.percentile(stats, 97.5)),
            "n_products": int(len(pids)), "n_boot": int(n_boot)}

# ------------------- Round-2 additions: audited cosine readout + records ----

NORM_FLOOR = 1e-12  # legacy name for additive SCORE epsilon, NOT a clamp


def retrieval_gradcam_audited(feat: torch.Tensor, pool_fn, z_other: torch.Tensor):
    """Audited variant of retrieval_gradcam. The query's OWN L2-normalization
    derivative is kept in the graph (only the opposite embedding is fixed);
    detaching the query norm while calling it the actual cosine gradient is
    exactly the negative control tested in tests/test_cosine_contract.py.

    Returns (signed_cam, positive_cam, record) where record carries the
    per-case audit fields required by the Round-2 contract."""
    feat = feat.detach().clone().requires_grad_(True)
    z = pool_fn(feat)
    zn, on = z.norm(), z_other.detach().norm()
    floor_hit = bool(zn < 1e-6 or on < 1e-6)
    s = (z * z_other.detach()).sum() / (zn * on + NORM_FLOOR)
    g = torch.autograd.grad(s, feat)[0]
    alpha = g.mean(dim=(1, 2))
    signed = torch.einsum("c,chw->hw", alpha, feat.detach())
    positive = signed.clamp_min(0)
    rec = {
        "score": float(s.detach()), "query_norm": float(zn.detach()), "other_norm": float(on.detach()),
        "grad_norm": float(g.norm()),
        "signed_map_max_abs": float(signed.abs().max()),
        "positive_map_sum": float(positive.sum()),
        "norm_floor_hit": floor_hit,  # legacy name, NOT an applied norm clamp
        "low_norm_diagnostic": floor_hit,
        "score_additive_epsilon": NORM_FLOOR,
        "degenerate_map": bool(float(positive.max() - positive.min()) < 1e-12),
        "nonfinite": bool(not torch.isfinite(signed).all()),
    }
    return signed, positive, rec


def score_decomposition(F: torch.Tensor, z_other: torch.Tensor):
    """SECONDARY, internally pre-specified readout (all four arms identically):
    D_u = (F_u · z_d) / (U * ||z_q|| * ||z_d||) with z_q = mean_u F_u, so that
    sum_u D_u equals the same stabilized score in exact arithmetic.
    Floating-point reductions have measured, dtype-specific tolerance.
    F is the feature actually pooled (M*H for BGER, H otherwise), (C,Hh,Ww).
    This is an algebraic decomposition of the forward value — NOT the score
    derivative, NOT causal attribution, and in BGER it directly uses the
    supervised route (never present it as a route-independent audit).
    Returns (signed_map, score); keep the signed map for the sum-to-score
    identity; any ReLU/normalization for localization is logged separately."""
    C, Hh, Ww = F.shape
    U = Hh * Ww
    z_q = F.mean(dim=(1, 2))
    denom = U * (z_q.norm() * z_other.norm() + NORM_FLOOR)
    D = torch.einsum("chw,c->hw", F, z_other) / denom
    s = (z_q * z_other).sum() / (z_q.norm() * z_other.norm() + NORM_FLOOR)
    return D, float(s.detach())


def product_bootstrap_shared(series_by_key: dict, q_pid: np.ndarray,
                             n_boot=10000, seed=0, stat=np.mean):
    """Same product-cluster bootstrap, but ONE set of resample draws shared
    across every entry of series_by_key (methods/seeds/metrics), as the
    Round-2 contract requires for paired comparisons."""
    rng = np.random.default_rng(seed)
    pids = np.unique(q_pid)
    groups = {p: np.flatnonzero(q_pid == p) for p in pids}
    draws = [rng.choice(pids, size=len(pids), replace=True) for _ in range(n_boot)]
    out = {}
    for key, series in series_by_key.items():
        series = np.asarray(series, dtype=float)
        stats = np.empty(n_boot)
        for b, take in enumerate(draws):
            idx = np.concatenate([groups[p] for p in take])
            stats[b] = stat(series[idx])
        out[key] = {"point": float(stat(series)),
                    "ci_lo": float(np.percentile(stats, 2.5)),
                    "ci_hi": float(np.percentile(stats, 97.5))}
    return out
