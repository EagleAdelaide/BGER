"""Author-adopted A3 fixed-model/fixed-gallery inference; no real-data run approval.

Query-weighted estimands; resample query-product clusters, keeping all queries
and all trained seeds together. Basic bootstrap CIs and centered-bootstrap
one-sided p-values are asymptotic approximations, not exact finite-sample tests.
They do not estimate training-seed/population or gallery uncertainty.
"""
from __future__ import annotations
import numpy as np


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("A nonempty finite probability vector is required")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    output = np.empty_like(adjusted); output[order] = adjusted
    return output


def paired_cluster_inference(differences, query_product_ids, null_boundaries,
                             *, n_boot=10000, random_seed=0, alpha=.05):
    """differences[key] is (seeds, queries), already paired A-minus-B.

    null_boundaries[key] defines H0: theta <= boundary (upper alternative).
    All keys share the same resampling indices and fixed seed aggregation.
    The caller fixes which keys belong in a multiplicity family before test.
    """
    keys = list(differences)
    if not keys or set(keys) != set(null_boundaries):
        raise ValueError("Null boundary keys must match contrast keys")
    if not isinstance(n_boot, int) or n_boot < 100:
        raise ValueError("At least 100 draws required for helper; protocol requires 10000")
    if not 0 < alpha < 1: raise ValueError("Invalid alpha")
    pids = np.asarray(query_product_ids)
    if pids.ndim != 1 or len(pids) == 0: raise ValueError("Query-product vector required")
    unique, inverse = np.unique(pids, return_inverse=True)
    if len(unique) < 2: raise ValueError("Cannot resample fewer than two clusters")
    arrays = {k: np.asarray(differences[k], dtype=float) for k in keys}
    shape = arrays[keys[0]].shape
    if len(shape) != 2 or shape[0] == 0 or shape[1] != len(pids):
        raise ValueError("Expected (seeds,queries) difference arrays")
    if any(a.shape != shape or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Identical shapes and complete finite pairs required")
    if not all(np.isfinite(null_boundaries[k]) for k in keys): raise ValueError("Invalid boundary")
    counts = np.bincount(inverse).astype(float)
    # Average SEEDS, not predictions. Query weights remain image-level weights.
    per_query = {k: a.mean(axis=0) for k, a in arrays.items()}
    sums = {k: np.bincount(inverse, weights=per_query[k], minlength=len(unique)) for k in keys}
    points = {k: float(per_query[k].mean()) for k in keys}
    replicates = {k: np.empty(n_boot) for k in keys}
    rng = np.random.default_rng(random_seed)
    for b in range(n_boot):
        draw = rng.integers(0, len(unique), size=len(unique))
        den = counts[draw].sum()
        for k in keys: replicates[k][b] = sums[k][draw].sum() / den
    result = {}
    raw_p = []
    for k in keys:
        observed = points[k]; boot = replicates[k]; boundary = float(null_boundaries[k])
        center = boot - observed
        degenerate = bool(np.ptp(boot) <= np.finfo(float).eps * 16)
        # Finite Monte Carlo tail; do not treat an uncentered percentile as p.
        p = 1.0 if degenerate else (1 + np.count_nonzero(center >= observed - boundary)) / (n_boot + 1)
        qlo, qhi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], method="linear")
        lower = 2 * observed - np.quantile(boot, 1 - alpha, method="linear")
        result[k] = {"point": observed, "per_seed_point": arrays[k].mean(axis=1).tolist(),
                     "null_boundary": boundary, "basic_two_sided_ci": [float(2*observed-qhi), float(2*observed-qlo)],
                     "basic_one_sided_lower": float(lower), "p_one_sided_centered_bootstrap": float(p),
                     "bootstrap_degenerate": degenerate, "reject_unadjusted": bool(p <= alpha),
                     "replicates": boot}
        raw_p.append(p)
    adj = holm_adjust(raw_p)
    for k, p in zip(keys, adj):
        result[k]["p_holm"] = float(p); result[k]["reject_holm"] = bool(p <= alpha)
    return {"contrasts": result, "n_seeds": shape[0], "n_queries": len(pids),
            "n_products": len(unique), "n_boot": n_boot, "seed": random_seed,
            "all_family_rejected": bool(all(result[k]["reject_holm"] for k in keys)),
            "scope": "fixed trained models and fixed gallery; query logo-identity clusters only (OSLD adapter stores the identity in product_id)",
            "method": "basic CIs; centered bootstrap upper-tail p; Holm; asymptotic approximation"}
