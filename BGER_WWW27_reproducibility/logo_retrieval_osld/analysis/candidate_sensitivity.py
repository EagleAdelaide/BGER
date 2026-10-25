#!/usr/bin/env python3
"""Candidate-sensitivity diagnostic for the fixed-candidate view (descriptive).

Pre-test secondary diagnostic (amendment A5, adopted for OSLD-SingleLogo-Retrieval-v1 on 2026-10-16). It reads
outputs that the frozen runner already writes for every selected run --
fixed_pairs.json and signed_maps/*.npz -- and needs no training and no new
model inference. It does not change any primary endpoint or hypothesis.

For each query q with its CE-seed-0-reference positive d+ and negative d-:
  * score change          ds = s(q,d+) - s(q,d-)
  * raw magnitudes        max|m|, sum ReLU(m) of the query-side signed map for d+ and d-
  * degeneracy            constant or non-finite signed map, per candidate
  * map distance          L1 distance between the ReLU-normalized maps (0..2),
                          defined only if both positive parts are non-zero
  * map correlation       Pearson r between the ReLU parts; UNDEFINED (NaN, counted)
                          if either part is constant -- never imputed as 1
  * verdict change        whether query-side IoU>=.5 localization at the run's
                          locked threshold differs between d+ and d-
A candidate-independent reference is computed with the identical map pipeline:
a fixed centre prior on the feature grid (Gaussian, sigma = grid/4). If the
score-conditioned maps localize no better than this reference, the evaluated
localization reflects product extent rather than candidate-specific evidence.

This quantifies candidate dependence of the readout. It is not a measure of
causal faithfulness and is not ground truth for why a candidate matched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from e1_common import GRID  # noqa: E402  (read-only import; code/ is unchanged)
from e1_evaluate import map_metrics  # noqa: E402

READOUTS = ("primary", "secondary")
TOL = 1e-12


def positive_part(m: np.ndarray):
    p = np.clip(np.asarray(m, dtype=np.float64), 0.0, None)
    finite = bool(np.isfinite(m).all())
    total = float(p.sum()) if finite else float("nan")
    return p, finite, total


def pair_stats(m_pos: np.ndarray, m_neg: np.ndarray) -> dict:
    """Distance/correlation between two signed maps of the same query."""
    out = {}
    for tag, m in (("pos", m_pos), ("neg", m_neg)):
        m = np.asarray(m, dtype=np.float64)
        finite = bool(np.isfinite(m).all())
        out[f"raw_max_abs_{tag}"] = float(np.abs(m).max()) if finite else float("nan")
        p, _, total = positive_part(m)
        out[f"raw_pos_sum_{tag}"] = total
        out[f"degenerate_{tag}"] = (not finite) or float(m.max() - m.min()) < TOL
    p1, f1, t1 = positive_part(m_pos)
    p2, f2, t2 = positive_part(m_neg)
    if f1 and f2 and t1 > TOL and t2 > TOL:
        out["l1_distance"] = float(np.abs(p1 / t1 - p2 / t2).sum())
    else:
        out["l1_distance"] = float("nan")
    if f1 and f2 and p1.std() > TOL and p2.std() > TOL:
        out["correlation"] = float(np.corrcoef(p1.ravel(), p2.ravel())[0, 1])
    else:
        out["correlation"] = float("nan")  # undefined, counted separately
    return out


def centre_prior(shape) -> torch.Tensor:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx, sy, sx = (h - 1) / 2, (w - 1) / 2, h / 4, w / 4
    g = np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    return torch.tensor(g, dtype=torch.float32)


def analyze_run(run_dir: Path, records: dict) -> dict:
    fixed = json.loads((run_dir / "fixed_pairs.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    identity = fixed.get("identity")
    by_query: dict = {}
    for p in fixed["pairs"]:
        sign = "positive" if p["view"].endswith("_positive") else "negative"
        by_query.setdefault(p["query_id"], {})[sign] = p
    rows = []
    for qid, d in sorted(by_query.items()):
        if set(d) != {"positive", "negative"}:
            raise ValueError(f"query {qid} lacks a positive/negative pair")
        zp = np.load(run_dir / "signed_maps" / d["positive"]["signed_maps_file"])
        zn = np.load(run_dir / "signed_maps" / d["negative"]["signed_maps_file"])
        row = {"query_id": qid, "score_pos": d["positive"]["score"],
               "score_neg": d["negative"]["score"],
               "score_change": d["positive"]["score"] - d["negative"]["score"]}
        for ro in READOUTS:
            st = pair_stats(zp[f"query_{ro}_signed"], zn[f"query_{ro}_signed"])
            row.update({f"{ro}_{k}": v for k, v in st.items()})
            idx = int(np.argmin(np.abs(GRID - summary["thresholds"][ro])))
            lp = d["positive"]["query"][ro]["hits"][idx][1]
            ln = d["negative"]["query"][ro]["hits"][idx][1]
            row[f"{ro}_loc_pos"], row[f"{ro}_loc_neg"] = int(lp), int(ln)
            row[f"{ro}_verdict_changes"] = int(lp != ln)
            shape = np.asarray(zp[f"query_{ro}_signed"]).shape
            ref = map_metrics(centre_prior(shape), records[qid])
            row[f"{ro}_centre_prior_loc"] = int(ref["hits"][idx][1])
        rows.append(row)
    return {"identity": identity, "run": run_dir.name, "rows": rows}


def aggregate(rows: list) -> dict:
    def mean(key, finite_only=True):
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)] if finite_only else v
        return float(v.mean()) if len(v) else None
    out = {"n_queries": len(rows), "mean_score_change": mean("score_change")}
    for ro in READOUTS:
        corr = np.array([r[f"{ro}_correlation"] for r in rows], dtype=float)
        out[ro] = {
            "degenerate_rate_pos": mean(f"{ro}_degenerate_pos", False),
            "degenerate_rate_neg": mean(f"{ro}_degenerate_neg", False),
            "median_raw_max_abs_pos": float(np.nanmedian([r[f"{ro}_raw_max_abs_pos"] for r in rows])),
            "mean_l1_distance": mean(f"{ro}_l1_distance"),
            "mean_correlation_defined": mean(f"{ro}_correlation"),
            "correlation_undefined_count": int(np.isnan(corr).sum()),
            "verdict_change_rate": mean(f"{ro}_verdict_changes", False),
            "loc_rate_pos": mean(f"{ro}_loc_pos", False),
            "loc_rate_neg": mean(f"{ro}_loc_neg", False),
            "centre_prior_loc_rate": mean(f"{ro}_centre_prior_loc", False),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", required=True, help="runner output directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    records = {r["image_id"]: r for r in
               json.loads((root / "technical_integrity.json").read_text())["image_records"]}
    result = {"diagnostic": "candidate_sensitivity_v1 (amendment A5, descriptive)",
              "runs": {}}
    identities = set()
    for run_dir in sorted((root / "test").iterdir()):
        if not (run_dir / "fixed_pairs.json").exists():
            continue
        r = analyze_run(run_dir, records)
        identities.add(r["identity"])
        result["runs"][r["run"]] = {"aggregate": aggregate(r["rows"]), "rows": r["rows"]}
    result["identity"] = sorted(identities)
    # Only outputs of an approved core run can carry scientific meaning.
    result["scientific_result"] = bool(identities) and identities <= {"OSLD_E1_CORE"}
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=True))
    print(json.dumps({k: v["aggregate"] for k, v in result["runs"].items()}, indent=1)[:2000])


if __name__ == "__main__":
    main()
