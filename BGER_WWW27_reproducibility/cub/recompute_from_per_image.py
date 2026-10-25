#!/usr/bin/env python3
"""Tier B: recompute per-run metrics and the paired bootstrap from per-image records.

Inputs (this directory):
  per_image/*.npz   for each of the 25 CUB runs: test-image ids, top-1
                    correctness, and packed hit bits hits[i, k, t] =
                    1(IoU_i(t) >= c_k) for IoU criteria c = (.3, .5, .7) and the
                    101 map thresholds t (from the saved per-image IoU matrices).
  data/cub_ev_r101_rows.csv, data/B1_existing_image_bootstrap.json
Checks:
  1. accuracy and MaxBoxAccV2 of every run equal the per-run records exactly;
  2. the paired image bootstrap of BGER(lambda=2) minus CAM-BCE(lambda=.5),
     2,000 resamples per seed with numpy default_rng(0) and V2 re-maximized over
     thresholds inside each resample, reproduces the intervals in the paper.
No model, image or GPU is needed. This is not a re-run of the models.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(method, lam, seed):
    z = np.load(HERE / "per_image" / f"{method}_lam{lam:g}_s{seed}.npz")
    n_t = int(z["n_thresholds"])
    hits = np.unpackbits(z["hits_packed"], axis=-1, count=n_t).astype(bool)
    return z["ids"], z["correct"], hits


def v2(hits, idx=None):
    h = hits if idx is None else hits[idx]
    return float(np.mean([h[:, k].mean(0).max() for k in range(h.shape[1])]))


def main():
    rows = list(csv.DictReader((HERE / "data" / "cub_ev_r101_rows.csv").open()))
    bad = 0
    for r in rows:
        ids, correct, hits = load(r["method"], float(r["lambda_value"]), int(r["seed"]))
        ok = v2(hits) == float(r["V2"]) and abs(correct.mean() - float(r["Acc"])) < 1e-12
        bad += not ok
    print(f"[1] {len(rows)} runs: accuracy/MaxBoxAccV2 mismatches = {bad}")

    ref = json.loads((HERE / "data" / "B1_existing_image_bootstrap.json").read_text())
    rng = np.random.default_rng(0)
    worst = 0.0
    for s in (0, 1, 2):
        ia, ca, ha = load("bger", 2.0, s)
        ib, cb, hb = load("cam_bce", 0.5, s)
        assert (ia == ib).all(), "test-image order differs"
        n = len(ia)
        dv2, dacc = [], []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            dv2.append(v2(ha, idx) - v2(hb, idx))
            dacc.append(float(ca[idx].mean() - cb[idx].mean()))
        got = {"dV2_point": round(v2(ha) - v2(hb), 4),
               "dV2_ci": [round(float(np.percentile(dv2, 2.5)), 4), round(float(np.percentile(dv2, 97.5)), 4)],
               "dAcc_point": round(float(ca.mean() - cb.mean()), 4),
               "dAcc_ci": [round(float(np.percentile(dacc, 2.5)), 4), round(float(np.percentile(dacc, 97.5)), 4)]}
        exp = ref[f"seed{s}"]
        flat_got = [got["dV2_point"], *got["dV2_ci"], got["dAcc_point"], *got["dAcc_ci"]]
        flat_exp = [exp["dV2_point"], *exp["dV2_ci"], exp["dAcc_point"], *exp["dAcc_ci"]]
        worst = max(worst, max(abs(a - b) for a, b in zip(flat_got, flat_exp)))
        print(f"[2] seed {s}: dV2 CI {got['dV2_ci']}  (paper {exp['dV2_ci']})")
    print(f"[2] max |difference| to the paper's rounded values = {worst:.4g}")
    sys.exit(1 if bad or worst > 0 else 0)


if __name__ == "__main__":
    main()
