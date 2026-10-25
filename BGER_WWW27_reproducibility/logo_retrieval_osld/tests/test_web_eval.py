"""Unit tests for paper_audits/www2027/code/web_eval.py (synthetic CPU only)."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import web_eval as W


def test_retrieval_metrics_exact():
    # 3 queries, 4 gallery; constructed ranks
    sim = np.array([[0.9, 0.1, 0.2, 0.0],    # q0 -> g0 (correct pid)
                    [0.1, 0.8, 0.7, 0.0],    # q1 -> g1 (wrong), g2 correct @2
                    [0.0, 0.1, 0.2, 0.9]])   # q2 -> g3 (correct)
    q = np.array([1, 2, 3]); g = np.array([1, 9, 2, 3])
    m = W.retrieval_metrics(sim, q, g, ks=(1, 2))
    assert m["recall@1"] == 2 / 3
    assert m["recall@2"] == 1.0
    # AP: q0=1, q1=1/2, q2=1 -> mAP = 2.5/3
    assert abs(m["mAP"] - (1 + 0.5 + 1) / 3) < 1e-12
    assert list(m["top1_correct"]) == [1, 0, 1]


def test_gradcam_matches_identity_for_gap():
    # For ungated GAP pooling, CAM weights must equal mu*r with mu=1/(hw)
    torch.manual_seed(0)
    C, Hh, Ww = 8, 5, 5
    feat = torch.randn(C, Hh, Ww, dtype=torch.float64)
    z_d = torch.randn(C, dtype=torch.float64)
    pool = lambda f: f.mean(dim=(1, 2))
    cam = W.retrieval_gradcam(feat, pool, z_d, relu=False)
    with torch.no_grad():
        z = pool(feat)
        s = W.cosine_score(z, z_d)
        r = z_d / (z.norm() * z_d.norm() + 1e-12) - s * z / (z.norm() ** 2 + 1e-12)
        alpha = r / (Hh * Ww)
        ref = torch.einsum("c,chw->hw", alpha, feat)
    assert float((cam - ref).abs().max()) < 1e-10


def test_degenerate_map_counts_as_failure_not_one():
    cam = np.zeros((7, 7))
    hit, degen = W.localization_hit(cam, (0, 0, 7, 7), thr=0.5)
    assert hit == 0 and degen == 1


def test_threshold_selection_frozen_rule():
    # map with a clear 3x3 blob at top-left; GT box matches blob
    cam = np.zeros((7, 7)); cam[0:3, 0:3] = 1.0
    thr, best = W.select_threshold([cam], [(0, 0, 3, 3)])
    assert best == 1.0
    hit, _ = W.localization_hit(cam, (0, 0, 3, 3), thr)
    assert hit == 1
    # and a mismatched GT fails
    hit2, _ = W.localization_hit(cam, (4, 4, 7, 7), thr)
    assert hit2 == 0


def test_j_at_1_and_marginals():
    j = W.joint_j_at_1([1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0])
    assert j["J@1"] == 0.25
    assert j["marginal_top1"] == 0.75
    assert j["marginal_loc_query"] == 0.75
    assert j["marginal_loc_gallery"] == 0.75


def test_product_bootstrap_groups_whole_products():
    # two products with opposite values; grouped resampling can only produce
    # means from {all A, mixed, all B}; with per-product values constant the
    # bootstrap statistics must lie in {0, .5, 1} exactly.
    per_q = np.array([1.0, 1.0, 0.0, 0.0])
    pid = np.array([10, 10, 20, 20])
    out = W.product_bootstrap(per_q, pid, n_boot=500, seed=1)
    assert out["point"] == 0.5 and out["n_products"] == 2
    assert out["ci_lo"] in (0.0, 0.5) and out["ci_hi"] in (0.5, 1.0)


def test_box_from_map_largest_component():
    cam = np.zeros((8, 8)); cam[0, 0] = 1.0          # 1-px comp
    cam[4:7, 4:7] = 1.0                              # 3x3 comp (largest)
    b = W.box_from_map(cam, 0.5)
    assert b == (4, 4, 7, 7)
