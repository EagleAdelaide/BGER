"""Round-2 cosine-readout contract tests for OUR web_eval implementation.
These mirror the external package's 12 synthetic checks but run against
paper_audits/www2027/code/web_eval.py — passing here is a statement about
OUR code, not about the external script and not about any dataset.
All float64, fixed global tolerances (never per-method)."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import web_eval as W

TOL = 1e-10          # analytic identities (float64)
TOL_FD = 1e-6        # finite differences
torch.manual_seed(0)
C, Hh, Ww = 16, 7, 7
U = Hh * Ww


def rand_feat():
    return torch.randn(C, Hh, Ww, dtype=torch.float64)


def gap(f):
    return f.mean(dim=(1, 2))


def analytic_r(z, d):
    s = (z @ d) / (z.norm() * d.norm())
    return d / (z.norm() * d.norm()) - s * z / (z.norm() ** 2), s


def test_actual_cosine_gradient_and_orthogonality():
    for _ in range(16):
        feat = rand_feat(); d = torch.randn(C, dtype=torch.float64)
        f = feat.clone().requires_grad_(True)
        z = gap(f)
        s = (z * d).sum() / (z.norm() * d.norm())
        g = torch.autograd.grad(s, f)[0]
        alpha = g.mean(dim=(1, 2))
        with torch.no_grad():
            r, s0 = analytic_r(gap(feat), d)
        assert float((alpha - r / U).abs().max()) < TOL          # alpha = r/U
        assert abs(float(r @ gap(feat))) < TOL                   # r.z = 0
        n2 = (1 - float(s0) ** 2) / float(gap(feat).norm() ** 2)
        assert abs(float(r.norm() ** 2) - n2) < TOL              # ||r||^2 identity


def test_ungated_signed_gradcam_sums_to_zero():
    feat = rand_feat(); d = torch.randn(C, dtype=torch.float64)
    signed, _, rec = W.retrieval_gradcam_audited(feat, gap, d)
    assert abs(float(signed.sum())) < TOL                        # r.z = 0 property
    assert not rec["nonfinite"]


def test_gated_alpha_identity():
    v = torch.randn(C, dtype=torch.float64); b = torch.randn((), dtype=torch.float64)
    def gated_pool(f):
        M = torch.sigmoid(torch.einsum("chw,c->hw", f, v) + b)
        return (M.unsqueeze(0) * f).mean(dim=(1, 2))
    for _ in range(8):
        feat = rand_feat(); d = torch.randn(C, dtype=torch.float64)
        f = feat.clone().requires_grad_(True)
        z = gated_pool(f)
        s = (z * d).sum() / (z.norm() * d.norm())
        alpha = torch.autograd.grad(s, f)[0].mean(dim=(1, 2))
        with torch.no_grad():
            M = torch.sigmoid(torch.einsum("chw,c->hw", feat, v) + b).reshape(-1)
            Hu = feat.reshape(C, -1).T
            r, _ = analytic_r(gated_pool(feat), d)
            mu = M.sum() / U ** 2
            beta = (M * (1 - M) * (Hu @ r)).sum() / U ** 2
            ref = mu * r + beta * v
        assert float((alpha - ref).abs().max()) < TOL


def test_norm_detach_negative_control():
    # forward score identical; gradient must differ when own norm is detached
    feat = rand_feat(); d = torch.randn(C, dtype=torch.float64)
    f1 = feat.clone().requires_grad_(True)
    z1 = gap(f1); s1 = (z1 * d).sum() / (z1.norm() * d.norm())
    g1 = torch.autograd.grad(s1, f1)[0]
    f2 = feat.clone().requires_grad_(True)
    z2 = gap(f2); s2 = (z2 * d).sum() / (z2.norm().detach() * d.norm())
    g2 = torch.autograd.grad(s2, f2)[0]
    assert abs(float(s1 - s2)) < TOL                              # same forward
    assert float((g1 - g2).abs().max()) > 1e-3                    # different grad


def test_similarity_one_gives_zero_gradient_and_degenerate_flag():
    feat = rand_feat()
    d = gap(feat).clone()                                         # exactly collinear
    signed, pos, rec = W.retrieval_gradcam_audited(feat, gap, d)
    assert abs(rec["score"] - 1.0) < 1e-9
    assert rec["grad_norm"] < 1e-9                                # stationary point
    assert rec["degenerate_map"]                                  # flagged, not imputed


def test_norm_floor_flag_near_zero_embedding():
    feat = rand_feat() * 1e-9
    d = torch.randn(C, dtype=torch.float64)
    _, _, rec = W.retrieval_gradcam_audited(feat, gap, d)
    assert rec["norm_floor_hit"]


def test_decomposition_sums_to_score_both_sides():
    for _ in range(8):
        Fq, Fd = rand_feat(), rand_feat()
        zq, zd = gap(Fq), gap(Fd)
        Dq, sq = W.score_decomposition(Fq, zd)
        Dd, sd = W.score_decomposition(Fd, zq)
        s = float((zq @ zd) / (zq.norm() * zd.norm()))
        assert abs(float(Dq.sum()) - s) < TOL and abs(sq - s) < TOL
        assert abs(float(Dd.sum()) - s) < TOL and abs(sd - s) < TOL   # symmetry


def test_finite_difference_matches_autograd():
    feat = rand_feat(); d = torch.randn(C, dtype=torch.float64)
    f = feat.clone().requires_grad_(True)
    z = gap(f); s = (z * d).sum() / (z.norm() * d.norm())
    g = torch.autograd.grad(s, f)[0]
    eps = 1e-6
    for (c, i, j) in [(0, 0, 0), (3, 2, 5), (15, 6, 6)]:
        fp = feat.clone(); fp[c, i, j] += eps
        fm = feat.clone(); fm[c, i, j] -= eps
        def sc(x):
            zz = gap(x); return float((zz * d).sum() / (zz.norm() * d.norm()))
        fd = (sc(fp) - sc(fm)) / (2 * eps)
        assert abs(fd - float(g[c, i, j])) < TOL_FD


def test_candidate_swap_changes_gradient():
    feat = rand_feat()
    d1 = torch.randn(C, dtype=torch.float64); d2 = torch.randn(C, dtype=torch.float64)
    s1, _, r1 = W.retrieval_gradcam_audited(feat, gap, d1)
    s2, _, r2 = W.retrieval_gradcam_audited(feat, gap, d2)
    assert float((s1 - s2).abs().max()) > 1e-6                    # synthetic: must differ
    # (on real data, candidate-insensitivity is a FINDING, not auto-bug)


def test_j_at_1_leq_min_marginal():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = rng.integers(0, 2, 50); q = rng.integers(0, 2, 50); g = rng.integers(0, 2, 50)
        j = W.joint_j_at_1(t, q, g)
        assert j["J@1"] <= min(j["marginal_top1"], j["marginal_loc_query"],
                               j["marginal_loc_gallery"]) + 1e-12


def test_shared_bootstrap_indices_across_methods():
    # identical series under two keys must give IDENTICAL intervals
    pid = np.repeat(np.arange(10), 3)
    x = np.random.default_rng(1).normal(size=30)
    out = W.product_bootstrap_shared({"m1": x, "m2": x.copy()}, pid,
                                     n_boot=300, seed=7)
    assert out["m1"] == out["m2"]
    # and a paired difference of identical series is exactly zero everywhere
    outd = W.product_bootstrap_shared({"d": x - x}, pid, n_boot=100, seed=7)
    assert outd["d"]["ci_lo"] == 0.0 and outd["d"]["ci_hi"] == 0.0


def test_nonfinite_map_flagged():
    feat = rand_feat(); feat[0, 0, 0] = torch.nan
    d = torch.randn(C, dtype=torch.float64)
    _, _, rec = W.retrieval_gradcam_audited(feat, gap, d)
    assert rec["nonfinite"]
