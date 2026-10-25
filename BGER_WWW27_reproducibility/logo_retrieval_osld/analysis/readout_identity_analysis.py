#!/usr/bin/env python3
"""Zero-new-inference readout analysis on SAVED core outputs (CPU; post hoc, descriptive).

1. Source correspondence (structural, random-init modules, no OSLD data): which tensor
   each arm pools into z (F), and whether the saved primary readout is taken at F.
2. Identity at the actual pooled input F (Stylianou-style decomposition D and the
   cosine Grad-CAM C at F):  C_F = D - s*rho*T,  T_u = F_u^T z / (U |z|^2),
   rho = |z||d| / (|z||d| + eps).  For the four arms whose saved primary IS taken at F
   (CE, Aux-BCE, CAM-BCE, ABN-box) T is recovered from every saved candidate view
   (own top-1, d+, d-) as (D - C)/(s*rho); agreement across views checks the identity on
   the saved maps; the norm share |C|/|D| and the shape agreement of D with T are
   reported. BGER's saved primary is at the pre-gate H, not F: the identity is NOT
   applied to it, and C_F of BGER cannot be computed from the caches (F is not saved).
3. A5 split: change measures (score, raw magnitude, normalized shape) vs localization
   measures, with the arithmetic bounds that hit rates place on the verdict-change rate.
4. Centre prior: at each arm's locked threshold (as in the paper) and, as a model-free
   reference, at its own threshold chosen on the official validation queries with the
   same 19-point rule (needs the release records for boxes; optional).
Usage: readout_identity_analysis.py --core <results/core> --a5 <candidate_sensitivity_A5.json>
       --out OUT.json [--records osld_records.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))
ARMS = ['ce', 'aux', 'bger', 'cam_bce', 'abn_box']
EPS = 1e-12
GRID = np.linspace(.05, .95, 19)


def structural_check():
    import torch
    from train_web_arms import build_model
    torch.manual_seed(0); out = {}
    x = torch.randn(1, 3, 64, 64)
    for arm in ARMS:
        m = build_model(5, arm, weights=None).eval()
        with torch.no_grad():
            H = m.features(x); F = m.pooled_features(H); _, _, z = m(x)
        out[arm] = dict(primary_readout_tensor='features(x) output (pre-gate H for BGER)',
                        pooled_input_equals_readout_tensor=bool(torch.equal(F, H)),
                        z_equals_gap_of_pooled_input=bool(torch.allclose(z, F.mean((2, 3)), atol=1e-6)))
    return out


def load_views(d):
    own = json.loads((d / 'own_top1.json').read_text())['pairs']
    fixed = json.loads((d / 'fixed_pairs.json').read_text())['pairs']
    z = np.load(d / 'query_signed_maps.npz'); ix = {str(n): i for i, n in enumerate(z['names'])}
    C = z['query_primary_signed'].astype(np.float64); D = z['query_secondary_signed'].astype(np.float64)
    views = {}
    for name, pairs in [('own', own), ('pos', [p for p in fixed if p['view'].endswith('_positive')]),
                        ('neg', [p for p in fixed if p['view'].endswith('_negative')])]:
        idx = [ix[p['signed_maps_file'][:-4]] for p in pairs]
        dg = [p['query']['diagnostic'] for p in pairs]
        s = np.array([g['score'] for g in dg]); a = np.array([g['query_norm'] for g in dg]); b = np.array([g['other_norm'] for g in dg])
        views[name] = dict(q=[p['query_id'] for p in pairs], C=C[idx].reshape(len(idx), -1), D=D[idx].reshape(len(idx), -1),
                           s=s, rho=a * b / (a * b + EPS))
    return views


def identity_analysis(core):
    rows = []
    for arm in ARMS:
        if arm == 'bger': continue
        for seed in range(3):
            v = load_views(core / 'test' / f'{arm}_s{seed}')
            order = {q: i for i, q in enumerate(v['own']['q'])}
            T = {}
            for name in ['own', 'pos', 'neg']:
                w = v[name]; perm = np.array([order[q] for q in w['q']])
                coef = w['s'] * w['rho']
                if np.any(np.abs(coef) < 1e-8): raise ValueError('score too close to zero for recovery')
                T[name] = np.empty_like(w['D']); T[name][perm] = (w['D'] - w['C']) / coef[:, None]
            dmax = max(float(np.abs(T['own'] - T['pos']).max()), float(np.abs(T['own'] - T['neg']).max()), float(np.abs(T['pos'] - T['neg']).max()))
            Tm = (T['own'] + T['pos'] + T['neg']) / 3
            share, cosDT, sumT = [], [], []
            for name in ['own', 'pos', 'neg']:
                w = v[name]; perm = np.array([order[q] for q in w['q']]); t = Tm[perm]
                share.append(np.linalg.norm(w['C'], axis=1) / np.linalg.norm(w['D'], axis=1))
                cosDT.append(np.sum(w['D'] * t, 1) / (np.linalg.norm(w['D'], axis=1) * np.linalg.norm(t, axis=1)))
            sumT = Tm.sum(1)
            rows.append(dict(arm=arm, seed=seed, n_queries=len(order), views=3,
                             max_abs_diff_recovered_T_across_views=dmax,
                             recovered_T_sum_median=float(np.median(sumT)), recovered_T_sum_maxdev_from_1=float(np.abs(sumT - 1).max()),
                             median_norm_ratio_C_over_D=float(np.median(np.concatenate(share))),
                             p90_norm_ratio_C_over_D=float(np.quantile(np.concatenate(share), .9)),
                             median_cosine_D_vs_T=float(np.median(np.concatenate(cosDT)))))
    return rows


def a5_split(a5):
    out = {}
    for arm in ARMS:
        acc = {ro: {} for ro in ['primary', 'secondary']}
        for seed in range(3):
            rows = a5['runs'][f'{arm}_s{seed}']['rows']
            ds = np.array([r['score_change'] for r in rows])
            for ro in ['primary', 'secondary']:
                mp = np.array([r[f'{ro}_raw_max_abs_pos'] for r in rows]); mn = np.array([r[f'{ro}_raw_max_abs_neg'] for r in rows])
                sp = np.array([r[f'{ro}_raw_pos_sum_pos'] for r in rows]); sn = np.array([r[f'{ro}_raw_pos_sum_neg'] for r in rows])
                lp = np.array([r[f'{ro}_loc_pos'] for r in rows]); ln = np.array([r[f'{ro}_loc_neg'] for r in rows])
                l1 = np.array([r[f'{ro}_l1_distance'] for r in rows], dtype=float); cr = np.array([r[f'{ro}_correlation'] for r in rows], dtype=float)
                vals = dict(median_abs_score_change=float(np.median(np.abs(ds))),
                            median_abs_log_rawmax_ratio=float(np.median(np.abs(np.log(mp / mn)))),
                            median_abs_log_possum_ratio=float(np.nanmedian(np.abs(np.log(sp / sn)))),
                            mean_l1=float(np.nanmean(l1)), mean_corr=float(np.nanmean(cr)), corr_undefined=int(np.isnan(cr).sum()),
                            loc_pos=float(lp.mean()), loc_neg=float(ln.mean()), verdict_change=float((lp != ln).mean()),
                            verdict_change_lower_bound=float(abs(lp.mean() - ln.mean())), verdict_change_upper_bound=float(min(1, lp.mean() + ln.mean())),
                            centre_prior_at_arm_threshold=float(np.mean([r[f'{ro}_centre_prior_loc'] for r in rows])))
                for k, x in vals.items(): acc[ro].setdefault(k, []).append(x)
        out[arm] = {ro: {k: (float(np.mean(x)) if k != 'corr_undefined' else int(np.sum(x))) for k, x in d.items()} for ro, d in acc.items()}
    return out


def prior_own_threshold(records_path, core):
    from candidate_sensitivity import centre_prior
    from e1_evaluate import map_metrics
    recs = json.loads(Path(records_path).read_text())
    prod = {r['image_id']: r for r in recs['product_records']}
    val = sorted(r['image_id'] for r in recs['product_records'] if r['split'] == 'val')
    test = json.loads((core / 'test' / 'ce_s0' / 'summary.json').read_text())['query_ids']
    prior = centre_prior((7, 7))
    hv = np.array([np.array(map_metrics(prior, prod[q])['hits'])[:, 1] for q in val]).mean(0)
    k = int(hv.argmax())
    ht = np.array([np.array(map_metrics(prior, prod[q])['hits'])[:, 1] for q in test])
    return dict(val_hit_rate_by_threshold=hv.tolist(), val_selected_threshold=float(GRID[k]), val_hit_rate=float(hv[k]),
                test_loc_at_val_threshold=float(ht[:, k].mean()), test_loc_by_threshold=ht.mean(0).tolist(),
                n_val=len(val), n_test=len(test), rule='same 19-point grid and IoU>=.5 hit-rate rule as the arms; chosen on official val only')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--core', required=True); ap.add_argument('--a5', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--records')
    a = ap.parse_args(); core = Path(a.core)
    res = dict(scope='post hoc, descriptive; saved maps/records only; no OSLD model inference; no change to any primary endpoint',
               structural=structural_check(), identity_at_F=identity_analysis(core),
               bger_note='BGER primary is Grad-CAM at pre-gate H; C_F for BGER needs F, which is not cached (only z embeddings and 7x7 signed maps are saved)',
               a5_split=a5_split(json.loads(Path(a.a5).read_text())))
    if a.records: res['centre_prior_own_threshold'] = prior_own_threshold(a.records, core)
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: res[k] for k in ['structural', 'identity_at_F']}, indent=1)[:4000])
    if 'centre_prior_own_threshold' in res: print({k: v for k, v in res['centre_prior_own_threshold'].items() if 'by_threshold' not in k})


if __name__ == '__main__':
    main()
