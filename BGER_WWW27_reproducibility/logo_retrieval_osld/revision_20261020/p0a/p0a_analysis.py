#!/usr/bin/env python3
"""P0-A analysis (CPU): BGER readout at pooled F vs pre-gate H and vs the other arms (exploratory).
Reads p0a/out/seed*/{dev,test}_records.json + maps, the core test summaries, and the release
records (for the centre prior). No model inference. Contrast intervals: basic 95% from 10,000
shared query-logo-identity cluster draws (rng 0), seed-averaged, conditional on the trained
models and gallery, unadjusted; no p-values (post-hoc, descriptive).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
E1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(E1 / 'code')); sys.path.insert(0, str(E1 / 'analysis'))
GRID = np.linspace(.05, .95, 19)


def kidx(t): return int(np.argmin(np.abs(GRID - t)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--p0a', default=str(HERE / 'out')); ap.add_argument('--run-root', required=True)
    ap.add_argument('--records'); ap.add_argument('--out', default=str(HERE / 'recomputed' / 'P0A_RESULTS.json'))
    a = ap.parse_args()
    if not a.records:
        print('SKIPPED_REQUIRES_LICENSED_DATA: p0a full analysis includes centre-prior overlap; supply --records. Saved-hit contrasts are verified by source/figures_src/verify_figure_numbers.py.')
        return
    from e1_common import GRID, save  # noqa: E402
    from e1_evaluate import map_metrics  # noqa: E402
    from candidate_sensitivity import centre_prior, pair_stats  # noqa: E402
    import torch  # noqa: E402
    P = Path(a.p0a); run = Path(a.run_root)
    recs = {r['image_id']: r for r in json.loads(Path(a.records).read_text())['product_records']}
    core = {arm: [json.loads((run / 'test' / f'{arm}_s{s}' / 'summary.json').read_text()) for s in range(3)] for arm in ['ce', 'aux', 'bger', 'cam_bce', 'abn_box']}
    ids = core['ce'][0]['query_ids']; pids = np.array(core['ce'][0]['query_products']); n = len(ids); qi = {q: i for i, q in enumerate(ids)}
    seeds = {}; QF = []; QH = []; checks = dict(max_identity_residual=0.0, max_z_vs_cache=0.0, max_cH_vs_saved=0.0, max_D_vs_saved=0.0, QH_recomputed_equals_saved=True, dev_H_threshold_matches=True)
    for s in range(3):
        lock = json.loads((P / f'seed{s}' / 'THRESHOLD_LOCK.json').read_text()); kF = kidx(lock['threshold']); kH = kidx(lock['frozen_H_threshold'])
        checks['dev_H_threshold_matches'] &= bool(lock['H_threshold_matches_freeze'])
        rows = json.loads((P / f'seed{s}' / 'test_records.json').read_text())['rows']
        maps = np.load(P / f'seed{s}' / 'test_maps.npz')
        for r in rows:
            checks['max_identity_residual'] = max(checks['max_identity_residual'], r['identity_residual'] or 0)
            c = r['checks']; checks['max_z_vs_cache'] = max(checks['max_z_vs_cache'], c['z_vs_cache_max_abs'])
            checks['max_cH_vs_saved'] = max(checks['max_cH_vs_saved'], c.get('cH_vs_saved_max_abs', 0)); checks['max_D_vs_saved'] = max(checks['max_D_vs_saved'], c.get('D_vs_saved_max_abs', 0))
        own = {r['query_id']: r for r in rows if r['view'] == 'own_top1'}
        pos = {r['query_id']: r for r in rows if r['view'].endswith('_positive')}; neg = {r['query_id']: r for r in rows if r['view'].endswith('_negative')}
        R = np.array(core['bger'][s]['primary']['per_query']['R']); Qsaved = np.array(core['bger'][s]['primary']['per_query']['Q'])
        qF = np.array([own[q]['F']['hits'][kF][1] for q in ids]); qH = np.array([own[q]['H']['hits'][kH][1] for q in ids])
        checks['QH_recomputed_equals_saved'] &= bool(np.array_equal(qH, Qsaved))
        from e1_common import digest
        prior = centre_prior((7, 7)); pri = np.array([map_metrics(prior, recs[q])['hits'][kF][1] for q in ids])
        # fixed-candidate diagnostics for C_F
        l1 = []; corr = []; raw = []; lp = []; ln = []
        for q in ids:
            mp = maps[digest([q, pos[q]['gallery_id'], pos[q]['view']]) + '__cF']; mn = maps[digest([q, neg[q]['gallery_id'], neg[q]['view']]) + '__cF']
            ps = pair_stats(mp, mn); l1.append(ps['l1_distance']); corr.append(ps['correlation']); raw.append(abs(np.log(ps['raw_max_abs_pos'] / ps['raw_max_abs_neg'])))
            lp.append(pos[q]['F']['hits'][kF][1]); ln.append(neg[q]['F']['hits'][kF][1])
        lp, ln = np.array(lp), np.array(ln)
        seeds[s] = dict(threshold_F=lock['threshold'], dev_hit_F=lock['dev_hit_rate'], threshold_H=lock['frozen_H_threshold'],
                        test=dict(R1=float(R.mean()), Loc_F=float(qF.mean()), Loc_H=float(qH.mean()), Jq_F=float((R * qF).mean()), Jq_H=float((R * qH).mean()),
                                  EIB_F=float(np.mean([own[q]['F']['EIB'] for q in ids])), EIB_H=float(np.mean([own[q]['H']['EIB'] for q in ids])),
                                  degenerate_F=int(sum(own[q]['F']['degenerate'] for q in ids)), nonfinite_F=int(sum(own[q]['F']['nonfinite'] for q in ids)),
                                  centre_prior_at_F_threshold=float(pri.mean()),
                                  Loc_F_correct_top1=float(qF[R == 1].mean()), Loc_F_wrong_top1=float(qF[R == 0].mean())),
                        fixed_F=dict(loc_pos=float(lp.mean()), loc_neg=float(ln.mean()), verdict_change=float((lp != ln).mean()),
                                     mean_l1=float(np.nanmean(l1)), mean_corr=float(np.nanmean(corr)), corr_undefined=int(np.isnan(corr).sum()),
                                     median_abs_log_rawmax_ratio=float(np.nanmedian(raw))))
        QF.append(qF); QH.append(qH)
    QF, QH = np.array(QF), np.array(QH)
    # contrasts (seed-averaged per-query differences; shared cluster draws)
    unique, inv = np.unique(pids, return_inverse=True); counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(0); draws = rng.integers(0, len(unique), size=(10000, len(unique)))
    def ci(diff):  # diff: (3, n)
        d = diff.mean(0); sums = np.bincount(inv, weights=d, minlength=len(unique)); rep = sums[draws].sum(1) / counts[draws].sum(1); pt = float(d.mean())
        return dict(point=pt, basic_ci=[float(2 * pt - np.quantile(rep, .975)), float(2 * pt - np.quantile(rep, .025))], per_seed=diff.mean(1).tolist(), exploratory=True)
    contrasts = {'BGER_F-BGER_H/Loc': ci(QF - QH)}
    for arm in ['ce', 'aux', 'cam_bce', 'abn_box']:
        Qa = np.array([core[arm][s]['primary']['per_query']['Q'] for s in range(3)]); contrasts[f'BGER_F-{arm}/Loc'] = ci(QF - Qa)
    mean = lambda k: dict(mean=float(np.mean([seeds[s]['test'][k] for s in range(3)])), sd=float(np.std([seeds[s]['test'][k] for s in range(3)], ddof=1)))
    out = dict(identity='OSLD_P0A_EXPLORATORY', status='post-hoc exploratory; original H2-logo (pre-gate H) remains failed; thresholds for C_F locked on official val before test',
               checks=checks, seeds=seeds, summary={k: mean(k) for k in ['R1', 'Loc_F', 'Loc_H', 'Jq_F', 'Jq_H', 'EIB_F', 'EIB_H', 'centre_prior_at_F_threshold', 'Loc_F_correct_top1', 'Loc_F_wrong_top1']},
               fixed_F_mean={k: float(np.mean([seeds[s]['fixed_F'][k] for s in range(3)])) for k in seeds[0]['fixed_F']},
               contrasts=contrasts, n_queries=n, n_clusters=len(unique), n_boot=10000)
    save(a.out, out); print(json.dumps(dict(checks=checks, summary=out['summary'], contrasts={k: (round(v['point'], 4), [round(x, 4) for x in v['basic_ci']]) for k, v in contrasts.items()}), indent=1))


if __name__ == '__main__':
    main()
