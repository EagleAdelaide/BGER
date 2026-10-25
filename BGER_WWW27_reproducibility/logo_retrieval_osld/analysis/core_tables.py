#!/usr/bin/env python3
"""Result tables from a finished OSLD core run (CPU; reads saved outputs only).

No model inference and no recomputation of the frozen statistics: per-seed values
come from test/<arm>_s<seed>/summary.json, the confirmatory family from
primary_inference.json, descriptive paired contrasts from descriptive_inference.json
(shared 10,000 query-logo-identity cluster draws), the development grid from
all_dev_candidates.json, and the A5 diagnostic from candidate_sensitivity output.
Usage: core_tables.py --run-root RUN --a5 A5.json --out DIR
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

ARM_ORDER = ['ce', 'aux', 'bger', 'cam_bce', 'abn_box']
NAMES = dict(ce='CE', aux='Aux-BCE', bger='BGER', cam_bce='CAM-BCE', abn_box='ABN-box')


def per_seed_row(arm, seed, s):
    p, q = s['primary'], s['secondary']; n = s['n_queries']
    return dict(arm=arm, seed=seed, R1=s['recall1'], R10=s['recall10'], mAP=s['mAP'],
                Loc_q=p['marginal_loc_query'], Jq1=p['Jq_at_1'], EIB_q=p['EIB_query'],
                degenerate_rate=p['degenerate_query'] / n, nonfinite=p['nonfinite_query'],
                threshold_primary=p['threshold'],
                Loc_q_secondary=q['marginal_loc_query'], Jq1_secondary=q['Jq_at_1'], EIB_q_secondary=q['EIB_query'],
                degenerate_rate_secondary=q['degenerate_query'] / n, threshold_secondary=q['threshold'],
                fixed_pos_Loc_q=s['fixed_candidate']['positive']['primary']['marginal_loc_query'],
                fixed_neg_Loc_q=s['fixed_candidate']['negative']['primary']['marginal_loc_query'],
                random_R1_reference=s['random_R1_reference'], n_queries=n, gallery_size=s['gallery_size'],
                max_score_error=s['max_ranking_readout_error'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-root', required=True)
    ap.add_argument('--a5', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); root = Path(a.run_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((root / 'pretest_freeze.json').read_text())
    arms = [x for x in ARM_ORDER if x in freeze['arms_tested']]
    rows = []
    for arm in arms:
        for seed in range(3):
            rows.append(per_seed_row(arm, seed, json.loads((root / 'test' / f'{arm}_s{seed}' / 'summary.json').read_text())))
    metrics = [k for k in rows[0] if k not in ('arm', 'seed', 'n_queries', 'gallery_size')]
    agg = {}
    for arm in arms:
        rr = [r for r in rows if r['arm'] == arm]
        agg[arm] = {m: dict(mean=float(np.mean([r[m] for r in rr])), sd=float(np.std([r[m] for r in rr], ddof=1)),
                            per_seed=[r[m] for r in rr]) for m in metrics}
    prim = json.loads((root / 'primary_inference.json').read_text())
    desc = json.loads((root / 'descriptive_inference.json').read_text())
    keymap = dict(R1='R1', R10='R10', mAP='mAP', Loc_q='primary/Q', Jq1='primary/J', EIB_q='primary/query/EIB',
                  Loc_q_secondary='secondary/Q', Jq1_secondary='secondary/J')
    contrasts = []
    for other in ['ce', 'aux', 'cam_bce', 'abn_box']:
        if other not in arms: continue
        for m, k in keymap.items():
            c = desc['paired_contrasts'].get(f'bger-{other}/{k}')
            if c is None: continue
            per_seed = [agg['bger'][m]['per_seed'][s] - agg[other][m]['per_seed'][s] for s in range(3)]
            contrasts.append(dict(contrast=f'BGER-{NAMES[other]}', metric=m, point=c['point'], basic_ci_low=c['basic_ci'][0],
                                  basic_ci_high=c['basic_ci'][1], per_seed=per_seed,
                                  same_sign_all_seeds=bool(all(x > 0 for x in per_seed) or all(x < 0 for x in per_seed)),
                                  role='descriptive'))
    confirm = {k: {kk: v for kk, v in c.items()} for k, c in prim['contrasts'].items()}
    dev = json.loads((root / 'all_dev_candidates.json').read_text())['candidates']
    dev_rows = [dict(arm=c['arm'], seed=c['seed'], lam=c['lambda_value'], R1=c['summary']['recall1'], R10=c['summary']['recall10'],
                     mAP=c['summary']['mAP'], Loc_q=c['summary']['primary']['marginal_loc_query'],
                     Jq1=c['summary']['primary']['Jq_at_1'], threshold_primary=c['summary']['thresholds']['primary'],
                     threshold_secondary=c['summary']['thresholds']['secondary'],
                     selected=bool(freeze['selected_lambdas'].get(c['arm']) == c['lambda_value'] or c['arm'] == 'ce'))
                for c in dev]
    a5 = json.loads(Path(a.a5).read_text())
    a5_rows = []
    for arm in arms:
        for ro in ['primary', 'secondary']:
            runs = [a5['runs'][f'{arm}_s{s}']['aggregate'] for s in range(3)]
            g = lambda k: [r[ro][k] for r in runs]
            a5_rows.append(dict(arm=arm, readout=ro, mean_score_change=float(np.mean([r['mean_score_change'] for r in runs])),
                                verdict_change_rate=float(np.mean(g('verdict_change_rate'))),
                                mean_l1_distance=float(np.mean([x for x in g('mean_l1_distance') if x is not None])),
                                mean_correlation_defined=float(np.mean([x for x in g('mean_correlation_defined') if x is not None])),
                                correlation_undefined_total=int(sum(g('correlation_undefined_count'))),
                                degenerate_rate_pos=float(np.mean(g('degenerate_rate_pos'))),
                                degenerate_rate_neg=float(np.mean(g('degenerate_rate_neg'))),
                                median_raw_max_abs_pos=float(np.median(g('median_raw_max_abs_pos'))),
                                loc_rate_pos=float(np.mean(g('loc_rate_pos'))), loc_rate_neg=float(np.mean(g('loc_rate_neg'))),
                                centre_prior_loc_rate=float(np.mean(g('centre_prior_loc_rate'))),
                                n_queries_per_seed=runs[0]['n_queries']))
    result = dict(identity=prim.get('identity'), scientific_inference=prim.get('scientific_inference'),
                  arms_tested=arms, incomplete_arms=freeze.get('incomplete_arms', {}), selected_lambdas=freeze['selected_lambdas'],
                  per_seed=rows, mean_sd=agg, confirmatory=dict(contrasts=confirm, all_family_rejected=prim['all_family_rejected'],
                  n_queries=prim['n_queries'], n_clusters=prim['n_products'], n_boot=prim['n_boot'], method=prim['method'],
                  intervals=prim.get('intervals')), descriptive_contrasts=contrasts, dev_grid=dev_rows, a5=a5_rows,
                  a5_scientific_result=a5.get('scientific_result'))
    (out / 'core_results.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    for name, data in [('per_seed.csv', rows), ('descriptive_contrasts.csv', contrasts), ('dev_grid.csv', dev_rows), ('a5_candidate_sensitivity.csv', a5_rows)]:
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(json.dumps(dict(arms=arms, confirmatory={k: dict(point=v['point'], ci=v['basic_two_sided_ci'], p_holm=v['p_holm'], reject=v['reject_holm'])
                                                     for k, v in confirm.items()}), indent=1))


if __name__ == '__main__':
    main()
