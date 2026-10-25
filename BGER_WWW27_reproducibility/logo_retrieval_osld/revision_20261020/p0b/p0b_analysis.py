#!/usr/bin/env python3
"""P0-B analysis (CPU): gate_only (lambda=0) and shifted_box controls vs CE / BGER / Aux-BCE (exploratory).
Per-seed test values; seed-averaged per-query contrasts with basic 95% intervals from 10,000 shared
query-identity cluster draws (rng 0); conditional on trained models and gallery; no p-values."""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--core-root', '--run-root', dest='core_root', type=Path, default=HERE.parents[1] / 'results/core')
ap.add_argument('--out', type=Path, default=HERE / 'recomputed' / 'P0B_RESULTS.json')
args = ap.parse_args(); RUN = args.core_root
def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
S = {a: [json.loads((RUN / 'test' / f'{a}_s{s}' / 'summary.json').read_text()) for s in range(3)] for a in ['ce', 'aux', 'bger', 'cam_bce', 'abn_box']}
S['gate_only'] = [json.loads((HERE / 'out' / 'test' / f'gate_only_l0_s{s}' / 'summary.json').read_text()) for s in range(3)]
S['shifted_box'] = [json.loads((HERE / 'out' / 'test' / f'shifted_box_l0.5_s{s}' / 'summary.json').read_text()) for s in range(3)]
ids = S['ce'][0]['query_ids']; pids = np.array(S['ce'][0]['query_products'])
for a, v in S.items():
    for x in v: assert x['query_ids'] == ids and x['query_products'] == list(pids), a
unique, inv = np.unique(pids, return_inverse=True); counts = np.bincount(inv).astype(float)
rng = np.random.default_rng(0); draws = rng.integers(0, len(unique), size=(10000, len(unique)))
def arr(a, k): return np.array([S[a][s]['primary']['per_query'][k] for s in range(3)], dtype=float)
def ci(diff):
    d = diff.mean(0); sums = np.bincount(inv, weights=d, minlength=len(unique)); rep = sums[draws].sum(1) / counts[draws].sum(1); pt = float(d.mean())
    return dict(point=pt, basic_ci=[float(2 * pt - np.quantile(rep, .975)), float(2 * pt - np.quantile(rep, .025))], per_seed=diff.mean(1).tolist(), exploratory=True)
rec = json.loads((HERE / 'out' / 'EXPERIMENT_RECORD.json').read_text())
per_seed = {a: [dict(R1=x['recall1'], R10=x['recall10'], mAP=x['mAP'], Loc=x['primary']['marginal_loc_query'], Jq=x['primary']['Jq_at_1'], EIB=x['primary']['EIB_query'],
                      degenerate=x['primary']['degenerate_query'], threshold=x['thresholds']['primary']) for x in S[a]] for a in ['gate_only', 'shifted_box']}
mean_sd = {a: {k: dict(mean=float(np.mean([r[k] for r in v])), sd=float(np.std([r[k] for r in v], ddof=1))) for k in ['R1', 'R10', 'mAP', 'Loc', 'Jq', 'EIB']} for a, v in per_seed.items()}
contrasts = {}
for x, y in [('gate_only', 'ce'), ('bger', 'gate_only'), ('shifted_box', 'ce'), ('bger', 'shifted_box'), ('gate_only', 'aux'), ('shifted_box', 'gate_only')]:
    contrasts[f'{x}-{y}/R1'] = ci(arr(x, 'R') - arr(y, 'R')); contrasts[f'{x}-{y}/Loc'] = ci(arr(x, 'Q') - arr(y, 'Q'))
out = dict(identity='OSLD_P0B_EXPLORATORY', per_seed=per_seed, mean_sd=mean_sd, contrasts=contrasts, shifted_box_stats=json.loads((HERE / 'out' / 'shifted_box_stats.json').read_text()),
           runs={k: dict(checkpoint_sha256=v['checkpoint_sha256'], template_sha256=v['template_sha256'], thresholds=v['thresholds'], wall_seconds=v['wall_seconds']) for k, v in rec['runs'].items()},
           n_queries=len(ids), n_clusters=len(unique), n_boot=10000)
save(args.out, out)
print(json.dumps(dict(mean_sd=mean_sd, contrasts={k: (round(v['point'] * 100, 2), [round(c * 100, 2) for c in v['basic_ci']]) for k, v in contrasts.items()}), indent=1))
