#!/usr/bin/env python3
"""Descriptive analyses fixed in analysis/DISPLAY_RULES.md (CPU; saved outputs only).

1. small-logo boundary: test-query box-area-fraction quartiles (annotation-only);
2. query Loc for correct vs wrong own top-1;
4. numeric map examples: seed-0 own-top1 primary query-side signed maps of every
   tested arm for queries chosen by the pre-stated rule (no photographs).
Usage: descriptive_extra.py --run-root RUN --out OUT.json
"""
import argparse, json
from pathlib import Path
import numpy as np

ARM_ORDER = ['ce', 'aux', 'bger', 'cam_bce', 'abn_box']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-root', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); root = Path(a.run_root)
    freeze = json.loads((root / 'pretest_freeze.json').read_text())
    arms = [x for x in ARM_ORDER if x in freeze['arms_tested']]
    recs = {r['image_id']: r for r in json.loads((root / 'technical_integrity.json').read_text())['image_records']}
    S = {(arm, s): json.loads((root / 'test' / f'{arm}_s{s}' / 'summary.json').read_text()) for arm in arms for s in range(3)}
    ids = S[arms[0], 0]['query_ids']
    def frac(q):
        r = recs[q]; b = r['box_xyxy_half_open']
        return (b[2] - b[0]) * (b[3] - b[1]) / (r['width'] * r['height'])
    area = np.array([frac(q) for q in ids])
    edges = np.quantile(area, [0, .25, .5, .75, 1])
    bins = np.clip(np.searchsorted(edges[1:-1], area, side='right'), 0, 3)
    out = dict(identity=S[arms[0], 0]['identity'], rule='analysis/DISPLAY_RULES.md', n_queries=len(ids),
               area_quartile_edges=edges.tolist(), bin_counts=np.bincount(bins, minlength=4).tolist(), small_logo={}, by_top1={})
    for arm in arms:
        rows = []
        for b in range(4):
            m = bins == b; per = []
            if not m.any():
                rows.append(dict(bin=b, n=0, R=None, Q=None, J=None, per_seed=[])); continue
            for s in range(3):
                pq = S[arm, s]['primary']['per_query']
                per.append(dict(R=float(np.mean(np.array(pq['R'])[m])), Q=float(np.mean(np.array(pq['Q'])[m])), J=float(np.mean(np.array(pq['J'])[m]))))
            rows.append(dict(bin=b, n=int(m.sum()), **{k: float(np.mean([p[k] for p in per])) for k in 'RQJ'}, per_seed=per))
        out['small_logo'][arm] = rows
        cor, wro = [], []
        for s in range(3):
            pq = S[arm, s]['primary']['per_query']; R = np.array(pq['R']); Q = np.array(pq['Q'])
            cor.append(dict(n=int(R.sum()), Q=float(Q[R == 1].mean()) if R.sum() else None))
            wro.append(dict(n=int((1 - R).sum()), Q=float(Q[R == 0].mean()) if (1 - R).sum() else None))
        mean = lambda xs: float(np.mean([x['Q'] for x in xs if x['Q'] is not None])) if any(x['Q'] is not None for x in xs) else None
        out['by_top1'][arm] = dict(correct=dict(Q=mean(cor), per_seed=cor), wrong=dict(Q=mean(wro), per_seed=wro))
    # 4. examples (rule fixed before results): CE seed-0 correctness defines the strata
    ce0 = np.array(S['ce', 0]['primary']['per_query']['R'])
    def pick(mask, qs):
        idx = np.flatnonzero(mask)
        order = idx[np.lexsort((np.array(ids)[idx], area[idx]))]
        return [int(order[min(len(order) - 1, max(0, int(np.ceil(q * len(order))) - 1))]) for q in qs] if len(order) else []
    chosen = [('correct_p25', i) for i in pick(ce0 == 1, [.25])] + [('correct_p50', i) for i in pick(ce0 == 1, [.5])] + \
             [('correct_p75', i) for i in pick(ce0 == 1, [.75])] + [('wrong_p50', i) for i in pick(ce0 == 0, [.5])]
    ex = []
    for label, i in chosen:
        q = ids[i]; r = recs[q]; item = dict(label=label, query_id=q, box_area_fraction=float(area[i]),
                                             box=r['box_xyxy_half_open'], width=r['width'], height=r['height'], arms={})
        for arm in arms:
            pairs = json.loads((root / 'test' / f'{arm}_s0' / 'own_top1.json').read_text())['pairs']
            p = next(p for p in pairs if p['query_id'] == q)
            m = np.load(root / 'test' / f'{arm}_s0' / 'signed_maps' / p['signed_maps_file'])['query_primary_signed']
            thr = S[arm, 0]['thresholds']['primary']; k = int(np.argmin(abs(np.linspace(.05, .95, 19) - thr)))
            item['arms'][arm] = dict(map=np.asarray(m).tolist(), top1_relevant=int(S[arm, 0]['primary']['per_query']['R'][i]),
                                     localized=int(p['query']['primary']['hits'][k][1]), score=p['score'],
                                     top1_template_identity_matches=bool(p['gallery_id'].split('/')[-1].rsplit('-', 1)[0] == r['product_id']))
        ex.append(item)
    out['examples'] = ex
    Path(a.out).write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print(json.dumps(dict(edges=out['area_quartile_edges'], bins=out['bin_counts'],
                          small_logo={k: [None if r['Q'] is None else round(r['Q'], 3) for r in v] for k, v in out['small_logo'].items()},
                          examples=[(e['label'], e['query_id']) for e in ex]), indent=1))


if __name__ == '__main__':
    main()
