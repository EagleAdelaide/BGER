#!/usr/bin/env python3
"""P0-C audit (CPU): evaluation consistency of the frozen references and size-stratified R@1; writes P0C_RESULTS.json."""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--core-root', '--run-root', dest='core_root', type=Path, default=HERE.parents[1] / 'results/core')
ap.add_argument('--out', type=Path, default=HERE / 'recomputed' / 'P0C_RESULTS.json')
ap.add_argument('--records', type=Path)
args = ap.parse_args(); RUN = args.core_root
def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')
rec = json.loads((HERE / 'out' / 'EXPERIMENT_RECORD.json').read_text())
ce = json.loads((RUN / 'test' / 'ce_s0' / 'summary.json').read_text()); ids = ce['query_ids']; pids = np.array(ce['query_products'])
core_rank = json.loads((RUN / 'test' / 'ce_s0' / 'ranking.json').read_text())['rows']
# Gallery identity is recorded in the shipped summary; annotation-derived strata need licensed records.
bins = edges = None
gal_ids = []
if args.records:
    raw = json.loads(args.records.read_text())
    recs = {r['image_id']: r for r in raw['product_records'] + raw['logo_records']}
    gal_ids = sorted(r['image_id'] for r in raw['logo_records'])
    area = np.array([(lambda b,r:(b[2]-b[0])*(b[3]-b[1])/(r['width']*r['height']))(recs[q]['box_xyxy_half_open'],recs[q]) for q in ids])
    edges = np.quantile(area,[0,.25,.5,.75,1]); bins = np.clip(np.searchsorted(edges[1:-1],area,side='right'),0,3)
else:
    print('SKIPPED_REQUIRES_LICENSED_DATA: frozen-reference area quartiles; supply --records')
unique, inv = np.unique(pids, return_inverse=True); counts = np.bincount(inv).astype(float); rng = np.random.default_rng(0); draws = rng.integers(0, len(unique), size=(10000, len(unique)))
def ci(d):
    sums = np.bincount(inv, weights=d, minlength=len(unique)); rep = sums[draws].sum(1) / counts[draws].sum(1); pt = float(d.mean())
    return dict(point=pt, basic_ci=[float(2 * pt - np.quantile(rep, .975)), float(2 * pt - np.quantile(rep, .025))], exploratory=True)
bger = np.array([json.loads((RUN / 'test' / f'bger_s{s}' / 'summary.json').read_text())['primary']['per_query']['R'] for s in range(3)], dtype=float).mean(0)
cearr = np.array([json.loads((RUN / 'test' / f'ce_s{s}' / 'summary.json').read_text())['primary']['per_query']['R'] for s in range(3)], dtype=float).mean(0)
out = dict(identity='OSLD_P0C_FROZEN_REFERENCE', provenance=json.loads((HERE / 'weights_PROVENANCE.json').read_text()), weights_sha256_at_run=rec['weights'], params=rec['params'],
           open_clip_version=rec['open_clip_version'], torch=rec['torch'], gpu=rec['gpu'], audit={}, results={})
for tag, r in rec['results'].items():
    s = json.loads((HERE / 'out' / tag / 'summary.json').read_text()); rows = json.loads((HERE / 'out' / tag / 'ranking.json').read_text())['rows']
    gids = sorted({x['gallery_id'] for x in rows} | set(gal_ids))
    chk = dict(query_ids_identical=s['query_ids'] == ids, query_products_identical=s['query_products'] == list(pids), gallery_size=s['gallery_size'], gallery_size_matches_core=s['gallery_size'] == ce['gallery_size'],
               top1_from_gallery=all(x['gallery_id'].startswith('logo-images/') for x in rows), relevance_rule='product_id equality (logo identity), as in the five arms',
               n_queries=s['n_queries'], mean_correct_equals_recall1=abs(np.mean([x['correct'] for x in rows]) - s['recall1']) < 1e-12)
    R = np.array(s['per_query_correct'], dtype=float)
    out['audit'][tag] = chk
    out['results'][tag] = dict(model=s['model'], preprocessing=s['preprocessing'], embedding_dim=s['embedding_dim'], recall1=s['recall1'], recall10=s['recall10'], mAP=s['mAP'], ms_per_image=s['ms_per_image'],
                               R1_by_box_area_quartile=([float(R[bins == b].mean()) for b in range(4)] if bins is not None else 'SKIPPED_REQUIRES_LICENSED_DATA'), contrast_vs_BGER_R1=ci(R - bger), contrast_vs_CE_R1=ci(R - cearr))
out['quartile_edges'] = edges.tolist() if edges is not None else None; out['quartile_counts'] = np.bincount(bins,minlength=4).tolist() if bins is not None else None
out['notes'] = ['Frozen encoders: without OSLD-specific fine-tuning or box supervision; same 1,237 queries, 20,848 templates, identity relevance, FP32 stabilized cosine ranking with the study tie-break.',
                'Preprocessing variants: study = deterministic whole-image 224x224 bilinear resize (same as the five arms); native = manual RGB float-tensor antialiased bicubic short-side resize and centre crop, not validated as equivalent to the official callable (can remove logos). Both run; neither chosen on test.',
                'Pretraining overlap with OSLD cannot be excluded: LAION-2B (OpenCLIP) and LVD-142M (DINOv2) are web-crawled and OSLD product/template images come from the Web; no deduplication against OSLD was possible.',
                'Architecture, objective (contrastive image-text / self-distillation) and pretraining data differ from the five ResNet-50 identity-classification arms; these are references, not matched arms.']
save(args.out, out)
for tag, r in out['results'].items(): print(tag, {k: round(r[k], 4) for k in ['recall1', 'recall10', 'mAP']}, 'byQ', r['R1_by_box_area_quartile'], 'vsBGER', round(r['contrast_vs_BGER_R1']['point'], 4), [round(x, 4) for x in r['contrast_vs_BGER_R1']['basic_ci']], out['audit'][tag]['query_ids_identical'], out['audit'][tag]['gallery_size_matches_core'])
