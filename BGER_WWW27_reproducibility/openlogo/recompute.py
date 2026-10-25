#!/usr/bin/env python3
"""Read-only, CPU reproduction; all outputs go to a new --out directory.

Tier 1 needs only shipped records, NumPy and pandas. Optional tiers require
explicit paths; no download, model construction, inference or training occurs.
"""
from pathlib import Path
import argparse, json, hashlib, subprocess, sys, os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def missing_import(exc):return {'status':'SKIP_MISSING_DEPENDENCY','reason':str(exc)}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source-root',type=Path,help='Optional manuscript source data/table binding check')
    ap.add_argument('--neural-root',type=Path,help='Root containing original project-relative chunk paths')
    ap.add_argument('--recompute-maps',action='store_true',help='With --neural-root, reproduce every test query map metric on CPU (torch/scipy)')
    ap.add_argument('--products-dir',type=Path,help='Licensed OpenLogo JPEGImages directory for actual retained training-loader target check')
    ap.add_argument('--checkpoint-root',type=Path,help='Optional binary digest check; directory containing paths in CHECKPOINTS.csv')
    a=ap.parse_args();root=a.root.resolve();a.out.mkdir(parents=True,exist_ok=False)
    result={'scope':'Local saved-record recomputation, not an independent training/raw-image audit', 'GPU':'NOT_USED'}
    proc=subprocess.run([sys.executable,str(root/'audit_saved_records.py'),'--root',str(root),'--out',str(a.out/'tier1')],stdout=(a.out/'tier1.log').open('w'),stderr=subprocess.STDOUT)
    if proc.returncode:raise SystemExit('Tier 1 failed; see tier1.log')
    v=load(a.out/'tier1/verification_summary.json')
    if v['failed']:raise RuntimeError('Saved record verification failed: '+str(v['failed']))
    result['tier1']={'status':'PASS','checks':v['checks'],'passed':v['passed'],'skipped_rows':v['skipped_rows'],'note':'Skipped raw-score rows are not PASS. Historical freeze hashes are not anonymous export checksums.'}
    if a.source_root:
        import csv
        for name in ['per_seed_and_mean_SD','paired_contrasts','coordinate_mixture_bounds']:
            assert sha(a.source_root/'data'/('openlogo_'+name+'.csv'))==sha(root/'tables'/(name+'.csv'))
        result['source_binding']={'status':'PASS','exact_csv_inputs':3}
    m=load(root/'protocol/manifests.json');th=load(root/'results/dev/thresholds.json')
    if not a.neural_root:
        result['tier2_raw_scores_maps']={'status':'SKIP_REQUIRES_UNSHIPPED_NEURAL_SHARDS','note':'No raw score/map regeneration is claimed by tier 1.'}
    else:
        checked=0;rows=0;maps_checked=0
        if a.recompute_maps:
            try:
                sys.path.insert(0,str(root/'reference_code'));import cpu_metrics as metric
                metric.initializer()
            except ImportError as e:
                result['optional_map_metrics']=missing_import(e);metric=None
        else:metric=None
        for phase in ['dev','test']:
            gallery=m['dev_gallery' if phase=='dev' else 'gallery'];gids=np.array([x['image_id'] for x in gallery])
            for arm in ['ce','aux','bger','cam_bce']:
                for seed in range(3):
                    cell=f'{arm}_s{seed}';d=root/'results'/phase/cell
                    marker=load(d/'NEURAL_COMPLETE.json');ranks=load(d/'ranking.json');offset=0
                    saved=np.load(d/'metrics.npz') if phase=='test' and metric else None
                    for chunk in marker['chunks']:
                        p=a.neural_root/chunk['path']
                        if not p.is_file():raise FileNotFoundError('Explicit neural root missing '+chunk['path'])
                        assert sha(p)==chunk['sha256'];checked+=1
                        with np.load(p,allow_pickle=False) as z:
                            scores=z['scores'];qmaps=z['query_maps']
                            assert np.isfinite(scores).all() and np.isfinite(qmaps).all()
                            assert scores.shape[1]==len(gallery)
                            # Actual score order, deterministic ID tie-break and positive rank.
                            for j,s in enumerate(scores):
                                i=offset+j;r=ranks[i];label=m[phase][i]['product_id']
                                order=np.lexsort((gids,-s));pos=[k for k,x in enumerate(gallery) if x['product_id']==label]
                                rank=min(int(np.flatnonzero(order==k)[0])+1 for k in pos)
                                assert int(order[0])==r['top_index'] and rank==r['positive_rank'];rows+=1
                                if saved is not None:
                                    for view in range(qmaps.shape[1]):
                                        for ro in range(qmaps.shape[2]):
                                            box,iou,eib,deg,non=metric.evaluate_map(qmaps[j,view,ro],m[phase][i],[th[cell]['indices'][ro]])
                                            assert np.array_equal(box[0],saved['pred_boxes'][i,view,ro])
                                            assert np.allclose(iou[0],saved['ious'][i,view,ro],rtol=0,atol=1e-12)
                                            assert np.allclose(eib,saved['EIB'][i,view,ro],rtol=0,atol=1e-12)
                                            assert deg==saved['degenerate'][i,view,ro] and non==saved['nonfinite'][i,view,ro]
                                            maps_checked+=1
                            offset+=len(scores)
                    assert offset==len(ranks)
                    if saved is not None:saved.close()
        result['tier2_raw_scores_maps']={'status':'PASS','chunks_sha256_and_finite':checked,'ranking_rows_from_scores':rows,'scope':'Full chunk integrity and stored-score ranking reductions; no model inference'}
        if maps_checked:result['optional_map_metrics']={'status':'PASS','test_query_view_readout_maps_recomputed':maps_checked,'note':'Native CPU pixel interpolation and both coordinate IoUs/EIB, same locked thresholds'}
        elif 'optional_map_metrics' not in result:result['optional_map_metrics']={'status':'SKIP_NOT_REQUESTED','flag':'--recompute-maps'}
    if not a.products_dir:
        result['tier3_licensed_loader']={'status':'SKIP_REQUIRES_LICENSED_IMAGES','note':'Not implied by stored target verification'}
    else:
        try:
            sys.path.insert(0,str(root/'reference_code/native'))
            from osld_data import OSLDImageDataset
            import torch
            torch.set_num_threads(1)
            rr=[dict(r) for r in m['train'] if r['has_box']]
            for r in rr:r['image_id']=str((a.products_dir/Path(r['image_id']).name).resolve())
            labels=load(root/'protocol/labels.json')
            with np.load(root/'audit/training_targets_both.npz',allow_pickle=False) as targets:
                ix={str(q):i for i,q in enumerate(targets['image_ids'])}
                for k in [0,1]:
                    records=[dict(r,box_xyxy_half_open=[r['raw_box'][0]-k,r['raw_box'][1]-k,*r['raw_box'][2:]]) for r in rr]
                    ds=OSLDImageDataset(Path('/'),records,labels)
                    for r,item in zip(records,ds):
                        if not Path(r['image_id']).is_file():raise FileNotFoundError(r['query_id'])
                        assert sha(r['image_id'])==r['sha256']
                        t=(torch.nn.functional.adaptive_avg_pool2d(item['mask'][None,None],(7,7))[0,0]>.5).float().numpy()
                        assert np.array_equal(t,targets['C'+str(k)][ix[r['query_id']]])
            result['tier3_licensed_loader']={'status':'PASS','retained_products_both_policies':len(rr),'scope':'Retained actual image digests and loader targets only, not global semantic or coordinate-definition certification'}
        except ImportError as e:result['tier3_licensed_loader']=missing_import(e)
    if a.checkpoint_root:
        import csv
        with (root/'results/CHECKPOINTS.csv').open() as f:
            ck=list(csv.DictReader(f))
        for r in ck:assert sha(a.checkpoint_root/r['checkpoint'])==r['checkpoint_sha256']
        result['checkpoint_binaries']={'status':'PASS_DIGEST_ONLY','count':len(ck),'note':'No tensors loaded'}
    else:result['checkpoint_binaries']={'status':'SKIP_REQUIRES_UNSHIPPED_CHECKPOINT_BINARIES'}
    write(a.out/'RECOMPUTE_REPORT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='tier1'},indent=2));print('Tier 1:',v['checks'],'checks')
if __name__=='__main__':main()
