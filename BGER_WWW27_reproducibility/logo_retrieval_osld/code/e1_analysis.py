"""One fixed three-hypothesis family, plus explicitly descriptive endpoints."""
import json
import numpy as np
from e1_common import save,GRID,digest
from e1_inference_candidate import paired_cluster_inference


def analyze(outputs,out,identity):
    by={(r['arm'],r['seed']):r for r in outputs}
    arms=sorted({a for a,_ in by})
    if not {'ce','aux','bger','cam_bce'}<=set(arms) or set(by)!={(a,s) for a in arms for s in range(3)}:raise ValueError('Analysis requires the selected arms x three seeds')
    ref=by['ce',0]['summary'];ids=ref['query_ids'];pids=ref['query_products']
    if any(r['summary']['query_ids']!=ids or r['summary']['query_products']!=pids for r in outputs):raise ValueError('Incomplete query/product pairing across methods/seeds')
    differences={k:np.array([by['bger',s]['summary']['primary']['per_query'][k] for s in range(3)])-
                   np.array([by['ce',s]['summary']['primary']['per_query'][k] for s in range(3)]) for k in ['R','Q']}
    # OSLD amendment: H1-logo R@1 non-inferiority (-0.01), H2-logo query Loc superiority; Holm over two.
    # Cluster unit = query logo identity (stored in the 'product_id' field by the OSLD adapter).
    inference=paired_cluster_inference(differences,pids,dict(R=-.01,Q=0.),n_boot=10000,random_seed=0)
    reps={k:v.pop('replicates') for k,v in inference['contrasts'].items()}
    np.savez_compressed(out/'primary_bootstrap_replicates.npz',identity=np.array(identity),**reps)
    inference.update(identity=identity,margin_rationale='Design tolerance: at most one additional top-1 error per 100 queries; not a field standard or CUB certification',
                     intervals='Unadjusted basic intervals; not simultaneous confidence intervals',
                     scientific_inference=identity!='SYNTHETIC_INTEGRATION')
    save(out/'primary_inference.json',inference)
    # Same product draws for all descriptive endpoints, fixed models/gallery.
    rng=np.random.default_rng(0);unique=np.unique(pids);groups=[np.flatnonzero(np.array(pids)==p) for p in unique]
    draws=rng.integers(0,len(unique),size=(10000,len(unique)))
    counts=np.stack([np.bincount(np.concatenate([groups[j] for j in draw]),minlength=len(ids)) for draw in draws])
    weights=counts/counts.sum(1,keepdims=True)
    series={};boot={}
    def put(key,arr,envelope=False):
        # Seeds averaged before query aggregation; V2 remaximized per draw.
        arr=np.array(arr,dtype=float)
        if envelope:
            each=np.einsum('bq,sqti->sbti',weights,arr).max(2).mean(2)
            point=float(arr.mean(1).max(1).mean(1).mean());rep=each.mean(0)
        else:
            point=float(arr.mean());rep=weights@arr.mean(0)
        boot[key]=rep
        series[key]=dict(point=point,basic_ci=[float(2*point-np.quantile(rep,.975)),float(2*point-np.quantile(rep,.025))],
                         exploratory=True,uncertainty='fixed models/gallery, query logo-identity clusters; unadjusted')
    for arm in arms:
        runs=[by[arm,s] for s in range(3)]
        pairs=[json.loads((r['path']/'own_top1.json').read_text())['pairs'] for r in runs]
        ranks=[json.loads((r['path']/'ranking.json').read_text())['rows'] for r in runs]
        for metric,field in [('R1','correct'),('R10','recall10'),('mAP','AP')]:put(f'{arm}/{metric}',[[q[field] for q in rr] for rr in ranks])
        for ro in ['primary','secondary']:
            for k in ['Q','J']:put(f'{arm}/{ro}/{k}',[r['summary'][ro]['per_query'][k] for r in runs])
            for side in ['query']:
                put(f'{arm}/{ro}/{side}/EIB',[[p[side][ro]['EIB'] for p in pp] for pp in pairs])
                put(f'{arm}/{ro}/{side}/V2',[[p[side][ro]['hits'] for p in pp] for pp in pairs],True)
        fixed=[json.loads((r['path']/'fixed_pairs.json').read_text())['pairs'] for r in runs]
        for sign in ['positive','negative']:
            pp=[[p for p in rr if p['view'].endswith('_'+sign)] for rr in fixed]
            for ro in ['primary','secondary']:
                for side in ['query']:
                    put(f'{arm}/fixed_{sign}/{ro}/{side}/EIB',[[p[side][ro]['EIB'] for p in rr] for rr in pp])
                    put(f'{arm}/fixed_{sign}/{ro}/{side}/V2',[[p[side][ro]['hits'] for p in rr] for rr in pp],True)
                    put(f'{arm}/fixed_{sign}/{ro}/{side}/Loc',[[p[side][ro]['hits'][int(np.argmin(abs(GRID-r['summary']['thresholds'][ro])))][1] for p in rr] for rr,r in zip(pp,runs)])
    contrasts={}
    pairs_=[('bger','ce'),('aux','ce'),('cam_bce','ce'),('bger','aux'),('bger','cam_bce')]+([('abn_box','ce'),('bger','abn_box')] if 'abn_box' in arms else [])
    for a,b in pairs_:
        for key in list(series):
            if not key.startswith(a+'/'):continue
            other=b+'/'+key.split('/',1)[1]
            delta=series[key]['point']-series[other]['point'];rep=boot[key]-boot[other]
            contrasts[a+'-'+b+'/'+key.split('/',1)[1]]=dict(point=delta,basic_ci=[float(2*delta-np.quantile(rep,.975)),float(2*delta-np.quantile(rep,.025))],exploratory=True)
    save(out/'descriptive_inference.json',dict(identity=identity,points=series,paired_contrasts=contrasts,
         n_boot=10000,rng_seed=0,shared_draws_sha256=digest(draws),no_additional_significance_claims=True))
    return inference
