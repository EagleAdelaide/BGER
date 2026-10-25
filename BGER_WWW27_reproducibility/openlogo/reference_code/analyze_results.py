#!/usr/bin/env python3
"""Paired exploratory inference, including image-wise convention-mixture bounds."""
import argparse,csv,itertools
from collections import defaultdict
import numpy as np
from b_common import *

def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mixture_series(delta):
    """delta[query,convention], paired arms under SAME convention per query."""
    return np.min(delta,axis=-1),np.max(delta,axis=-1)
class SharedBootstrap:
    def __init__(self,groups,classes):
        self.groups=np.asarray(groups);self.classes=np.asarray(classes);self.unique=np.unique(groups)
        self.gix=[np.flatnonzero(self.groups==g) for g in self.unique]
        self.classes_unique=np.unique(classes);self.cix=[np.flatnonzero(self.classes==c) for c in self.classes_unique]
        self.group_classes=[[j for j,ix in enumerate(self.cix) if self.groups[ix[0]]==g] for g in self.unique]
        rng=np.random.default_rng(0);self.draws=rng.integers(0,len(self.unique),size=(10000,len(self.unique)))
        self.counts=np.array([np.bincount(d,minlength=len(self.unique)) for d in self.draws],dtype=np.float64)
        self.den_query=self.counts@np.array([len(ix) for ix in self.gix])
        self.den_class=self.counts@np.array([len(ix) for ix in self.group_classes])
    def point_boot(self,x,weighting='query'):
        x=np.asarray(x,dtype=float);assert x.ndim==1 and np.isfinite(x).all()
        if weighting=='query':
            point=float(x.mean());sums=np.array([x[ix].sum() for ix in self.gix]);reps=(self.counts@sums)/self.den_query
        else:
            means=np.array([x[ix].mean() for ix in self.cix]);point=float(means.mean())
            sums=np.array([means[ix].sum() for ix in self.group_classes]);reps=(self.counts@sums)/self.den_class
        return point,reps
    def estimate(self,x,weighting='query'):
        point,boots=self.point_boot(x,weighting)
        return dict(point=point,ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'outputs/analysis');a=ap.parse_args();a.out.mkdir(exist_ok=False)
    assert (ROOT/'outputs/test/ALL_METRICS_COMPLETE.json').exists()
    manifests=load(ROOT/'protocol/manifests.json');qr=manifests['test'];ids=[r['query_id'] for r in qr];groups=[r['group'] for r in qr];classes=[r['product_id'] for r in qr]
    gallery_index={r['image_id']:i for i,r in enumerate(manifests['gallery'])}
    fixed=load(ROOT/'outputs/test/CE_SEED0_FIXED_CANDIDATES.json')['pairs']
    positive_indices=np.array([gallery_index[r['positive']] for r in fixed]);negative_indices=np.array([gallery_index[r['negative']] for r in fixed])
    boot=SharedBootstrap(groups,classes)
    np.savez_compressed(a.out/'shared_group_bootstrap_draws.npz',groups=np.array(boot.unique),draws=boot.draws)
    series={};seedrows=[];diagnostics=[];matrices={};rankdata={}
    def add(key,arr):
        arr=np.array(arr,dtype=np.float64);assert arr.shape==(3,len(qr))
        series[key]=arr.mean(0)
        for weighting in ['query','class_macro']:
            vals=[boot.point_boot(x,weighting)[0] for x in arr]
            seedrows.append(dict(key=key,weighting=weighting,seed0=vals[0],seed1=vals[1],seed2=vals[2],mean=float(np.mean(vals)),sample_SD=float(np.std(vals,ddof=1))))
    for arm in ARMS:
        rank=[];metric=[];score_changes=[]
        for seed in range(3):
            p=ROOT/f'outputs/test/{arm}_s{seed}';r=load(p/'ranking.json')
            assert [x['query_id'] for x in r]==ids
            n=np.load(p/'metrics.npz');marker=load(p/'METRICS_COMPLETE.json');assert sha(p/'metrics.npz')==marker['metrics_sha256']
            rank.append(r);metric.append({k:n[k] for k in n.files});n.close()
            chunks=load(p/'NEURAL_COMPLETE.json')['chunks'];scores=[]
            for chunk in chunks:
                with np.load(PROJECT/chunk['path']) as z:scores.append(z['scores'])
            scores=np.concatenate(scores);score_changes.append(scores[np.arange(len(qr)),positive_indices]-scores[np.arange(len(qr)),negative_indices])
        rankdata[arm]=rank;matrices[arm]=metric
        correct=np.array([[r['correct'] for r in rs] for rs in rank])
        for field,name in [('correct','R1'),('recall10','R10'),('AP','MRR_mAP')]:add(f'{arm}/retrieval/{name}',[[r[field] for r in rs] for rs in rank])
        add(f'{arm}/candidate_score_change',score_changes)
        readouts=['H','decomposition']+(['F'] if arm=='bger' else [])
        for j,readout in enumerate(readouts):
            for v,view in enumerate(['own','fixed_positive','fixed_negative']):
                for k,convention in enumerate(['C0','C1']):
                    hits=np.array([n['hits'][:,v,j,k] for n in metric]);eib=np.array([n['EIB'][:,v,j,k] for n in metric])
                    add(f'{arm}/{readout}/{view}/Loc/{convention}',hits)
                    add(f'{arm}/{readout}/{view}/EIB/{convention}',eib)
                    if view=='own':add(f'{arm}/{readout}/own/Joint/{convention}',hits*correct)
                flips=np.array([n['hits'][:,v,j,0]!=n['hits'][:,v,j,1] for n in metric],dtype=float)
                add(f'{arm}/{readout}/{view}/coordinate_verdict_flip',flips)
                add(f'{arm}/{readout}/{view}/degenerate',[n['degenerate'][:,v,j] for n in metric])
            for k,conv in enumerate(['C0','C1']):
                add(f'{arm}/{readout}/candidate_verdict_change/{conv}',[n['hits'][:,1,j,k]!=n['hits'][:,2,j,k] for n in metric])
            for seed,n in enumerate(metric):
                row=dict(arm=arm,seed=seed,readout=readout,n_queries=len(qr))
                for field in ['candidate_positive_sum','candidate_negative_sum','candidate_positive_max_abs','candidate_negative_max_abs','candidate_normalized_l1','candidate_correlation']:
                    val=n[field][:,j];good=np.isfinite(val)
                    row[field+'_mean_defined']=float(val[good].mean()) if good.any() else ''
                    row[field+'_undefined']=int((~good).sum())
                diagnostics.append(row)
    prior=np.load(ROOT/'outputs/test/centre_prior_metrics.npz')
    for k,conv in enumerate(['C0','C1']):
        add(f'centre_prior/Loc/{conv}',np.repeat(prior['hits'][None,:,k],3,axis=0))
        add(f'centre_prior/EIB/{conv}',np.repeat(prior['EIB'][None,:,k],3,axis=0))
    add('centre_prior/coordinate_verdict_flip',np.repeat((prior['hits'][:,0]!=prior['hits'][:,1])[None,:],3,axis=0))
    estimates=[]
    for key,arr in series.items():
        for weighting in ['query','class_macro']:
            estimates.append(dict(key=key,weighting=weighting,**boot.estimate(arr,weighting),interval='95% percentile; unadjusted exploratory; conditional on trained models/gallery'))
    contrasts=[];bounds=[]
    pairs=[('bger','ce'),('aux','ce'),('cam_bce','ce'),('bger','aux'),('bger','cam_bce'),('aux','cam_bce')]
    for first,second in pairs:
        for key in list(series):
            if not key.startswith(first+'/'):continue
            other=second+'/'+key.split('/',1)[1]
            if other not in series:continue
            delta=series[key]-series[other]
            for weighting in ['query','class_macro']:
                contrasts.append(dict(contrast=first+'-'+second,endpoint=key.split('/',1)[1],weighting=weighting,
                    **boot.estimate(delta,weighting),interval='95% paired group-cluster percentile; exploratory; conditional'))
        for ro in ['H','decomposition']:
            for endpoint in ['Loc','Joint']:
                delta=np.stack([series[f'{first}/{ro}/own/{endpoint}/{c}']-series[f'{second}/{ro}/own/{endpoint}/{c}'] for c in ['C0','C1']],axis=1)
                lo,hi=mixture_series(delta)
                for weighting in ['query','class_macro']:
                    l=boot.estimate(lo,weighting);h=boot.estimate(hi,weighting)
                    c0=boot.estimate(delta[:,0],weighting)['point'];c1=boot.estimate(delta[:,1],weighting)['point']
                    bounds.append(dict(contrast=first+'-'+second,readout=ro,endpoint=endpoint,weighting=weighting,
                        C0_effect=c0,C1_effect=c1,mixture_lower_bound=l['point'],mixture_upper_bound=h['point'],
                        bound_type='Deterministic specified-two-policy image-wise annotation-sensitivity bound; NOT a95%CI',
                        lower_estimation_ci_low=l['ci_low'],lower_estimation_ci_high=l['ci_high'],upper_estimation_ci_low=h['ci_low'],upper_estimation_ci_high=h['ci_high'],
                        bound_bootstrap_label='Separate conditional exploratory uncertainty for each endpoint, not simultaneous coverage',
                        coordinate_verdict='coordinate-sensitive/inconclusive' if l['point']<=0<=h['point'] or c0*c1<0 else 'same effect sign within specified two-policy mixture set'))
    for readout,base in [('F','H'),('decomposition','H')]:
        for endpoint in ['Loc','Joint','EIB']:
            for conv in ['C0','C1']:
                one=f'bger/{readout}/own/{endpoint}/{conv}';two=f'bger/{base}/own/{endpoint}/{conv}'
                for weighting in ['query','class_macro']:
                    contrasts.append(dict(contrast=f'bger_{readout}-bger_{base}',endpoint=f'own/{endpoint}/{conv}',weighting=weighting,
                        **boot.estimate(series[one]-series[two],weighting),interval='95% paired group-cluster percentile; exploratory; conditional'))
    csvout(a.out/'per_seed_and_mean_SD.csv',seedrows);csvout(a.out/'estimates_cluster_CI.csv',estimates)
    csvout(a.out/'paired_contrasts.csv',contrasts);csvout(a.out/'coordinate_mixture_bounds.csv',bounds);csvout(a.out/'candidate_diagnostics_per_seed.csv',diagnostics)
    aggregates=[]
    for arm in ARMS:
        for ro in ['H','decomposition']+(['F'] if arm=='bger' else []):
            rr=[r for r in diagnostics if r['arm']==arm and r['readout']==ro]
            for field in [k for k in rr[0] if k.endswith('_mean_defined')]:
                vals=[float(r[field]) for r in rr if r[field]!='']
                aggregates.append(dict(arm=arm,readout=ro,metric=field,mean=float(np.mean(vals)) if vals else '',
                    sample_SD=float(np.std(vals,ddof=1)) if len(vals)>1 else '',defined_seeds=len(vals),
                    interpretation='Descriptive mean/SD of per-seed defined-case means; undefined counts reported separately'))
    csvout(a.out/'candidate_diagnostics_mean_SD.csv',aggregates)
    # Long-form per-query records are generated from saved ranking + map metrics.
    perquery=[]
    for arm in ARMS:
        for seed in range(3):
            n=matrices[arm][seed]
            for i,r in enumerate(rankdata[arm][seed]):
                for j,ro in enumerate(['H','decomposition']+(['F'] if arm=='bger' else [])):
                    for k,c in enumerate(['C0','C1']):
                        perquery.append(dict(arm=arm,seed=seed,query_id=r['query_id'],released_class=r['released_class'],group=r['group'],readout=ro,convention=c,
                            R1=r['correct'],R10=r['recall10'],MRR_mAP=r['AP'],Loc=int(n['hits'][i,0,j,k]),Joint=int(n['hits'][i,0,j,k])*r['correct'],
                            EIB=float(n['EIB'][i,0,j,k]),degenerate=bool(n['degenerate'][i,0,j]),
                            fixed_positive_Loc=int(n['hits'][i,1,j,k]),fixed_negative_Loc=int(n['hits'][i,2,j,k]),
                            coordinate_verdict_flip=bool(n['hits'][i,0,j,0]!=n['hits'][i,0,j,1])))
    csvout(a.out/'test_per_query_long.csv',perquery)
    save(a.out/'ANALYSIS_COMPLETE.json',dict(protocol=ID,status='ANALYSIS_COMPLETE',bootstrap_draws=10000,shared_draws_sha256=sha(a.out/'shared_group_bootstrap_draws.npz'),
        n_queries=len(qr),n_groups=len(boot.unique),n_classes=len(boot.classes_unique),models=12,seeds_averaged_before_sampling=True,
        primary_weighting='query',class_macro=True,official_coordinate_semantics='UNVERIFIED',A='NOT_RUN_SOURCE_UNSEEN_UNRESOLVED',
        files={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},completed_utc=now()))
    print('Analysis completed from all12 models and same C0/C1 saved outputs',flush=True)
if __name__=='__main__':main()
