#!/usr/bin/env python3
"""CPU saved-record audit. No checkpoint inference, training, images or network.
Tier 1: rankings, locked hit arrays, fixed signed maps and shared identity bootstrap.
Tier 2: --records licensed OSLD annotation records for area strata / centre prior.
A successful tier 1 never implies that skipped tier 2 checks passed.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
ARMS = ['ce','aux','bger','cam_bce','abn_box']
GRID = np.linspace(.05,.95,19)

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,default=ROOT)
    ap.add_argument('--supp-root',type=Path,default=ROOT.parent/'BGER_WWW27_reproducibility')
    ap.add_argument('--core-root',type=Path); ap.add_argument('--revision-root',type=Path)
    ap.add_argument('--records',type=Path); ap.add_argument('--report',type=Path,default=ROOT/'verification_report.json')
    a=ap.parse_args(); D=a.source_root/'data'; E=a.supp_root/'logo_retrieval_osld'
    C=a.core_root or E/'results/core'; V=a.revision_root or E/'revision_20261020'
    if not (C/'pretest_freeze.json').is_file():
        raise SystemExit('MISSING_SHIPPED_RECORDS: extract supplementary adjacent to source or supply --supp-root/--core-root')
    checks=[]
    def ck(name,actual,expected,tol=1e-10):
        aa=np.asarray(actual); ee=np.asarray(expected)
        good=aa.shape==ee.shape and np.allclose(aa,ee,rtol=0,atol=tol,equal_nan=False)
        checks.append(dict(name=name,passed=bool(good)))
        if not good: print('MISMATCH',name,str(actual)[:180],str(expected)[:180],flush=True)
    core=read(D/'osld_core_results.json'); pa=read(D/'p0a_results.json'); pb=read(D/'p0b_results.json'); pc=read(D/'p0c_results.json')
    ra=read(D/'osld_readout_analysis.json'); desc=read(D/'osld_descriptive.json')
    S={}; arrays={}; pair_cache={}
    for arm in ARMS+['gate_only','shifted_box']:
        arrays[arm]={k:[] for k in ['R1','R10','mAP','primary/Q','primary/J','primary/query/EIB','secondary/Q']}
        for seed in range(3):
            if arm in ARMS: run=C/'test'/f'{arm}_s{seed}'
            else: run=V/'p0b/out/test'/f"{dict(gate_only='gate_only_l0',shifted_box='shifted_box_l0.5')[arm]}_s{seed}"
            sm=read(run/'summary.json'); S[arm,seed]=sm
            rows=read(run/'ranking.json')['rows']; own=read(run/'own_top1.json')['pairs']; own={p['query_id']:p for p in own}
            ids=sm['query_ids']; rr={r['query_id']:r for r in rows}
            ck(f'{arm}/{seed}/query_order',[r['query_id']==q for r,q in zip(rows,ids)],[True]*len(ids))
            ck(f'{arm}/{seed}/query_identity_pairing',sm['query_products']==S['ce',0]['query_products'],True)
            for key,rkey,summarykey in [('R1','correct','recall1'),('R10','recall10','recall10'),('mAP','AP','mAP')]:
                ar=np.array([rr[q][rkey] for q in ids],dtype=float); arrays[arm][key].append(ar)
                ck(f'{arm}/{seed}/{key}/ranking_vs_summary',ar.mean(),sm[summarykey])
                if arm in ARMS: ck(f'{arm}/{seed}/{key}/source',ar.mean(),core['mean_sd'][arm][key]['per_seed'][seed])
                else: ck(f'{arm}/{seed}/{key}/source',ar.mean(),pb['per_seed'][arm][seed][dict(R1='R1',R10='R10',mAP='mAP')[key]])
            for ro in ['primary','secondary']:
                k=int(np.argmin(np.abs(GRID-sm['thresholds'][ro])))
                q=np.array([own[i]['query'][ro]['hits'][k][1] for i in ids],dtype=float)
                eib=np.array([own[i]['query'][ro]['EIB'] for i in ids])
                ck(f'{arm}/{seed}/{ro}/hits_vs_per_query',q,sm[ro]['per_query']['Q'])
                ck(f'{arm}/{seed}/{ro}/loc',q.mean(),sm[ro]['marginal_loc_query'])
                R=arrays[arm]['R1'][-1]; ck(f'{arm}/{seed}/{ro}/joint',(q*R).mean(),sm[ro]['Jq_at_1'])
                ck(f'{arm}/{seed}/{ro}/EIB',eib.mean(),sm[ro]['EIB_query'])
                arrays[arm][ro+'/Q'].append(q)
                if ro=='primary':
                    arrays[arm]['primary/J'].append(q*R); arrays[arm]['primary/query/EIB'].append(eib)
                    if arm in ARMS: ck(f'{arm}/{seed}/Loc/source',q.mean(),core['mean_sd'][arm]['Loc_q']['per_seed'][seed])
                    else: ck(f'{arm}/{seed}/Loc/source',q.mean(),pb['per_seed'][arm][seed]['Loc'])
            if arm in ARMS: pair_cache[arm,seed]=(run,own)
        arrays[arm]={k:np.array(v) for k,v in arrays[arm].items()}
    ids=S['ce',0]['query_ids']; pids=S['ce',0]['query_products']
    unique,inv=np.unique(pids,return_inverse=True); counts=np.bincount(inv).astype(float)
    draws=np.random.default_rng(0).integers(0,len(unique),size=(10000,len(unique))); den=counts[draws].sum(1)
    def boot(diff):
        d=np.asarray(diff,dtype=float).mean(0); pt=float(d.mean())
        sums=np.bincount(inv,weights=d,minlength=len(unique)); rep=sums[draws].sum(1)/den
        return pt,[float(2*pt-np.quantile(rep,.975)),float(2*pt-np.quantile(rep,.025))]
    def contrast(name,diff,c):
        pt,ci=boot(diff); ck(name+'/point',pt,c['point']); ck(name+'/CI',ci,c.get('basic_ci',c.get('basic_two_sided_ci')))
    for key,c in read(D/'osld_contrasts_all.json').items():
        pair,metric=key.split('/',1); x,y=pair.split('-'); contrast('core/'+key,arrays[x][metric]-arrays[y][metric],c)
    # The two prespecified tests and their original Holm family, using the shipped pure-numpy implementation.
    sys.path.insert(0,str(E/'code'))
    from e1_inference_candidate import paired_cluster_inference
    conf=paired_cluster_inference({'R':arrays['bger']['R1']-arrays['ce']['R1'],'Q':arrays['bger']['primary/Q']-arrays['ce']['primary/Q']},pids,{'R':-.01,'Q':0.})
    for k,c in core['confirmatory']['contrasts'].items():
        for m in ['point','basic_two_sided_ci','basic_one_sided_lower','p_holm','reject_holm']:
            ck('confirmatory/'+k+'/'+m,conf['contrasts'][k][m],c[m])
    # Figure 4: recompute all displayed hit rates and shape changes directly from fixed-pair maps.
    for arm in ARMS:
        metrics={ro:{k:[] for k in ['loc_pos','loc_neg','mean_l1','mean_corr','verdict_change']} for ro in ['primary','secondary']}
        for seed in range(3):
            run,_=pair_cache[arm,seed]; fixed=read(run/'fixed_pairs.json')['pairs']
            pos={p['query_id']:p for p in fixed if p['view'].endswith('_positive')}; neg={p['query_id']:p for p in fixed if p['view'].endswith('_negative')}
            with np.load(run/'query_signed_maps.npz') as z:
                ix={str(n):i for i,n in enumerate(z['names'])}
                for ro in ['primary','secondary']:
                    saved=z['query_'+ro+'_signed']
                    P=np.array([saved[ix[pos[q]['signed_maps_file'][:-4]]] for q in ids],dtype=float).reshape(len(ids),-1)
                    N=np.array([saved[ix[neg[q]['signed_maps_file'][:-4]]] for q in ids],dtype=float).reshape(len(ids),-1)
                    P=np.clip(P,0,None); N=np.clip(N,0,None)
                    valid=(P.sum(1)>1e-12)&(N.sum(1)>1e-12)
                    l1=np.abs(P[valid]/P[valid].sum(1)[:,None]-N[valid]/N[valid].sum(1)[:,None]).sum(1)
                    valid=(P.std(1)>1e-12)&(N.std(1)>1e-12); p=P[valid]-P[valid].mean(1)[:,None]; n=N[valid]-N[valid].mean(1)[:,None]
                    corr=(p*n).sum(1)/np.sqrt((p*p).sum(1)*(n*n).sum(1))
                    k=int(np.argmin(np.abs(GRID-S[arm,seed]['thresholds'][ro])))
                    lp=np.array([pos[q]['query'][ro]['hits'][k][1] for q in ids]); ln=np.array([neg[q]['query'][ro]['hits'][k][1] for q in ids])
                    for m,v in dict(loc_pos=lp.mean(),loc_neg=ln.mean(),mean_l1=l1.mean(),mean_corr=corr.mean(),verdict_change=(lp!=ln).mean()).items():metrics[ro][m].append(v)
        for ro,ms in metrics.items():
            for m,v in ms.items():ck(f'fixed_maps/{arm}/{ro}/{m}',np.mean(v),ra['a5_split'][arm][ro][m])
    # Pooled-hook hit records and lock files; original H2 is kept separate.
    qf=[]
    for seed in range(3):
        out=V/'p0a/out'/f'seed{seed}'; lock=read(out/'THRESHOLD_LOCK.json'); rows=read(out/'test_records.json')['rows']
        own={r['query_id']:r for r in rows if r['view']=='own_top1'}; k=int(np.argmin(np.abs(GRID-lock['threshold'])))
        q=np.array([own[i]['F']['hits'][k][1] for i in ids]); qf.append(q)
        ck(f'p0a/{seed}/Loc_F',q.mean(),pa['seeds_Loc_F'][seed]); ck(f'p0a/{seed}/Jq_F',(q*arrays['bger']['R1'][seed]).mean(),pa['seeds'][str(seed)]['test']['Jq_F'])
        ck(f'p0a/{seed}/EIB_F',np.mean([own[i]['F']['EIB'] for i in ids]),pa['seeds'][str(seed)]['test']['EIB_F'])
        dev=read(out/'dev_records.json')['rows']; dev=[r for r in dev if r['view']=='own_top1']; hit=np.mean([np.asarray(r['F']['hits'])[:,1] for r in dev],axis=0)
        ck(f'p0a/{seed}/validation_threshold',GRID[int(hit.argmax())],lock['threshold'])
    qf=np.array(qf)
    for key,c in pa['contrasts'].items():
        other=key.split('/')[0].split('-')[1]; rhs=arrays['bger' if other=='BGER_H' else other]['primary/Q']
        contrast('p0a/'+key,qf-rhs,c)
    for key,c in pb['contrasts'].items():
        pair,m=key.split('/'); x,y=pair.split('-'); metric='R1' if m=='R1' else 'primary/Q'
        contrast('p0b/'+key,arrays[x][metric]-arrays[y][metric],c)
    for tag,r in pc['results'].items():
        rows=read(V/'p0c/out'/tag/'ranking.json')['rows']; rr={r['query_id']:r for r in rows}
        R=np.array([rr[q]['correct'] for q in ids]); ck('p0c/'+tag+'/R1',R.mean(),r['recall1'])
        ck('p0c/'+tag+'/R10',np.mean([rr[q]['recall10'] for q in ids]),r['recall10'])
        ck('p0c/'+tag+'/mAP',np.mean([rr[q]['AP'] for q in ids]),r['mAP'])
        for other in ['bger','ce']:contrast('p0c/'+tag+'/vs_'+other,R[None,:]-arrays[other]['R1'],r['contrast_vs_'+('BGER' if other=='bger' else 'CE')+'_R1'])
    tier1_n=len(checks); tier2='SKIPPED_REQUIRES_LICENSED_DATA'
    if a.records:
        if not a.records.is_file():raise SystemExit('Explicit --records does not exist: '+str(a.records))
        records=read(a.records)
        raw=records['product_records']+records['logo_records']
        digest=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
        if digest!=read(C/'pretest_freeze.json')['records_digest']:raise SystemExit('Licensed records digest differs from frozen release')
        recs={r['image_id']:r for r in records['product_records']}
        area=[]
        for q in ids:
            r=recs[q]; b=r['box_xyxy_half_open']; area.append((b[2]-b[0])*(b[3]-b[1])/(r['width']*r['height']))
        edges=np.quantile(area,[0,.25,.5,.75,1]); bins=np.clip(np.searchsorted(edges[1:-1],area,side='right'),0,3)
        ck('licensed/quartile_edges',edges,desc['area_quartile_edges']); ck('licensed/bin_counts',np.bincount(bins,minlength=4),desc['bin_counts'])
        for arm in ARMS:
            for b in range(4):
                for key,m in [('R','R1'),('Q','primary/Q'),('J','primary/J')]:ck(f'licensed/size/{arm}/{b}/{key}',arrays[arm][m][:,bins==b].mean(),desc['small_logo'][arm][b][key])
        for tag,r in pc['results'].items():
            rows=read(V/'p0c/out'/tag/'ranking.json')['rows']; rr={x['query_id']:x for x in rows}; R=np.array([rr[q]['correct'] for q in ids])
            ck('licensed/p0c/'+tag+'/size',[R[bins==b].mean() for b in range(4)],r['R1_by_box_area_quartile'])
        # Centre-prior regeneration uses the original CPU map pipeline; requires torch/scipy.
        sys.path.insert(0,str(E/'analysis'))
        from readout_identity_analysis import prior_own_threshold
        prior=prior_own_threshold(a.records,C)
        for k in ['val_hit_rate_by_threshold','val_selected_threshold','val_hit_rate','test_loc_at_val_threshold','test_loc_by_threshold']:
            ck('licensed/prior/'+k,prior[k],ra['centre_prior_own_threshold'][k])
        tier2='PASS' if all(c['passed'] for c in checks[tier1_n:]) else 'FAIL'
    failed=[c for c in checks if not c['passed']]
    report=dict(scope='Saved-record CPU audit only; no model inference or training',tier1=dict(status='PASS' if not any(not c['passed'] for c in checks[:tier1_n]) else 'FAIL',checks=tier1_n),tier2=dict(status=tier2,checks=len(checks)-tier1_n,requirements='Licensed osld_records.json; torch, scipy for prior pipeline'),tier3=dict(status='NOT_RUN_NOT_AUTHORIZED',requirements='Original images/checkpoints and fresh authorization'),checks=checks,n_failed=len(failed),n_checks=len(checks))
    write(a.report,report); print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2),flush=True)
    return 1 if failed else 0
if __name__=='__main__':sys.exit(main())
