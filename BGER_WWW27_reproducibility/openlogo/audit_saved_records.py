from pathlib import Path
import json,hashlib,re,csv
from collections import Counter
import numpy as np
import pandas as pd
import argparse
parser=argparse.ArgumentParser(description='CPU audit of the completed BGER derived-OpenLogo archive; no neural inference.')
parser.add_argument('--root', type=Path, required=True, help='Extracted BGER_OPENLOGO_B_CONVENTION_STABLE directory')
parser.add_argument('--out', type=Path, default=Path('independent_audit_output'), help='Directory for audit outputs')
args=parser.parse_args()
B=args.root.resolve()
if not (B/'protocol/manifests.json').is_file():
    parser.error('--root must contain protocol/manifests.json')
OUT=args.out.resolve(); OUT.mkdir(parents=True,exist_ok=True)
checks=[]
def ck(name, ok, detail=None):
 checks.append({'check':name,'pass':bool(ok),'detail':detail})
def eq(name,x,y,tol=1e-12):
 a=np.asarray(x);b=np.asarray(y);ok=a.shape==b.shape and np.allclose(a,b,atol=tol,rtol=0,equal_nan=True)
 ck(name,ok,{'max_abs_error':float(np.nanmax(np.abs(a-b))) if a.size and a.shape==b.shape else None})
L=lambda p:json.loads((B/p).read_text())
hashes=[]
for line in (B/'SHA256SUMS').read_text().splitlines():
 h,rel=line.split(None,1);p=B/rel.strip();hashes.append((rel,p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h))
ck('All declared SHA256 records',all(v for k,v in hashes),{'records':len(hashes),'failed':[k for k,v in hashes if not v]})
m=L('protocol/manifests.json');th=L('results/dev/thresholds.json')
for split,n in [('train',1002),('dev',1466),('test',6706),('gallery',352),('dev_gallery',108)]:ck(split+' manifest count',len(m[split])==n)
groups={s:set(r['group'] for r in m[s]) for s in ['train','dev','test']}
classes={s:set(r['product_id'] for r in m[s]) for s in ['train','dev','test']}
for a,b in [('train','dev'),('train','test'),('dev','test')]:
 ck(a+'/'+b+' disjoint groups',not groups[a]&groups[b]);ck(a+'/'+b+' disjoint released labels',not classes[a]&classes[b])
with np.load(B/'audit/training_targets_both.npz',allow_pickle=False) as t:
 stable=(t['C0']==t['C1']).all((1,2))&(t['C0'].sum((1,2))>0)&(t['C1'].sum((1,2))>0)
 stable224=(t['C0_intermediate_224']==t['C1_intermediate_224']).all((1,2))
 ck('Stored 965 paired training target counts',len(stable)==965 and int(stable.sum())==949)
 eq('Final target stability agrees with block-expanded intermediate',stable.astype(int),stable224.astype(int))
 mids={r['query_id'] for r in m['train'] if r['has_box']}
 ck('Training membership equals stored stable targets',mids==set(t['image_ids'][stable]))
 def reconstruct(row,delta):
  x,y,u,v=row['raw_box'];x-=delta;y-=delta
  a=np.zeros((7,7),np.float32)
  x0=max(0,min(6,int(np.floor(x/row['width']*7))));y0=max(0,min(6,int(np.floor(y/row['height']*7))))
  x1=max(x0+1,min(7,int(np.ceil(u/row['width']*7))));y1=max(y0+1,min(7,int(np.ceil(v/row['height']*7))))
  a[y0:y1,x0:x1]=1
  return a
 idx={str(x):i for i,x in enumerate(t['image_ids'])}
 for delta in [0,1]:
  rr=[r for r in m['train'] if r['has_box']]
  eq('Retained target reconstruction C'+str(delta),np.stack([reconstruct(r,delta) for r in rr]),t['C'+str(delta)][[idx[r['query_id']] for r in rr]])

series={};raw_by_arm={};arrays={};rank_count=0;iou_count=0
qr=m['test'];ids=[r['query_id'] for r in qr];N=len(qr)
labels=np.array([r['product_id'] for r in qr]);gs=np.array([r['group'] for r in qr]);ug,gi=np.unique(gs,return_inverse=True);uc,ci=np.unique(labels,return_inverse=True)
gsize=np.bincount(gi);csize=np.bincount(ci);cgroup=np.array([gi[np.flatnonzero(ci==j)[0]] for j in range(len(uc))])
ck('Released class belongs to one identity group',all(len(set(gs[ci==j]))==1 for j in range(len(uc))))
with np.load(B/'results/shared_group_bootstrap_draws.npz') as z:
 ck('Bootstrap group ordering',np.array_equal(ug,z['groups']))
 draws=np.random.default_rng(0).integers(0,len(ug),size=(10000,len(ug)))
 ck('Bootstrap independently regenerated draws',np.array_equal(draws,z['draws']))
counts=np.zeros((10000,len(ug)),float)
np.add.at(counts,(np.repeat(np.arange(10000),len(ug)),draws.ravel()),1)
den=counts@gsize
cden=counts@np.bincount(cgroup,minlength=len(ug))
def estimate(x,w='query'):
 if w=='query': point=float(x.mean());s=np.bincount(gi,weights=x,minlength=len(ug));v=counts@s/den
 else:
  means=np.bincount(ci,weights=x,minlength=len(uc))/csize;point=float(means.mean());s=np.bincount(cgroup,weights=means,minlength=len(ug));v=counts@s/cden
 return np.array([point,*np.quantile(v,[.025,.975])])
def add(key,a):
 a=np.array(a,dtype=float);assert a.shape==(3,N);raw_by_arm[key]=a;series[key]=a.mean(0)
def calc_iou(pred,gt):
 # pred [N,view,readout,4], gt [N,4]
 g=gt[:,None,None,:]
 size=np.maximum(0,np.minimum(pred[...,2:],g[...,2:])-np.maximum(pred[...,:2],g[...,:2]))
 inter=np.prod(size,axis=-1);pa=np.prod(np.maximum(0,pred[...,2:]-pred[...,:2]),axis=-1);ga=np.prod(g[...,2:]-g[...,:2],axis=-1)
 v=np.divide(inter,pa+ga-inter,out=np.zeros_like(inter,dtype=float),where=pa+ga-inter>0)
 return np.where(pred[...,0]<0,0,v)
for arm in ['ce','aux','bger','cam_bce']:
 rr=[];mm=[]
 for seed in range(3):
  key=f'{arm}_s{seed}'
  for phase in ['dev','test']:
   ranks=L(f'results/{phase}/{key}/ranking.json');rank_count+=len(ranks)
   ck(phase+'/'+key+' query order', [r['query_id'] for r in ranks]==[r['query_id'] for r in m[phase]])
   ranks_num=np.array([r['positive_rank'] for r in ranks])
   eq(phase+'/'+key+' R1 from rank',[r['correct'] for r in ranks],(ranks_num==1).astype(int))
   eq(phase+'/'+key+' R10 from rank',[r['recall10'] for r in ranks],(ranks_num<=10).astype(int))
   eq(phase+'/'+key+' AP from reciprocal rank',[r['AP'] for r in ranks],1/ranks_num)
   gallery=m['dev_gallery' if phase=='dev' else 'gallery'];truth=[r['product_id'] for r in m[phase]]
   eq(phase+'/'+key+' top-ID labels',[r['correct'] for r in ranks],[int(gallery[r['top_index']]['product_id']==t) for r,t in zip(ranks,truth)])
   with np.load(B/f'results/{phase}/{key}/metrics.npz') as z:metrics={k:z[k].copy() for k in z.files}
   raw=np.array([r['raw_box'] for r in m[phase]],dtype=float)
   for delta in [0,1]:
    gt=raw.copy();gt[:,:2]-=delta
    computed=calc_iou(metrics['pred_boxes'],gt);iou_count+=computed.size
    eq(phase+'/'+key+f' IoU C{delta} from boxes',computed,metrics['ious'][...,delta],1e-12)
    eq(phase+'/'+key+f' hits C{delta}',(computed>=.5).astype(int),metrics['hits'][...,delta])
   ck(phase+'/'+key+' all map metrics finite',np.isfinite(metrics['ious']).all() and np.isfinite(metrics['EIB']).all())
   if phase=='dev':
    with np.load(B/f'results/dev/{key}/dev_grid_own.npz') as dv:
     best=(dv['ious'][...,0]>=.5).mean(0).argmax(1)
     eq(key+' dev threshold optimal grid index',best,th[key]['indices'])
   else:rr.append(ranks);mm.append(metrics)
 arrays[arm]=mm
 correct=np.array([[r['correct'] for r in rs] for rs in rr])
 for f,name in [('correct','R1'),('recall10','R10'),('AP','MRR_mAP')]:add(f'{arm}/retrieval/{name}',[[r[f] for r in rs] for rs in rr])
 for j,ro in enumerate(['H','decomposition']+(['F'] if arm=='bger' else [])):
  for v,view in enumerate(['own','fixed_positive','fixed_negative']):
   for k,c in enumerate(['C0','C1']):
    hit=np.array([x['hits'][:,v,j,k] for x in mm]);add(f'{arm}/{ro}/{view}/Loc/{c}',hit);add(f'{arm}/{ro}/{view}/EIB/{c}',[x['EIB'][:,v,j,k] for x in mm])
    if view=='own':add(f'{arm}/{ro}/own/Joint/{c}',hit*correct)
   add(f'{arm}/{ro}/{view}/coordinate_verdict_flip',[x['hits'][:,v,j,0]!=x['hits'][:,v,j,1] for x in mm]);add(f'{arm}/{ro}/{view}/degenerate',[x['degenerate'][:,v,j] for x in mm])
  for k,c in enumerate(['C0','C1']):add(f'{arm}/{ro}/candidate_verdict_change/{c}',[x['hits'][:,1,j,k]!=x['hits'][:,2,j,k] for x in mm])
with np.load(B/'results/test/centre_prior_metrics.npz') as pr:
 for k,c in enumerate(['C0','C1']):
  for field,ep in [('hits','Loc'),('EIB','EIB')]:add('centre_prior/'+ep+'/'+c,np.tile(pr[field][None,:,k],(3,1)))
 add('centre_prior/coordinate_verdict_flip',np.tile((pr['hits'][:,0]!=pr['hits'][:,1])[None],(3,1)))
computed_table=[];skipped=[]
for r in pd.read_csv(B/'tables/per_seed_and_mean_SD.csv').to_dict('records'):
 key=r['key'];w=r['weighting']
 if key not in raw_by_arm:skipped.append({'table':'per_seed','key':key,'reason':'score shards not shipped'});continue
 a=raw_by_arm[key]
 vals=a.mean(1) if w=='query' else np.array([(np.bincount(ci,weights=x)/csize).mean() for x in a])
 eq('per_seed/'+key+'/'+w,np.r_[vals,vals.mean(),vals.std(ddof=1)],[r[k] for k in ['seed0','seed1','seed2','mean','sample_SD']])
for r in pd.read_csv(B/'tables/estimates_cluster_CI.csv').to_dict('records'):
 key=r['key'];w=r['weighting']
 if key not in series:skipped.append({'table':'estimates','key':key,'reason':'score shards not shipped'});continue
 e=estimate(series[key],w);eq('estimates/'+key+'/'+w,e,[r[k] for k in ['point','ci_low','ci_high']]);computed_table.append(dict(key=key,weighting=w,point=e[0],ci_low=e[1],ci_high=e[2]))
contrasts=[]
for r in pd.read_csv(B/'tables/paired_contrasts.csv').to_dict('records'):
 a,b=r['contrast'].split('-');ep=r['endpoint'];w=r['weighting']
 def key(a):return a.replace('bger_F','bger/F').replace('bger_H','bger/H').replace('bger_decomposition','bger/decomposition')+'/'+ep
 ka,kb=key(a),key(b)
 if ka not in series or kb not in series:skipped.append({'table':'contrasts','key':r['contrast']+'/'+ep,'reason':'score shards not shipped'});continue
 e=estimate(series[ka]-series[kb],w);eq('contrast/'+r['contrast']+'/'+ep+'/'+w,e,[r[k] for k in ['point','ci_low','ci_high']]);contrasts.append(dict(contrast=r['contrast'],endpoint=ep,weighting=w,point=e[0],ci_low=e[1],ci_high=e[2]))
for r in pd.read_csv(B/'tables/coordinate_mixture_bounds.csv').to_dict('records'):
 a,b=r['contrast'].split('-');ep=r['endpoint'];ro=r['readout'];w=r['weighting']
 d=np.stack([series[f'{a}/{ro}/own/{ep}/{c}']-series[f'{b}/{ro}/own/{ep}/{c}'] for c in ['C0','C1']],1)
 lo,hi=estimate(d.min(1),w),estimate(d.max(1),w)
 eq('mixbound/'+r['contrast']+'/'+ro+'/'+ep+'/'+w,np.r_[lo,hi],[r[k] for k in ['mixture_lower_bound','lower_estimation_ci_low','lower_estimation_ci_high','mixture_upper_bound','upper_estimation_ci_low','upper_estimation_ci_high']])
# Additional exploratory paired prior comparisons; NOT prespecified.
new=[]
for arm in ['ce','aux','bger','cam_bce']:
 for c in ['C0','C1']:
  e=estimate(series[f'{arm}/H/own/Loc/{c}']-series[f'centre_prior/Loc/{c}'])
  new.append(dict(contrast=arm+'-centre_prior',coordinate=c,point=e[0],ci_low=e[1],ci_high=e[2],scope='new descriptive review calculation, not original protocol'))
# BGER-F vs CE not present in supplied pair table, recompute from saved records.
for c in ['C0','C1']:
 e=estimate(series[f'bger/F/own/Loc/{c}']-series[f'ce/H/own/Loc/{c}'])
 new.append(dict(contrast='bger_F-ce_H',coordinate=c,point=e[0],ci_low=e[1],ci_high=e[2],scope='new descriptive review calculation, not original protocol'))
pd.DataFrame(new).to_csv(OUT/'additional_descriptive_comparisons.csv',index=False)
pd.DataFrame(contrasts).to_csv(OUT/'recomputed_paired_contrasts.csv',index=False)
pd.DataFrame(computed_table).to_csv(OUT/'recomputed_estimates.csv',index=False)
summary={'checks':len(checks),'passed':sum(x['pass'] for x in checks),'failed':[x for x in checks if not x['pass']], 'declared_hash_records':len(hashes),'ranking_rows_checked':rank_count,'pairwise_bbox_IoU_values_recomputed':iou_count,'per_seed_table_rows_checked':sum(x['check'].startswith('per_seed/') for x in checks),'estimate_table_rows_checked':len(computed_table),'contrast_rows_checked':len(contrasts),'sensitivity_bound_rows_checked':48,'skipped_rows':skipped,'scope':'Metadata, saved ranking reductions, stored predicted-box IoU, threshold locks, target tensors and conditional statistics; not re-training/inference, raw-score validation or raw-pixel audit.'}
(OUT/'verification_summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'verification_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='skipped_rows'},indent=2));print('skipped table rows',len(skipped));print(pd.DataFrame(new).to_string(index=False))
