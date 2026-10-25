#!/usr/bin/env python3
"""Source-faithful pixel localization, same saved maps under C0/C1.

CPU-only scheduler job, independently parallelized from GPU neural inference.
"""
import argparse,os,time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from b_common import *

STRUCTURE=np.array([[0,1,0],[1,1,1],[0,1,0]])
def initializer():torch.set_num_threads(1)
def centre_prior():
    y,x=np.mgrid[0:7,0:7]
    return np.exp(-.5*(((y-3)/(7/4))**2+((x-3)/(7/4))**2)).astype(np.float32)
def largest_box(cam,t):
    lab,n=ndimage.label(cam>=t,structure=STRUCTURE)
    if not n:return [-1,-1,-1,-1]
    counts=np.bincount(lab.ravel());counts[0]=0
    y,x=np.where(lab==counts.argmax())
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
def iou(pred,gt):return 0. if pred[0]<0 else W.iou(pred,gt)
def evaluate_map(signed,r,indices):
    resized=F.interpolate(torch.from_numpy(signed.copy()).clamp_min(0)[None,None],size=(r['height'],r['width']),mode='bilinear',align_corners=False)[0,0].numpy().astype(np.float64)
    cam,degen=W.normalize_map(resized)
    boxes=np.array([[-1]*4 if degen else largest_box(cam,float(GRID[j])) for j in indices],dtype=np.int32)
    gt=[box(r['raw_box'],k) for k in [0,1]]
    ious=np.array([[iou(p,b) for b in gt] for p in boxes],dtype=np.float64)
    energy=float(cam.sum());eib=[]
    for b in gt:
        x,y,u,v=map(int,b)
        eib.append(0. if degen or energy==0 else float(cam[y:v,x:u].sum()/energy))
    return boxes,ious,np.array(eib),bool(degen),bool(not np.isfinite(resized).all())
def own_task(args):
    r,maps=args
    outs=[evaluate_map(m,r,list(range(19))) for m in maps]
    return np.stack([x[0] for x in outs]),np.stack([x[1] for x in outs]),np.stack([x[2] for x in outs]),np.array([x[3] for x in outs]),np.array([x[4] for x in outs])
def fixed_task(args):
    r,maps,indices=args
    outs=[[evaluate_map(m,r,[indices[j]]) for j,m in enumerate(view)] for view in maps]
    return (np.array([[x[0][0] for x in view] for view in outs]),np.array([[x[1][0] for x in view] for view in outs]),
            np.array([[x[2] for x in view] for view in outs]),np.array([[x[3] for x in view] for view in outs]),
            np.array([[x[4] for x in view] for view in outs]))
def diag_maps(maps):
    # Shape/correlation of positive native7x7 maps, not spatial supervision maps.
    pos,neg=maps[:,1],maps[:,2]
    pp=np.maximum(pos,0).astype(np.float64);pn=np.maximum(neg,0).astype(np.float64)
    ps=pp.sum((-2,-1));ns=pn.sum((-2,-1));den=(ps>1e-12)&(ns>1e-12)
    l1=np.full(ps.shape,np.nan);l1[den]=np.abs(pp[den]/ps[den,None,None]-pn[den]/ns[den,None,None]).sum((-2,-1))
    a=pp-pp.mean((-2,-1),keepdims=True);b=pn-pn.mean((-2,-1),keepdims=True)
    aa=np.sqrt((a*a).sum((-2,-1)));bb=np.sqrt((b*b).sum((-2,-1)))
    good=(pp.std((-2,-1))>1e-12)&(pn.std((-2,-1))>1e-12)
    corr=np.full(ps.shape,np.nan);corr[good]=(a*b).sum((-2,-1))[good]/(aa*bb)[good]
    return dict(candidate_positive_sum=ps,candidate_negative_sum=ns,candidate_positive_max_abs=np.abs(pos).max((-2,-1)),
                candidate_negative_max_abs=np.abs(neg).max((-2,-1)),candidate_normalized_l1=l1,candidate_correlation=corr,
                candidate_undefined_correlation=~good)
def load_maps(path):
    done=load(path/'NEURAL_COMPLETE.json');arrays=[];scores=[];emb=[];diags=[]
    assert sha(path/'ranking.json')==done['ranking_sha256']
    for c in done['chunks']:
        p=PROJECT/c['path'];assert sha(p)==c['sha256']
        with np.load(p) as n:
            arrays.append(n['query_maps']);scores.append(n['scores']);emb.append(n['query_embeddings']);diags.append(n['diagnostics'])
    maps=np.concatenate(arrays);scores=np.concatenate(scores);emb=np.concatenate(emb);diags=np.concatenate(diags)
    assert len(maps)==done['n_queries']
    # Verify stored ranking reductions and readout diagnostics at sampled rows.
    gz=torch.from_numpy(np.load(path/'gallery_embeddings.npy'))
    for i in np.linspace(0,len(maps)-1,10,dtype=int):
        z=torch.from_numpy(emb[i]);recon=torch.cat([((g*z).sum(1)/(g.norm(dim=1)*z.norm()+1e-12)) for g in gz.split(64)]).numpy()
        assert np.max(np.abs(recon-scores[i]))<=3e-6
    assert np.isfinite(scores).all() and np.nanmax(diags[...,4:])<=3e-6
    return maps,done
def save_npz(path,**arrays):
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['dev','test']);ap.add_argument('--workers',type=int,default=16);a=ap.parse_args()
    setup();m=load(ROOT/'protocol/manifests.json');qr=m[a.phase]
    thresholds=load(ROOT/'outputs/dev/thresholds.json') if a.phase=='test' else {}
    start=time.monotonic();complete=load(ROOT/'outputs/ALL_TRAINING_COMPLETE.json')
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initializer) as pool:
        for cell in complete['models']:
            arm,seed=cell['arm'],cell['seed'];key=f'{arm}_s{seed}';path=ROOT/f'outputs/{a.phase}/{key}'
            output=path/'metrics.npz';marker=path/'METRICS_COMPLETE.json'
            # CPU pixel metrics may overlap later GPU cells, but only consume
            # an atomically completed, hash-bound neural cell.
            wait_start=time.monotonic()
            while not (path/'NEURAL_COMPLETE.json').exists():
                if time.monotonic()-wait_start>3600:
                    raise RuntimeError('Completed neural cell unavailable after1h: '+key)
                time.sleep(5)
            if marker.exists():
                if a.phase=='dev':thresholds[key]=load(marker)['thresholds']
                continue
            maps,done=load_maps(path);nr=maps.shape[2]
            if a.phase=='dev':
                values=list(pool.map(own_task,[(r,maps[i,0]) for i,r in enumerate(qr)],chunksize=8))
                ownboxes=np.stack([x[0] for x in values]);ownious=np.stack([x[1] for x in values]);owneib=np.stack([x[2] for x in values])
                owndeg=np.stack([x[3] for x in values]);ownnon=np.stack([x[4] for x in values])
                indices=(ownious[:,:,:,0]>=.5).mean(0).argmax(1)
                thresholds[key]=dict(indices=indices.tolist(),values=[float(GRID[j]) for j in indices],readouts=done['readouts'],
                                     rule='C0 development own-top1 IoU>=.5, maximum count, first grid point on ties')
                vals=list(pool.map(fixed_task,[(r,maps[i,1:],indices) for i,r in enumerate(qr)],chunksize=8))
                pred=np.empty((len(qr),3,nr,4),np.int32);ious=np.zeros((len(qr),3,nr,2));eib=np.zeros_like(ious)
                degen=np.zeros((len(qr),3,nr),bool);non=np.zeros_like(degen)
                for j,idx in enumerate(indices):
                    pred[:,0,j]=ownboxes[:,j,idx];ious[:,0,j]=ownious[:,j,idx];eib[:,0,j]=owneib[:,j];degen[:,0,j]=owndeg[:,j];non[:,0,j]=ownnon[:,j]
                for i,val in enumerate(vals):pred[i,1:],ious[i,1:],eib[i,1:],degen[i,1:],non[i,1:]=val
                save_npz(path/'dev_grid_own.npz',boxes=ownboxes,ious=ownious)
            else:
                indices=thresholds[key]['indices']
                vals=list(pool.map(fixed_task,[(r,maps[i],indices) for i,r in enumerate(qr)],chunksize=8))
                pred=np.stack([x[0] for x in vals]);ious=np.stack([x[1] for x in vals]);eib=np.stack([x[2] for x in vals]);degen=np.stack([x[3] for x in vals]);non=np.stack([x[4] for x in vals])
            save_npz(output,pred_boxes=pred,ious=ious,hits=(ious>=.5).astype(np.uint8),EIB=eib,degenerate=degen,nonfinite=non,**diag_maps(maps))
            save(marker,dict(protocol=ID,phase=a.phase,arm=arm,seed=seed,n_queries=len(qr),thresholds=thresholds[key],
                C0_C1_same_neural_maps=True,thresholds_retuned_C1=False,metrics_sha256=sha(output),neural_manifest_sha256=sha(path/'NEURAL_COMPLETE.json'),completed_utc=now()))
            print(json.dumps(dict(event='metrics_complete',phase=a.phase,cell=key,elapsed=time.monotonic()-start)),flush=True)
        priorpath=ROOT/f'outputs/{a.phase}/centre_prior_metrics.npz'
        if not priorpath.exists():
            if a.phase=='dev':
                vals=list(pool.map(own_task,[(r,np.stack([centre_prior()])) for r in qr],chunksize=8))
                ious=np.stack([v[1][0] for v in vals]);idx=int((ious[:,:,0]>=.5).mean(0).argmax())
                thresholds['centre_prior']=dict(index=idx,value=float(GRID[idx]),rule='C0 development own-query IoU>=.5 grid19 tie lowest')
                pred=np.stack([v[0][0,idx] for v in vals]);eib=np.stack([v[2][0] for v in vals]);ious=ious[:,idx]
            else:
                idx=thresholds['centre_prior']['index'];vals=list(pool.map(fixed_task,[(r,np.stack([[centre_prior()]]),[idx]) for r in qr],chunksize=8))
                pred=np.stack([v[0][0,0] for v in vals]);ious=np.stack([v[1][0,0] for v in vals]);eib=np.stack([v[2][0,0] for v in vals])
            save_npz(priorpath,pred_boxes=pred,ious=ious,hits=(ious>=.5).astype(np.uint8),EIB=eib)
    if a.phase=='dev':
        save(ROOT/'outputs/dev/thresholds.json',thresholds)
        paths=[ROOT/'outputs/dev/thresholds.json',ROOT/'outputs/ALL_TRAINING_COMPLETE.json',ROOT/'protocol/PRE_GPU_FREEZE.json',ROOT/'protocol/EXECUTION_CODE_FREEZE.json']
        paths+=list((ROOT/'scripts').glob('*.py'))
        files={str(p.relative_to(PROJECT)):sha(p) for p in paths}
        files.update({cell['checkpoint']:cell['checkpoint_sha256'] for cell in complete['models']})
        save(ROOT/'protocol/PRE_TEST_FREEZE.json',dict(protocol=ID,locked_utc=now(),threshold_selection='C0 derived development only',
            C1_retuning=False,no_test_outputs=True,files=files))
    save(ROOT/f'outputs/{a.phase}/ALL_METRICS_COMPLETE.json',dict(protocol=ID,phase=a.phase,cells=12,completed_utc=now(),
        node=os.uname().nodename,slurm_job_id=os.getenv('SLURM_JOB_ID'),wall_seconds=time.monotonic()-start))
if __name__=='__main__':main()
