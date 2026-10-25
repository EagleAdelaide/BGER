#!/usr/bin/env python3
"""Scoped B pilot/train/neural readouts. CPU box metrics run separately."""
import argparse,copy,math,os,random,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision as tv
from b_common import *
import resources

def log(event,**kw):
    row=dict(utc=now(),event=event,**kw)
    with open(ROOT/'logs/events.jsonl','a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)
def check_code():
    verify_freeze()
    for rel,h in load(ROOT/'protocol/EXECUTION_CODE_FREEZE.json')['files'].items():
        assert sha(PROJECT/rel)==h,rel
def initial(seed,labels):
    path=ROOT/f'outputs/initial_seed{seed}.pt'
    if path.exists():return path
    init=load(ROOT/'protocol/config.json')['cached_init'];p=Path(init['path'])
    assert p.is_file() and sha(p)==init['sha256']
    os.environ['TORCH_HOME']=str(p.parents[2])
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    # Cached V1 is present and checked. Forbid any fallback model download.
    def no_download(*a,**k):raise RuntimeError('Unplanned download forbidden')
    torch.hub.download_url_to_file=no_download
    m=WebBGER(len(labels),'ce',weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
    state={k:v.detach().cpu() for k,v in m.state_dict().items()}
    torch.save(dict(state=state,seed=seed,labels=labels,state_hash=state_hash(state),cached_v1_sha256=init['sha256']),path)
    log('shared_initialization',seed=seed,path=str(path),state_hash=state_hash(state),file_sha256=sha(path))
    return path
def model(seed,arm):
    labels=load(ROOT/'protocol/labels.json');p=initial(seed,labels)
    c=torch.load(p,map_location='cpu',weights_only=False)
    m=WebBGER(len(labels),arm,weights=None);m.load_state_dict(c['state'])
    assert state_hash(m.state_dict())==c['state_hash']
    return m.cuda(),c['state_hash']
def train_inputs(ds):
    path=ROOT/'outputs/training_inputs.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    items=list(DataLoader(ds,batch_size=32,shuffle=False,num_workers=8))
    data={key:torch.cat([r[key] for r in items]) for key in ['x','label','mask','has_box','effective_target']}
    data['ids']=sum([r['id'] for r in items],[])
    assert data['ids']==[r['image_id'] for r in ds.records]
    assert torch.equal(effective_target(data['mask']),data['effective_target'])
    torch.save(data,path)
    save(ROOT/'outputs/training_input_cache_receipt.json',dict(file=str(path.relative_to(ROOT)),sha256=sha(path),
        source_records_digest=digest(ds.records),preprocessing='native deterministic OSLDImageDataset; no image modification',
        x_tensor_hash=tensor_hash(data['x']),target_tensor_hash=tensor_hash(data['effective_target']),ids=data['ids']))
    return data
def batch(data,indices):
    return [data[key][indices].cuda(non_blocking=False) for key in ['x','label','mask','has_box']]
def score(z,gallery):
    return torch.cat([((g*z).sum(1)/(g.norm(dim=1)*z.norm()+1e-12)) for g in gallery.split(64)])
def maps_pair(m,h,zother,expected,arm):
    ro=[];diagnostics=[]
    primary,_,diag=W.retrieval_gradcam_audited(h,lambda f:m.pool(f[None])[0],zother)
    F=m.pooled_features(h[None])[0].detach()
    with torch.no_grad():secondary,s=W.score_decomposition(F,zother)
    assert abs(diag['score']-expected)<=3e-6 and abs(s-expected)<=3e-6
    assert abs(float(secondary.sum())-s)<=3e-6
    ro=[primary.detach().cpu().numpy(),secondary.cpu().numpy()]
    diagnostics=[diag['score'],diag['grad_norm'],diag['signed_map_max_abs'],diag['positive_map_sum'],abs(diag['score']-expected),abs(float(secondary.sum())-s)]
    if arm=='bger':
        fm,_,fd=W.retrieval_gradcam_audited(F,lambda f:f.mean((1,2)),zother)
        assert abs(fd['score']-expected)<=3e-6
        ro.append(fm.detach().cpu().numpy())
    return np.stack(ro),np.asarray(diagnostics,dtype=np.float64)
def pilot(ds):
    data=train_inputs(ds['train']);batches=load(ROOT/'protocol/sampler_batches.json')
    indices={x:i for i,x in enumerate(data['ids'])};b=batches[0]
    ix=[indices[x] for x in b['ids']];x,y,mask,has=batch(data,ix)
    from train_web_arms import step_losses as native_loss
    rows=[]
    for arm in ARMS:
        m,init=model(0,arm);m.train();state={k:v.cpu().clone() for k,v in m.state_dict().items()}
        native,parts=native_loss(m,x,y,mask,has,0 if arm=='ce' else .5,arm)
        native.backward();grads={k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None}
        expected=float(native.detach());del native
        m.zero_grad(set_to_none=True);m.load_state_dict(state)
        actual,parts=step_loss(m,x,y,mask,has,0 if arm=='ce' else .5,arm);actual.backward()
        assert float(actual.detach())==expected
        assert all(torch.equal(p.grad,grads[k]) for k,p in m.named_parameters() if p.grad is not None)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        del grads,actual,state;m.zero_grad(set_to_none=True)
        opt=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,weight_decay=1e-4,nesterov=True)
        torch.cuda.synchronize();t=time.monotonic()
        for _ in range(4):
            loss,_=step_loss(m,x,y,mask,has,0 if arm=='ce' else .5,arm)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        torch.cuda.synchronize();stepsec=(time.monotonic()-t)/4
        m.eval();t=time.monotonic()
        with torch.no_grad():
            h=m.features(ds['dev'][0]['x'][None].cuda())[0];z=m.pool(h[None])[0]
            gh=m.features(ds['dev_gallery'][0]['x'][None].cuda())[0];zg=m.pool(gh[None])[0]
        torch.cuda.synchronize();embedsec=(time.monotonic()-t)/2
        t=time.monotonic();sc=float(score(z,zg[None])[0]);maps,diag=maps_pair(m,h,zg,sc,arm)
        torch.cuda.synchronize();mapsec=time.monotonic()-t
        rows.append(dict(arm=arm,loss_exact_native=expected,gradients_exact_native=True,step_seconds=stepsec,
                         embedding_single_seconds=embedsec,pair_maps_seconds=mapsec,score_errors=diag[-2:].tolist(),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        del m,opt;torch.cuda.empty_cache()
    hw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True)
    save(ROOT/'outputs/PILOT_RESULT.json',dict(status='CORRECTNESS_THROUGHPUT_PILOT_COMPLETED',hardware=hw,
        measurements=rows,training_steps_all_models=len(batches)*4,dev_queries=len(ds['dev']),test_queries=len(ds['test']),test_metrics_exposed=False))
def train(ds):
    data=train_inputs(ds['train']);batches=load(ROOT/'protocol/sampler_batches.json');ix={x:i for i,x in enumerate(data['ids'])}
    results=[]
    for seed in range(3):
        for arm in ARMS:
            out=ROOT/f'outputs/models/{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'TRAINING_COMPLETE.json').exists():results.append(load(out/'TRAINING_COMPLETE.json'));continue
            m,init=model(seed,arm);opt=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,weight_decay=1e-4,nesterov=True)
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            start=time.monotonic();epoch0=0;steps=0;hist=[]
            checkpoint=out/'checkpoint.pt'
            if checkpoint.exists():
                c=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert c['protocol']==ID and c['seed']==seed and c['arm']==arm
                m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);epoch0=c['epoch'];steps=c['steps'];hist=c['history']
                for st in opt.state.values():
                    for k,val in st.items():
                        if torch.is_tensor(val):st[k]=val.cuda()
            for epoch in range(epoch0,15):
                m.train();lr=.005*(1+math.cos(math.pi*epoch/15))
                for g in opt.param_groups:g['lr']=lr
                losses=[]
                for b in [b for b in batches if b['seed']==seed and b['epoch']==epoch]:
                    indices=[ix[x] for x in b['ids']];x,y,mask,has=batch(data,indices)
                    assert torch.equal(effective_target(mask).cpu(),data['effective_target'][indices])
                    loss,parts=step_loss(m,x,y,mask,has,0 if arm=='ce' else .5,arm)
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward()
                    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
                    opt.step();steps+=1;losses.append(parts)
                hist.append(dict(epoch=epoch+1,lr=lr,batches=len(losses),mean_ce=float(np.mean([p['ce'] for p in losses])),
                                 mean_box_bce=float(np.mean([p.get('box_bce',0) for p in losses]))))
                tmp=checkpoint.with_suffix('.tmp');torch.save(dict(protocol=ID,arm=arm,seed=seed,epoch=epoch+1,steps=steps,
                    initial_state_hash=init,model={k:v.detach().cpu() for k,v in m.state_dict().items()},optimizer=opt.state_dict(),history=hist),tmp);tmp.replace(checkpoint)
                log('train_epoch',arm=arm,seed=seed,**hist[-1])
            result=dict(protocol=ID,arm=arm,seed=seed,epochs=15,steps=steps,initial_state_hash=init,checkpoint=str(checkpoint.relative_to(PROJECT)),
                        checkpoint_sha256=sha(checkpoint),final_state_hash=state_hash(m.state_dict()),history=hist,wall_seconds=time.monotonic()-start,complete=True)
            save(out/'TRAINING_COMPLETE.json',result);results.append(result);del m,opt;torch.cuda.empty_cache()
    assert len(results)==12
    save(ROOT/'outputs/ALL_TRAINING_COMPLETE.json',dict(protocol=ID,completed_utc=now(),models=results))
def embed_gallery(m,ds):
    H=[];Z=[]
    with torch.no_grad():
        for item in DataLoader(ds,batch_size=1,shuffle=False,num_workers=8):
            h=m.features(item['x'].cuda())[0];z=m.pool(h[None])[0]
            H.append(h.detach());Z.append(z.detach())
    return H,torch.stack(Z)
def neural(ds,phase,limit=None):
    complete=load(ROOT/'outputs/ALL_TRAINING_COMPLETE.json');assert len(complete['models'])==12
    if phase=='test':
        lock=load(ROOT/'protocol/PRE_TEST_FREEZE.json')
        for rel,h in lock['files'].items():assert sha(PROJECT/rel)==h,rel
    qd=ds[phase];gd=ds['dev_gallery' if phase=='dev' else 'gallery']
    fixed_path=ROOT/f'outputs/{phase}/CE_SEED0_FIXED_CANDIDATES.json'
    for cell in complete['models']:
        arm,seed=cell['arm'],cell['seed'];out=ROOT/f'outputs/{phase}/{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'NEURAL_COMPLETE.json').exists():continue
        ck=PROJECT/cell['checkpoint'];assert sha(ck)==cell['checkpoint_sha256']
        m=WebBGER(len(load(ROOT/'protocol/labels.json')),arm,weights=None).cuda()
        m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']);m.eval()
        before=state_hash(m.state_dict());gh,gz=embed_gallery(m,gd)
        gids=np.array([r['image_id'] for r in gd.records]);gclasses=np.array([r['product_id'] for r in gd.records])
        fixed=load(fixed_path) if fixed_path.exists() else None
        assert fixed is not None or (arm=='ce' and seed==0)
        ranks=[];generated_fixed=[];chunk_maps=[];chunk_gmaps=[];chunk_diag=[];chunk_scores=[];chunk_z=[]
        chunks=[];start=time.monotonic();chunk_size=100
        loader=DataLoader(qd,batch_size=1,shuffle=False,num_workers=8)
        for i,item in enumerate(loader):
            if limit and i>=limit:break
            r=qd.records[i]
            with torch.no_grad():
                h=m.features(item['x'].cuda())[0].detach();z=m.pool(h[None])[0].detach();scores=score(z,gz).cpu().numpy()
            assert np.isfinite(scores).all();order=np.lexsort((gids,-scores));positive=np.flatnonzero(gclasses[order]==r['product_id']);assert len(positive)==1
            top=int(order[0]);rankpos=int(positive[0]);pos=int(order[rankpos]);neg=int(next(j for j in order if gclasses[j]!=r['product_id']))
            ranks.append(dict(query_id=r['query_id'],image_id=r['image_id'],released_class=r['product_id'],group=r['group'],
                top_gallery_id=str(gids[top]),top_index=top,score=float(scores[top]),correct=int(rankpos==0),recall10=int(rankpos<10),AP=1/(rankpos+1),positive_rank=rankpos+1))
            if fixed is None:
                f=dict(query_id=r['query_id'],positive=str(gids[pos]),negative=str(gids[neg]));generated_fixed.append(f)
                choices=[top,pos,neg]
            else:
                f=fixed['pairs'][i];assert f['query_id']==r['query_id'];choices=[top,int(np.flatnonzero(gids==f['positive'])[0]),int(np.flatnonzero(gids==f['negative'])[0])]
            qmaps=[];gmaps=[];diags=[];paircache={}
            for j in choices:
                if j not in paircache:
                    a,diag=maps_pair(m,h,gz[j],float(scores[j]),arm)
                    b,gdiag=maps_pair(m,gh[j],z,float(scores[j]),arm)
                    paircache[j]=(a,b,np.stack([diag,gdiag]))
                a,b,diag=paircache[j];qmaps.append(a);gmaps.append(b);diags.append(diag)
            chunk_maps.append(np.stack(qmaps));chunk_gmaps.append(np.stack(gmaps));chunk_diag.append(np.stack(diags));chunk_scores.append(scores);chunk_z.append(z.cpu().numpy())
            if (i+1)%chunk_size==0 or i+1==len(qd):
                num=len(chunks);path=out/f'neural_{num:04d}.npz'
                np.savez_compressed(path,query_maps=np.stack(chunk_maps),gallery_maps=np.stack(chunk_gmaps),diagnostics=np.stack(chunk_diag),
                    scores=np.stack(chunk_scores),query_embeddings=np.stack(chunk_z))
                chunks.append(dict(path=str(path.relative_to(PROJECT)),sha256=sha(path),n_queries=len(chunk_maps)))
                chunk_maps=[];chunk_gmaps=[];chunk_diag=[];chunk_scores=[];chunk_z=[]
                log('neural_chunk',phase=phase,arm=arm,seed=seed,queries=i+1,total=len(qd),wall_seconds=time.monotonic()-start)
        assert len(ranks)==len(qd)
        if fixed is None:
            save(fixed_path,dict(protocol=ID,phase=phase,reference='CE seed0',created_utc=now(),pairs=generated_fixed,
                rule='highest exact cosine relevant/nonrelevant; score-descending gallery-id ties',saved_before_other_models_maps=True))
        assert before==state_hash(m.state_dict())
        save(out/'ranking.json',ranks)
        np.save(out/'gallery_embeddings.npy',gz.cpu().numpy())
        save(out/'NEURAL_COMPLETE.json',dict(protocol=ID,phase=phase,arm=arm,seed=seed,checkpoint_sha256=cell['checkpoint_sha256'],
            n_queries=len(qd),n_gallery=len(gd),query_records_digest=digest(qd.records),gallery_records_digest=digest(gd.records),
            views=['own_top1','CEseed0_positive','CEseed0_negative'],readouts=['H','decomposition']+(['F'] if arm=='bger' else []),
            sides=['query','gallery'],ranking_sha256=sha(out/'ranking.json'),fixed_pairs_sha256=sha(fixed_path),chunks=chunks,
            eval_state_unchanged=True,completed_utc=now(),wall_seconds=time.monotonic()-start))
        del m,gh,gz;torch.cuda.empty_cache()
    save(ROOT/f'outputs/{phase}/ALL_NEURAL_COMPLETE.json',dict(protocol=ID,phase=phase,completed_utc=now(),cells=12))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['pilot','train','dev','test']);ap.add_argument('--key',required=True);a=ap.parse_args()
    resources.start(a.key);start=time.monotonic();status='FAILED'
    try:
        setup()
        count=torch.cuda.device_count()
        hardware=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True)
        log('cuda_visibility',visible_count=count,CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),
            SLURM_JOB_GPUS=os.getenv('SLURM_JOB_GPUS'),hardware=hardware.stdout.strip(),hardware_error=hardware.stderr.strip())
        assert count==1
        check_code();ds=datasets()
        if a.phase=='pilot':pilot(ds)
        elif a.phase=='train':train(ds)
        else:neural(ds,a.phase)
        status='COMPLETED'
    except Exception:
        log('FAILURE',phase=a.phase,traceback=traceback.format_exc());raise
    finally:resources.finish(a.key,status,time.monotonic()-start)
if __name__=='__main__':main()
