"""Actual optimization and resumable epoch/step state; no epoch selection."""
from pathlib import Path
import math,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from train_web_arms import WebBGER,step_losses,build_model
from e1_common import RECIPE,state_hash,save,sha,digest


def init_template(n_ids,seed,out,weights,log):
    path=out/f'initial_seed{seed}.pt'
    if not path.exists():
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        model=WebBGER(n_ids,'ce',weights=weights)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save(dict(state=state,seed=seed,n_ids=n_ids),path)
        log('common_initial_state_created',seed=seed,state_hash=state_hash(state))
    return path

def model_from(path,arm,device):
    c=torch.load(path,map_location='cpu',weights_only=False)
    state=c.get('model',c.get('state'))
    if arm=='abn_box' and 'model' not in c:
        # common per-seed template: shared tensors loaded, attention branch locked-seed init
        from abn_box import load_common_template
        model=build_model(state['head.weight'].shape[0],arm,weights=None,init_seed=int(c['seed']))
        load_common_template(model,state)
    else:
        model=build_model(state['head.weight'].shape[0],arm,weights=None);model.load_state_dict(state)
    return model.to(device)

def train(arm,lam,seed,template,dataset,out,settings,log,identity,stop_after_steps=None):
    out.mkdir(parents=True,exist_ok=True);checkpoint=out/'checkpoint.pt'
    device=settings['device'];epochs=settings['epochs'];max_steps=settings['max_steps'];batch=settings['batch']
    signature=digest(dict(arm=arm,lam=lam,seed=seed,template=sha(template),recipe=RECIPE,settings=settings,
                          input_ids=[(r['image_id'],r['sha256']) for r in dataset.records]))
    model=model_from(template,arm,device);initial_hash=state_hash(model.state_dict())
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=1e-4,nesterov=True)
    epoch=offset=steps=0;order_log=[];zero_targets=0
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    augmentation_rng=torch.Generator().manual_seed(seed+300000)
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if c['signature']!=signature:raise ValueError('Resume configuration/input identity mismatch')
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer'])
        for st in opt.state.values():
            for k,v in st.items():
                if torch.is_tensor(v):st[k]=v.to(device)
        epoch,offset,steps=c['epoch'],c['offset'],c['steps'];order_log=c['sample_order'];zero_targets=c['zero_grid_targets']
        torch.set_rng_state(c['torch_rng']);np.random.set_state(c['numpy_rng']);random.setstate(c['python_rng']);augmentation_rng.set_state(c['augmentation_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(c['cuda_rng'])
        log('checkpoint_resumed',arm=arm,seed=seed,epoch=epoch,offset=offset,optimizer_steps=steps)
    def checkpoint_save():
        c=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),
               epoch=epoch,offset=offset,steps=steps,signature=signature,sample_order=order_log,
               zero_grid_targets=zero_targets,initial_state_hash=initial_hash,
               torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
               augmentation_rng=augmentation_rng.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],
               sampler_rule='seed+100000+epoch; independent Generator; deterministic input transforms',identity=identity)
        tmp=checkpoint.with_suffix('.tmp');torch.save(c,tmp);tmp.replace(checkpoint)
        save(out/'training_state.json',dict(identity=identity,arm=arm,lam=lam,seed=seed,completed_epochs=epoch,
                next_batch=offset,optimizer_steps=steps,zero_grid_targets=zero_targets,initial_state_hash=initial_hash,
                sample_order=order_log,checkpoint_sha256=sha(checkpoint),complete=epoch==epochs))
    start=time.monotonic();newsteps=0
    every=int(settings.get('checkpoint_every',200));timing=bool(settings.get('timing',False))
    step_times=[];data_times=[];sync=(lambda:torch.cuda.synchronize()) if device=='cuda' else (lambda:None)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    while epoch<epochs:
        model.train();gen=torch.Generator().manual_seed(seed+100000+epoch)
        # Workers only parallelize file I/O/decoding; order comes from `gen`, transforms are deterministic.
        loader=DataLoader(dataset,batch_size=batch,shuffle=True,generator=gen,num_workers=int(settings.get('loader_workers',0)),
                          pin_memory=(device=='cuda'),drop_last=True)
        n_batches=len(loader) if max_steps is None else min(max_steps,len(loader))
        if n_batches<1:raise ValueError('No complete training batch')
        lr=.005*(1+math.cos(math.pi*epoch/15))
        for g in opt.param_groups:g['lr']=lr
        tick=time.perf_counter()
        for k,item in enumerate(loader):
            if k<offset:tick=time.perf_counter();continue
            if k>=n_batches:break
            got=time.perf_counter()
            x=item['x'].to(device);y=item['label'].to(device);mask=item['mask'].to(device);has=item['has_box'].to(device)
            if not torch.isfinite(x).all():raise ValueError('Nonfinite training input')
            loss,parts=step_losses(model,x,y,mask,has,lam,arm)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            # Existing Web adaptive-average >.5 target, never force positive pixels.
            grid=(F.adaptive_avg_pool2d(mask[:,None],(7,7))[:,0]>.5)
            zero_targets+=int(((grid.flatten(1).sum(1)==0)&has).sum())  # boxed images only
            opt.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite training gradient')
            opt.step();steps+=1;newsteps+=1;offset=k+1;order_log.append(dict(epoch=epoch,batch=k,image_ids=item['id']))
            if timing:sync()
            done=time.perf_counter();data_times.append(got-tick);step_times.append(done-got)
            if steps%50==0 or timing or settings.get('device')=='cpu':
                log('optimizer_step',arm=arm,seed=seed,lam=lam,epoch=epoch+1,batch=k,lr=lr,loss=parts,zero_grid_targets=zero_targets)
            # Full state every `every` steps and at epoch end (per-step saving is infeasible at scale).
            if steps%every==0:checkpoint_save()
            tick=time.perf_counter()
            if stop_after_steps is not None and newsteps>=stop_after_steps:
                return checkpoint,False
        epoch+=1;offset=0;checkpoint_save()
    peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
    if newsteps:
        # One timing file per call that trained; a resumed call never overwrites earlier timing.
        tpath=out/'timing.json'
        if tpath.exists():tpath=out/f'timing_call{len(list(out.glob("timing*.json")))+1}.json'
        save(tpath,dict(identity=identity,arm=arm,lam=lam,seed=seed,steps_this_call=newsteps,batch=batch,
             step_seconds=step_times,data_wait_seconds=data_times,peak_cuda_bytes=peak,checkpoint_every=every,
             wall_seconds=time.monotonic()-start))
    log('training_complete',arm=arm,seed=seed,lam=lam,epochs=epochs,steps=steps,steps_this_call=newsteps,elapsed_cpu_or_wall_seconds=time.monotonic()-start,
        checkpoint_sha256=sha(checkpoint),initial_state_hash=initial_hash,peak_cuda_bytes=peak)
    return checkpoint,True
