"""B-derived experiment primitives. No inherited OSLD hypothesis/config gates."""
from pathlib import Path
import datetime, hashlib, json, sys, os
import numpy as np
import torch
import torch.nn.functional as TF

ROOT=Path(os.environ.get('OPENLOGO_RUN_ROOT',Path(__file__).resolve().parents[1])).resolve()
PROJECT=Path(os.environ.get('OPENLOGO_DATA_ROOT',ROOT/'UNSUPPLIED_LICENSED_DATA')).resolve()
NATIVE=Path(__file__).resolve().parent/'native'
sys.path.insert(0,str(NATIVE))
from osld_data import OSLDImageDataset
from train_web_arms import WebBGER,balanced_bce
import web_eval as W

ID='OpenLogo-Derived-ConventionStable-B-v1'
ARMS=('ce','aux','bger','cam_bce')
GRID=np.linspace(.05,.95,19)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def box(raw,k):
    x,y,u,v=raw
    return [x-int(k),y-int(k),u,v]
def valid_box(b,w,h):return bool(np.isfinite(b).all() and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h)
def effective_target(mask,shape=(7,7)):
    # EXACT native loss transformation, extracted as the actual B loss adapter.
    return (TF.adaptive_avg_pool2d(mask[:,None],shape)[:,0]>.5).float()
def balanced_weights(target):
    p=target.sum().clamp_min(1.);n=(1-target).sum().clamp_min(1.)
    return torch.where(target>.5,(p+n)/(2*p),(p+n)/(2*n))
def step_loss(model,x,y,mask,has,lam,arm):
    logits,H,z=model(x)
    loss=TF.cross_entropy(logits,y);parts={'ce':float(loss.detach())}
    if arm!='ce' and bool(has.any()):
        target=effective_target(mask,H.shape[-2:])[has]
        if arm in ('aux','bger'):sp=model.gate(H)[:,0]
        elif arm=='cam_bce':sp=torch.einsum('bc,bchw->bhw',model.head.weight[y],H)
        else:raise ValueError(arm)
        sup=balanced_bce(sp[has],target);loss=loss+lam*sup;parts['box_bce']=float(sup.detach())
    return loss,parts
class ProductionDataset(OSLDImageDataset):
    def __init__(self,records,labels=None):super().__init__(PROJECT,records,labels)
    def __getitem__(self,i):
        item=super().__getitem__(i)
        # This final target adapter is also called by the actual production loss.
        item['effective_target']=effective_target(item['mask'][None])[0]
        return item
def datasets():
    m=load(ROOT/'protocol/manifests.json');labels=load(ROOT/'protocol/labels.json')
    return {k:ProductionDataset(v,labels) for k,v in m.items()}
def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def verify_freeze():
    f=load(ROOT/'protocol/PRE_GPU_FREEZE.json')
    for rel,h in f['files'].items():
        if sha(PROJECT/rel)!=h:raise RuntimeError('Frozen dependency changed: '+rel)
    return f
