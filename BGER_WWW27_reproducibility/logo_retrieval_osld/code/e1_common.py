"""Shared serialization, locked recipe, validation-state protection."""
from pathlib import Path
from contextlib import contextmanager
import hashlib, json, datetime
import numpy as np
import torch

ARMS = ('ce','aux','bger','cam_bce','abn_box')
GRID = np.linspace(.05,.95,19)
RECIPE = dict(epochs=15, batch=64, input_size=224, interpolation='bilinear',
              antialias=True, augmentation='none', weights='IMAGENET1K_V1',
              optimizer='SGD', momentum=.9, weight_decay=1e-4, nesterov=True,
              learning_rate='.005*(1+cos(pi*epoch/15))', drop_last=True,
              dtype='float32', tf32=False, amp=False, score_epsilon=1e-12,
              score_chunk=64, score_atol=3e-6, secondary_sum_atol=3e-6,
              embedding_batch=1, dev_fraction=.15, dev_seed=0,
              threshold_grid=GRID.tolist(), inference_draws=10000,
              inference_seed=0, margin=.01, alpha=.05,
              protocol='OSLD-SingleLogo-Retrieval-v1', target_rule='any_positive_overlap_7x7',
              confirmatory_family=['H1-logo: BGER-CE R@1 > -0.01','H2-logo: BGER-CE query Loc > 0'],
              cluster_unit='query logo identity', development_split='official val')

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
def native(x):
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    return x
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(native(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)
def append_jsonl(path,row):
    with open(path,'a') as f:f.write(json.dumps(native(row),allow_nan=False)+'\n')
def digest(data):return hashlib.sha256(json.dumps(native(data),sort_keys=True).encode()).hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def buffer_hash(model):return state_hash(dict(model.named_buffers()))
@contextmanager
def evaluation(model, log=None):
    modes={m:m.training for m in model.modules()};before=buffer_hash(model)
    model.eval()
    try: yield
    finally:
        for m,mode in modes.items():m.training=mode
        after=buffer_hash(model)
        if log is not None:log('validation_state',before=before,after=after,modes_restored=all(m.training==v for m,v in modes.items()))
        if before!=after: raise RuntimeError('Validation mutated model buffers')

def log_writer(path, identity):
    def log(event,**kwargs):
        row=dict(utc=now(),identity=identity,event=event,**native(kwargs))
        with open(path,'a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    return log
