#!/usr/bin/env python3
"""Versioned, matched CUB repair; no historical checkpoint is overwritten.

Training, validation, and GPU smoke use development data only. Evaluation is a
separate entry point. Checkpoints include optimizer/RNG state for exact resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from methods.bger import BGERResNet50, box_to_grid
from datasets.cub import CUBDataset, load_cub_index, make_cub_splits

ARMS = ('ce', 'bger', 'aux', 'residual', 'cam_bce')
OUT = ROOT / 'results/cvpr2027/cub'
CKPT = ROOT / 'checkpoints/cvpr2027/cub'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def balanced_bce(m, b):
    p = b.mean((1, 2, 3)).clamp(.02, .98)
    return (-(.5 / p) * (b * torch.log(m + 1e-6)).mean((1, 2, 3))
            - (.5 / (1-p)) * ((1-b) * torch.log(1-m + 1e-6)).mean((1, 2, 3))).mean()


class MatchedModel(BGERResNet50):
    def __init__(self, arm, pretrained=True):
        super().__init__(200, pretrained)
        self.arm = arm
        self.detach_gate = False
        if arm in ('ce', 'cam_bce'):
            self.gate.requires_grad_(False)

    def forward(self, x, return_spatial=False, target=None):
        h = self.features(x)
        m = torch.sigmoid(self.gate(h))
        route = m.detach() if self.detach_gate else m
        routed = h * route if self.arm == 'bger' else (
            h * (1 + .5 * route) if self.arm == 'residual' else h)
        logits = self.classifier(routed.mean((2, 3)))
        if not return_spatial:
            return logits
        if self.arm == 'cam_bce':
            if target is None:
                raise ValueError('Training CAM proxy requires ground-truth class')
            m = torch.sigmoid((h * self.classifier.weight[target, :, None, None]).sum(1, keepdim=True))
        return logits, m


def data():
    idx = load_cub_index(str(ROOT / 'data/CUB_200_2011'))
    splits = make_cub_splits(idx, .1, .1, 0)
    return splits, {k: CUBDataset(idx, ids, 224, 'eval') for k, ids in splits.items()}


@torch.no_grad()
def validate(model, loader, device):
    modes = {m: m.training for m in model.modules()}
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    model.eval()
    correct = total = 0
    loss_sum = 0.
    try:
        for b in loader:
            x, y = b['image'].to(device), b['label'].to(device)
            logits, m = model(x, return_spatial=True, target=y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            if model.arm != 'ce':
                loss_sum += float(balanced_bce(m, box_to_grid(b['bbox'].to(device), m.shape[-1]))) * len(y)
        for k, v in model.named_buffers():
            if not torch.equal(v, buffers[k]):
                raise RuntimeError('Validation mutated buffer ' + k)
    finally:
        for module, mode in modes.items():
            module.training = mode
    return {'accuracy': correct / total, 'spatial_loss': loss_sum / total, 'n': total}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Training/smoke must run on an allocated GPU node')
    device = 'cuda'
    torch.set_num_threads(min(4, args.workers or 1))
    seed_all(args.seed)
    splits, ds = data()
    tag = f'{args.arm}_lam{args.lam:g}_s{args.seed}'
    out = OUT / ('smoke' if args.stage == 'smoke' else 'runs') / tag
    out.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    final = CKPT / (tag + '.pt')
    last = CKPT / (tag + '.resume.pt')
    config = {'arm': args.arm, 'seed': args.seed, 'lambda': args.lam,
              'epochs': 15, 'batch_size': 64, 'precision': 'FP32-no-TF32',
              'residual_alpha': .5, 'selection': 'max validation accuracy; last tie',
              'code_sha256': file_hash(__file__),
              'split_sha256': hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()}
    if final.exists() and args.stage != 'smoke':
        saved = torch.load(final, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise RuntimeError('Existing final checkpoint configuration mismatch')
        print('COMPLETE', final, flush=True)
        return
    model = MatchedModel(args.arm, True).to(device)
    init_hash = state_hash(model.state_dict())
    generator = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                          lr=.01, momentum=.9, weight_decay=1e-4, nesterov=True)
    first_epoch, best, best_state, history = 0, -1., None, []
    if last.exists() and args.stage != 'smoke':
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise RuntimeError('Resume configuration mismatch')
        model.load_state_dict(saved['state_dict'])
        opt.load_state_dict(saved['optimizer'])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v): state[k] = v.to(device)
        first_epoch, best, best_state, history = saved['epoch'], saved['best'], saved['best_state'], saved['history']
        generator.set_state(saved['loader_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    atomic_json(out/'config.json', dict(config, init_state_sha256=init_hash,
                command=sys.argv, scheduler_job_id=os.environ.get('SLURM_JOB_ID'),
                python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), splits=splits))
    train_ds = Subset(ds['train'], range(128)) if args.stage == 'smoke' else ds['train']
    val_ds = Subset(ds['val'], range(16)) if args.stage == 'smoke' else ds['val']
    tl = DataLoader(train_ds, batch_size=8 if args.stage == 'smoke' else 64,
                    shuffle=True, generator=generator, num_workers=args.workers,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=args.workers)
    end_epoch = 1 if args.stage in ('smoke', 'pilot') else 15
    started = time.time()
    for ep in range(first_epoch, end_epoch):
        for group in opt.param_groups: group['lr'] = .005 * (1 + np.cos(np.pi * ep / 15))
        model.train()
        loss_sum, n = 0., 0
        ep_start = time.time()
        for step, b in enumerate(tl):
            x, y, bbox = b['image'].to(device), b['label'].to(device), b['bbox'].to(device)
            opt.zero_grad(set_to_none=True)
            logits, m = model(x, return_spatial=True, target=y)
            spatial = balanced_bce(m, box_to_grid(bbox, m.shape[-1])) if args.arm != 'ce' else logits.new_zeros(())
            loss = F.cross_entropy(logits, y) + args.lam * spatial
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradients')
            opt.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
            if args.stage == 'smoke' and step == 1: break
        val = validate(model, vl, device)
        if val['accuracy'] >= best:
            best = val['accuracy']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {'epoch': ep+1, 'train_loss': loss_sum/n, 'train_n': n,
               'validation': val, 'seconds': time.time()-ep_start,
               'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
        history.append(row)
        print(json.dumps(dict(tag=tag, **row)), flush=True)
        atomic_json(out/'history.json', history)
        if args.stage != 'smoke':
            atomic_save({'config': config, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
                'epoch': ep+1, 'best': best, 'best_state': best_state, 'history': history,
                'init_state_sha256': init_hash, 'loader_rng': generator.get_state(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}, last)
    if args.stage == 'train':
        atomic_save({'state_dict': best_state, 'config': config, 'history': history,
                     'best_validation_accuracy': best, 'init_state_sha256': init_hash}, final)
        atomic_json(out/'completion.json', {'checkpoint': str(final.relative_to(ROOT)),
                    'checkpoint_sha256': file_hash(final), 'seconds_this_invocation': time.time()-started,
                    'epochs': 15, 'state': 'COMPLETE'})
    elif args.stage == 'smoke':
        model.eval()
        x = next(iter(vl))['image'][:2].to(device)
        with torch.no_grad():
            normal = model(x)
            model.detach_gate = True
            detached = model(x)
        error = float((normal-detached).abs().max())
        if error != 0: raise RuntimeError('Detach changed forward logits')
        atomic_json(out/'passed.json', {'stage': 'smoke', 'forward_detach_max_error': error,
                    'finite_loss_and_gradients': True, 'validation_buffers_unchanged': True,
                    'init_state_sha256': init_hash})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lam', type=float, default=2.)
    ap.add_argument('--stage', choices=['smoke', 'pilot', 'train'], required=True)
    ap.add_argument('--workers', type=int, default=4)
    run(ap.parse_args())
