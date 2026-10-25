#!/usr/bin/env python3
"""P0-C: Proxy Anchor metric-learning reference on the identical OSLD protocol (exploratory, post-test).

Recipe fixed before any run (official Proxy-Anchor-CVPR2020 defaults for ResNet-50, MIT licence;
re-implemented here, deviations disclosed):
  backbone ResNet-50 ImageNet V1 (as the five arms), BN frozen (official --bn-freeze 1);
  embedding = Linear(2048 -> 512) of (GAP + GMP), L2-normalized; proxies: one per training identity
  (3,516), Kaiming-normal; Proxy Anchor loss alpha=32, margin=0.1; AdamW lr 1e-4 (proxies x100),
  weight decay 1e-4, gradient clip 10, StepLR(step 5, gamma .5), batch 120 random (drop_last),
  60 epochs, 5 warm-up epochs with the backbone frozen; seeds 0-2.
  Deviations: inputs are the study's deterministic whole-image 224x224 resize for training and
  evaluation (no RandomResizedCrop/flip, no centre crop: crops remove logos and evaluation must
  use the full product image); epoch selection by official-validation Recall@1 every 10 epochs
  (official code selects on the test split); no test-set selection.
Evaluation: identical queries/galleries/relevance; FP32 stabilized cosine ranking; primary
readout = Grad-CAM of the deployed cosine at the stage-4 feature map through the real graph
(GAP+GMP -> linear -> L2 norm) for own top-1 and the CE-seed-0 fixed pairs; threshold locked on
validation (19-point grid). No forward decomposition (undefined under max pooling).
This is a REFERENCE with its own recipe, not a single-factor matched control.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv

HERE = Path(__file__).resolve().parent
E1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(E1 / 'code')); sys.path.insert(0, str(HERE.parent / 'common'))
from e1_common import GRID, save, sha, digest, now, log_writer, evaluation  # noqa: E402
from e1_evaluate import rank, map_metrics  # noqa: E402
import web_eval as W  # noqa: E402
from osld_data import build_sets, OSLDImageDataset  # noqa: E402

RECIPE_PA = dict(embedding=512, alpha=32.0, margin=0.1, lr=1e-4, proxy_lr_mult=100, weight_decay=1e-4, clip=10.0, lr_step=5, lr_gamma=0.5,
                 epochs=60, batch=120, warm=5, bn_freeze=True, pooling='GAP+GMP', input='study 224 whole image, no augmentation',
                 select='val R@1 every 10 epochs (incl. final)', source='sung-yeon-kim/Proxy-Anchor-CVPR2020 defaults (MIT)')


class ProxyAnchorNet(nn.Module):
    def __init__(self, weights):
        super().__init__()
        m = tv.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.embedding = nn.Linear(2048, RECIPE_PA['embedding'])
        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out'); nn.init.zeros_(self.embedding.bias)
        for mod in self.backbone.modules():
            if isinstance(mod, nn.BatchNorm2d): mod.weight.requires_grad_(False); mod.bias.requires_grad_(False)
    def features(self, x): return self.backbone(x)
    def pooled_features(self, H): return H
    def pool(self, H):  # real deployed graph: (GAP + GMP) -> linear -> L2 normalize
        z = self.embedding(H.mean((2, 3)) + H.amax((2, 3)))
        return F.normalize(z, dim=1)
    def forward(self, x):
        H = self.features(x); z = self.pool(H); return None, H, z
    def train(self, mode=True):
        super().train(mode)
        if mode:
            for mod in self.backbone.modules():
                if isinstance(mod, nn.BatchNorm2d): mod.eval()
        return self


class ProxyAnchorLoss(nn.Module):
    def __init__(self, n_classes, dim, margin, alpha):
        super().__init__(); self.proxies = nn.Parameter(torch.randn(n_classes, dim)); nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.n, self.mrg, self.alpha = n_classes, margin, alpha
    def forward(self, X, T):
        P = F.normalize(self.proxies, dim=1); cos = F.linear(X, P)
        P_one = F.one_hot(T, self.n).float(); N_one = 1 - P_one
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg)); neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        with_pos = torch.nonzero(P_one.sum(0) != 0).squeeze(1); num_valid = len(with_pos)
        P_sum = torch.where(P_one == 1, pos_exp, torch.zeros_like(pos_exp)).sum(0); N_sum = torch.where(N_one == 1, neg_exp, torch.zeros_like(neg_exp)).sum(0)
        return torch.log(1 + P_sum).sum() / num_valid + torch.log(1 + N_sum).sum() / self.n


@torch.no_grad()
def embed(model, ds, device, workers):
    out = []
    for item in torch.utils.data.DataLoader(ds, batch_size=64, num_workers=workers, shuffle=False, collate_fn=lambda b: dict(x=torch.stack([i['x'] for i in b]))):
        _, _, z = model(item['x'].to(device)); out.append(z.float().cpu())
    return torch.cat(out)


def readout_pairs(model, qd, gd, qz, gz, pairs, device, workers, out_npz):
    """Grad-CAM of the deployed cosine at H for (query, gallery, view) pairs; returns rows + maps."""
    gi = {r['image_id']: i for i, r in enumerate(gd.records)}; by = {}
    for q, g, v in pairs: by.setdefault(q, []).append((g, v))
    rows, maps = [], {}
    loader = torch.utils.data.DataLoader(qd, batch_size=1, shuffle=False, num_workers=workers, collate_fn=lambda b: b[0])
    with evaluation(model, None):
        for i, item in enumerate(loader):
            q = item['id']
            if q not in by: continue
            with torch.no_grad(): H = model.features(item['x'][None].to(device))[0]
            for g, v in by[q]:
                zd = gz[gi[g]].to(device)
                signed, _, rec = W.retrieval_gradcam_audited(H, lambda f: model.pool(f[None])[0], zd)
                mm = map_metrics(signed, qd.records[i])
                rows.append(dict(query_id=q, gallery_id=g, view=v, score=rec['score'], hits=mm['hits'], EIB=mm['EIB'], degenerate=mm['degenerate'], nonfinite=mm['nonfinite'],
                                 raw_max_abs=rec['signed_map_max_abs'] if np.isfinite(rec['signed_map_max_abs']) else None))
                maps[digest([q, g, v])] = signed.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(out_npz, **maps)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', required=True); ap.add_argument('--data-root', required=True); ap.add_argument('--records', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--no-reserve', action='store_true'); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    raise PermissionError('V2 release: fresh inference/training is not authorized. This archive is for saved-record CPU reproduction; original allocation guards remain below.')
    run = Path(args.run_root); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); root = Path(args.data_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; reserved = 0
    if not args.no_reserve:
        from revision_guard import reserve; reserved = reserve('p0c_proxy_anchor')
    identity = 'OSLD_P0C_PROXY_ANCHOR_REFERENCE'; log = log_writer(out / 'events.jsonl', identity)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    scanned = json.loads(Path(args.records).read_text()); freeze = json.loads((run / 'pretest_freeze.json').read_text())
    if digest(scanned['product_records'] + scanned['logo_records']) != freeze['records_digest']: raise PermissionError('records digest != freeze')
    built = build_sets(scanned); sets, labels = built['sets'], built['labels']
    if digest(labels) != freeze['label_map_digest']: raise PermissionError('label map differs')
    fixed = json.loads((run / 'fixed_candidate_manifest.json').read_text())['pairs']
    if args.smoke:   # tiny CPU plumbing check: few queries, galleries restricted to their relevant templates + 60 others
        for qk, gk in [('dev_query', 'dev_gallery'), ('query', 'gallery')]:
            sets[qk] = sorted(sets[qk], key=lambda r: r['image_id'])[:12]; ids = {r['product_id'] for r in sets[qk]}
            rel = [r for r in sets[gk] if r['product_id'] in ids]; oth = [r for r in sorted(sets[gk], key=lambda r: r['image_id']) if r['product_id'] not in ids][:60]
            sets[gk] = rel + oth
        qids = {r['image_id'] for r in sets['query']}; gids = {r['image_id'] for r in sets['gallery']}
        fixed = [f for f in fixed if f['query_id'] in qids and f['positive'] in gids and f['negative'] in gids]
        sets['train'] = sets['train'][:240]
    ds = {k: OSLDImageDataset(root, v, labels if k == 'train' else None) for k, v in sets.items()}
    R = dict(RECIPE_PA); epochs = 2 if args.smoke else R['epochs']
    reg = dict(identity=identity, recipe=R, created_utc=now(), reserved_gpu_seconds=reserved, slurm_job_id=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__,
               device=device, records_digest=freeze['records_digest'], fixed_manifest_sha256=sha(run / 'fixed_candidate_manifest.json'), seeds={})
    rp = out / 'EXPERIMENT_RECORD.json'
    if rp.exists(): reg['seeds'] = json.loads(rp.read_text()).get('seeds', {})
    for seed in [int(s) for s in args.seeds.split(',')]:
        sd = out / f'seed{seed}'; sd.mkdir(exist_ok=True)
        if (sd / 'test_summary.json').exists(): log('seed_reused', seed=seed); continue
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = ProxyAnchorNet(tv.models.ResNet50_Weights.IMAGENET1K_V1 if not args.smoke else None).to(device)
        crit = ProxyAnchorLoss(len(labels), R['embedding'], R['margin'], R['alpha']).to(device)
        opt = torch.optim.AdamW([dict(params=[p for n, p in model.named_parameters() if not n.startswith('embedding') and p.requires_grad], lr=R['lr']),
                                 dict(params=model.embedding.parameters(), lr=R['lr']), dict(params=crit.parameters(), lr=R['lr'] * R['proxy_lr_mult'])], weight_decay=R['weight_decay'])
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=R['lr_step'], gamma=R['lr_gamma'])
        best = None; t0 = time.monotonic(); steps = 0
        for epoch in range(epochs):
            frozen = epoch < R['warm']
            for mod in model.backbone.modules():
                for p in mod.parameters(recurse=False): p.requires_grad_(not frozen and not isinstance(mod, nn.BatchNorm2d))
            model.train(); gen = torch.Generator().manual_seed(seed + 500000 + epoch)
            loader = torch.utils.data.DataLoader(ds['train'], batch_size=R['batch'], shuffle=True, generator=gen, num_workers=args.workers, drop_last=True, pin_memory=device == 'cuda',
                                                 collate_fn=lambda b: dict(x=torch.stack([i['x'] for i in b]), label=torch.tensor([i['label'] for i in b])))
            losses = []
            for k, item in enumerate(loader):
                if args.smoke and k >= 2: break
                _, _, z = model(item['x'].to(device)); loss = crit(z, item['label'].to(device))
                if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_value_(list(model.parameters()) + list(crit.parameters()), R['clip']); opt.step(); steps += 1; losses.append(float(loss))
            sched.step(); log('epoch', seed=seed, epoch=epoch + 1, loss=float(np.mean(losses)), frozen_backbone=frozen, lr=sched.get_last_lr()[0], elapsed=time.monotonic() - t0)
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                model.eval(); qz, gz = embed(model, ds['dev_query'], device, args.workers), embed(model, ds['dev_gallery'], device, args.workers)
                r1 = float(np.mean([r['correct'] for r in rank(qz, gz, ds['dev_query'].records, ds['dev_gallery'].records)]))
                log('val', seed=seed, epoch=epoch + 1, recall1=r1)
                if best is None or r1 > best[0]:
                    best = (r1, epoch + 1); torch.save(dict(model=model.state_dict(), proxies=crit.state_dict(), epoch=epoch + 1, val_recall1=r1, seed=seed, recipe=R), sd / 'best.pt')
        # ---- selected checkpoint: dev readout -> threshold lock -> test (once)
        model.load_state_dict(torch.load(sd / 'best.pt', map_location=device)['model']); model.eval(); ck = sha(sd / 'best.pt')
        qz, gz = embed(model, ds['dev_query'], device, args.workers), embed(model, ds['dev_gallery'], device, args.workers)
        drows = rank(qz, gz, ds['dev_query'].records, ds['dev_gallery'].records)
        dpairs = readout_pairs(model, ds['dev_query'], ds['dev_gallery'], qz, gz, [(r['query_id'], r['gallery_id'], 'own_top1') for r in drows], device, args.workers, sd / 'dev_maps.npz')
        hits = np.array([p['hits'] for p in dpairs])[:, :, 1]; k = int(hits.mean(0).argmax())
        lock = dict(identity=identity, seed=seed, checkpoint_sha256=ck, selected_epoch=best[1], val_recall1=best[0], threshold=float(GRID[k]), dev_loc=float(hits.mean(0)[k]), locked_utc=now())
        save(sd / 'THRESHOLD_LOCK.json', lock); save(sd / 'dev_ranking.json', dict(identity=identity, rows=drows)); save(sd / 'dev_pairs.json', dict(identity=identity, pairs=dpairs))
        qz, gz = embed(model, ds['query'], device, args.workers), embed(model, ds['gallery'], device, args.workers)
        trows = rank(qz, gz, ds['query'].records, ds['gallery'].records)
        tpairs = [(r['query_id'], r['gallery_id'], 'own_top1') for r in trows] + [(r['query_id'], r[s_], f'ce_seed0_reference_{s_}') for r in fixed for s_ in ('positive', 'negative')]
        tp = readout_pairs(model, ds['query'], ds['gallery'], qz, gz, tpairs, device, args.workers, sd / 'test_maps.npz')
        own = [p for p in tp if p['view'] == 'own_top1']; corr = np.array([r['correct'] for r in trows]); Q = np.array([p['hits'][k][1] for p in own])
        summ = dict(identity=identity, seed=seed, checkpoint_sha256=ck, selected_epoch=best[1], threshold=lock['threshold'],
                    recall1=float(corr.mean()), recall10=float(np.mean([r['recall10'] for r in trows])), mAP=float(np.mean([r['AP'] for r in trows])),
                    loc=float(Q.mean()), Jq=float((corr * Q).mean()), EIB=float(np.mean([p['EIB'] for p in own])), degenerate=int(sum(p['degenerate'] for p in own)),
                    query_ids=[r['query_id'] for r in trows], query_products=[r['product_id'] for r in trows], per_query=dict(R=corr.tolist(), Q=Q.tolist()),
                    train_steps=steps, train_wall_seconds=time.monotonic() - t0)
        save(sd / 'test_ranking.json', dict(identity=identity, rows=trows)); save(sd / 'test_pairs.json', dict(identity=identity, pairs=tp)); save(sd / 'test_summary.json', summ)
        torch.save(dict(query=qz, gallery=gz), sd / 'test_embeddings.pt')
        reg['seeds'][seed] = {k_: summ[k_] for k_ in ['checkpoint_sha256', 'selected_epoch', 'threshold', 'recall1', 'recall10', 'mAP', 'loc', 'Jq', 'train_wall_seconds']}
        save(rp, reg); log('seed_done', seed=seed, **reg['seeds'][seed]); del model, crit, opt
        if device == 'cuda': torch.cuda.empty_cache()
    log('proxy_anchor_complete')


if __name__ == '__main__':
    main()
