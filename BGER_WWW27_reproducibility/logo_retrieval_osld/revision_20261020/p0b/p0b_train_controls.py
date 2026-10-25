#!/usr/bin/env python3
"""P0-B: controls separating the on-path gate from correct box supervision (exploratory, post-test).

Two controls under the frozen core recipe (ResNet-50 V1 init, 224 px whole images, 15 epochs,
batch 64, SGD/Nesterov, cosine lr, FP32, final checkpoint), the SAME per-seed initial template
as the core arms (runs/core_osld_v1/initial_seed{s}.pt) and the same training split:
  gate_only   : BGER architecture with lambda = 0 (gate on the pooling path, no box loss)
  shifted_box : BGER lambda = .5 trained on train-only corrupted boxes: each boxed training image
                receives a same-size box placed at a random position (fixed corruption seed
                20261020, identical for all training seeds; IoU with the true box < .1 where a
                placement allows). Evaluation always uses the true boxes.
Order: gate_only seeds 0,1,2 then shifted_box seeds 0,1,2 (the essential control first).
Per run: dev evaluation (official val) locks thresholds, then one test evaluation on the same
1,237 queries / 20,848 templates with the core CE-seed-0 fixed candidates. Outputs under
osld_revision_20261020/p0b/out; the core run directory is read only.
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
import numpy as np
import torch, torchvision as tv

HERE = Path(__file__).resolve().parent
E1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(E1 / 'code')); sys.path.insert(0, str(HERE.parent / 'common'))
from e1_common import RECIPE, save, sha, digest, now, log_writer  # noqa: E402
from e1_train import train, model_from  # noqa: E402
import e1_evaluate  # noqa: E402
from e1_evaluate import evaluate  # noqa: E402
from osld_data import build_sets, OSLDImageDataset  # noqa: E402
import web_eval as W  # noqa: E402

CORRUPTION_SEED = 20261020
CONTROLS = [('gate_only', 0.0), ('shifted_box', 0.5)]


def shift_boxes(records, rng):
    """Same-size box at a random position inside the image; prefer IoU<.1 with the true box."""
    out, ious = [], []
    for r in records:
        r = copy.deepcopy(r)
        if r.get('has_box'):
            x1, y1, x2, y2 = r['box_xyxy_half_open']; w, h = x2 - x1, y2 - y1; W_, H_ = r['width'], r['height']
            best = None
            for _ in range(50):
                nx = float(rng.uniform(0, max(W_ - w, 0))); ny = float(rng.uniform(0, max(H_ - h, 0)))
                cand = [nx, ny, nx + w, ny + h]; io = W.iou(cand, r['box_xyxy_half_open'])
                if best is None or io < best[1]: best = (cand, io)
                if io < 0.1: break
            r['box_xyxy_half_open'] = best[0]; r['true_box_xyxy_half_open'] = [x1, y1, x2, y2]; r['iou_with_true_box'] = best[1]; ious.append(best[1])
        out.append(r)
    return out, dict(n_boxed=len(ious), mean_iou_with_true=float(np.mean(ious)), frac_iou_below_0p1=float(np.mean(np.array(ious) < .1)), max_iou=float(max(ious)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', required=True); ap.add_argument('--data-root', required=True); ap.add_argument('--records', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--controls', default='gate_only,shifted_box')
    ap.add_argument('--cpu-threads', type=int, default=8)
    args = ap.parse_args()
    raise PermissionError('V2 release: fresh inference/training is not authorized. This archive is for saved-record CPU reproduction; original allocation guards remain below.')
    run = Path(args.run_root); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); root = Path(args.data_root)
    from revision_guard import reserve
    reserved = reserve('p0b_gate_controls')
    identity = 'OSLD_P0B_EXPLORATORY'; log = log_writer(out / 'events.jsonl', identity)
    torch.set_num_threads(args.cpu_threads); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    e1_evaluate.LOADER_WORKERS = 6
    scanned = json.loads(Path(args.records).read_text()); freeze = json.loads((run / 'pretest_freeze.json').read_text())
    if digest(scanned['product_records'] + scanned['logo_records']) != freeze['records_digest']: raise PermissionError('records digest != freeze')
    built = build_sets(scanned); sets, labels = built['sets'], built['labels']
    if digest(labels) != freeze['label_map_digest']: raise PermissionError('label map differs from the core study')
    base = dict(identity=identity, device='cuda', weights='V1', convention='zero_half_open', epochs=15, max_steps=None, batch=64,
                checkpoint_every=200, timing=False, loader_workers=6)
    shifted, shift_stats = shift_boxes(sets['train'], np.random.default_rng(CORRUPTION_SEED))
    save(out / 'shifted_box_stats.json', dict(identity=identity, corruption_seed=CORRUPTION_SEED, **shift_stats, rule=shift_boxes.__doc__))
    ds = {k: OSLDImageDataset(root, v, labels if k == 'train' else None) for k, v in sets.items()}
    ds_shift = OSLDImageDataset(root, shifted, labels)
    fixed = json.loads((run / 'fixed_candidate_manifest.json').read_text())['pairs']
    weights = tv.models.ResNet50_Weights.IMAGENET1K_V1
    reg = dict(identity=identity, experiment='p0b_gate_controls', created_utc=now(), reserved_gpu_seconds=reserved, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0), recipe=RECIPE, settings=base, records_digest=freeze['records_digest'],
               fixed_manifest_sha256=sha(run / 'fixed_candidate_manifest.json'), runs={})
    rp = out / 'EXPERIMENT_RECORD.json'
    if rp.exists(): reg['runs'] = json.loads(rp.read_text()).get('runs', {})
    seeds = [int(s) for s in args.seeds.split(',')]; controls = [c for c in CONTROLS if c[0] in args.controls.split(',')]
    for name, lam in controls:
        for seed in seeds:
            tag = f'{name}_l{lam:g}_s{seed}'; rd = out / 'models' / tag; t0 = time.monotonic()
            template = run / f'initial_seed{seed}.pt'
            settings = dict(base, control=name, corruption_seed=CORRUPTION_SEED if name == 'shifted_box' else None)
            log('run_start', tag=tag, template_sha256=sha(template))
            ckpt, complete = train('bger', lam, seed, template, ds_shift if name == 'shifted_box' else ds['train'], rd, settings, log, identity)
            if not complete: raise RuntimeError('incomplete training ' + tag)
            t_train = time.monotonic() - t0; ck_sha = sha(ckpt)
            model = model_from(ckpt, 'bger', 'cuda'); t1 = time.monotonic()
            dev = evaluate(model, ds['dev_query'], ds['dev_gallery'], 'cuda', rd / 'dev', log, identity,
                           binding=dict(stage='dev', checkpoint_sha256=ck_sha, source_digest=sha(Path(__file__))))
            lock = dict(identity=identity, tag=tag, checkpoint_sha256=ck_sha, thresholds=dev['thresholds'], dev_recall1=dev['recall1'],
                        dev_loc=dev['primary']['marginal_loc_query'], locked_utc=now())
            lp = rd / 'THRESHOLD_LOCK.json'
            if lp.exists() and json.loads(lp.read_text())['thresholds'] != lock['thresholds']: raise ValueError('lock changed')
            save(lp, lock); t_dev = time.monotonic() - t1; t2 = time.monotonic()
            test = evaluate(model, ds['query'], ds['gallery'], 'cuda', out / 'test' / tag, log, identity, thresholds=dev['thresholds'], fixed_pairs=fixed,
                            binding=dict(stage='test', checkpoint_sha256=ck_sha, source_digest=sha(Path(__file__)), fixed_pair_manifest_sha256=sha(run / 'fixed_candidate_manifest.json')))
            reg['runs'][tag] = dict(control=name, lam=lam, seed=seed, checkpoint_sha256=ck_sha, template_sha256=sha(template), thresholds=dev['thresholds'],
                                    dev=dict(recall1=dev['recall1'], loc=dev['primary']['marginal_loc_query']),
                                    test=dict(recall1=test['recall1'], recall10=test['recall10'], mAP=test['mAP'], loc=test['primary']['marginal_loc_query'], Jq=test['primary']['Jq_at_1']),
                                    wall_seconds=dict(train=t_train, dev=t_dev, test=time.monotonic() - t2))
            save(rp, reg); log('run_done', tag=tag, **reg['runs'][tag]['test'])
            del model; torch.cuda.empty_cache()
    log('p0b_complete', runs=sorted(reg['runs']))


if __name__ == '__main__':
    main()
