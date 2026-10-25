#!/usr/bin/env python3
"""Audited checkpoint evaluation and paired per-image corruption records."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
from scripts.cvpr2027.run_matched_cub import MatchedModel, data, file_hash, atomic_json, seed_all
from attribution.gradcam import GradCAM, normalize_cam
from datasets.cub import bbox_to_mask
from metrics.maxboxacc_v2 import (per_image_ious, summarize_ious, THRESHOLDS,
    half_open_boxes_to_inclusive, METRIC_VERSION)
from metrics.grounding import energy_in_region, pointing_game


def spatial_eval(model, ds, checkpoint, out, seed, arm, workers, training_config=None):
    cam = GradCAM(model, model.features)
    records = {op: [] for op in ('gradcam', 'detached', 'head_weight', 'hirescam', 'gradcampp')}
    ids, labels, logits_all, eib, pg, detach_err, head_err = [], [], [], [], [], [], []
    start = time.time()
    zero_standard=zero_detached=near_degenerate_head=0
    for b in DataLoader(ds, batch_size=24, num_workers=workers, shuffle=False):
        x = b['image'].cuda()
        maps, logits, targets = cam(x, out_size=(224,224))
        zero_standard += int((np.ptp(maps,axis=(1,2))==0).sum())
        ids.extend(b['sample_id'].tolist()); labels.extend(b['label'].tolist()); logits_all.extend(logits)
        boxes = [half_open_boxes_to_inclusive(box[None], 224, 224) for box in b['bbox'].numpy()]
        for i, heat in enumerate(maps):
            records['gradcam'].append(per_image_ious(heat, boxes[i]))
            mask = bbox_to_mask(b['bbox'][i].numpy(), 224)
            eib.append(energy_in_region(heat, mask)); pg.append(pointing_game(heat,mask))
        if arm in ('bger', 'ce'):
            model.detach_gate = True
            dm, dl, _ = cam(x, targets, out_size=(224,224))
            model.detach_gate = False
            zero_detached += int((np.ptp(dm,axis=(1,2))==0).sum())
            detach_err.append(float(np.max(np.abs(logits-dl))))
            h = cam._acts.detach()
            weight = model.classifier.weight.detach()[torch.as_tensor(targets, device='cuda')]
            hm = F.relu((h * weight[:,:,None,None]).sum(1))
            hm = F.interpolate(hm[:,None],(224,224),mode='bilinear',align_corners=False)[:,0]
            span=hm.flatten(1).max(1).values-hm.flatten(1).min(1).values
            with torch.no_grad():
                mu=(torch.sigmoid(model.gate(h)).sum((1,2,3))/(h.shape[-1]*h.shape[-2])**2
                    if arm=='bger' else torch.full_like(span,1/(h.shape[-1]*h.shape[-2])))
            near_degenerate_head += int((span*mu<=1e-8).sum())
            hm = normalize_cam(hm).detach().cpu().numpy()
            head_err.extend(np.max(np.abs(dm-hm), axis=(1,2)).tolist())
            for i in range(len(dm)):
                records['detached'].append(per_image_ious(dm[i],boxes[i]))
                records['head_weight'].append(per_image_ious(hm[i],boxes[i]))
            for op in ('hirescam','gradcampp'):
                om, _, _ = cam(x,targets,out_size=(224,224),method=op)
                for i in range(len(om)): records[op].append(per_image_ious(om[i], boxes[i]))
        if len(ids) % 480 == 0: print('Evaluated',len(ids),'/',len(ds),flush=True)
    cam.remove()
    logits_all = np.asarray(logits_all)
    probs = torch.softmax(torch.from_numpy(logits_all),1).numpy()
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == np.asarray(labels)
    metadata = {'checkpoint_sha256': file_hash(checkpoint), 'evaluator_sha256': file_hash(ROOT/'metrics/maxboxacc_v2.py'),
        'attribution_sha256': file_hash(ROOT/'attribution/gradcam.py'), 'metric_version': METRIC_VERSION,
        'conditioning':'predicted class', 'hook':'features post-ReLU', 'seed':seed,'method':arm,
        'geometry':'224 square; half-open continuous GT floor lower/ceil upper-1 inclusive',
        'eib_geometry':'legacy round-to-nearest half-open box mask at 224',
        'empty_map':'reference origin sentinel, exact minmax constant zero',
        'training_config':training_config, 'evaluation_runner_sha256':file_hash(__file__)}
    metadata['split_sha256'] = __import__('hashlib').sha256(json.dumps(ids).encode()).hexdigest()
    arrays = {'ids':np.asarray(ids),'labels':np.asarray(labels),'logits':logits_all,
        'confidence':conf,'correctness':correct,'eib':np.asarray(eib),'pg':np.asarray(pg),
        'thresholds':THRESHOLDS,'metadata_json':np.asarray(json.dumps(metadata))}
    summary = {'method':arm,'seed':seed,'N':len(ids),'accuracy':float(correct.mean()),
        'EIB':float(np.mean(eib)),'PG':float(np.mean(pg)),
        'high_confidence_count':int((conf>=.9).sum()),
        'HCU_count':int(((conf>=.9)&(np.asarray(eib)<.5)).sum()),
        'HCU_percent':float(100*((conf>=.9)&(np.asarray(eib)<.5)).mean()),
        'high_confidence_coverage':float((conf>=.9).mean()),
        'NLL':float(F.cross_entropy(torch.from_numpy(logits_all),torch.tensor(labels))),
        'seconds':time.time()-start, 'metadata':metadata,
        'zero_standard_maps':zero_standard}
    for op, values in records.items():
        if values:
            arrays['ious' if op=='gradcam' else 'ious_'+op] = np.asarray(values)
            summary[op] = summarize_ious(values)
    if detach_err:
        summary['detach_forward_max_error'] = max(detach_err)
        summary['detached_head_weight_map_max_error'] = max(head_err)
        summary['detached_head_weight_map_mean_max_error'] = float(np.mean(head_err))
        summary['zero_detached_maps']=zero_detached
        summary['detached_head_raw_span_at_most_normalization_epsilon']=near_degenerate_head
        if max(detach_err)>1e-6: raise RuntimeError('Detach changed forward logits')
    np.savez_compressed(out/'spatial_inputs.npz', **arrays)
    atomic_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


@torch.no_grad()
def corruption_eval(model, ds, checkpoint, out, seed, arm, workers):
    original = ROOT/'results/bger_3seed_subset.json'
    manifest = json.loads(original.read_text())
    wanted = manifest['sample_ids']
    lookup = {sid:i for i,sid in enumerate(ds.ids)}
    subset = Subset(ds,[lookup[sid] for sid in wanted])
    mean = torch.tensor((.485,.456,.406),device='cuda')[None,:,None,None]
    std = torch.tensor((.229,.224,.225),device='cuda')[None,:,None,None]
    path = out/'corruption.csv'
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['method','seed','sample_id','region','kind','severity','clean_correct','corrupt_correct'])
        writer.writeheader()
        for b in DataLoader(subset,batch_size=32,num_workers=workers,shuffle=False):
            x,y=b['image'].cuda(),b['label'].cuda()
            clean=(model(x).argmax(1)==y).cpu().numpy()
            masks=torch.tensor(np.stack([bbox_to_mask(box.numpy(),224) for box in b['bbox']]),device='cuda')[:,None]
            pixels=x*std+mean
            for kind in ('noise','blur'):
                for severity in (1,2,3):
                    if kind=='noise':
                        # Per-image CPU RNG: identical realizations across methods,
                        # model seeds, batch boundaries, and both spatial regions.
                        noise=torch.stack([torch.randn((3,224,224),generator=torch.Generator().manual_seed(
                            20271009 + int(sid)*100 + severity)) for sid in b['sample_id']]).cuda()
                        perturbed=(pixels+noise*{1:.1,2:.25,3:.5}[severity]).clamp(0,1)
                    else:
                        ks,sigma={1:(9,3.),2:(21,7.),3:(41,15.)}[severity]
                        perturbed=TF.gaussian_blur(pixels,ks,sigma)
                    for region in ('background','object'):
                        mask=1-masks if region=='background' else masks
                        xc=(pixels*(1-mask)+perturbed*mask-mean)/std
                        correct=(model(xc).argmax(1)==y).cpu().numpy()
                        for sid,c,cc in zip(b['sample_id'],clean,correct):
                            writer.writerow(dict(method=arm,seed=seed,sample_id=int(sid),region=region,
                                kind=kind,severity=severity,clean_correct=int(c),corrupt_correct=int(cc)))
    temp.replace(path)
    atomic_json(out/'corruption_manifest.json',{'subset_manifest_sha256':file_hash(original),
        'N':len(subset),'seed':seed,'method':arm,'checkpoint_sha256':file_hash(checkpoint),
        'perturbation_rng':'CPU torch.Generator(20271009+sample_id*100+severity), shared across methods/seeds/regions',
        'analysis':'equal weight noise/blur x severity1/2/3; per-image paired clean minus corrupt'})


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--stage',choices=['spatial','corruption','all'],default='all')
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Requires allocated GPU node')
    torch.set_num_threads(4);seed_all(0)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config'];arm=cfg['arm'];seed=cfg['seed']
    model=MatchedModel(arm,False).cuda();model.load_state_dict(saved['state_dict']);model.eval()
    _,datasets=data()
    out=ROOT/'results/cvpr2027/cub/evaluation'/args.checkpoint.stem;out.mkdir(parents=True,exist_ok=True)
    if args.stage in ('spatial','all'): spatial_eval(model,datasets['test'],args.checkpoint,out,seed,arm,args.workers,cfg)
    if args.stage in ('corruption','all') and arm in ('ce','bger'):
        corruption_eval(model,datasets['test'],args.checkpoint,out,seed,arm,args.workers)
