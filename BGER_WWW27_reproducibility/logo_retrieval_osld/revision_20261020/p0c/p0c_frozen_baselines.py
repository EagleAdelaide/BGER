#!/usr/bin/env python3
"""P0-C: frozen pretrained visual-embedding retrieval references on the identical OSLD protocol
(1,237 single-logo test queries, 20,848 canonical templates, identity relevance, whole images;
no training, no fine-tuning, no use of boxes). Retrieval only; no attribution readout.

Models (official weights, downloaded on the login node, hashes recorded):
  dinov2_vitb14  : facebookresearch/dinov2 ViT-B/14, CLS token (768-d)
  openclip_vitb32: OpenCLIP ViT-B-32, pretrained laion2b_s34b_b79k, image tower (512-d)
Preprocessing variants (disclosed, both reported):
  study : the five arms' deterministic whole-image 224x224 bilinear resize (no crop)
  native: manual tensor bicubic short-side resize + centre crop 224, which can cut logos
Ranking uses the study's FP32 stabilized cosine with stable tie-breaking (e1_evaluate.rank).
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

HERE = Path(__file__).resolve().parent
E1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(E1 / 'code')); sys.path.insert(0, str(HERE.parent / 'common'))
from e1_common import save, sha, digest, now  # noqa: E402
from e1_evaluate import rank  # noqa: E402
from osld_data import build_sets  # noqa: E402

MEAN, STD = [.485, .456, .406], [.229, .224, .225]
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


class Imgs(torch.utils.data.Dataset):
    def __init__(self, root, records, variant, model):
        self.root = Path(root); self.records = sorted(records, key=lambda r: r['image_id']); self.variant = variant; self.model = model
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(self.root / r['image_id']) as im:
            im = im.convert('RGB'); x = TF.pil_to_tensor(im).float() / 255
        mean, std = (CLIP_MEAN, CLIP_STD) if self.model == 'openclip_vitb32' else (MEAN, STD)
        if self.variant == 'study':
            x = TF.resize(x, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
        elif self.model == 'dinov2_vitb14':   # implemented tensor geometry: resize 256 (bicubic) + centre crop 224
            x = TF.center_crop(TF.resize(x, 256, interpolation=InterpolationMode.BICUBIC, antialias=True), 224)
        else:                                  # OpenCLIP: resize shortest side 224 (bicubic) + centre crop 224
            x = TF.center_crop(TF.resize(x, 224, interpolation=InterpolationMode.BICUBIC, antialias=True), 224)
        return TF.normalize(x, mean, std), i


def load_model(name, wdir, device):
    if name == 'dinov2_vitb14':
        m = torch.hub.load(str(wdir / 'hub' / 'facebookresearch_dinov2_main'), 'dinov2_vitb14', source='local', pretrained=False)
        sd = torch.load(wdir / 'dinov2_vitb14_pretrain.pth', map_location='cpu'); m.load_state_dict(sd)
        return m.to(device).eval(), (lambda x: m(x))
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=str(wdir / 'open_clip_vitb32_laion2b_s34b_b79k.bin'))
    m = m.to(device).eval()
    return m, (lambda x: m.encode_image(x))


@torch.no_grad()
def embed(fn, ds, device, workers, bs=128):
    out = []; t0 = time.time()
    for x, _ in torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False):
        z = fn(x.to(device)).float()
        if not torch.isfinite(z).all(): raise ValueError('nonfinite embedding')
        out.append(z.cpu())
    return torch.cat(out), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', required=True); ap.add_argument('--data-root', required=True); ap.add_argument('--records', required=True)
    ap.add_argument('--weights', required=True); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--models', default='dinov2_vitb14,openclip_vitb32'); ap.add_argument('--variants', default='study,native')
    ap.add_argument('--no-reserve', action='store_true')
    args = ap.parse_args()
    raise PermissionError('V2 release: fresh inference/training is not authorized. This archive is for saved-record CPU reproduction; original allocation guards remain below.')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reserved = 0
    if not args.no_reserve:
        from revision_guard import reserve; reserved = reserve('p0c_frozen_baselines')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    scanned = json.loads(Path(args.records).read_text()); freeze = json.loads((Path(args.run_root) / 'pretest_freeze.json').read_text())
    if digest(scanned['product_records'] + scanned['logo_records']) != freeze['records_digest']: raise PermissionError('records digest != freeze')
    sets = build_sets(scanned)['sets']; wdir = Path(args.weights)
    ref = json.loads((Path(args.run_root) / 'test' / 'ce_s0' / 'summary.json').read_text())
    reg = dict(identity='OSLD_P0C_FROZEN_REFERENCE', created_utc=now(), reserved_gpu_seconds=reserved, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
               torch=torch.__version__, device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
               weights={p.name: sha(p) for p in wdir.iterdir() if p.is_file()}, results={})
    try:
        import open_clip; reg['open_clip_version'] = open_clip.__version__
    except Exception as e: reg['open_clip_version'] = str(e)
    for name in args.models.split(','):
        model, fn = load_model(name, wdir, device)
        reg.setdefault('params', {})[name] = sum(p.numel() for p in model.parameters())
        for variant in args.variants.split(','):
            tag = f'{name}__{variant}'; od = out / tag
            if (od / 'summary.json').exists(): reg['results'][tag] = json.loads((od / 'summary.json').read_text()); continue
            qd, gd = Imgs(args.data_root, sets['query'], variant, name), Imgs(args.data_root, sets['gallery'], variant, name)
            qz, tq = embed(fn, qd, device, args.workers); gz, tg = embed(fn, gd, device, args.workers)
            rows = rank(qz, gz, qd.records, gd.records)
            if [r['query_id'] for r in rows] != ref['query_ids']: raise ValueError('query order differs from the core study')
            summ = dict(identity=reg['identity'], model=name, preprocessing=variant, embedding_dim=int(qz.shape[1]),
                        recall1=float(np.mean([r['correct'] for r in rows])), recall10=float(np.mean([r['recall10'] for r in rows])), mAP=float(np.mean([r['AP'] for r in rows])),
                        n_queries=len(rows), gallery_size=len(gd.records), embed_seconds=dict(query=tq, gallery=tg),
                        ms_per_image=1000 * (tq + tg) / (len(qd) + len(gd)), query_ids=ref['query_ids'], query_products=ref['query_products'],
                        per_query_correct=[r['correct'] for r in rows])
            save(od / 'ranking.json', dict(identity=reg['identity'], rows=rows)); save(od / 'summary.json', summ)
            torch.save(dict(query=qz, gallery=gz), od / 'embeddings.pt')
            reg['results'][tag] = summ; print(tag, {k: round(summ[k], 4) for k in ['recall1', 'recall10', 'mAP']}, f'{summ["ms_per_image"]:.1f} ms/img', flush=True)
        del model; torch.cuda.empty_cache()
    save(out / 'EXPERIMENT_RECORD.json', {k: v for k, v in reg.items() if k != 'results'} | dict(results={k: {kk: vv for kk, vv in v.items() if kk not in ('query_ids', 'query_products', 'per_query_correct')} for k, v in reg['results'].items()}))


if __name__ == '__main__':
    main()
