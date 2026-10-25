#!/usr/bin/env python3
"""P0-A: BGER ranking-score Grad-CAM at the POOLED tensor F = M*H (post-hoc, exploratory).

For each frozen BGER checkpoint (lambda=.5, seeds 0-2) and each query, with the query image,
candidate embeddings, deployed cosine score and candidate selection all unchanged:
  (1) C_H : Grad-CAM at pre-gate H  (original primary; recomputed and checked against saved maps)
  (2) C_F : Grad-CAM at the pooled tensor F = sigmoid(gate(H)) * H  (NEW readout)
  (3) D   : forward decomposition at F (original secondary; checked against saved maps)
Hook checks per pair: GAP(F) equals the cached embedding z_q; score equals the saved ranking
score; autograd C_F equals the analytic D - s*rho*T (T_u = F_u.z/(U|z|^2)); C_H equals the
saved primary map. Threshold for C_F is chosen ONLY on the official validation split
(own top-1, 19-point grid, IoU>=.5 hit rate), frozen, then the test split is evaluated once:
own top-1 (incl. wrong templates) and the CE-seed-0 fixed d+/d- pairs.
Outputs per seed/split: records JSON (hits at 19 thresholds x 3 IoU levels, EIB, degeneracy,
raw magnitudes, checks) and an NPZ of C_F / recomputed C_H maps. Resumable per (seed, split).
Nothing in runs/core_osld_v1 is modified.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(E1 / 'code')); sys.path.insert(0, str(HERE.parent / 'common'))
from e1_common import GRID, save, sha, digest, now, evaluation  # noqa: E402
from e1_train import model_from  # noqa: E402
from e1_evaluate import map_metrics, score_row  # noqa: E402
import web_eval as W  # noqa: E402
from osld_data import scan, build_sets, OSLDImageDataset  # noqa: E402

EPS = 1e-12


def load_records(args):
    if args.fixture:
        s = scan(args.data_root, workers=1); return s, 'SYNTHETIC_INTEGRATION'
    s = json.loads(Path(args.records).read_text())
    freeze = json.loads((Path(args.run_root) / 'pretest_freeze.json').read_text())
    if digest(s['product_records'] + s['logo_records']) != freeze['records_digest']: raise PermissionError('records digest != pre-test freeze')
    return s, 'OSLD_P0A_EXPLORATORY'


def readouts(model, x, zd_list, device):
    """One forward; per candidate: C_H (pre-gate), C_F (pooled), D (forward decomposition), T, checks."""
    with torch.no_grad():
        H = model.features(x[None].to(device))[0]
        F = model.pooled_features(H[None])[0]
        z = F.mean((1, 2))
    U = F.shape[1] * F.shape[2]
    out = []
    for zd in zd_list:
        zd = zd.to(device)
        cH, _, recH = W.retrieval_gradcam_audited(H, lambda f: model.pool(f[None])[0], zd)
        cF, _, recF = W.retrieval_gradcam_audited(F, lambda f: f.mean((1, 2)), zd)
        with torch.no_grad():
            D, sD = W.score_decomposition(F, zd)
            a, b = z.norm(), zd.norm(); K = a * b + EPS; s = (z * zd).sum() / K; rho = a * b / K
            T = torch.einsum('chw,c->hw', F, z) / (U * a * a)
            ident = (D - s * rho * T - cF).abs().max()
        out.append(dict(cH=cH.detach().cpu().numpy().astype(np.float32), cF=cF.detach().cpu().numpy().astype(np.float32),
                        D=D.detach().cpu().numpy().astype(np.float32), T=T.detach().cpu().numpy().astype(np.float32),
                        score=float(s), score_H=recH['score'], score_F=recF['score'], rho=float(rho), a=float(a), b=float(b),
                        identity_residual=float(ident), recF=recF, recH=recH))
    return out, z.detach().cpu()


def run_split(args, model, device, qd, qz_cache, gz_cache, gindex, pairs, saved_dir, identity, out, split, log):
    """pairs: list of (query_id, gallery_id, view). saved_dir: original evaluation dir with signed_maps (for checks)."""
    rec_path, map_path = out / f'{split}_records.json', out / f'{split}_maps.npz'
    if rec_path.exists() and map_path.exists():
        log(f'{split}: reuse existing records'); return json.loads(rec_path.read_text())
    qindex = {r['image_id']: i for i, r in enumerate(qd.records)}
    by_q = {}
    for q, g, v in pairs: by_q.setdefault(q, []).append((g, v))
    saved_own = {}
    for f in ['own_top1.json', 'fixed_pairs.json']:
        p = saved_dir / f
        if p.exists():
            for pr in json.loads(p.read_text())['pairs']: saved_own[(pr['query_id'], pr['gallery_id'], pr['view'])] = pr['signed_maps_file']
    rows = []; maps = {}; t0 = time.time(); zdiff = 0.0
    loader = torch.utils.data.DataLoader(qd, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=lambda b: b[0])
    with evaluation(model, None):
        for i, item in enumerate(loader):
            q = item['id']
            if q not in by_q: continue
            cands = by_q[q]
            res, z = readouts(model, item['x'], [gz_cache[gindex[g]] for g, _ in cands], device)
            zdiff = max(zdiff, float((z - qz_cache[qindex[q]]).abs().max()))
            rec = qd.records[i]
            for (g, v), r in zip(cands, res):
                mF = map_metrics(torch.from_numpy(r['cF']), rec); mH = map_metrics(torch.from_numpy(r['cH']), rec)
                key = (q, g, v); chk = dict(z_vs_cache_max_abs=float((z - qz_cache[qindex[q]]).abs().max()))
                if key in saved_own:
                    sv = np.load(saved_dir / 'signed_maps' / saved_own[key])
                    chk['cH_vs_saved_max_abs'] = float(np.abs(sv['query_primary_signed'] - r['cH']).max())
                    chk['D_vs_saved_max_abs'] = float(np.abs(sv['query_secondary_signed'] - r['D']).max())
                rows.append(dict(query_id=q, gallery_id=g, view=v, score=r['score'], score_H=r['score_H'], score_F=r['score_F'],
                                 identity_residual=r['identity_residual'], checks=chk,
                                 F=dict(hits=mF['hits'], EIB=mF['EIB'], degenerate=mF['degenerate'], nonfinite=mF['nonfinite'],
                                        raw_max_abs=r['recF']['signed_map_max_abs'], positive_sum=r['recF']['positive_map_sum']),
                                 H=dict(hits=mH['hits'], EIB=mH['EIB'], degenerate=mH['degenerate'], nonfinite=mH['nonfinite'],
                                        raw_max_abs=r['recH']['signed_map_max_abs'], positive_sum=r['recH']['positive_map_sum'])))
                name = digest([q, g, v]); maps[name + '__cF'] = r['cF']; maps[name + '__cH'] = r['cH']; maps[name + '__T'] = r['T']
            if len(rows) % 300 < len(cands): log(f'{split}: {len(rows)} pairs, {time.time() - t0:.0f}s, zdiff {zdiff:.2e}')
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list): return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o): return None   # undefined diagnostics are stored as null, never imputed
        return o
    rows = clean(rows)
    np.savez_compressed(map_path.with_suffix('.tmp.npz'), identity=np.array(identity), **maps); map_path.with_suffix('.tmp.npz').replace(map_path)
    save(rec_path, dict(identity=identity, split=split, n_pairs=len(rows), wall_seconds=time.time() - t0, rows=rows))
    return json.loads(rec_path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', required=True); ap.add_argument('--data-root', required=True)
    ap.add_argument('--records'); ap.add_argument('--fixture', action='store_true'); ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    raise PermissionError('V2 release: fresh inference/training is not authorized. This archive is for saved-record CPU reproduction; original allocation guards remain below.')
    run = Path(args.run_root); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = 'cpu' if args.cpu else 'cuda'
    if not args.fixture:
        from revision_guard import reserve
        reserved = reserve('p0a_bger_pooled_readout')
    else: reserved = 0
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    logf = open(out / 'log.txt', 'a')
    def log(m): print(m, flush=True); logf.write(f'{now()} {m}\n'); logf.flush()
    scanned, identity = load_records(args)
    sets = build_sets(scanned)['sets']; root = Path(args.data_root)
    dq, dg = OSLDImageDataset(root, sets['dev_query']), OSLDImageDataset(root, sets['dev_gallery'])
    tq, tg = OSLDImageDataset(root, sets['query']), OSLDImageDataset(root, sets['gallery'])
    dgi = {r['image_id']: i for i, r in enumerate(dg.records)}; tgi = {r['image_id']: i for i, r in enumerate(tg.records)}
    freeze = json.loads((run / 'pretest_freeze.json').read_text())
    fixed = json.loads((run / 'fixed_candidate_manifest.json').read_text())['pairs']
    registry = dict(identity=identity, experiment='p0a_bger_pooled_readout', created_utc=now(), reserved_gpu_seconds=reserved,
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__, device=device,
                    gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None, run_root=str(run), freeze_sha256=sha(run / 'pretest_freeze.json'),
                    fixed_manifest_sha256=sha(run / 'fixed_candidate_manifest.json'), code_sha256={p.name: sha(p) for p in [Path(__file__), E1 / 'code' / 'web_eval.py', E1 / 'code' / 'e1_evaluate.py']},
                    seeds={})
    for seed in [int(s) for s in args.seeds.split(',')]:
        mdir = run / 'models' / f'bger_l0.5_s{seed}'; ck = mdir / 'checkpoint.pt'
        fz = next(c for c in freeze['checkpoints'] if c['arm'] == 'bger' and c['seed'] == seed)
        h = sha(ck)
        if h != fz['checkpoint_sha256']: raise PermissionError(f'checkpoint sha mismatch seed {seed}')
        if fz['lambda_value'] != 0.5: raise PermissionError('unexpected selected lambda')
        log(f'seed {seed}: checkpoint {h[:12]} verified against freeze')
        model = model_from(ck, 'bger', device)
        so = out / f'seed{seed}'; so.mkdir(exist_ok=True)
        # ---- validation split: own top-1 only -> threshold lock for C_F (and recheck of H threshold)
        demb = torch.load(mdir / 'dev' / 'embeddings.pt', map_location='cpu', weights_only=False)
        drank = json.loads((mdir / 'dev' / 'ranking.json').read_text())['rows']
        dpairs = [(r['query_id'], r['gallery_id'], 'own_top1') for r in drank]
        drec = run_split(args, model, device, dq, demb['query'], demb['gallery'], dgi, dpairs, mdir / 'dev', identity, so, 'dev', log)
        hitsF = np.array([r['F']['hits'] for r in drec['rows']])[:, :, 1]; hitsH = np.array([r['H']['hits'] for r in drec['rows']])[:, :, 1]
        kF = int(hitsF.mean(0).argmax()); kH = int(hitsH.mean(0).argmax())
        lock = dict(identity=identity, seed=seed, readout='C_F', threshold=float(GRID[kF]), dev_hit_rate=float(hitsF.mean(0)[kF]),
                    dev_hit_rate_by_threshold=hitsF.mean(0).tolist(), recomputed_H_threshold=float(GRID[kH]), frozen_H_threshold=fz['thresholds']['primary'],
                    H_threshold_matches_freeze=bool(abs(GRID[kH] - fz['thresholds']['primary']) < 1e-9), locked_utc=now(), rule='argmax over 19-point grid of IoU>=.5 hit rate on official val own top-1; frozen before test')
        lp = so / 'THRESHOLD_LOCK.json'
        if lp.exists():
            old = json.loads(lp.read_text())
            if abs(old['threshold'] - lock['threshold']) > 1e-9: raise ValueError('threshold lock changed on resume')
        else: save(lp, lock)
        log(f'seed {seed}: C_F threshold locked {lock["threshold"]:.2f} (dev hit {lock["dev_hit_rate"]:.4f}); H threshold recheck {lock["H_threshold_matches_freeze"]}')
        # ---- test split: own top-1 + CE-seed-0 fixed pairs, evaluated once after the lock
        temb = torch.load(run / 'test' / f'bger_s{seed}' / 'embeddings.pt', map_location='cpu', weights_only=False)
        trank = json.loads((run / 'test' / f'bger_s{seed}' / 'ranking.json').read_text())['rows']
        tpairs = [(r['query_id'], r['gallery_id'], 'own_top1') for r in trank]
        for r in fixed: tpairs += [(r['query_id'], r['positive'], 'ce_seed0_reference_positive'), (r['query_id'], r['negative'], 'ce_seed0_reference_negative')]
        run_split(args, model, device, tq, temb['query'], temb['gallery'], tgi, tpairs, run / 'test' / f'bger_s{seed}', identity, so, 'test', log)
        registry['seeds'][seed] = dict(checkpoint_sha256=h, threshold_C_F=lock['threshold'])
        del model
        if device == 'cuda': torch.cuda.empty_cache()
    save(out / 'EXPERIMENT_RECORD.json', registry); log('P0-A complete')


if __name__ == '__main__':
    main()
