#!/usr/bin/env python3
"""CPU check on REAL OSLD train/dev records (no test, no GPU, no selection):
loader decode/geometry/target, one optimizer step per arm at batch 4 with the
real class count, and one dev query/gallery readout pair. Not a result."""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from osld_data import build_sets, OSLDImageDataset
from train_web_arms import build_model, step_losses
from e1_evaluate import PairReader, embed
import torchvision as tv
torch.manual_seed(0); torch.set_num_threads(4)
import os
root = Path(os.environ["OSLD_ROOT"])  # extracted official OSLD release
recs = json.loads(Path(os.environ["OSLD_RECORDS"]).read_text())  # output of code/osld_release_check.py
b = build_sets(recs); sets, labels = b["sets"], b["labels"]
tr = OSLDImageDataset(root, sets["train"], labels)
boxed = [i for i, r in enumerate(tr.records) if r["has_box"]][:2]
unboxed = [i for i, r in enumerate(tr.records) if not r["has_box"]][:2]
items = [tr[i] for i in boxed + unboxed]
x = torch.stack([it["x"] for it in items]); y = torch.tensor([it["label"] for it in items])
mask = torch.stack([it["mask"] for it in items]); has = torch.tensor([it["has_box"] for it in items])
out = {"n_labels": len(labels), "mask_cells_boxed": [int((m[::32, ::32] > 0).sum()) for m in mask[:2]],
       "mask_sum_unboxed": [float(m.sum()) for m in mask[2:]], "labels_ok": bool((y >= 0).all())}
for arm in ("ce", "aux", "bger", "cam_bce", "abn_box"):
    m = build_model(len(labels), arm, weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
    opt = torch.optim.SGD(m.parameters(), lr=.01, momentum=.9, nesterov=True, weight_decay=1e-4)
    t0 = time.time(); loss, parts = step_losses(m, x, y, mask, has, 2.0, arm)
    opt.zero_grad(); loss.backward(); opt.step()
    out[arm] = {"loss_parts": parts, "finite": bool(torch.isfinite(loss)), "cpu_step_s": round(time.time() - t0, 2),
                "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2)}
    if arm in ("bger", "abn_box"):
        m.eval()
        dq = OSLDImageDataset(root, sets["dev_query"][:1]); dg = OSLDImageDataset(root, [r for r in sets["dev_gallery"] if r["product_id"] == sets["dev_query"][0]["product_id"]][:1])
        with torch.no_grad():
            qz = torch.stack([m(dq[0]["x"][None])[2][0]]); gz = torch.stack([m(dg[0]["x"][None])[2][0]])
        pr = PairReader(m, dq, dg, qz, gz, "cpu", lambda *a, **k: None)
        res = pr.pair(dq.records[0]["image_id"], dg.records[0]["image_id"], "check")
        out[arm]["readout_query_scored"] = res["query"]["primary"] is not None
        out[arm]["readout_gallery_unscored"] = res["gallery"]["primary"] is None
        out[arm]["score_err"] = res["query"]["diagnostic"]["ranking_score_error"]
print(json.dumps(out, indent=1))
