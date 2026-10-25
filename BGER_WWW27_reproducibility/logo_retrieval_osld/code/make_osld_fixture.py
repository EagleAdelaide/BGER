#!/usr/bin/env python3
"""Generate a tiny OSLD-format fixture of NOISE images (SYNTHETIC_INTEGRATION).
Never real data. Covers: single-logo, multi-box, no-logo, unknown-box and
small-box cases; identity-disjoint train/val/test; distractor-only identity."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = Path(a.out); rng = np.random.default_rng(0)
for d in ("product-images", "logo-images", "annotations"):
    (out / d).mkdir(parents=True, exist_ok=True)
ids = {"train": ["alpha", "beta", "gamma", "delta"], "val": ["eps", "zeta"], "test": ["eta", "theta"]}
distractor = ["iota"]
files = {}
def img(path, w, h):
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=95)
for ident in sum(ids.values(), []) + distractor:
    for k in (1, 2):
        img(out / "logo-images" / f"{ident}-{k}.jpg", 64, 48)
ann = {}
for split, idl in ids.items():
    ann[split] = {}
    n = 0
    for ident in idl:
        for rep in range(3 if split == "train" else 2):
            name = f"{split}{n:05d}.jpg"; n += 1
            w, h = 120, 100
            img(out / "product-images" / name, w, h)
            x1, y1 = int(rng.integers(5, 60)), int(rng.integers(5, 50))
            bw = 6 if rep == 0 else 30          # one small logo per identity
            ann[split][name] = [[[x1, y1, x1 + bw, y1 + bw], f"{ident}-{1 + rep % 2}.jpg"]]
    for extra, boxes in (("multi", [[[5, 5, 30, 30], f"{idl[0]}-1.jpg"], [[40, 40, 70, 70], f"{idl[-1]}-1.jpg"]]),
                         ("nologo", []), ("unknown", [[[5, 5, 30, 30], "__unknown__"]])):
        name = f"{split}{n:05d}.jpg"; n += 1
        img(out / "product-images" / name, 120, 100)
        ann[split][name] = boxes
    (out / "annotations" / f"osld-{split}.json").write_text(json.dumps(ann[split]))
for p in sorted(out.rglob("*")):
    if p.is_file() and p.name != "SYNTHETIC_INTEGRATION.json":
        files[str(p.relative_to(out))] = hashlib.sha256(p.read_bytes()).hexdigest()
(out / "SYNTHETIC_INTEGRATION.json").write_text(json.dumps(dict(
    identity="SYNTHETIC_INTEGRATION", generator="make_osld_fixture.py",
    note="generated noise images in OSLD layout; not OSLD data", files=files), indent=1))
print(len(files), "fixture files")
