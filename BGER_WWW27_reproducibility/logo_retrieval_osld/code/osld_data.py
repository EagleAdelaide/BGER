"""OSLD adapter for OSLD-SingleLogo-Retrieval-v1 (CPU data checks, cohort, sets, loader).

Release format (release Readme.txt): annotations/osld-{train,val,test}.json map a
product-image file name to [[bbox, canonical_logo_file], ...]; bbox = (x1, y1, x2,
y2) = (left, top, right, bottom); canonical_logo_file "__unknown__" marks an
unlabeled box that cannot be used for matching; [] means no logo. A logo
identity is the canonical file name without its trailing "-<number>.jpg"
(e.g. "amazon-3.jpg" -> "amazon"). Identities are logo classes as released,
not brands and not product SKUs.

Coordinates are used as continuous half-open boxes [x1, x2) x [y1, y2) in pixels
of the released image; a box is valid iff 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
Invalid boxes are excluded and counted, never clamped.

The field name 'product_id' in records is kept for compatibility with the
shared ranking/analysis code and holds the LOGO IDENTITY here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

SPLITS = ("train", "val", "test")
GRID = 7            # ResNet-50 final feature grid at 224x224 input
INPUT = 224
_SUFFIX = re.compile(r"^(?P<cls>.+)-(?P<idx>\d+)\.jpg$")


def logo_identity(fname: str) -> str:
    m = _SUFFIX.match(fname)
    if not m:
        raise ValueError(f"Unparseable canonical logo file name: {fname}")
    return m.group("cls")


def _file_info(path: str):
    p = Path(path)
    raw = p.read_bytes()
    out = {"sha256": hashlib.sha256(raw).hexdigest(), "ok": False}
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            small = im.convert("L").resize((9, 8), Image.BILINEAR)
            a = np.asarray(small, dtype=np.int16)
            dh = (a[:, 1:] > a[:, :-1]).flatten()
            out.update(ok=True, width=w, height=h,
                       dhash="".join("1" if b else "0" for b in dh),
                       decoded_rgb_sha256=hashlib.sha256(np.asarray(im).tobytes()).hexdigest())
    except Exception as exc:  # undecodable image is recorded, not hidden
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def grid_target_any_overlap(box, w, h, g=GRID):
    """OSLD amendment: a grid cell is positive iff it has positive-area overlap
    with the box (lower floor / upper ceil on the g x g grid, clipped)."""
    x1, y1, x2, y2 = box
    gx1 = max(0, min(g - 1, math.floor(x1 * g / w)))
    gy1 = max(0, min(g - 1, math.floor(y1 * g / h)))
    gx2 = max(gx1 + 1, min(g, math.ceil(x2 * g / w)))
    gy2 = max(gy1 + 1, min(g, math.ceil(y2 * g / h)))
    t = np.zeros((g, g), dtype=np.float32)
    t[gy1:gy2, gx1:gx2] = 1.0
    return t


def grid_target_old_rule(box, w, h, g=GRID, s=INPUT):
    """Inherited Web rule: 224 mask (floor/ceil) -> adaptive average pool -> > .5."""
    x1, y1, x2, y2 = [box[0] * s / w, box[1] * s / h, box[2] * s / w, box[3] * s / h]
    m = np.zeros((s, s), dtype=np.float32)
    m[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)] = 1.0
    pooled = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(m)[None, None], (g, g))[0, 0].numpy()
    return (pooled > 0.5).astype(np.float32)


def scan(root, workers: int = 4) -> dict:
    """Full-release technical scan: decode every product and logo image, parse all
    annotations, define the single-logo cohort with explicit exclusion reasons,
    check identity disjointness and exact/near duplicates. No model is used."""
    root = Path(root)
    ann = {s: json.loads((root / "annotations" / f"osld-{s}.json").read_text()) for s in SPLITS}
    prod_files = sorted(p.name for p in (root / "product-images").iterdir() if p.is_file())
    logo_files = sorted(p.name for p in (root / "logo-images").iterdir() if p.is_file())
    paths = [str(root / "product-images" / f) for f in prod_files] + [str(root / "logo-images" / f) for f in logo_files]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        infos = list(ex.map(_file_info, paths, chunksize=64))
    pinfo = dict(zip(prod_files, infos[:len(prod_files)]))
    linfo = dict(zip(logo_files, infos[len(prod_files):]))

    report = {"release_files": {"product_images": len(prod_files), "logo_images": len(logo_files)},
              "undecodable": {"product": sorted(f for f, i in pinfo.items() if not i["ok"]),
                              "logo": sorted(f for f, i in linfo.items() if not i["ok"])}}
    # canonical gallery
    logo_records = []
    bad_logo_names = []
    for f in logo_files:
        try:
            ident = logo_identity(f)
        except ValueError:
            bad_logo_names.append(f)
            continue
        i = linfo[f]
        if i["ok"]:
            logo_records.append(dict(image_id=f"logo-images/{f}", role="canonical", product_id=ident,
                                     logo_identity=ident, sha256=i["sha256"], decoded_rgb_sha256=i["decoded_rgb_sha256"],
                                     width=i["width"], height=i["height"], dhash=i["dhash"],
                                     box_xyxy_half_open=None, has_box=False))
    report["logo_file_name_unparseable"] = bad_logo_names
    canon_ids = defaultdict(list)
    for r in logo_records:
        canon_ids[r["product_id"]].append(r["image_id"])
    report["gallery"] = {"canonical_images_decodable": len(logo_records), "identities": len(canon_ids),
                         "templates_per_identity": dict(Counter(len(v) for v in canon_ids.values()))}

    # product images and cohort
    product_records, reasons, split_ids = [], {s: Counter() for s in SPLITS}, {s: set() for s in SPLITS}
    annotated_names = set()
    for s in SPLITS:
        for img, boxes in ann[s].items():
            annotated_names.add(img)
            for _, lf in boxes:
                if lf != "__unknown__":
                    split_ids[s].add(logo_identity(lf))
            i = pinfo.get(img)
            if i is None:
                reasons[s]["image_file_missing"] += 1
                continue
            if not i["ok"]:
                reasons[s]["undecodable"] += 1
                continue
            if len(boxes) == 0:
                reasons[s]["no_logo"] += 1
                continue
            if len(boxes) > 1:
                reasons[s]["multiple_boxes"] += 1
                continue
            box, lf = boxes[0]
            if lf == "__unknown__":
                reasons[s]["single_box_unlabeled"] += 1
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            if not (0 <= x1 < x2 <= i["width"] and 0 <= y1 < y2 <= i["height"]):
                reasons[s]["invalid_coordinates"] += 1
                continue
            ident = logo_identity(lf)
            if f"logo-images/{lf}" not in {x for x in canon_ids.get(ident, [])}:
                reasons[s]["matched_template_missing_or_undecodable"] += 1
                continue
            reasons[s]["included_single_logo"] += 1
            product_records.append(dict(image_id=f"product-images/{img}", role="product", split=s,
                                        product_id=ident, logo_identity=ident, matched_template=f"logo-images/{lf}",
                                        sha256=i["sha256"], decoded_rgb_sha256=i["decoded_rgb_sha256"],
                                        width=i["width"], height=i["height"], dhash=i["dhash"],
                                        box_xyxy_half_open=[x1, y1, x2, y2], has_box=True))
    report["annotation_entries"] = {s: len(ann[s]) for s in SPLITS}
    report["product_files_without_annotation"] = sorted(set(prod_files) - annotated_names)[:50]
    report["product_files_without_annotation_n"] = len(set(prod_files) - annotated_names)
    report["cohort_accounting"] = {s: dict(reasons[s]) for s in SPLITS}
    box_counts = {s: dict(Counter(len(b) for b in ann[s].values())) for s in SPLITS}
    unknown = {s: sum(lf == "__unknown__" for b in ann[s].values() for _, lf in b) for s in SPLITS}
    report["boxes_per_image_histogram"] = box_counts
    report["unlabeled_boxes"] = unknown
    report["labeled_identities_per_split"] = {s: len(split_ids[s]) for s in SPLITS}
    report["identity_overlap"] = {f"{a}&{b}": len(split_ids[a] & split_ids[b])
                                  for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    cohort_ids = {s: {r["product_id"] for r in product_records if r["split"] == s} for s in SPLITS}
    report["cohort_identities"] = {s: len(cohort_ids[s]) for s in SPLITS}
    all_split_ids = set().union(*split_ids.values())
    report["gallery_identities_not_in_any_split"] = len(set(canon_ids) - all_split_ids)
    report["split_identities_without_template"] = {s: len(split_ids[s] - set(canon_ids)) for s in SPLITS}

    # duplicates: exact bytes, exact decoded pixels, near (dHash Hamming <= 4) across splits
    by_sha = defaultdict(list)
    for r in product_records + logo_records:
        by_sha[r["decoded_rgb_sha256"]].append(r["image_id"])
    exact = [v for v in by_sha.values() if len(v) > 1]
    report["exact_pixel_duplicate_groups"] = len(exact)
    report["exact_pixel_duplicate_examples"] = exact[:20]
    cross_split_exact = []
    split_of = {r["image_id"]: r.get("split", "gallery") for r in product_records + logo_records}
    for v in exact:
        sp = {split_of[x] for x in v if split_of[x] != "gallery"}
        if len(sp) > 1:
            cross_split_exact.append(v)
    report["cross_split_exact_product_duplicates"] = cross_split_exact[:50]
    report["cross_split_exact_product_duplicates_n"] = len(cross_split_exact)
    bits = {r["image_id"]: int(r["dhash"], 2) for r in product_records}
    near = []
    by_split = defaultdict(list)
    for r in product_records:
        by_split[r["split"]].append(r["image_id"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        A = np.array([bits[x] for x in by_split[a]], dtype=np.uint64)
        for y in by_split[b]:
            x = (A ^ np.uint64(bits[y])).view(np.uint8).reshape(-1, 8)
            d = np.unpackbits(x, axis=1).sum(1)
            for j in np.flatnonzero(d <= 4):
                near.append((by_split[a][j], y, int(d[j])))
    report["suspected_cross_split_near_duplicates_dhash_le4"] = near[:100]
    report["suspected_cross_split_near_duplicates_dhash_le4_n"] = len(near)

    # training-target geometry (train cohort): old rule vs any-overlap rule
    tr = [r for r in product_records if r["split"] == "train"]
    old0 = new0 = 0
    occ_old, occ_new, area = [], [], []
    for r in tr:
        b, w, h = r["box_xyxy_half_open"], r["width"], r["height"]
        to, tn = grid_target_old_rule(b, w, h), grid_target_any_overlap(b, w, h)
        old0 += int(to.sum() == 0); new0 += int(tn.sum() == 0)
        occ_old.append(float(to.mean())); occ_new.append(float(tn.mean()))
        area.append((b[2] - b[0]) * (b[3] - b[1]) / (w * h))
    q = lambda v: {p: float(np.quantile(v, p / 100)) for p in (5, 25, 50, 75, 95)} if v else {}
    report["train_target_geometry"] = {
        "n_train_cohort": len(tr), "zero_target_rate_old_avgpool_gt_half": old0 / max(1, len(tr)),
        "zero_target_rate_any_overlap": new0 / max(1, len(tr)),
        "grid_occupancy_old_quantiles": q(occ_old), "grid_occupancy_new_quantiles": q(occ_new),
        "box_area_fraction_quantiles": q(area),
        "note": "coarse 7x7 grid supervision, not a mask; evaluation always uses the original box"}
    sizes = Counter((r["width"], r["height"]) for r in product_records)
    report["product_image_sizes_top"] = [[list(k), v] for k, v in sizes.most_common(10)]
    report["boxes_touching_right_or_bottom_edge"] = sum(
        r["box_xyxy_half_open"][2] == r["width"] or r["box_xyxy_half_open"][3] == r["height"] for r in product_records)
    report["status"] = "TECHNICAL_CHECKS_PASS" if not report["undecodable"]["product"] and report["identity_overlap"] == {
        "train&val": 0, "train&test": 0, "val&test": 0} else "TECHNICAL_CHECKS_REVIEW"
    return {"report": report, "product_records": product_records, "logo_records": logo_records}


def build_sets(scan_out: dict) -> dict:
    """OSLD-SingleLogo-Retrieval-v1 sets. Development uses the official val split;
    test is touched only by the core runner after the pre-test freeze."""
    prod, logos = scan_out["product_records"], scan_out["logo_records"]
    ids = {s: sorted({r["product_id"] for r in prod if r["split"] == s}) for s in SPLITS}
    train_ids = set(ids["train"])
    val_ids = set(ids["val"])
    train = [r for r in prod if r["split"] == "train"] + [r for r in logos if r["product_id"] in train_ids]
    sets = {
        "train": train,
        "dev_query": [r for r in prod if r["split"] == "val"],
        # train+val templates only; test-identity templates never enter development
        "dev_gallery": [r for r in logos if r["product_id"] in train_ids | val_ids],
        "query": [r for r in prod if r["split"] == "test"],
        "gallery": list(logos),  # full decodable canonical library (distractors included)
    }
    labels = {pid: j for j, pid in enumerate(sorted(train_ids))}
    leak = {r["product_id"] for r in sets["train"]} & ({r["product_id"] for r in sets["dev_query"]} |
                                                     {r["product_id"] for r in sets["query"]})
    if leak:
        raise ValueError(f"Identity leakage into training: {sorted(leak)[:5]}")
    return {"sets": sets, "labels": labels}


class OSLDImageDataset(Dataset):
    """Full released image -> 224x224 (bilinear, antialias), ImageNet normalization.
    'mask' is a 224x224 block-constant upsampling of the 7x7 any-overlap target so
    that the shared adaptive-average-pool > .5 step reproduces the amended grid
    target exactly; images without a box have has_box=False and a zero mask that
    never enters a spatial loss."""

    def __init__(self, root, records, labels=None):
        self.root = Path(root)
        self.records = sorted(records, key=lambda r: r["image_id"])
        self.labels = labels or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        p = self.root / r["image_id"]
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            if (w, h) != (r["width"], r["height"]):
                raise ValueError("Image geometry changed after scan: " + r["image_id"])
            x = TF.pil_to_tensor(im).float() / 255
        x = TF.resize(x, [INPUT, INPUT], interpolation=InterpolationMode.BILINEAR, antialias=True)
        x = TF.normalize(x, [.485, .456, .406], [.229, .224, .225])
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite loader input")
        if r["has_box"]:
            t = grid_target_any_overlap(r["box_xyxy_half_open"], w, h)
            box = torch.tensor(r["box_xyxy_half_open"])
        else:
            t = np.zeros((GRID, GRID), dtype=np.float32)
            box = torch.full((4,), float("nan"))
        mask = torch.from_numpy(np.kron(t, np.ones((INPUT // GRID, INPUT // GRID), dtype=np.float32)))
        return {"x": x, "label": self.labels.get(r["product_id"], -1), "mask": mask,
                "has_box": bool(r["has_box"]), "index": index, "id": r["image_id"],
                "product": r["product_id"], "box": box, "size": torch.tensor([h, w])}
