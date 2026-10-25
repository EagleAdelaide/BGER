#!/usr/bin/env python3
"""Technical checks only; never downloads data or certifies legal access.

A successful scan is TECHNICAL_CHECKS_PASS, never 'OK to train'. Coordinate
convention must be supplied from the actual release documentation. Near-duplicate
review, release/crop provenance, authorization and real pipeline integration are
separate requirements. Generated test fixtures are not DeepFashion evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from PIL import Image


def _records(path: Path, min_fields: int) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Missing count/header: {path.name}")
    expected = int(lines[0].strip())
    rows = [line.split() for line in lines[2:] if line.strip()]
    if expected != len(rows):
        raise ValueError(f"Declared/actual count mismatch: {path.name}")
    if any(len(row) < min_fields for row in rows):
        raise ValueError(f"Malformed row: {path.name}")
    return rows


def _image_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or relative.as_posix() != name:
        raise ValueError(f"Unsafe image path: {name}")
    result = (root / Path(*relative.parts)).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes data root: {name}")
    return result


def _box(values: tuple[int, int, int, int], convention: str):
    x0, y0, x1, y1 = values
    if convention == "zero_half_open": return x0, y0, x1, y1
    if convention == "zero_inclusive": return x0, y0, x1 + 1, y1 + 1
    if convention == "one_inclusive": return x0 - 1, y0 - 1, x1, y1
    raise ValueError("Unverified bbox coordinate convention")


def scan(root: Path | str, bbox_convention: str | None) -> dict:
    root = Path(root).resolve()
    report = {"status": "NOT_VERIFIED", "authorization_verified": False,
              "bbox_convention": bbox_convention, "errors": [],
              "unverified": ["legal_access", "release_and_crop_provenance",
                             "near_duplicate_review", "real_pipeline_integration"]}
    if bbox_convention not in {"zero_half_open", "zero_inclusive", "one_inclusive"}:
        report["errors"].append("Coordinate convention has not been established")
        return report
    try:
        partition = _records(root / "Eval/list_eval_partition.txt", 3)
        bbox_rows = _records(root / "Anno/list_bbox_inshop.txt", 7)
        if not partition: raise ValueError("Empty partition file")
        rows = []
        for fields in partition:
            if len(fields) != 3: raise ValueError("Partition row must have three fields")
            image, item, split = fields
            if split not in {"train", "query", "gallery"}: raise ValueError(f"Unknown split: {split}")
            rows.append((image, item, split))
        names = [name for name, _, _ in rows]
        if len(set(names)) != len(names): raise ValueError("Image ID occurs in multiple partition rows")
        boxes = {}
        for fields in bbox_rows:
            name = fields[0]
            if name in boxes: raise ValueError(f"Duplicate box row: {name}")
            boxes[name] = _box(tuple(map(int, fields[-4:])), bbox_convention)
        pids = {sp: {item for _, item, s in rows if s == sp} for sp in ("train", "query", "gallery")}
        counts = Counter(s for _, _, s in rows)
        if any(counts[s] == 0 for s in pids): raise ValueError("A required partition is empty")
        report["counts"] = {sp: {"images": counts[sp], "items": len(pids[sp])} for sp in pids}
        overlap = pids["train"] & (pids["query"] | pids["gallery"])
        uncovered = pids["query"] - pids["gallery"]
        report["train_test_item_overlap_n"] = len(overlap)
        report["query_items_without_gallery"] = len(uncovered)
        if overlap: report["errors"].append("Training/test identity overlap")
        if uncovered: report["errors"].append("Query identity lacks a relevant gallery image")
        checks, missing, bad_boxes, unreadable = [], [], [], []
        byte_groups, pixel_groups = defaultdict(list), defaultdict(list)
        for name, item, split in rows:
            path = _image_path(root, name)
            if not path.is_file():
                missing.append(name); continue
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    rgb = image.convert("RGB")
                    pixel_payload = f"{width}x{height}|RGB|".encode() + rgb.tobytes()
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                pixel_sha = hashlib.sha256(pixel_payload).hexdigest()
                byte_groups[sha].append((name, split))
                pixel_groups[pixel_sha].append((name, split))
                box = boxes.get(name)
                if box is None or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    bad_boxes.append(name)
                checks.append({"image_id": name, "product_id": item, "split": split,
                               "width": width, "height": height, "box_xyxy_half_open": box,
                               "sha256": sha, "decoded_rgb_sha256": pixel_sha})
            except (OSError, ValueError) as exc:
                unreadable.append({"image_id": name, "reason": str(exc)})
        dup_bytes = [v for v in byte_groups.values() if len(v) > 1]
        dup_pixels = [v for v in pixel_groups.values() if len(v) > 1]
        report.update(scanned_images=len(checks), missing_files=missing,
                      invalid_or_missing_boxes=bad_boxes, unreadable_images=unreadable,
                      exact_byte_duplicate_groups=dup_bytes,
                      exact_decoded_duplicate_groups=dup_pixels, image_records=checks)
        if missing: report["errors"].append("Missing images in full scan")
        if bad_boxes: report["errors"].append("Missing or out-of-bounds boxes")
        if unreadable: report["errors"].append("Unreadable image files")
        cross = [v for v in dup_pixels if "train" in {s for _, s in v}
                 and ({"query", "gallery"} & {s for _, s in v})]
        if cross:
            report["errors"].append("Exact decoded image duplicated across train/test; no silent exclusion")
        if dup_pixels: report["unverified"].append("duplicate_group_disposition")
        report["status"] = "FAIL" if report["errors"] else "TECHNICAL_CHECKS_PASS"
    except (OSError, ValueError) as exc:
        report["status"] = "FAIL"; report["errors"].append(str(exc))
    return report


def development_manifests(report: dict, outdir: Path, *, fraction=.15, seed=0) -> dict:
    """Inherited train-ID-only split; technical manifest, not training approval."""
    if report["status"] != "TECHNICAL_CHECKS_PASS":
        raise ValueError("Refuse manifest generation after failed technical scan")
    if not 0 < fraction < 1: raise ValueError("Invalid dev fraction")
    by_id = defaultdict(list)
    for rec in report["image_records"]:
        if rec["split"] == "train": by_id[rec["product_id"]].append(rec["image_id"])
    eligible = sorted(pid for pid, images in by_id.items() if len(images) >= 2)
    if len(eligible) < 2: raise ValueError("Insufficient eligible train identities for held-out dev")
    k = max(1, int(len(eligible) * fraction))
    selected = set(random.Random(seed).sample(eligible, k))
    train, queries, gallery = [], [], []
    for pid in sorted(by_id):
        names = sorted(by_id[pid])
        if pid in selected:
            queries.append((names[0], pid)); gallery.extend((name, pid) for name in names[1:])
        else: train.extend((name, pid) for name in names)
    if not train: raise ValueError("Empty fitting partition after dev split")
    labels = {pid: j for j, pid in enumerate(sorted({pid for _, pid in train}))}
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_after_dev.txt", train), ("dev_query.txt", queries), ("dev_gallery.txt", gallery)):
        (outdir / name).write_text("".join(f"{im}\t{pid}\n" for im, pid in rows))
    (outdir / "train_label_map.json").write_text(json.dumps(labels, indent=2)+"\n")
    return {"seed": seed, "dev_fraction": fraction, "dev_product_ids": sorted(selected),
            "train_images": len(train), "dev_queries": len(queries), "dev_gallery": len(gallery),
            "training_label_count": len(labels)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="technical_integrity_report.json")
    ap.add_argument("--bbox-convention", choices=["zero_half_open", "zero_inclusive", "one_inclusive"])
    ap.add_argument("--dev-frac", type=float, default=.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hash-images", action="store_true", help="Compatibility flag; full hashing is always done")
    args = ap.parse_args()
    report = scan(args.root, args.bbox_convention)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if report["status"] == "TECHNICAL_CHECKS_PASS":
        try:
            report["dev_manifests"] = development_manifests(report, out.parent,
                                                            fraction=args.dev_frac, seed=args.seed)
        except ValueError as exc:
            report["status"] = "FAIL"; report["errors"].append(str(exc))
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "image_records"}, indent=2))
    raise SystemExit(0 if report["status"] == "TECHNICAL_CHECKS_PASS" else 2)


if __name__ == "__main__": main()
