#!/usr/bin/env python3
"""Write the OSLD data receipt consumed by the B0/core guards.

Every evidence entry is a file path plus its SHA256 (re-verified by the guard).
Authorization basis: public CC BY-NC 4.0 release (research only) obtained from
the official link under the author's explicit written instruction. This script
does not grant core approval and does not touch any model output.
"""
import argparse, hashlib, json
from pathlib import Path
from e1_common import digest, save


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--release", required=True)
ap.add_argument("--provenance", required=True)
ap.add_argument("--instruction", required=True, help="author instruction file adopting OSLD")
ap.add_argument("--records", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rel, prov = Path(a.release).resolve(), Path(a.provenance).resolve()
recs = json.loads(Path(a.records).read_text())
rep = json.loads(Path(a.report).read_text())
if rep["status"] != "TECHNICAL_CHECKS_PASS":
    raise SystemExit("Technical checks did not pass; no receipt written")
entry = lambda p: {"path": str(Path(p).resolve()), "sha256": sha(p)}
receipt = {
    "root": str(rel),
    "authorization": "AUTHOR_INSTRUCTED_PUBLIC_LICENSE",
    "authorization_note": "CC BY-NC 4.0, research purposes only; images/logos owned by their rights owners; "
                          "obtained from the official Dropbox link named in github.com/mubastan/osld",
    "release_crop_provenance": "RELEASE_README_DOCUMENTED",
    "access_evidence": entry(a.instruction),
    "license_evidence": entry(rel / "LICENSE"),
    "coordinate_evidence": entry(rel / "Readme.txt"),
    "release_evidence": entry(prov / "osld.zip.sha256"),
    "bbox_convention": "zero_half_open",
    "bbox_convention_note": "Readme states (x1,y1,x2,y2)=(left,top,right,bottom) without inclusivity; "
                            "used as continuous half-open pixel boxes, invalid boxes excluded not clamped",
    "annotation_hashes": {f"annotations/osld-{s}.json": sha(rel / "annotations" / f"osld-{s}.json")
                          for s in ("train", "val", "test")},
    "records_file": str(Path(a.records).resolve()),
    "technical_records_digest": digest(recs["product_records"] + recs["logo_records"]),
    "technical_report_sha256": sha(a.report),
}
save(a.out, receipt)
print(json.dumps({k: v for k, v in receipt.items() if k in ("authorization", "bbox_convention", "technical_records_digest")}, indent=1))
