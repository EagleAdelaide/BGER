#!/usr/bin/env python3
"""CPU-only technical check of the received OSLD release (no model involved)."""
import argparse, json
from pathlib import Path
from osld_data import scan, build_sets
from e1_common import save, digest

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
s = scan(a.root, a.workers)
b = build_sets(s)
rep = s["report"]
rep["protocol_sets"] = {k: {"images": len(v), "identities": len({r["product_id"] for r in v}),
                            "with_box": sum(bool(r["has_box"]) for r in v)} for k, v in b["sets"].items()}
rep["training_label_count"] = len(b["labels"])
rel = {}
for key, gal in (("dev", "dev_gallery"), ("test", "gallery")):
    q = b["sets"]["dev_query" if key == "dev" else "query"]
    counts = {}
    for r in b["sets"][gal]:
        counts[r["product_id"]] = counts.get(r["product_id"], 0) + 1
    n = len(b["sets"][gal])
    rel[key] = {"gallery_size": n, "random_R@1_reference": sum(counts.get(r["product_id"], 0) / n for r in q) / max(1, len(q)),
                "queries_without_relevant_template": sum(counts.get(r["product_id"], 0) == 0 for r in q)}
rep["random_retrieval_reference"] = rel
rep["records_digest"] = digest(s["product_records"] + s["logo_records"])
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
save(out / "osld_release_report.json", rep)
save(out / "osld_records.json", {"product_records": s["product_records"], "logo_records": s["logo_records"]})
print(json.dumps({k: rep[k] for k in ("release_files", "cohort_accounting", "identity_overlap", "protocol_sets",
                                      "train_target_geometry", "random_retrieval_reference", "status")}, indent=1))
