# Derived OpenLogo protocol notes

The adopted prospective amendment defines C0=[xmin,ymin,xmax,ymax) and
C1=[xmin-1,ymin-1,xmax,ymax). Official release semantics remain UNVERIFIED.
All arms train on the identical valid effective-target subset (949/965 products),
with 53 exemplars; all development/test queries and gallery are unchanged.
The common any-overlap 7x7 -> block-expanded 224 -> pooled >.5 target pipeline
is in reference_code/native/osld_data.py and reference_code/b_common.py.
No original OSLD hypothesis or selection rule was amended.

The scientific C0 threshold lock preceded test output. One test launch failed
the single-device visibility assertion before checkpoint loading or output;
an A100 retry succeeded. Engineering changes added diagnostics and disclosed
the failed reservation; no model, target, threshold or statistical rule changed.
All 12 models completed 15 epochs / 225 updates. Charged reservations total
29,700 GPU-seconds (8.25 of 24 hours); actual GPU walltime totals 2,763 seconds.
This export and integration performed no training or inference and carry no
authorization to spend a historical balance. Transfer A remains
NOT_RUN_SOURCE_UNSEEN_UNRESOLVED. No official benchmark replication is claimed.

Public author-linked archive digests, gallery inventory, declared identity
groups and all-product exclusion decisions are in data_provenance/. They are
reused audit evidence, not newly verified raw images or signed publisher hashes.
Group disjointness is not a perfect exhaustive semantic-contamination audit.
Data are for academic research with copyrights retained by original owners;
no images, weights or raw neural shards are bundled.

reference_code/ contains the actual numerical training/evaluation definitions
for inspection, with portable data/run bindings and disabled scheduler execution.
It is not the archive-only reproduction entry point. Original code hashes in
the historical protocol locks refer to internal originals. EXPORT_BINDINGS.json
discloses derivative changes; SHA256SUMS binds the anonymous export itself.
