# Derived OpenLogo: anonymous saved-record release

CE/Aux-BCE/BGER/CAM-BCE were independently trained with seeds 0/1/2.
This is not OSLD transfer or official benchmark replication. Coordinate
semantics remain UNVERIFIED. See PROTOCOL_NOTES.md, protocol/config.json,
manifests, shared batches and the full data_provenance/ exclusion ledger.

## Archive-only CPU endpoint recomputation

Use Python >=3.9, NumPy and pandas (requirements-cpu.txt). From any directory:

```sh
OPENBLAS_NUM_THREADS=1 python /path/to/openlogo/recompute.py --out /new/output_directory
# Also bind the manuscript generator inputs:
python openlogo/recompute.py --source-root ../BGER_WWW27_source --out /new/check
```

The output directory must not exist. No source record is overwritten. The
command checks this export's SHA256SUMS, dev/test ranking reductions, stored
predicted-box IoUs under C0/C1, paired targets, declared split separation,
C0 threshold locks, per-seed means/SD, query/class-macro endpoints, paired
percentile intervals and deterministic two-policy sensitivity bounds. It
uses shipped bootstrap draws and checks their deterministic seed. It emits
recomputed_estimates.csv, recomputed_paired_contrasts.csv and detailed checks.
Candidate-score-change rows require unshipped scores and are explicitly
skipped. Extra descriptive prior/F comparisons emitted by the audit are
review-stage computations, not retroactively frozen experimental endpoints;
they are not used in the paper.

Scientific arrays, rankings, thresholds and table values preserve the run.
EXPORT_BINDINGS.json discloses metadata/path derivatives. Historical lock
hashes identify internal originals; they are not silently recomputed or
claimed to bind changed export metadata. SHA256SUMS binds the current export.

## Optional saved neural score/map tier

No raw shards or checkpoint binaries are included. An explicitly supplied
extraction root must contain the project-relative paths recorded in
results/SAVED_NEURAL_OUTPUT_LOCATIONS.json:

```sh
python openlogo/recompute.py --out /new/raw_check --neural-root /external/neural_bundle
python openlogo/recompute.py --out /new/map_check --neural-root /external/neural_bundle --recompute-maps
```

Without --neural-root: SKIP_REQUIRES_UNSHIPPED_NEURAL_SHARDS. With it: all
chunk hashes/finiteness and score-to-ranking reductions are checked.
--recompute-maps additionally needs torch/torchvision/scipy and reproduces
all test query/view/readout prediction boxes, IoU and EIB with the native
CPU map pipeline and both policies at locked thresholds. It does not
regenerate maps from checkpoints. Explicit missing files fail; missing
optional dependencies are SKIP. This optional path is separate from the
tested archive-only endpoint; archive-only full reproduction is not claimed.

## Optional licensed data / checkpoint tiers

Obtain the data from author links in data_provenance/. Academic-use
restrictions and owners' copyrights apply. No dataset image is bundled.

```sh
python openlogo/recompute.py --out /new/target_check --products-dir /licensed/openlogo/JPEGImages
python openlogo/recompute.py --out /new/checkpoint_check --checkpoint-root /external/checkpoint_tree
```

The first checks retained file hashes and actual native-loader targets under
both policies, not official semantics or global annotation correctness. The
second checks all 12 binary digests at paths in results/CHECKPOINTS.csv;
it never loads tensors. Without inputs, both report distinct SKIP statuses.
No images/checkpoints are fetched. The cached ImageNet initializer digest is
in config.json; a future rerun needs an explicit weights binding.

reference_code/ contains actual numerical definitions for inspection, with
OPENLOGO_DATA_ROOT and OPENLOGO_RUN_ROOT replacing the old workspace. Public
relative manifest IDs are retained. Reconstruct that relative layout under
an explicit data root only for a separately authorized rerun. Scheduler
execution is disabled; these are not the CPU recompute entry point and this
release grants no GPU authorization.

The independent review checked saved records, not raw licensed images,
checkpoint tensors, training or map generation. Integration checks are
local recomputations/builds, not independent reviewer reproduction. Group
mapping, coordinate uncertainty and unknown base-pretraining overlap retain
their stated limits. Transfer A remains NOT_RUN_SOURCE_UNSEEN_UNRESOLVED.
