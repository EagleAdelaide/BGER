# Anonymous review code and saved-record evidence

Reviewer entry: https://anonymous.4open.science/r/BGER-6E9C/

This release supplies the research code, frozen protocols, redistributable saved
outputs, numerical tables and figure generators supporting the paper. Start with
[README_REPRODUCE.md](README_REPRODUCE.md) and [CLAIM_TO_FILE.csv](CLAIM_TO_FILE.csv).
The directory names preserve the tested sibling layout used by the reproduction
scripts; they do not imply separate OpenReview attachments.

- `BGER_WWW27_source/`: scientific table/figure generators, their exact input
  records, editable tables and generated figures. The manuscript itself is
  delivered separately; no manuscript PDF or full manuscript text is published here.
- `BGER_WWW27_reproducibility/logo_retrieval_osld/`: matched OSLD code, pre-test
  freeze, saved query records, original H1/H2 statistics and exploratory post-hoc
  hook/control/frozen-reference evidence.
- `BGER_WWW27_reproducibility/openlogo/`: derived OpenLogo protocol, group splits,
  exclusions, paired C0/C1 targets, rankings/metrics, shared cluster draws and
  archive-only CPU recomputation.
- `BGER_WWW27_reproducibility/cub/` and `cub_training/`: related classification
  evidence and the actual matched-generation implementation. CUB is a different
  task; it is not an OSLD/OpenLogo retrieval replication.
- `BGER_WWW27_reproducibility/EXPANDED_TABLES.pdf`: the two-page numerical
  companion, with editable sources in `BGER_WWW27_source/tables/`.

## Reproduction boundaries

Saved-record recomputation is not fresh training, checkpoint inference or neural
map generation. No licensed image pixels, pretrained weights, checkpoint binaries,
embeddings or OpenLogo raw neural shards are included or downloaded. Annotation-
dependent analyses require separately obtained licensed inputs and report SKIP
when missing. Official OpenLogo coordinate semantics remain UNVERIFIED; paired
target agreement protects only the specified retained cohort and C0/C1 policies.
The unknown pretraining overlap and declared-group limitations remain.

H1 is the original OSLD non-inferiority hypothesis; H2 localization superiority
failed. Hook/controls/frozen references and independently trained derived OpenLogo
remain exploratory. Seed SDs are not confidence intervals. Conditional pointwise
intervals are not simultaneous or training-population uncertainty.

`RELEASE_SHA256SUMS.txt` binds this release; `RELEASE_BINDINGS.json` identifies
unchanged baseline scientific bytes and explicit metadata derivatives. Historical
freeze hashes identify original records and are not silently replaced by export
hashes. Internal approval notes, scheduler scripts and identifiers are excluded.

Third-party credits and license notices are retained in
[RIGHTS_AND_ATTRIBUTION.md](RIGHTS_AND_ATTRIBUTION.md) and `THIRD_PARTY_LICENSES/`.
The release does not grant new rights to the datasets or model weights.

Clean-clone CPU validation: [verification/CPU_REPORT.json](verification/CPU_REPORT.json).
All available saved-record layers passed; absent licensed inputs and neural
artifacts remain explicit skips. The scientific manifest binds the tested bytes
across documentation-only finalization.


Figure 2 synchronization: the current five-arm primary-H figure and exact CPU
coordinate/vector regeneration evidence are documented in README_REPRODUCE.md
and verification/FIGURE2_PRIMARY_H_REPORT.json. Exploratory BGER-F remains in the
Table 3/Section 6.1 records. Other published figure assets are retained; this
synchronization changes only Figure 2 presentation and its reproduction bindings.
The earlier broad CPU report is historical; the fresh check is figure-only.
