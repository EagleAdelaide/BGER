# CPU reproduction from a clean checkout

Use Python 3.9+ with `reproduce/requirements-cpu.txt`. The optional tensor/metric
fixtures additionally use the packages in `reproduce/requirements-fixtures.txt`.
The recorded training environment is preserved in the historical requirements;
it is distinct from this tested CPU environment. No command below downloads data,
weights or models, and no command performs training or checkpoint inference.

Run from the repository root, using an output directory that does not exist:

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python reproduce/run_cpu_checks.py --out /tmp/bger_review_check
```

The driver verifies the release manifest, compiles Python sources without
executing them, checks synthetic numeric contracts, recomputes OSLD matched and
post-hoc statistics, recomputes derived OpenLogo endpoints/intervals from saved
records, and recomputes CUB per-run results and its 2,000-draw image bootstrap.
It regenerates tables and figures in a temporary copy and checks the numerical
table bytes. Each layer has its own log and output hashes in `CPU_REPORT.json`.
Missing optional dependencies are explicit SKIP, never PASS. Scientific
differences are FAIL. The input tree is not overwritten.

The clean-clone verification accompanying this release masks the original project
and disables network; all outputs go to a new external directory. It verifies
saved records, not independently trained models or the licensed raw images.

## Individual entry points

```sh
# Matched OSLD + saved post-hoc hook/control/reference reductions
python BGER_WWW27_source/figures_src/verify_figure_numbers.py --report /tmp/osld_saved.json

# Original OSLD H1/H2 and descriptive shared query-identity bootstrap
python BGER_WWW27_reproducibility/logo_retrieval_osld/recompute_core.py --work /tmp/osld_core

# Derived OpenLogo, with exact generator-input binding
python BGER_WWW27_reproducibility/openlogo/recompute.py \
  --source-root BGER_WWW27_source --out /tmp/openlogo_saved

# Related CUB classification evidence
python BGER_WWW27_reproducibility/cub/recompute_from_per_image.py
```

OSLD `unlabeled` means a box without its canonical-logo association, as classified
by `logo_retrieval_osld/code/osld_release_check.py`; it does not establish that an
image contains no logo. See `BGER_WWW27_source/data/osld_release_summary.json` for the exact counts.
The smallest OSLD box-area quartile has original upper boundary 0.02046 of image
area (2.046%, displayed as approximately 2.05%). Query assignments are unchanged.

## Correct implementation and optional layers

The CUB matched implementation is
`cub_training/scripts/cvpr2027/run_matched_cub.py` (`MatchedModel`, `balanced_bce`).
`cub_training/methods/bger.py` supplies a common model utility and retains an older
pilot loss; that pilot loss is not the loss used for the reported matched runs.
The paper describes the actual matched trainer.

The OSLD training/evaluation definitions are in `logo_retrieval_osld/code/`;
post-hoc code/configuration and seed records are in `revision_20261020/`.
OpenLogo uses `openlogo/reference_code/b_common.py` and its `native/` definitions;
its actual paired-target adapter is distinct from the OSLD release loader.
The synthetic fixture suite runs no trained model, checkpoint or dataset input.
The archived GPU scheduler/budget fixture is excluded from that CPU suite because
the exported orchestration is deliberately disabled.

Optional licensed OSLD annotation checks require `--records` in the documented
verifier/recompute entry points and otherwise SKIP. OpenLogo's optional raw-shard,
licensed-loader and checkpoint-digest flags are documented in `openlogo/README.md`;
none of those inputs is supplied or implied by the archive-only PASS layer.
Training/re-inference is not authorized by this release. Obtain any future inputs
from the cited official providers under their terms, never from hidden workspace
paths. No reviewer needs to contact the authors or visit a named project origin.
