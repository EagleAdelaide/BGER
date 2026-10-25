# Scientific table and figure sources

This folder supplies saved generator inputs, tables and figures. The manuscript
PDF and full manuscript LaTeX are delivered separately; they are not published by
this code/evidence release. Use the repository-root README_REPRODUCE.md and
reproduce/run_cpu_checks.py for the current tested saved-record workflow.

The driver generates in an external temporary copy, preserving the checkout.
Generators are scripts/make_tables.py, scripts/make_osld_tables.py,
scripts/make_osld_results.py, scripts/make_figure.py,
figures_src/make_publication_figures.py, scripts/make_openlogo.py and
scripts/make_expanded_tables.py. Figure 1 remains editable TikZ and synthetic
artwork, with scripts/build_figure1.sh; no dataset pixels or generated concept
images are imported. requirements-build.txt describes the generator toolchain.
Expanded tables are optional repository evidence and are not appended to the
12-page manuscript. Historical verification JSONs in figure manifests are
provenance; current checks are separately bound in the release verification report.

All scientific input values and hypothesis/uncertainty definitions are retained.
Missing raw/licensed/checkpoint tiers remain SKIP. No training, inference or GPU
execution is carried out by the saved-record workflow.


For the accepted Figure 2 only, run `python figures_src/make_figure2_primary_h.py`
from this folder (Python environment in requirements-build.txt). This regenerates
only Figure 2 and its manifest entry. For exact coordinate and vector regression,
use the repository-root figure-only command in README_REPRODUCE.md. The isolated
Figure 2 style does not alter the other legacy generators.
