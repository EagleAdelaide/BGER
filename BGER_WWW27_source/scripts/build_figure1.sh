#!/usr/bin/env bash
# Only Figure 1: editable TikZ -> vector PDF; optional vector SVG and 600-dpi PNG.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1792108800}"
export FORCE_SOURCE_DATE=1
mkdir -p figures
pdflatex -interaction=nonstopmode -halt-on-error -jobname=fig01_supervision_placement -output-directory=figures figures_src/fig01_standalone.tex > build_figure1.log
pdflatex -interaction=nonstopmode -halt-on-error -jobname=fig01_artwork -output-directory=figures figures_src/fig01_artwork_standalone.tex > build_figure1_assets.log
cp figures/fig01_supervision_placement.pdf figures/fig01_instance_grounded.pdf
if command -v pdftocairo >/dev/null 2>&1; then
  pdftocairo -svg figures/fig01_instance_grounded.pdf figures/fig01_instance_grounded.svg
  pdftocairo -png -singlefile -r 600 figures/fig01_instance_grounded.pdf figures/fig01_instance_grounded
else
  printf '%s\n' 'SVG/PNG export skipped: pdftocairo is not on PATH; vector PDF and editable TikZ are built.'
fi
