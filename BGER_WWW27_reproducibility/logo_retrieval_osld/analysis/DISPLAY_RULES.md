# Display and descriptive-analysis rules (fixed before any core test output exists)

Written 2026-10-16 while core segment 1 (job 16377607) was still queued: no
OSLD test-model output exists. These rules choose what is shown; they do not
change any endpoint, threshold, hypothesis or statistic of
OSLD-SingleLogo-Retrieval-v1. Everything below is descriptive.

1. **Small-logo boundary.** Test queries are binned by box-area fraction
   (box area / image area) into quartiles computed from the test annotations alone
   (model-independent). For each arm: query Loc (primary, locked threshold),
   R@1 and Jq@1 per bin, seed-averaged, with per-seed values. Reported for all
   arms whichever leads.
2. **Correct vs wrong top-1.** Query Loc (primary) separately for queries whose
   own top-1 template is relevant and not relevant, per arm, seed-averaged.
3. **A5 candidate sensitivity.** `analysis/candidate_sensitivity.py` on the saved
   fixed-pair maps of all tested runs; per arm (seed-averaged): score change,
   verdict-change rate, L1 distance, defined-correlation mean and undefined
   count, degenerate rates for d+ and d-, Loc for d+ and d-, and the centre-prior
   reference at the same locked threshold.
4. **Map examples (numeric only; no OSLD photographs are reproduced).** Seed-0
   primary query-side signed maps for the own top-1 pair, drawn as feature-grid
   heat values with the query box projected onto the grid, for all tested arms.
   Queries: among test queries whose CE-seed-0 own top-1 is relevant, those at
   the 25th, 50th and 75th percentile (nearest rank, ties by image ID) of
   box-area fraction; plus, among queries whose CE-seed-0 own top-1 is not
   relevant, the one at the median box-area fraction. Shown whether or not they
   favour any arm. Maps are read from saved `signed_maps/*.npz`; nothing is
   generated or re-rendered from a model.
5. **Failures.** Degenerate and non-finite map counts per arm and readout,
   any `FAILURES.jsonl` records, and incomplete arms are reported in full.
