#!/usr/bin/env python3
"""Recompute the OSLD core statistics and tables from the shipped per-query records (CPU only).

Step 1 (no data needed): rerun the frozen inference (code/e1_analysis.analyze: H1-logo/H2-logo
with Holm and every descriptive contrast, 10,000 shared query-identity draws) on results/core and
compare with the shipped primary_inference.json / descriptive_inference.json.
Step 3 (post hoc, saved maps only): the readout analysis of the revised paper
(analysis/readout_identity_analysis.py; the centre prior with its own validation
threshold needs the records of step 2).
Step 2 (needs your copy of the OSLD release records, from code/osld_release_check.py):
restore the per-pair map files, rerun analysis/candidate_sensitivity.py (A5) and
analysis/descriptive_extra.py, then analysis/core_tables.py, and compare with results/analysis/.

Usage:  python recompute_core.py --work /tmp/osld_recompute [--records <dir>/osld_records.json]
"""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
EXPECTED_RECORDS_DIGEST = 'cd19f0f24960acf19a0c23ed33975e288d6c2fcc31cfd9eb8af49c7d9f65bf7f'


def same_json(a, b, tol=0.0):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    return a == b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--records')
    a = ap.parse_args(); W = Path(a.work); W.mkdir(parents=True, exist_ok=True)
    core = HERE / 'results' / 'core'
    from e1_analysis import analyze
    freeze = json.loads((core / 'pretest_freeze.json').read_text())
    outputs = []
    for arm in freeze['arms_tested']:
        for s in range(3):
            d = core / 'test' / f'{arm}_s{s}'
            outputs.append(dict(arm=arm, seed=s, path=d, summary=json.loads((d / 'summary.json').read_text())))
    out = W / 'inference'; out.mkdir(exist_ok=True)
    analyze(outputs, out, freeze['identity'])
    report = {}
    for name in ['primary_inference.json', 'descriptive_inference.json']:
        report[name] = same_json(json.loads((out / name).read_text()), json.loads((core / name).read_text()))
    print('step 1 (inference from per-query records):', report)
    if a.records:
        recs = json.loads(Path(a.records).read_text())
        from e1_common import digest
        if digest(recs['product_records'] + recs['logo_records']) != EXPECTED_RECORDS_DIGEST:
            raise SystemExit('Records differ from the release used in the study')
        run = W / 'run'
        if run.exists(): shutil.rmtree(run)
        shutil.copytree(core, run, ignore=shutil.ignore_patterns('query_signed_maps.npz'))
        (run / 'technical_integrity.json').write_text(json.dumps(dict(identity=freeze['identity'],
                                                      image_records=recs['product_records'] + recs['logo_records'])))
        for d in sorted((core / 'test').iterdir()):
            z = np.load(d / 'query_signed_maps.npz'); md = run / 'test' / d.name / 'signed_maps'; md.mkdir(parents=True, exist_ok=True)
            for i, n in enumerate(z['names']):
                np.savez_compressed(md / f'{n}.npz', identity=np.array(freeze['identity']),
                                    query_primary_signed=z['query_primary_signed'][i], query_secondary_signed=z['query_secondary_signed'][i])
        py = sys.executable
        subprocess.run([py, str(HERE / 'analysis' / 'candidate_sensitivity.py'), '--run-root', str(run), '--out', str(W / 'A5.json')], check=True)
        subprocess.run([py, str(HERE / 'analysis' / 'descriptive_extra.py'), '--run-root', str(run), '--out', str(W / 'descriptive_extra.json')], check=True)
        subprocess.run([py, str(HERE / 'analysis' / 'core_tables.py'), '--run-root', str(run), '--a5', str(W / 'A5.json'), '--out', str(W / 'tables')], check=True)
        ship = HERE / 'results' / 'analysis'
        report['A5_aggregate'] = same_json({k: v['aggregate'] for k, v in json.loads((W / 'A5.json').read_text())['runs'].items()},
                                           {k: v['aggregate'] for k, v in json.loads((ship / 'candidate_sensitivity_A5.json').read_text())['runs'].items()}, 1e-12)
        report['descriptive_extra'] = same_json(json.loads((W / 'descriptive_extra.json').read_text()), json.loads((ship / 'descriptive_extra.json').read_text()), 1e-12)
        report['core_results'] = same_json(json.loads((W / 'tables' / 'core_results.json').read_text()), json.loads((ship / 'core_results.json').read_text()), 1e-12)
        print('step 2 (A5, descriptive, tables):', report)
    else:
        print('SKIPPED_REQUIRES_LICENSED_DATA: A5 annotation overlap, size strata, numeric-example boxes and centre-prior threshold; supply --records')
    # step 3: post-hoc readout analysis (identity on saved maps, A5 split; prior with records)
    cmd = [sys.executable, str(HERE / 'analysis' / 'readout_identity_analysis.py'), '--core', str(core),
           '--a5', str(HERE / 'results' / 'analysis' / 'candidate_sensitivity_A5.json'), '--out', str(W / 'readout_identity_analysis.json')]
    if a.records: cmd += ['--records', a.records]
    subprocess.run(cmd, check=True)
    mine = json.loads((W / 'readout_identity_analysis.json').read_text())
    ship = json.loads((HERE / 'results' / 'analysis' / 'readout_identity_analysis.json').read_text())
    keys = ['structural', 'identity_at_F', 'a5_split'] + (['centre_prior_own_threshold'] if a.records else [])
    report['readout_identity_analysis'] = all(same_json(mine[k], ship[k], 1e-12) for k in keys)
    print('step 3 (readout analysis):', report['readout_identity_analysis'])
    (W / 'recompute_report.json').write_text(json.dumps(report, indent=1) + '\n')
    sys.exit(0 if all(report.values()) else 1)


if __name__ == '__main__':
    main()
