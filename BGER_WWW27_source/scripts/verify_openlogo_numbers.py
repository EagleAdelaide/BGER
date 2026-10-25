#!/usr/bin/env python3
"""CPU endpoint recomputation and source-input binding for the new evidence."""
from pathlib import Path
import argparse, json, os, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source-root',type=Path,default=ROOT)
ap.add_argument('--supp-root',type=Path,default=ROOT.parent/'BGER_WWW27_reproducibility')
# Other build arguments belong to the original OSLD verifier, run separately.
args,_=ap.parse_known_args()
supp=args.supp_root/'openlogo'
if not (supp/'recompute.py').is_file():raise SystemExit('MISSING_OPENLOGO_SAVED_RECORDS: extract the updated supplement adjacent to source, or use --supp-root')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='bger_openlogo_cpu_') as tmp:
    out=Path(tmp)/'recomputed'
    proc=subprocess.run([sys.executable,str(supp/'recompute.py'),'--root',str(supp),'--source-root',str(args.source_root),'--out',str(out)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (ROOT/'openlogo_verification.log').write_text(proc.stdout)
    if proc.returncode:raise SystemExit('OpenLogo saved-record verification failed; see openlogo_verification.log')
    report=json.loads((out/'RECOMPUTE_REPORT.json').read_text())
    (ROOT/'openlogo_verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('OpenLogo saved records:',report['tier1']['passed'],'checks; raw score rows skipped:',len(report['tier1']['skipped_rows']))
