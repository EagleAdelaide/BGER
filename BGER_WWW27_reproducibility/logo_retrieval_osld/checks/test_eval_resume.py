#!/usr/bin/env python3
"""Targeted CPU checks of the evaluation-resume / OOM-guard path (SYNTHETIC_INTEGRATION).

Uses a generated OSLD-format noise fixture (make_osld_fixture.py); never real data.
Covers: fresh completion; verified read-back of a completed evaluation (no embed, no
readout); interruption mid-readout and resume that computes only the missing chunks
with byte-identical outputs; identity conflict, tampered output, unbound outputs,
embedding and partial-chunk conflicts all stop without overwriting; dev path
(thresholds locked from pairs); train_guarded retry; and two end-to-end fixture runs
(full rerun = pure reuse; ABN-box OOM -> recorded incomplete arm, four-arm test).
Usage: python test_eval_resume.py --work <empty dir>
"""
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
import torch
import e1_evaluate as E
import e1_runner as R
from osld_data import scan, build_sets, OSLDImageDataset
from train_web_arms import build_model
from e1_common import sha

ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); a = ap.parse_args()
work = Path(a.work).resolve(); work.mkdir(parents=True, exist_ok=True)
fx = work / 'fixture'
subprocess.run([sys.executable, str(CODE / 'make_osld_fixture.py'), '--out', str(fx)], check=True)
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
results = []
def check(name, ok, **info):
    results.append(dict(check=name, passed=bool(ok), **info)); print(('PASS ' if ok else 'FAIL ') + name, info or '', flush=True)
def hashes(d):
    return {f: sha(d / f) for f in E.EVAL_OUTPUTS + ('EVALUATION_COMPLETE.json', 'EVALUATION_IDENTITY.json', 'ranking.json')}
def raises(fn, text):
    try: fn()
    except ValueError as e: return text in str(e)
    return False
events = []
log = lambda ev, **k: events.append(ev)

sets_ = build_sets(scan(fx, workers=1)); sets, labels = sets_['sets'], sets_['labels']
qd, gd = OSLDImageDataset(fx, sets['query']), OSLDImageDataset(fx, sets['gallery'])
dq, dg = OSLDImageDataset(fx, sets['dev_query']), OSLDImageDataset(fx, sets['dev_gallery'])
torch.manual_seed(0); model = build_model(len(labels), 'bger', weights=None)
qz, gz = E.embed(model, qd, 'cpu', log), E.embed(model, gd, 'cpu', log)
fixed = E.rank(qz, gz, qd.records, gd.records)
thr = dict(primary=0.5, secondary=0.5)
bind = dict(stage='test', checkpoint_sha256='c' * 64, source_digest='s' * 64, fixed_pair_manifest_sha256='m' * 64)
E.PAIR_CHUNK = 3
n_jobs = len(qd.records) * 3
orig_pair, orig_embed = E.PairReader.pair, E.embed
calls = dict(pair=0, embed=0)
def counting_pair(self, *x): calls['pair'] += 1; return orig_pair(self, *x)
def counting_embed(*x): calls['embed'] += 1; return orig_embed(*x)
E.PairReader.pair, E.embed = counting_pair, counting_embed
run = lambda d, t=thr, b=bind, f=fixed, q=qd, g=gd: E.evaluate(model, q, g, 'cpu', d, log, 'SYNTHETIC_INTEGRATION', thresholds=t, fixed_pairs=f, binding=b)

# 1 fresh
A = work / 'evalA'; sA = run(A); hA = hashes(A)
check('fresh_complete', (A / 'EVALUATION_COMPLETE.json').exists() and calls['pair'] == n_jobs and calls['embed'] == 2,
      pairs=calls['pair'], jobs=n_jobs, chunks=len(list((A / 'partial').glob('chunk_*.json'))))
# 2 reuse: no embedding, no readout, identical files and summary
calls.update(pair=0, embed=0); mt = (A / 'summary.json').stat().st_mtime_ns
sA2 = run(A)
check('complete_read_back_not_recomputed', calls == dict(pair=0, embed=0) and sA2 == sA and hashes(A) == hA
      and (A / 'summary.json').stat().st_mtime_ns == mt and events[-1] == 'evaluation_reused')
# 3 interruption after 5 readouts (chunk 0 complete, chunk 1 partial), then resume
B = work / 'evalB'; calls.update(pair=0, embed=0)
def failing_pair(self, *x):
    if calls['pair'] >= 5: raise RuntimeError('simulated node failure')
    return counting_pair(self, *x)
E.PairReader.pair = failing_pair
try: run(B); interrupted = False
except RuntimeError: interrupted = True
E.PairReader.pair = counting_pair
partial_before = sorted(p.name for p in (B / 'partial').glob('chunk_*.json'))
calls.update(pair=0, embed=0); sB = run(B)
att = [json.loads(l) for l in (B / 'evaluation_attempts.jsonl').read_text().splitlines()]
check('interrupted_resume_fills_only_missing', interrupted and partial_before == ['chunk_00000.json'] and calls['embed'] == 0
      and calls['pair'] == n_jobs - 3 and att[-1]['chunks_reused'] == 1 and att[-1]['resumed'] is True,
      pairs_recomputed=calls['pair'], expected=n_jobs - 3, embed_calls=calls['embed'])
check('resumed_outputs_identical_to_uninterrupted', sB == sA and all(sha(B / f) == sha(A / f) for f in E.EVAL_OUTPUTS))
# 4 identity conflict (different thresholds / fixed pairs / checkpoint) -> stop, no overwrite
check('threshold_change_refused', raises(lambda: run(A, t=dict(primary=.45, secondary=.5)), 'identity conflict') and hashes(A) == hA)
check('fixed_pair_change_refused', raises(lambda: run(A, f=fixed[::-1]), 'identity conflict') and hashes(A) == hA)
check('checkpoint_change_refused', raises(lambda: run(A, b=dict(bind, checkpoint_sha256='d' * 64)), 'identity conflict') and hashes(A) == hA)
# 5 tampered completed output -> stop, not recompute
raw = (A / 'summary.json').read_bytes(); (A / 'summary.json').write_bytes(raw + b' ')
calls.update(pair=0, embed=0)
check('tampered_complete_refused', raises(lambda: run(A), 'missing/changed') and calls == dict(pair=0, embed=0))
(A / 'summary.json').write_bytes(raw)
# 6 unbound outputs (no identity file) -> refuse
C = work / 'evalC'; C.mkdir(); (C / 'summary.json').write_text('{}')
check('unbound_outputs_refused', raises(lambda: run(C), 'Unbound evaluation outputs'))
# 7 embedding cache identity conflict
check('embedding_identity_refused', raises(lambda: E.cached_embeddings(model, qd, gd, 'cpu', A, log, dict(bind, checkpoint_sha256='e' * 64, identity='SYNTHETIC_INTEGRATION')), 'different identity'))
# 8 partial chunk conflict
D = work / 'evalD'; calls.update(pair=0)
E.PairReader.pair = failing_pair
try: run(D)
except RuntimeError: pass
E.PairReader.pair = counting_pair
cp = D / 'partial' / 'chunk_00000.json'; c = json.loads(cp.read_text()); c['jobs'][0][1] = 'logo-images/other.jpg'; cp.write_text(json.dumps(c))
check('partial_chunk_conflict_refused', raises(lambda: run(D), 'Partial readout identity conflict'))
# 9 dev path: thresholds locked from pairs; interrupted dev resume identical
dbind = dict(stage='dev', checkpoint_sha256='c' * 64, source_digest='s' * 64)
F1, F2 = work / 'devF1', work / 'devF2'
s1 = E.evaluate(model, dq, dg, 'cpu', F1, log, 'SYNTHETIC_INTEGRATION', binding=dbind)
calls.update(pair=0); E.PairReader.pair = lambda self, *x: (_ for _ in ()).throw(RuntimeError('x')) if calls['pair'] >= 3 else counting_pair(self, *x)
try: E.evaluate(model, dq, dg, 'cpu', F2, log, 'SYNTHETIC_INTEGRATION', binding=dbind)
except RuntimeError: pass
E.PairReader.pair = counting_pair
s2 = E.evaluate(model, dq, dg, 'cpu', F2, log, 'SYNTHETIC_INTEGRATION', binding=dbind)
check('dev_resume_identical_with_locked_thresholds', s1 == s2 and s1['thresholds'] == s2['thresholds'] and 'fixed_candidate' not in s1)
E.PairReader.pair, E.embed = orig_pair, orig_embed

# 10 train_guarded: one OOM then success; repeated OOM -> ABN incomplete / core arm stops
(work / 'tg').mkdir(); seq = []
def fake_train(arm, *x):
    seq.append(arm)
    if arm == 'abn_box_once' and len(seq) == 1: raise torch.cuda.OutOfMemoryError('mock OOM')
    if arm in ('abn_box', 'bger'): raise torch.cuda.OutOfMemoryError('mock OOM')
    return 'ckpt', True
orig_train = R.train; R.train = fake_train
ok1 = R.train_guarded('abn_box_once', .5, 0, None, None, Path('x'), dict(device='cpu'), log, 'S', work / 'tg') == ('ckpt', True)
seq.clear()
try: R.train_guarded('abn_box', .5, 0, None, None, Path('x'), dict(device='cpu'), log, 'S', work / 'tg'); ok2 = False
except R.ArmIncomplete: ok2 = len(seq) == 2
try: R.train_guarded('bger', .5, 0, None, None, Path('x'), dict(device='cpu'), log, 'S', work / 'tg'); ok3 = False
except RuntimeError as e: ok3 = 'Repeated CUDA OOM' in str(e)
R.train = orig_train
nfail = len((work / 'tg' / 'FAILURES.jsonl').read_text().splitlines())
check('train_guarded_retry_then_record', ok1 and ok2 and ok3 and nfail == 5, failure_records=nfail)

# 11 end-to-end fixture runner: fresh, then identical rerun = pure reuse
def runner(out):
    R.run(SimpleNamespace(mode='fixture', data_root=str(fx), out=str(out), fixture_epochs=2, fixture_steps=1, cpu_threads=2))
O = work / 'run_full'; t0 = time.time(); runner(O); t_fresh = time.time() - t0
snap = {str(p.relative_to(O)): sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.suffix in ('.json', '.npz', '.pt') and p.name not in ('RUN_STATUS.json',)}
n_events = len((O / 'events.jsonl').read_text().splitlines())
t0 = time.time(); runner(O); t_rerun = time.time() - t0
ev = [json.loads(l) for l in (O / 'events.jsonl').read_text().splitlines()[n_events:]]
snap2 = {k: sha(O / k) for k in snap}
reused = sum(e['event'] == 'evaluation_reused' for e in ev); fresh_eval = sum(e['event'] == 'evaluation_complete' for e in ev)
trained = sum(e.get('steps_this_call', 0) for e in ev if e['event'] == 'training_complete')
st = json.loads((O / 'RUN_STATUS.json').read_text())
changed = sorted(k for k in snap if snap[k] != snap2[k])
check('runner_rerun_is_pure_reuse', reused == 34 and fresh_eval == 0 and trained == 0 and st['status'] == 'COMPLETE'
      and st['five_arm_study_complete'] is True and set(changed) <= {'all_dev_candidates.json', 'primary_inference.json', 'descriptive_inference.json', 'technical_integrity.json', 'primary_bootstrap_replicates.npz'},
      evaluations_reused=reused, fresh_evaluations=fresh_eval, new_steps=trained, changed_files=changed,
      fresh_seconds=round(t_fresh, 1), rerun_seconds=round(t_rerun, 1))
check('runner_rerun_deterministic_files_unchanged', not changed, changed=changed)

# 12 end-to-end: ABN-box OOM in every attempt -> recorded incomplete, four-arm freeze/test/analysis
def abn_oom(arm, *x, **k):
    if arm == 'abn_box': raise torch.cuda.OutOfMemoryError('mock OOM (fixture)')
    return orig_train(arm, *x, **k)
R.train = abn_oom; O2 = work / 'run_abn_oom'; runner(O2); R.train = orig_train
st2 = json.loads((O2 / 'RUN_STATUS.json').read_text()); fz = json.loads((O2 / 'pretest_freeze.json').read_text())
fails = [json.loads(l) for l in (O2 / 'FAILURES.jsonl').read_text().splitlines()]
check('abn_failure_recorded_incomplete_not_dropped', st2['status'] == 'COMPLETE' and st2['five_arm_study_complete'] is False
      and st2['arms_tested'] == ['aux', 'bger', 'cam_bce', 'ce'] and 'abn_box' in fz['incomplete_arms']
      and fz['incomplete_arms']['abn_box']['not_attempted'] == [1, 2] and len(fails) == 2
      and not (O2 / 'test' / 'abn_box_s0').exists(), incomplete=fz['incomplete_arms'], oom_records=len(fails))

# 13 resume after a recorded ABN failure: freeze replayed, ABN not retrained, everything reused
n2 = len((O2 / 'events.jsonl').read_text().splitlines()); fz_raw = (O2 / 'pretest_freeze.json').read_bytes()
runner(O2)
ev2 = [json.loads(l) for l in (O2 / 'events.jsonl').read_text().splitlines()[n2:]]
check('resume_after_abn_failure_replays_freeze', (O2 / 'pretest_freeze.json').read_bytes() == fz_raw
      and sum(e['event'] == 'frozen_failure_replayed' for e in ev2) == 1 and sum(e['event'] == 'evaluation_complete' for e in ev2) == 0
      and sum(e.get('steps_this_call', 0) for e in ev2 if e['event'] == 'training_complete') == 0
      and json.loads((O2 / 'RUN_STATUS.json').read_text())['five_arm_study_complete'] is False)

ok = all(r['passed'] for r in results)
report = dict(identity='SYNTHETIC_INTEGRATION', scope='evaluation resume / OOM guard affected path only', passed=ok,
              n_checks=len(results), checks=results, code_digest=R.code_digest(),
              code_files={p.name: sha(p) for p in sorted(CODE.glob('*.py'))})
(work / 'eval_resume_report.json').write_text(json.dumps(report, indent=1) + '\n')
print('ALL PASS' if ok else 'FAILURES PRESENT'); sys.exit(0 if ok else 1)
