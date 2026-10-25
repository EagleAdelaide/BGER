"""Allocation guard for the 2026-10-20 revision round (author approval: <=10.5 GPU-h, per-task caps).

Reserves the FULL scheduler walltime x GPUs of the running Slurm job at job start against
`revision_allocation_ledger.json`, refusing if the task cap or the overall cap would be
exceeded, if the job was already reserved, or if no Slurm allocation exists. Actual elapsed
time is recorded separately by `record_actual()` after the job (from sacct); it never
refunds a reservation. Never touches the core/B0/B1 ledgers.
"""
import fcntl, json, os, subprocess, sys
from pathlib import Path

E1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(E1 / 'code'))
from e1_guards import parse_allocation  # noqa: E402
from e1_common import now, save  # noqa: E402

LEDGER = Path(__file__).resolve().parents[1] / 'revision_allocation_ledger.json'
TASKS = ('p0a_bger_pooled_readout', 'p0b_gate_controls', 'p0c_frozen_baselines', 'p0c_proxy_anchor')


def reserve(task):
    raise PermissionError('No new GPU allocation is authorized for the V2 evidence-preserving patch')
    if task not in TASKS: raise PermissionError('Unknown revision task ' + task)
    job = os.environ.get('SLURM_JOB_ID')
    if not job: raise PermissionError('Revision GPU work requires a Slurm allocation (no SLURM_JOB_ID)')
    raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
    seconds, info = parse_allocation(raw, job, os.getuid())
    if not LEDGER.is_file(): raise PermissionError('revision ledger missing; never create from a job')
    with LEDGER.with_suffix('.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.loads(LEDGER.read_text())
        if any(r['job_id'] == job for r in d['reservations']): raise PermissionError('job already reserved')
        used_task = sum(r['reserved_gpu_seconds'] for r in d['reservations'] if r['task'] == task)
        used_all = sum(r['reserved_gpu_seconds'] for r in d['reservations'])
        if used_task + seconds > d['task_caps_gpu_seconds'][task]:
            raise PermissionError(f'task cap exceeded for {task}: {used_task}+{seconds} > {d["task_caps_gpu_seconds"][task]}')
        if d['initial_charged'] + used_all + seconds > d['cap_allocated_gpu_seconds']:
            raise PermissionError('overall revision cap would be exceeded')
        d['reservations'].append(dict(job_id=job, task=task, utc=now(), reserved_gpu_seconds=seconds,
                                      state='RESERVED_FULL_ALLOCATION_INCLUDING_FAILURES', actual_elapsed_seconds=None, scheduler=info))
        save(LEDGER, d)
    return seconds


def record_actual():
    """Fill actual elapsed seconds (sacct) for finished jobs; reservations are never reduced."""
    d = json.loads(LEDGER.read_text())
    for r in d['reservations']:
        if r.get('actual_elapsed_seconds') is None:
            out = subprocess.run(['sacct', '-n', '-X', '-P', '-j', r['job_id'], '-o', 'State,Elapsed'], capture_output=True, text=True).stdout.strip()
            if out and out.split('|')[0] not in ('RUNNING', 'PENDING', ''):
                st, el = out.split('|')[:2]
                p = [int(x) for x in el.replace('-', ':').split(':')]
                while len(p) < 4: p.insert(0, 0)
                r['actual_elapsed_seconds'] = ((p[0] * 24 + p[1]) * 60 + p[2]) * 60 + p[3]; r['final_state'] = st
    save(LEDGER, d)
    return d


if __name__ == '__main__':
    print(json.dumps({k: v for k, v in record_actual().items() if k != 'basis'}, indent=1, default=str)[:3000])
