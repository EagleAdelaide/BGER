"""Fail-closed local authorization and cumulative allocation reservation.

These are execution guards, not a claim to authenticate legal documents. The
author supplies actual access/release evidence; technical PASS never grants it.
No CLI switch grants core approval. A tracked policy change is required.
"""
from pathlib import Path
import os,json,subprocess,re,fcntl
from e1_common import save,sha,digest,now
ROOT=Path(__file__).resolve().parents[1]
# B0 plans (train/official-val only). 'preflight' is a short GPU check of the same
# path; 'full' measures unit costs. Both are charged to the same cumulative ledger.
B0_PLANS={'preflight':dict(b0_steps=2,b0_dev_queries=8,b0_readout_queries=4),
          'full':dict(b0_steps=30,b0_dev_queries=60,b0_readout_queries=20)}


def code_digest():return digest({p.name:sha(p) for p in sorted((ROOT/'code').glob('*.py'))})
def verify_evidence(entry,root):
    path=Path(entry['path']).resolve()
    if not path.is_file() or sha(path)!=entry['sha256']:raise PermissionError('Evidence file missing or hash differs')
    return path

def _b0_plan():
    plan=os.environ.get('OSLD_B0_PLAN','full')
    if plan not in B0_PLANS:raise PermissionError('Unknown B0 plan')
    return plan

def gate(mode,data_root,out,fixture_epochs=2,fixture_steps=1):
    policy=json.loads((ROOT/'configs/authorization.json').read_text())
    if mode=='fixture':
        manifest=Path(data_root)/'SYNTHETIC_INTEGRATION.json'
        if not manifest.is_file():raise PermissionError('Fixture requires generated-file identity manifest')
        m=json.loads(manifest.read_text())
        if m.get('identity')!='SYNTHETIC_INTEGRATION' or m.get('generator') not in ('make_cpu_fixture.py','make_osld_fixture.py'):raise PermissionError('Not a declared generated fixture')
        if len(m['files'])>128:raise PermissionError('Fixture file count exceeds CPU integration bound')
        for name,h in m['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise PermissionError('Unsafe fixture path')
            if sha(Path(data_root)/name)!=h:raise PermissionError('Fixture input changed: '+name)
        if not(1<=fixture_epochs<=2 and 1<=fixture_steps<=2):raise PermissionError('Fixture bounded to at most two epochs / two steps each')
        return dict(identity='SYNTHETIC_INTEGRATION',device='cpu',weights=None,
                    convention='zero_half_open',epochs=fixture_epochs,max_steps=fixture_steps,batch=2,
                    checkpoint_every=1,timing=False,loader_workers=2)
    if mode=='core' and not policy['core']['approved']:raise PermissionError('CORE_NOT_AUTHORIZED: 15-run core remains frozen')
    if mode=='b0' and not policy['B0']['conditional_authorization']:raise PermissionError('B0 not authorized')
    if policy['data']['status']!='VERIFIED_ACCESS_AND_DATA' or not policy['data']['receipt']:
        raise PermissionError('BLOCKED_DATA_ACCESS: technical fixtures do not confer access')
    receipt_path=Path(policy['data']['receipt'])
    receipt=json.loads((receipt_path if receipt_path.is_absolute() else ROOT/receipt_path).read_text())
    if Path(receipt['root']).resolve()!=Path(data_root).resolve():raise PermissionError('Receipt/data root mismatch')
    # OSLD: public CC BY-NC 4.0 release obtained from the official link under an explicit
    # author instruction; both must be evidenced by hashed files (checked below).
    if receipt.get('authorization') not in ('AUTHOR_VERIFIED','AUTHOR_INSTRUCTED_PUBLIC_LICENSE'):raise PermissionError('Authorization NOT_VERIFIED')
    if receipt.get('release_crop_provenance') not in ('AUTHOR_VERIFIED','RELEASE_README_DOCUMENTED'):raise PermissionError('Release provenance NOT_VERIFIED')
    for key in ['access_evidence','coordinate_evidence','release_evidence']:verify_evidence(receipt[key],data_root)
    if receipt.get('bbox_convention') not in {'zero_half_open','zero_inclusive','one_inclusive'}:raise PermissionError('Actual release coordinate convention NOT_VERIFIED')
    for name,h in receipt['annotation_hashes'].items():
        if sha(Path(data_root)/name)!=h:raise PermissionError('Annotation receipt mismatch')
    ep=policy['engineering']['receipt']
    if not ep:raise PermissionError('Engineering receipt absent')
    ep=Path(ep)
    eng=json.loads((ep if ep.is_absolute() else ROOT/ep).read_text())
    if eng.get('status')!='COMPLETE' or eng.get('code_digest')!=code_digest():raise PermissionError('Engineering receipt stale / incomplete')
    if mode=='core':
        # Future core approval must include a separately bounded allocation plan.
        plan=policy['core'].get('allocation_plan')
        if not plan or not policy['core'].get('approval_reference'):raise PermissionError('Core requires explicit approved allocation plan')
        cap=plan['cap_allocated_gpu_seconds'];initial=plan['already_charged_gpu_seconds']
    else:cap=policy['B0']['cap_allocated_gpu_seconds'];initial=policy['B0']['initial_charged_gpu_seconds']
    reserve_allocation(mode,cap,initial)
    # OSLD B0: real batch-64 timing on train, bounded dev readouts; never official test.
    return dict(identity='OSLD_B0_TIMING' if mode=='b0' else 'OSLD_E1_CORE',device='cuda',
                weights='V1',convention=receipt['bbox_convention'],epochs=1 if mode=='b0' else 15,
                max_steps=B0_PLANS[_b0_plan()]['b0_steps'] if mode=='b0' else None,batch=64,
                **(dict(b0_plan=_b0_plan(),**B0_PLANS[_b0_plan()]) if mode=='b0' else {}),
                expected_records_digest=receipt['technical_records_digest'],
                records_file=str(receipt['records_file']),checkpoint_every=200,timing=(mode=='b0'),loader_workers=6,
                b0_deadline_seconds=float(os.environ.get('OSLD_B0_DEADLINE_S','0')) if mode=='b0' else 0.0)

def reserve_allocation(mode,cap,initial):
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise PermissionError('Real runs require a Slurm bounded GPU allocation')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
    # Installed cluster is Slurm 20.11.7: use its supported oneliner format,
    # not the newer --json interface. Keep the raw scheduler evidence.
    seconds,info=parse_allocation(raw,job,os.getuid())
    policy=json.loads((ROOT/'configs/authorization.json').read_text())
    ledger_root=Path(policy['budget_ledger_root'])
    if not ledger_root.is_absolute():raise PermissionError('Authoritative ledger root must be absolute')
    ledger=ledger_root/f'{mode}_allocation_ledger.json';lock=ledger.with_suffix('.lock')
    if not ledger.is_file():raise PermissionError('Authoritative cumulative ledger missing; never initialize/reset from a new checkout')
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        d=json.loads(ledger.read_text())
        if d['initial_charged']!=initial:raise PermissionError('Budget baseline mismatch; no reset')
        if any(r['job_id']==job for r in d['reservations']):raise PermissionError('Allocation already consumed/reserved; no unaccounted retry')
        if initial+sum(r['reserved_gpu_seconds'] for r in d['reservations'])+seconds>cap:raise PermissionError('Cumulative allocated GPU cap would be exceeded')
        d['reservations'].append(dict(job_id=job,utc=now(),reserved_gpu_seconds=seconds,state='RESERVED_FULL_ALLOCATION_INCLUDING_FAILURES',scheduler=info))
        save(ledger,d)
    # Entire scheduler time limit remains charged conservatively. No auto refund.


def parse_allocation(raw,job,uid):
    def field(key):
        m=re.search(r'(?:^|\s)'+re.escape(key)+r'=([^\s]+)',raw)
        if not m:raise PermissionError('Cannot verify scheduler field '+key)
        return m.group(1)
    if field('JobId')!=str(job) or field('JobState')!='RUNNING':raise PermissionError('Not this active allocation')
    owner=re.fullmatch(r'[^()]+\((\d+)\)',field('UserId'))
    if not owner or int(owner.group(1))!=uid:raise PermissionError('Allocation owner mismatch')
    time=field('TimeLimit')
    m=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})',time)
    if not m:raise PermissionError('Cannot verify finite scheduler TimeLimit')
    day,hour,minute,second=[int(x or 0) for x in m.groups()]
    if minute>=60 or second>=60:raise PermissionError('Invalid scheduler time')
    wall=(day*24+hour)*3600+minute*60+second
    tres=field('AllocTRES') if re.search(r'(?:^|\s)AllocTRES=',raw) else field('TRES')
    entries=dict(x.split('=',1) for x in tres.split(','))
    if 'gres/gpu' in entries:gpus=int(entries['gres/gpu'])
    else:gpus=sum(int(v) for k,v in entries.items() if k.startswith('gres/gpu:'))
    if wall<=0 or gpus<=0:raise PermissionError('Cannot verify positive allocated GPU wall-time')
    return wall*gpus,dict(raw_scontrol_oneliner=raw,job_id=job,gpus=gpus,time_limit_seconds=wall)
