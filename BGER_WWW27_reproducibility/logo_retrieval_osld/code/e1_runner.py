#!/usr/bin/env python3
"""Guarded production orchestration for OSLD-SingleLogo-Retrieval-v1.

Modes: fixture (CPU, generated OSLD-format files), b0 (bounded real timing on
train/official-val only; official test is never built), core (requires an
approved allocation plan). Arms: CE, Aux-BCE, BCE-BGER, CAM-BCE, ABN-box.
"""
from pathlib import Path
import argparse,gc,json,os,signal,socket,sys,time,traceback
import torch
import torchvision as tv
from e1_common import RECIPE,ARMS,save,sha,digest,now,log_writer,append_jsonl
from e1_guards import gate,code_digest
from osld_data import scan,build_sets,OSLDImageDataset
from e1_train import init_template,train,model_from
import e1_evaluate
from e1_evaluate import embed,rank,evaluate,choose,cached_embeddings
from e1_analysis import analyze


def load_records(settings,root,mode):
    if mode=='fixture':
        return scan(root,workers=1)
    s=json.loads(Path(settings['records_file']).read_text())
    if digest(s['product_records']+s['logo_records'])!=settings['expected_records_digest']:
        raise PermissionError('Records differ from the verified data receipt')
    return s


def b0_subsets(sets,n_queries):
    """Deterministic train/dev-only timing subsets (no official test)."""
    q=sorted(sets['dev_query'],key=lambda r:r['image_id'])[:n_queries]
    ids={r['product_id'] for r in q}
    rel=[r for r in sets['dev_gallery'] if r['product_id'] in ids]
    other=[r for r in sorted(sets['dev_gallery'],key=lambda r:r['image_id']) if r['product_id'] not in ids]
    return q,rel+other[:max(0,1000-len(rel))]


def release(device,log,where):
    """Free genuinely unreferenced objects and cached CUDA blocks between runs (allocator only;
    no change of batch, precision, BN, loss or resolution)."""
    gc.collect()
    if device=='cuda':
        torch.cuda.empty_cache()
        log('cuda_memory',where=where,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())


class ArmIncomplete(Exception):pass


def train_guarded(arm,lam,seed,template,dataset,rd,settings,log,identity,out):
    """One retry from the last checkpoint after a CUDA OOM, after releasing memory outside the
    exception scope. A second OOM is recorded; ABN-box is then marked incomplete (pre-stated
    rule: it enters test only if all of its planned runs complete); any other arm stops the run."""
    for attempt in (1,2):
        oom=None
        try:return train(arm,lam,seed,template,dataset,rd,settings,log,identity)
        except torch.cuda.OutOfMemoryError as exc:oom=str(exc)[:2000]
        rec=dict(utc=now(),event='cuda_oom',arm=arm,lam=lam,seed=seed,attempt=attempt,error=oom,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        append_jsonl(out/'FAILURES.jsonl',rec);log('cuda_oom',**{k:v for k,v in rec.items() if k not in ('utc','event')})
        release(settings['device'],log,'after_oom')
    if arm!='abn_box':raise RuntimeError(f'Repeated CUDA OOM in core arm {arm}')
    raise ArmIncomplete(arm)


def run(args):
    root=Path(args.data_root).resolve();out=Path(args.out).resolve()
    settings=gate(args.mode,root,out,args.fixture_epochs,args.fixture_steps)
    out.mkdir(parents=True,exist_ok=True);identity=settings.pop('identity')
    log=log_writer(out/'events.jsonl',identity)
    lock=dict(identity=identity,recipe=RECIPE,settings=settings,data_root=str(root),source_digest=code_digest())
    if (out/'locked_config.json').exists() and json.loads((out/'locked_config.json').read_text())!=lock:raise ValueError('Output contains a different frozen config / source / input')
    save(out/'locked_config.json',lock)
    torch.set_num_threads(args.cpu_threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    e1_evaluate.LOADER_WORKERS=int(settings.get('loader_workers',0))
    if settings['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('Reserved real GPU allocation has no CUDA device')
    t_start=time.monotonic()
    def on_term(signum,frame):
        append_jsonl(out/'FAILURES.jsonl',dict(utc=now(),event='SIGTERM_interrupted',slurm_job_id=os.environ.get('SLURM_JOB_ID')))
        raise SystemExit(143)
    signal.signal(signal.SIGTERM,on_term)
    append_jsonl(out/'RUN_ATTEMPTS.jsonl',dict(utc=now(),mode=args.mode,slurm_job_id=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname(),
        gpu=torch.cuda.get_device_name(0) if settings['device']=='cuda' else None,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if settings['device']=='cuda' else None,
        complete_checkpoints=sorted(p.parent.name for p in out.glob('models/*/training_state.json') if json.loads(p.read_text())['complete']),
        complete_evaluations=sorted(str(p.parent.relative_to(out)) for p in out.glob('*/*/EVALUATION_COMPLETE.json'))+
                             sorted(str(p.parent.relative_to(out)) for p in out.glob('models/*/dev/EVALUATION_COMPLETE.json'))))
    scanned=load_records(settings,root,args.mode)
    built=build_sets(scanned);sets,labels=built['sets'],built['labels']
    save(out/'technical_integrity.json',dict(identity=identity,image_records=scanned['product_records']+scanned['logo_records']))
    if args.mode=='b0':
        sets['dev_query'],sets['dev_gallery']=b0_subsets(sets,settings['b0_dev_queries'])
        sets.pop('query');sets.pop('gallery')   # official test is never built in B0
    ds={k:OSLDImageDataset(root,v,labels if k=='train' else None) for k,v in sets.items()}
    weights=None if args.mode=='fixture' else tv.models.ResNet50_Weights.IMAGENET1K_V1
    log('data_ready',counts={k:len(v) for k,v in sets.items()},fit_id_count=len(labels),
        boxed_train=sum(bool(r['has_box']) for r in sets['train']),coordinates=settings['convention'],
        weights='NONE_OFFLINE' if weights is None else 'IMAGENET1K_V1',prep_seconds=time.monotonic()-t_start)
    paths={};selected={};all_dev=[];resume_exercised=False;b0_units=[];b0_fixed=None;b0_untested=[];incomplete={}
    # After a pre-test freeze, a failure it records is replayed (not retried), so a resume
    # reproduces the frozen arm set instead of retraining an arm the freeze excluded.
    frozen_failed={a:r['failed_run'] for a,r in json.loads((out/'pretest_freeze.json').read_text()).get('incomplete_arms',{}).items()} if (out/'pretest_freeze.json').exists() else {}
    seeds=[0] if args.mode=='b0' else [0,1,2]
    for seed in seeds:
        template=init_template(len(labels),seed,out,weights,log)
        for arm in ARMS:
            if arm in incomplete:
                log('arm_run_not_attempted_after_failure',arm=arm,seed=seed);incomplete[arm]['not_attempted'].append(seed);continue
            if args.mode=='b0' and settings.get('b0_deadline_seconds'):
                # Stop before an arm that would not finish within the allocation; record it as untested.
                last=b0_units[-1]['train_wall_seconds']+b0_units[-1]['dev_eval_wall_seconds'] if b0_units else 0.0
                if time.monotonic()-t_start+1.3*last>settings['b0_deadline_seconds']:
                    log('b0_arm_skipped_deadline',arm=arm);b0_untested.append(arm);continue
            if args.mode=='b0':lams=[0.] if arm=='ce' else [2.]
            else:lams=[0.] if arm=='ce' else ([.5,2.] if seed==0 else [selected[arm]])
            candidates=[]
            for lam in lams:
                tag=f'{arm}_l{lam:g}_s{seed}';rd=out/'models'/tag
                if args.mode=='fixture' and not resume_exercised:
                    train(arm,lam,seed,template,ds['train'],rd,settings,log,identity,stop_after_steps=1)
                    resume_exercised=True
                t0=time.monotonic();release(settings['device'],log,'before_train_'+tag)
                try:
                    if frozen_failed.get(arm)==tag:log('frozen_failure_replayed',arm=arm,run=tag);raise ArmIncomplete(arm)
                    ckpt,complete=train_guarded(arm,lam,seed,template,ds['train'],rd,settings,log,identity,out)
                except ArmIncomplete:
                    incomplete[arm]=dict(failed_run=tag,not_attempted=[]);candidates=[];break
                t_train=time.monotonic()-t0
                if not complete:raise RuntimeError('Incomplete final checkpoint cannot enter selection')
                release(settings['device'],log,'before_dev_eval_'+tag)
                ckpt_sha=sha(ckpt);model=model_from(ckpt,arm,settings['device'])
                t0=time.monotonic()
                if args.mode=='b0' and arm=='ce':
                    # CE-reference fixed pairs on the timing subset, mirroring the test rule
                    qz,gz=embed(model,ds['dev_query'],settings['device'],log),embed(model,ds['dev_gallery'],settings['device'],log)
                    b0_fixed=rank(qz,gz,ds['dev_query'].records,ds['dev_gallery'].records)[:settings['b0_readout_queries']]
                summary=evaluate(model,ds['dev_query'],ds['dev_gallery'],settings['device'],rd/'dev',log,identity,
                                 fixed_pairs=b0_fixed if args.mode=='b0' else None,
                                 embeddings=(qz,gz) if (args.mode=='b0' and arm=='ce') else None,
                                 binding=dict(stage='dev',checkpoint_sha256=ckpt_sha,source_digest=code_digest()))
                t_eval=time.monotonic()-t0
                del model
                c=dict(arm=arm,seed=seed,lambda_value=lam,summary=summary,checkpoint=str(ckpt),checkpoint_sha256=ckpt_sha)
                all_dev.append(c);candidates.append(dict(c,**{'lambda':lam}))
                b0_units.append(dict(arm=arm,lam=lam,train_wall_seconds=t_train,dev_eval_wall_seconds=t_eval,
                                     dev_queries=len(ds['dev_query']),dev_gallery=len(ds['dev_gallery']),
                                     fixed_pair_readouts=0 if args.mode!='b0' else 2*len(b0_fixed),
                                     timing_file=str(rd/'timing.json')))
                if args.mode=='b0':
                    save(out/'RUN_STATUS.json',dict(identity=identity,status='B0_PARTIAL_IN_PROGRESS',official_test_accessed=False,
                         units=b0_units,elapsed_wall_seconds=time.monotonic()-t_start))
            if not candidates or arm in incomplete:continue
            choice=choose(candidates)
            if seed==0:selected[arm]=choice['lambda']
            paths[arm,seed]=choice
    # Pre-stated: an arm with a failed planned run enters no test evaluation (all of its seeds dropped).
    for arm in incomplete:
        for key in [k for k in paths if k[0]==arm]:paths.pop(key)
        selected.pop(arm,None)
    save(out/'all_dev_candidates.json',dict(identity=identity,candidates=all_dev))
    if args.mode=='b0':
        result=dict(identity=identity,status='BOUNDED_TRAIN_DEV_TIMING_COMPLETE',official_test_accessed=False,
                    units=b0_units,untested_arms=b0_untested,total_wall_seconds=time.monotonic()-t_start,
                    note='timing only; dev numbers of 30-step models are not results and select nothing',
                    allocation_accounting='full scheduler allocation reserved including validation/readouts/overhead; no automatic refund')
        save(out/'RUN_STATUS.json',result);log('b0_complete',result=result);return result
    freeze=dict(identity=identity,created_utc=now(),kind='internal pre-test freeze',source_digest=code_digest(),
                selected_lambdas=selected,records_digest=digest(scanned['product_records']+scanned['logo_records']),
                label_map_digest=digest(labels),arms_tested=sorted({a for a,_ in paths}),incomplete_arms=incomplete,
                candidates_sha256=sha(out/'all_dev_candidates.json'),
                checkpoints=[dict(arm=a,seed=s,lambda_value=c['lambda'],checkpoint_sha256=c['checkpoint_sha256'],
                                  thresholds=c['summary']['thresholds']) for (a,s),c in paths.items()],
                analysis=dict(family=['H1-logo R>-0.01','H2-logo Q>0'],reference='CE',target='BGER',view='own_top1',readout='primary',
                              method='query logo-identity cluster centered/basic bootstrap; fixed-seed mean; Holm',draws=10000,rng_seed=0,alpha=.05),
                fixed_pair_rule='CE seed0 final; highest relevant and highest non-relevant canonical template; exact score then canonical image ID')
    freeze_path=out/'pretest_freeze.json'
    if freeze_path.exists():
        previous=json.loads(freeze_path.read_text())
        comparable=lambda x:{k:v for k,v in x.items() if k!='created_utc'}
        if comparable(previous)!=comparable(freeze):raise ValueError('Pre-test freeze changed on resume')
    else:save(freeze_path,freeze)
    log('all_selection_locked_before_test',freeze_sha256=sha(freeze_path))
    # Official test loader/embeddings first touched after all selected models and dev locks.
    release(settings['device'],log,'before_test')
    test_binding=lambda c:dict(stage='test',checkpoint_sha256=c['checkpoint_sha256'],source_digest=code_digest())
    ref=paths['ce',0];m=model_from(ref['checkpoint'],'ce',settings['device'])
    # Cached under test/ce_s0 and read back by its evaluation and by any resume (no re-embedding).
    qz,gz=cached_embeddings(m,ds['query'],ds['gallery'],settings['device'],out/'test'/'ce_s0',log,dict(test_binding(ref),identity=identity))
    fixed=rank(qz,gz,ds['query'].records,ds['gallery'].records)
    if any(r['negative'] is None for r in fixed):raise ValueError('CE reference fixed pairs require non-relevant candidates')
    pair_manifest=dict(identity=identity,reference_checkpoint_sha256=ref['checkpoint_sha256'],
                       created_utc=now(),pretest_freeze_sha256=sha(out/'pretest_freeze.json'),
                       description='CE-reference-conditioned; not model-neutral hard negatives',pairs=fixed)
    pairpath=out/'fixed_candidate_manifest.json'
    if pairpath.exists():
        old=json.loads(pairpath.read_text())
        if old['pairs']!=fixed or old['reference_checkpoint_sha256']!=ref['checkpoint_sha256'] or old['pretest_freeze_sha256']!=sha(out/'pretest_freeze.json'):raise ValueError('Fixed pair manifest cannot be changed on resume')
    else:save(pairpath,pair_manifest)
    log('fixed_pairs_persisted_before_maps',manifest_sha256=sha(pairpath));del m,qz,gz
    results=[]
    for (arm,seed),c in paths.items():
        if sha(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('Checkpoint changed after selection lock')
        release(settings['device'],log,f'before_test_{arm}_s{seed}')
        m=model_from(c['checkpoint'],arm,settings['device']);rd=out/'test'/f'{arm}_s{seed}'
        summary=evaluate(m,ds['query'],ds['gallery'],settings['device'],rd,log,identity,
                         thresholds=c['summary']['thresholds'],fixed_pairs=fixed,
                         binding=dict(test_binding(c),fixed_pair_manifest_sha256=sha(pairpath),pretest_freeze_sha256=sha(freeze_path)))
        results.append(dict(arm=arm,seed=seed,path=rd,summary=summary));del m
    inference=analyze(results,out,identity)
    result=dict(identity=identity,status='COMPLETE',production_entrypoint='code/e1_runner.py',optimizer_steps=sum(json.loads((Path(c['checkpoint']).parent/'training_state.json').read_text())['optimizer_steps'] for c in all_dev),
                model_runs=len(all_dev),selected_test_runs=len(results),new_gpu_seconds=0 if args.mode=='fixture' else 'see_reserved_allocation_ledger',
                pretrained=weights is not None,resume_exercised=resume_exercised,
                max_score_error=max(r['summary']['max_ranking_readout_error'] for r in results),
                max_secondary_error=max((r['summary']['max_secondary_additivity_error'] for r in results if r['summary']['max_secondary_additivity_error'] is not None),default=None),
                wrong_top1_included=sum(r['summary']['incorrect_top1_included'] for r in results),
                official_osld_results=args.mode=='core',statistical_scope=inference['scope'],
                arms_tested=sorted({a for a,_ in paths}),incomplete_arms=incomplete,
                five_arm_study_complete=not incomplete and len({a for a,_ in paths})==len(ARMS))
    save(out/'RUN_STATUS.json',result);log('run_complete',result=result);return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['fixture','b0','core'],required=True)
    ap.add_argument('--data-root',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fixture-epochs',type=int,default=2);ap.add_argument('--fixture-steps',type=int,default=1)
    ap.add_argument('--cpu-threads',type=int,default=2)
    args=ap.parse_args()
    try:run(args)
    except Exception as exc:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        rec=dict(utc=now(),mode=args.mode,identity='SYNTHETIC_INTEGRATION' if args.mode=='fixture' else 'REAL_EXECUTION_NOT_COMPLETED',slurm_job_id=os.environ.get('SLURM_JOB_ID'),error=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
        save(out/'FAILURE.json',rec);append_jsonl(out/'FAILURES.jsonl',rec)
        traceback.print_exc();sys.exit(2)
if __name__=='__main__':main()
