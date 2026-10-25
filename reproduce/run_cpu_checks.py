#!/usr/bin/env python3
"""Saved-record reproduction; no network, images, weights, training or checkpoint inference."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,argparse,shutil,re,importlib.metadata
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'BGER_WWW27_source';R=ROOT/'BGER_WWW27_reproducibility'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def equal(a,b,tol=1e-12):
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k],tol) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(equal(x,y,tol) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)<=tol
    return a==b

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'scope':'CPU saved-record recomputation plus synthetic code contracts; no training, checkpoint inference or image input',
            'layers':{},'python':sys.version.split()[0],'dependencies':{}}
    for name in ['numpy','pandas','scipy','torch','torchvision','matplotlib','opencv-python']:
        try:report['dependencies'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:report['dependencies'][name]='MISSING'
    def save():
        (out/'CPU_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,command,cwd=ROOT):
        log=out/(name+'.log')
        with log.open('w') as f:
            result=subprocess.run([str(x) for x in command],cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        rec={'status':'PASS' if result.returncode==0 else 'FAIL','returncode':result.returncode,'log':log.name,'log_sha256':sha(log)}
        report['layers'][name]=rec;print(name,rec['status'],flush=True);save()
        return rec
    def skip(name,reason):
        report['layers'][name]={'status':'SKIP','reason':reason};save()
    # Validate the current export, not historical internal freeze hashes.
    failures=[];count=0
    for line in (ROOT/'RELEASE_SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1);count+=1
        if not (ROOT/name).is_file() or sha(ROOT/name)!=digest:failures.append(name)
    report['layers']['release_manifest']={'status':'FAIL' if failures else 'PASS','files':count,'failed':failures}
    errors=[];sources=sorted(ROOT.rglob('*.py'))
    for f in sources:
        try:compile(f.read_text(),str(f.relative_to(ROOT)),'exec')
        except Exception as exc:errors.append({'file':str(f.relative_to(ROOT)),'error':str(exc)})
    report['layers']['python_syntax']={'status':'FAIL' if errors else 'PASS','files':len(sources),'errors':errors};save()
    py=sys.executable
    if report['dependencies']['torch']=='MISSING' or report['dependencies']['torchvision']=='MISSING':
        skip('synthetic_numeric_contracts','Optional tensor fixture dependencies are missing')
    else:
        rec=run('synthetic_numeric_contracts',[py,ROOT/'reproduce/numeric_contracts.py'])
        text=(out/rec['log']).read_text()
        match=re.search(r'Ran (\d+) tests?',text)
        if match:rec['tests_run']=int(match.group(1))
        rec['historical_gpu_budget_fixture']='SKIP_DISABLED_ORCHESTRATION'
    rec=run('osld_saved_reductions',[py,S/'figures_src/verify_figure_numbers.py','--report',out/'OSLD_SAVED.json'])
    if rec['status']=='PASS':
        v=json.loads((out/'OSLD_SAVED.json').read_text())
        rec.update(checks=v['n_checks'],failed=v['n_failed'],tier1=v['tier1'],tier2=v['tier2'])
        if v['n_failed']:rec['status']='FAIL'
    rec=run('osld_original_bootstrap_and_readout',[py,R/'logo_retrieval_osld/recompute_core.py','--work',out/'osld_core'])
    if rec['status']=='PASS':
        v=json.loads((out/'osld_core/recompute_report.json').read_text());rec['exact_comparisons']=v
        if not all(v.values()):rec['status']='FAIL'
    rec=run('posthoc_gate_controls',[py,R/'logo_retrieval_osld/revision_20261020/p0b/p0b_analysis.py','--out',out/'P0B_RESULTS.json'])
    if rec['status']=='PASS':
        got=json.loads((out/'P0B_RESULTS.json').read_text());expected=json.loads((S/'data/p0b_results.json').read_text())
        rec['all_saved_results_equal']=equal(got,expected)
        if not rec['all_saved_results_equal']:rec['status']='FAIL'
    rec=run('frozen_references',[py,R/'logo_retrieval_osld/revision_20261020/p0c/p0c_audit.py','--out',out/'P0C_RESULTS.json'])
    if rec['status']=='PASS':
        got=json.loads((out/'P0C_RESULTS.json').read_text());expected=json.loads((S/'data/p0c_results.json').read_text())
        tested=[];bad=[]
        for tag,vals in got['results'].items():
            for field in ['recall1','recall10','mAP','contrast_vs_BGER_R1','contrast_vs_CE_R1']:
                tested.append(tag+'/'+field)
                if not equal(vals[field],expected['results'][tag][field]):bad.append(tag+'/'+field)
        rec.update(saved_metric_comparisons=len(tested),failed_comparisons=bad,annotation_quartiles='SKIP_REQUIRES_LICENSED_ANNOTATIONS')
        if bad:rec['status']='FAIL'
    rec=run('openlogo_archive_only',[py,R/'openlogo/recompute.py','--source-root',S,'--out',out/'openlogo'])
    if rec['status']=='PASS':
        v=json.loads((out/'openlogo/RECOMPUTE_REPORT.json').read_text())
        rec.update(checks=v['tier1']['checks'],passed=v['tier1']['passed'],skipped_raw_score_rows=len(v['tier1']['skipped_rows']),
                   source_binding=v.get('source_binding'),raw_shards=v['tier2_raw_scores_maps'],licensed_loader=v['tier3_licensed_loader'],checkpoint_binaries=v['checkpoint_binaries'])
        if v['tier1']['status']!='PASS':rec['status']='FAIL'
    rec=run('cub_saved_records_and_bootstrap',[py,R/'cub/recompute_from_per_image.py'])
    if rec['status']=='PASS':rec.update(runs=25,paired_image_bootstrap_seeds=3,draws_per_seed=2000)
    # Regenerate only in an external copy, never in this checkout.
    generated=out/'generated_sources';shutil.copytree(S,generated)
    commands=[('cub_tables',['scripts/make_tables.py']),('osld_data_table',['scripts/make_osld_tables.py']),
              ('osld_results_tables',['scripts/make_osld_results.py']),('cub_plot',['scripts/make_figure.py']),
              ('publication_figures',['figures_src/make_publication_figures.py']),('openlogo_tables_figures',['scripts/make_openlogo.py']),
              ('expanded_table_wrappers',['scripts/make_expanded_tables.py'])]
    gen_ok=True
    for name,cmd in commands:
        rec=run('generate_'+name,[py,*cmd],generated)
        gen_ok=gen_ok and rec['status']=='PASS'
    if gen_ok:
        differences=[];checked=0
        for original in sorted((S/'tables').glob('*.tex')):
            checked+=1
            if sha(original)!=sha(generated/'tables'/original.name):differences.append(original.name)
        report['layers']['generated_numerical_tables']={'status':'FAIL' if differences else 'PASS','tables_checked':checked,'different_files':differences}
        figure_files=sorted(f for f in (S/'figures').glob('*') if f.suffix in ['.pdf','.svg','.png'])
        matches=[str(f.relative_to(S)) for f in figure_files if (generated/f.relative_to(S)).exists() and sha(f)==sha(generated/f.relative_to(S))]
        report['layers']['figure_export_bytes']={'status':'PASS' if len(matches)==len(figure_files) else 'RENDER_BYTE_DIFFERENCE',
            'checked':len(figure_files),'byte_identical':len(matches),'different_files':[str(f.relative_to(S)) for f in figure_files if str(f.relative_to(S)) not in matches],
            'scope':'Unchanged input records and generator code are separately hash bound; render-byte differences require review, not a numerical PASS claim.'}
    else:skip('generated_numerical_tables','One or more generators failed')
    if shutil.which('pdflatex'):
        rec=run('figure1_vector_build',['bash','scripts/build_figure1.sh'],generated)
        if rec['status']=='PASS':rec['scope']='Synthetic editable TikZ only; no dataset imagery or image generation'
    else:skip('figure1_vector_build','Optional TeX toolchain unavailable')
    skip('osld_licensed_annotations','Licensed full OSLD annotation records not supplied; prior/strata and full P0-A annotation-dependent recomputation are not inferred from saved-hit checks')
    skip('fresh_training_checkpoint_inference','Not authorized or supplied; no GPU, fresh training, pretrained downloads or checkpoint inference')
    # Hash all produced evidence except the report that contains these hashes.
    report['output_hashes']={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='CPU_REPORT.json'}
    report['failed_layers']=[name for name,rec in report['layers'].items() if rec['status']=='FAIL']
    report['status']='FAIL' if report['failed_layers'] else 'PASS_WITH_EXPLICIT_SKIPS'
    save();print(report['status'],report['failed_layers'],flush=True)
    return 1 if report['failed_layers'] else 0

if __name__=='__main__':
    raise SystemExit(main())
