#!/usr/bin/env python3
"""CPU figures/tables from shipped, audited records. No images or models."""
from pathlib import Path
import shutil
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
T=R/'tables'; O=R/'figures'; D=R/'data'
def rows(name):
    with (D/('openlogo_'+name+'.csv')).open() as f:return list(csv.DictReader(f))
seed={(r['key'],r['weighting']):r for r in rows('per_seed_and_mean_SD')}
contr={(r['contrast'],r['endpoint'],r['weighting']):r for r in rows('paired_contrasts')}
def val(k,w='query'):return float(seed[k,w]['mean'])*100
def sd(k,w='query'):return float(seed[k,w]['sample_SD'])*100
def recomputed_mean_sd(k):
    r=seed[k,'query'];values=np.array([float(r[f'seed{i}'])*100 for i in range(3)])
    mean=float(values.mean());sample_sd=float(values.std(ddof=1))
    np.testing.assert_allclose([mean,sample_sd],[float(r['mean'])*100,float(r['sample_SD'])*100],rtol=0,atol=1e-12)
    return mean,sample_sd
arms=['ce','aux','bger','cam_bce'];names=['CE','Aux-BCE','BGER','CAM-BCE']
text=r'''\begin{table}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Derived OpenLogo (\%): query-weighted mean $\pm$ sample SD over
three seeds for R@1 and primary $H$ localization; SD is not a confidence
interval. Joint@1 is a mean and requires retrieval and localization.
Thresholds are $C_0$ development-locked; prior localization is
7.46\% ($C_0$), 7.50\% ($C_1$).}
\label{tab:openlogo}
\begin{tabular}{@{}lrrr@{}}
\toprule Arm & R@1 & $H$ Loc ($C_0$) & Joint@1\\
\midrule
'''
for a,n in zip(arms,names):
    rm,rs=recomputed_mean_sd(a+'/retrieval/R1');lm,ls=recomputed_mean_sd(a+'/H/own/Loc/C0')
    text+=f"{n} & ${rm:.2f}\\pm{rs:.2f}$ & ${lm:.2f}\\pm{ls:.2f}$ & {val(a+'/H/own/Joint/C0'):.2f} \\\\\n"
text+=r'\bottomrule\end{tabular}\end{table}'+'\n'
T.mkdir(exist_ok=True);(T/'openlogo_main.tex').write_text(text)
shutil.copyfile(R/'table_sources/cub_osld_comparison.tex',T/'cub_osld_comparison.tex')
(T/'osld_contrasts_artifact.tex').write_text((T/'osld_contrasts_all.tex').read_text().replace('ref{fig:effects}','ref*{main-fig:effects}'))
text=r'''\begin{table*}[t]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Derived OpenLogo additional endpoints (percentages). First four
numeric columns: mean $\pm$ sample SD across three seeds; other columns:
three-seed means. $H$ is primary; D is forward decomposition. Macro R@1
weights released classes equally. These seed SDs are not confidence intervals.}
\label{tab:openlogo-full}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Arm & R@1 & R@10 & MRR=AP & $H$ Loc $C_0$ & $H$ Loc $C_1$ & D Loc $C_0$ & D Loc $C_1$ & Macro R@1\\
\midrule
'''
for a,n in zip(arms,names):
    ks=[a+'/retrieval/'+v for v in ['R1','R10','MRR_mAP']]+[a+'/H/own/Loc/C0']
    values=[f'${val(k):.2f}\\pm{sd(k):.2f}$' for k in ks]
    values += [f'{val(a+k):.2f}' for k in ['/H/own/Loc/C1','/decomposition/own/Loc/C0','/decomposition/own/Loc/C1']]
    values += [f"{val(a+'/retrieval/R1','class_macro'):.2f}"]
    text+=n+' & '+' & '.join(values)+r'\\'+'\n'
text+=r'''\bottomrule\end{tabular}
\par\smallskip\raggedright BGER $F$ localization: 3.63\% ($C_0$), 3.70\% ($C_1$).
Primary coordinate-verdict flips: CE 0.094\%, Aux-BCE 0.144\%, BGER 0.050\%,
CAM-BCE 0.129\%. All per-seed values, C1 joint rates, class macros and
conditional percentile intervals remain in the machine-readable tables.
\end{table*}
'''
(T/'openlogo_full.tex').write_text(text)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'BGER-OpenLogo-integration','axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8.5})
core=json.loads((D/'osld_core_results.json').read_text());allc=json.loads((D/'osld_contrasts_all.json').read_text())
conf=core['confirmatory']['contrasts']
data=[('OSLD  P  BGER − CE','o','#c75423',conf['R'],conf['Q']),
      ('OSLD  D  CAM-BCE − CE','s','#197d61',allc['cam_bce-ce/R1'],allc['cam_bce-ce/primary/Q']),
      ('OpenLogo  E  BGER − CE','^','#c75423',contr['bger-ce','retrieval/R1','query'],contr['bger-ce','H/own/Loc/C0','query']),
      ('OpenLogo  E  CAM-BCE − CE','^','#197d61',contr['cam_bce-ce','retrieval/R1','query'],contr['cam_bce-ce','H/own/Loc/C0','query'])]
fig,axes=plt.subplots(1,2,figsize=(7,1.95),sharey=True)
fig.subplots_adjust(left=.32,right=.98,top=.78,bottom=.24,wspace=.23)
for j,ax in enumerate(axes):
    for y,row in zip([3,2,1,0],data):
        d=row[3+j];lo,hi=d.get('basic_two_sided_ci',d.get('basic_ci',[d.get('ci_low'),d.get('ci_high')]))
        ax.plot([float(lo)*100,float(hi)*100],[y,y],color=row[2],lw=1.4)
        ax.plot(float(d['point'])*100,y,marker=row[1],ms=5,color=row[2])
    ax.axvline(0,color='#555555',lw=.6)
    ax.axhline(1.5,color='#999999',lw=.6)
    ax.grid(axis='x',color='#dddddd',lw=.5);ax.set_axisbelow(True)
    ax.set_ylim(-.55,3.55);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.set_title(['Recall@1 difference (pp)','Primary localization difference (pp)'][j],fontsize=8.5,loc='left',pad=10)
axes[0].set_yticks([3,2,1,0]);axes[0].set_yticklabels([r[0] for r in data])
axes[0].set_xlim(-.6,4.2);axes[1].set_xlim(-4,17)
axes[1].set_xticks([-2,0,5,10,15])
manifest={}
def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(O/f'{name}.{ext}',dpi=400,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None)
    plt.close(fig)
    manifest[name]=[name+'.'+ext for ext in ['pdf','svg','png']]
save(fig,'fig03_paired_effects')
fig,ax=plt.subplots(figsize=(3.33,1.85))
for i,a in enumerate(arms):
    ys=[val(a+'/H/own/Loc/'+c) for c in ['C0','C1']]
    ax.plot([i-.1,i+.1],ys,lw=1,color='#555555')
    ax.plot(i-.1,ys[0],'o',ms=4,color='#2271a8')
    ax.plot(i+.1,ys[1],'s',ms=4,mfc='white',mec='#a84319')
ax.axhline(val('centre_prior/Loc/C0'),ls='--',lw=.8,color='#555555')
ax.axhline(val('centre_prior/Loc/C1'),ls=':',lw=.8,color='#777777')
ax.set_xticks(range(4));ax.set_xticklabels(names,fontsize=8)
ax.set_ylabel('Primary localization (%)');ax.set_ylim(2.5,8)
ax.legend(handles=[plt.Line2D([],[],color='#2271a8',marker='o',ls='',label='C0'),plt.Line2D([],[],color='#a84319',marker='s',mfc='white',ls='',label='C1')],ncol=2,frameon=False,loc='upper left',bbox_to_anchor=(0,.86),fontsize=8)
ax.grid(axis='y',lw=.4,color='#dddddd');fig.tight_layout(pad=.4)
save(fig,'fig06_coordinate_sensitivity')
manifest['inputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in list(D.glob('openlogo_*.csv'))+[D/'osld_core_results.json',D/'osld_contrasts_all.json']}
manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(O/'CROSS_DATASET_EFFECTS.json').write_text(json.dumps([{'row':row[0],'marker':row[1],'metrics':[{'point':float(d['point']),'ci':[float(v) for v in d.get('basic_two_sided_ci',d.get('basic_ci',[d.get('ci_low'),d.get('ci_high')]))]} for d in row[3:]]} for row in data],indent=2)+'\n')
table5={a:{'R1':recomputed_mean_sd(a+'/retrieval/R1'),'H_Loc_C0':recomputed_mean_sd(a+'/H/own/Loc/C0')} for a in arms}
directions=[]
for a in ['bger','cam_bce']:
    for ep in ['retrieval/R1','H/own/Loc/C0','H/own/Loc/C1']:
        diffs=[100*(float(seed[a+'/'+ep,'query'][f'seed{i}'])-float(seed['ce/'+ep,'query'][f'seed{i}'])) for i in range(3)]
        assert all(d<0 for d in diffs) if a=='bger' and ep.startswith('H/') else all(d>0 for d in diffs)
        directions.append({'contrast':a+'-ce','endpoint':ep,'seed_differences_pp':diffs})
(O/'OPENLOGO_TABLE5_RECOMPUTATION.json').write_text(json.dumps({'unit':'percent / percentage points','mean_and_sample_SD_from_seed_rows':table5,'paired_seed_directions':directions,'SD_is_not_CI':True},indent=2)+'\n')
(O/'OPENLOGO_FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
all_manifest=json.loads((O/'FIGURE_MANIFEST.json').read_text())
all_manifest['fig03_paired_effects']={'inputs':manifest['inputs'],'generator':'scripts/make_openlogo.py','script_sha256':manifest['script_sha256'],'files':manifest['fig03_paired_effects'],'supersedes':'The original OSLD-only Figure 3 is replaced by the separate-dataset contrast view; all OSLD contrast records remain unchanged.'}
all_manifest['fig06_coordinate_sensitivity']={'inputs':manifest['inputs'],'generator':'scripts/make_openlogo.py','files':manifest['fig06_coordinate_sensitivity']}
(O/'FIGURE_MANIFEST.json').write_text(json.dumps(all_manifest,indent=2)+'\n')
print('OpenLogo main/appendix tables and cross-dataset/sensitivity figures regenerated')
