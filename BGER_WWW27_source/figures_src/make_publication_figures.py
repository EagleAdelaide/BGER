#!/usr/bin/env python3
"""Deterministic, source-linked publication figures for the WWW 2027 revision.

Every plotted number is read from the frozen result files (never typed from the paper):
  data/osld_core_results.json      per-seed test values, mean/SD, confirmatory + descriptive contrasts
  data/osld_contrasts_all.json     all paired contrasts (shared cluster draws)
  data/osld_readout_analysis.json  A5 split (magnitude/shape/verdict), centre prior at own threshold
  data/osld_descriptive.json       box-area strata, correct/wrong split
  data/p0a_results.json            (optional) P0-A BGER readout at pooled F  -> exploratory rows
  data/p0b_results.json            (optional) P0-B gate controls              -> exploratory rows
Outputs PDF (vector, Type-42 fonts) + SVG + PNG preview, and a manifest with input hashes.
Identity is never colour-alone: each arm has a fixed marker shape and a direct label.
"""
import argparse, hashlib, json, sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'; OUT = ROOT / 'figures'
ARMS = ['ce', 'aux', 'bger', 'cam_bce', 'abn_box']
NAME = dict(ce='CE', aux='Aux-BCE', bger='BGER', cam_bce='CAM-BCE', abn_box='ABN-box')
# validated reference palette slots 1-3 for the three arms the argument turns on; neutral inks for the two
# supporting arms. Marker shape + direct label carry identity in every panel (never colour alone).
COL = dict(ce='#2a78d6', bger='#eb6834', cam_bce='#1baf7a', aux='#52514e', abn_box='#9a9890')
MK = dict(ce='o', aux='s', bger='D', cam_bce='^', abn_box='v')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'font.family': 'DejaVu Sans', 'svg.hashsalt': 'BGER-WWW27-V2'})
COLW, TEXTW = 3.33, 7.0


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, optional=False):
    p = DATA / name
    if not p.exists():
        if optional: return None
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def savefig(fig, name, manifest, inputs):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None} if ext == 'pdf' else ({'Date': None} if ext == 'svg' else None))
    plt.close(fig)
    manifest[name] = dict(inputs={i: sha(DATA / i) for i in inputs}, files=[f'{name}.{e}' for e in ['pdf', 'svg', 'png']])


def fig02(core, ra, p0a, manifest):
    """Recall@1 vs own-top-1 query localization; seeds small, means large; prior as a reference line."""
    fig, ax = plt.subplots(figsize=(COLW, 2.55))
    ms = core['mean_sd']
    for arm in ARMS:
        xs = np.array(ms[arm]['R1']['per_seed']) * 100; ys = np.array(ms[arm]['Loc_q']['per_seed']) * 100
        ax.scatter(xs, ys, marker=MK[arm], s=16, facecolors='none', edgecolors=COL[arm], linewidths=0.8, zorder=3)
        ax.scatter([xs.mean()], [ys.mean()], marker=MK[arm], s=52, facecolors=COL[arm], edgecolors='white', linewidths=0.6, zorder=5)
    lab = dict(ce=(2.8, 1.0, 'right'), aux=(1.0, 8.7, 'left'), bger=(6.5, 0.7, 'center'), cam_bce=(5.3, 19.6, 'left'), abn_box=(0.65, 3.0, 'left'))
    for arm, (x, y, ha) in lab.items():
        txt = ('BGER-H' if arm == 'bger' else NAME[arm])
        ax.annotate(txt, (ms[arm]['R1']['mean'] * 100, ms[arm]['Loc_q']['mean'] * 100), xytext=(x, y), textcoords='data', fontsize=7.5, color=COL[arm] if arm not in ('aux', 'abn_box') else INK2, ha=ha, va='center',
                    arrowprops=dict(arrowstyle='-', color=GRID, lw=0.6, shrinkA=0, shrinkB=4))
    prior = ra['centre_prior_own_threshold']['test_loc_at_val_threshold'] * 100
    ax.axhline(prior, color=INK2, ls=(0, (3, 2)), lw=0.7, zorder=1)
    ax.text(0.02, 1.08, f'dashed: centre prior {prior:.1f}%', transform=ax.transAxes, fontsize=7, color=INK2, va='bottom')
    if p0a:
        xs = np.array(p0a['seeds_R1']) * 100; ys = np.array(p0a['seeds_Loc_F']) * 100
        ax.scatter(xs, ys, marker=MK['bger'], s=16, facecolors='none', edgecolors=COL['bger'], linewidths=0.8, zorder=3, linestyle=':')
        ax.scatter([xs.mean()], [ys.mean()], marker=MK['bger'], s=110, facecolors='none', edgecolors=COL['bger'], linewidths=1.3, zorder=6, linestyle=(0, (2, 1)))
        ax.annotate('BGER-F', (xs.mean(), ys.mean()), xytext=(6.25, 8.1), textcoords='data', fontsize=7.5, color=COL['bger'], ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='-', color=GRID, lw=0.6, shrinkA=0, shrinkB=6))
    ax.set_xlabel('Recall@1 (%)'); ax.set_ylabel('Query-logo localization (%)\nown top-1, IoU $\\geq$ .5')
    ax.set_xlim(0, 8.3); ax.set_ylim(0, 22); ax.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.text(0.02, 0.98, 'small: seeds 0/1/2   large: mean', transform=ax.transAxes, fontsize=6.5, color=INK2, va='top')
    savefig(fig, 'fig02_retrieval_vs_localization', manifest, ['osld_core_results.json', 'osld_readout_analysis.json'] + (['p0a_results.json'] if p0a else []))


def fig03(core, allc, p0a, p0b, manifest):
    """Six central comparisons at two-column width; full contrasts in appendix tables."""
    conf = core['confirmatory']['contrasts']
    rows = [('P  BGER-H − CE', 'P', conf['R'], conf['Q']),
            ('D  CAM-BCE − CE', 'D', allc['cam_bce-ce/R1'], allc['cam_bce-ce/primary/Q']),
            ('D  BGER-H − Aux-BCE', 'D', allc['bger-aux/R1'], allc['bger-aux/primary/Q']),
            ('E  BGER-F − CE', 'E', None, p0a['contrasts']['BGER_F-ce/Loc']),
            ('E  BGER-H − gate-only', 'E', p0b['contrasts']['bger-gate_only/R1'], p0b['contrasts']['bger-gate_only/Loc']),
            ('E  BGER-H − shifted-box', 'E', p0b['contrasts']['bger-shifted_box/R1'], p0b['contrasts']['bger-shifted_box/Loc'])]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.45), sharey=True)
    fig.subplots_adjust(left=.285, right=.985, top=.85, bottom=.24, wspace=.18)
    y = np.arange(len(rows))[::-1]
    styles = {'P': (COL['bger'], 'o'), 'D': (INK2, 's'), 'E': (COL['cam_bce'], '^')}
    for ax, col, title in [(axes[0], 2, 'Recall@1 difference (pp)'), (axes[1], 3, 'Localization difference (pp)')]:
        for yi, row in zip(y, rows):
            c = row[col]
            if c is None:
                ax.text(.97, yi, '—', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8)
                continue
            lo, hi = c.get('basic_two_sided_ci', c.get('basic_ci')); pt = c['point']
            color, marker = styles[row[1]]
            ax.plot([lo*100, hi*100], [yi, yi], color=color, lw=1.4, zorder=3)
            ax.plot([pt*100], [yi], marker=marker, ms=4.5, color=color, zorder=4)
        ax.axvline(0, color=INK2, lw=.6)
        ax.set_title(title, fontsize=8, loc='left')
        ax.grid(True, axis='x', color=GRID, lw=.4); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.set_ylim(-.6, len(rows)-.45)
        ax.axhline(4.5, color=GRID, lw=.6); ax.axhline(2.5, color=GRID, lw=.6)
    axes[0].set_xlim(-2, 5); axes[1].set_xlim(-5, 18)
    axes[0].axvline(-1, color=COL['bger'], ls=(0,(2,2)), lw=.7)
    # The dashed H1 margin is explained in the caption; no title-area annotation.
    axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows], fontsize=8)
    handles = [plt.Line2D([], [], color=styles[k][0], marker=styles[k][1], lw=1.2, ms=4,
               label=lab) for k, lab in [('P','P: prespecified'),('D','D: descriptive'),('E','E: exploratory')]]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(.58, .015), handlelength=1.3, columnspacing=1.5)
    savefig(fig, 'fig03_paired_effects', manifest, ['osld_core_results.json', 'osld_contrasts_all.json', 'p0a_results.json', 'p0b_results.json'])


def fig04(ra, manifest):
    """Candidate dependence split: localization for d+/d- (dumbbell) and shape change (L1) per readout."""
    sp = ra['a5_split']; prior = ra['centre_prior_own_threshold']['test_loc_at_val_threshold'] * 100
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 2.95), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.35))
    fig.subplots_adjust(left=.20, right=.98, top=.88, bottom=.25, wspace=.35)
    y = np.arange(len(ARMS))[::-1]
    ax = axes[0]
    for yi, arm in zip(y, ARMS):
        for ro, off, ms_ in [('primary', 0.17, 5), ('secondary', -0.17, 3.5)]:
            v = sp[arm][ro]; lo, hi = v['loc_neg'] * 100, v['loc_pos'] * 100
            ax.plot([lo, hi], [yi + off, yi + off], color=COL[arm] if ro == 'primary' else INK2, lw=1.2 if ro == 'primary' else 1.0, ls='-' if ro == 'primary' else ':', zorder=3)
            ax.plot([lo], [yi + off], marker='o', ms=ms_, mfc='white', mec=COL[arm] if ro == 'primary' else INK2, mew=0.9, zorder=4)
            ax.plot([hi], [yi + off], marker=MK[arm] if ro == 'primary' else 'o', ms=ms_, color=COL[arm] if ro == 'primary' else INK2, zorder=4)
    ax.axvline(prior, color=INK2, ls=(0, (3, 2)), lw=0.7, zorder=1); ax.text(prior + 0.8, y[0] + 0.5, 'prior', fontsize=7, color=INK2, va='bottom')
    ax.set_yticks(y); ax.set_yticklabels([NAME[a] for a in ARMS], fontsize=7); ax.set_xlabel('Localization (%)\nopen $d^-$ $\\to$ filled $d^+$', fontsize=7)
    ax.set_title('Extent hit rate', fontsize=8, loc='left', pad=8); ax.grid(True, axis='x', color=GRID, lw=0.4); ax.set_axisbelow(True); ax.set_ylim(-0.6, len(ARMS) - 0.4 + 0.5)
    fig.legend(handles=[plt.Line2D([],[],color=INK2,marker='s',lw=1.2,label='Grad-CAM'), plt.Line2D([],[],color=INK2,marker='o',ls=':',lw=1.0,label='Forward')], loc='lower center', bbox_to_anchor=(.54,.005), ncol=2, frameon=False, fontsize=7.5)
    ax = axes[1]
    for yi, arm in zip(y, ARMS):
        for ro, off in [('primary', 0.17), ('secondary', -0.17)]:
            v = sp[arm][ro]
            ax.plot([v['mean_l1']], [yi + off], marker=MK[arm] if ro == 'primary' else 'o', ms=5 if ro == 'primary' else 3.5, color=COL[arm] if ro == 'primary' else INK2, zorder=4)
    ax.set_xlim(0, 2); ax.set_yticks(y); ax.set_yticklabels([]); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('L1 distance (0--2),\nnormalized $d^+$ vs $d^-$', fontsize=7); ax.set_title('Shape change', fontsize=8, loc='left', pad=12); ax.set_ylim(-0.6, len(ARMS) - 0.4 + 0.5); ax.grid(True, axis='x', color=GRID, lw=0.4); ax.set_axisbelow(True)
    for ax in axes: ax.spines['left'].set_visible(False)
    savefig(fig, 'fig04_candidate_dependency', manifest, ['osld_readout_analysis.json'])


def fig05(desc, manifest):
    """R@1 and localization by box-area quartile (separate panels)."""
    e = desc['area_quartile_edges']; n = desc['bin_counts']
    labels = [f'Q{i + 1}' for i in range(4)]
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 2.45), sharex=True)
    for ax, key, title in [(axes[0], 'R', 'Recall@1 (%)'), (axes[1], 'Q', 'Query localization (%)')]:
        for arm in ARMS:
            v = [100 * r[key] for r in desc['small_logo'][arm]]
            ax.plot(range(4), v, marker=MK[arm], ms=4, color=COL[arm], lw=1.1, mfc=COL[arm] if arm in ('ce', 'bger', 'cam_bce') else 'white', mew=0.9, zorder=3)
        ax.set_title(title, fontsize=8, loc='left'); ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5); ax.grid(True, axis='y', color=GRID, lw=0.4); ax.set_axisbelow(True); ax.set_ylim(bottom=0); ax.set_xlim(-0.3, 3.3)
    fig.text(0.5, -0.035, 'Test-query box-area quartile (Q1: smallest)', ha='center', va='top', fontsize=7.5, color=INK2)
    handles = [plt.Line2D([], [], marker=MK[a], color=COL[a], lw=1.1, ms=4, mfc=COL[a] if a in ('ce', 'bger', 'cam_bce') else 'white', label=NAME[a]) for a in ARMS]
    axes[1].legend(handles=handles, loc='upper left', frameon=False, fontsize=7.5, ncol=1, handlelength=1.5, borderaxespad=0.2)
    fig.subplots_adjust(wspace=0.35)
    savefig(fig, 'fig05_size_strata', manifest, ['osld_descriptive.json'])


def main():
    global DATA, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=ROOT)
    args = ap.parse_args(); DATA = args.source_root / 'data'; OUT = args.source_root / 'figures'
    OUT.mkdir(parents=True, exist_ok=True)
    core, allc, ra, desc = load('osld_core_results.json'), load('osld_contrasts_all.json'), load('osld_readout_analysis.json'), load('osld_descriptive.json')
    p0a, p0b = load('p0a_results.json', True), load('p0b_results.json', True)
    manifest = {}
    fig02(core, ra, p0a, manifest); fig03(core, allc, p0a, p0b, manifest); fig04(ra, manifest); fig05(desc, manifest)
    manifest['_meta'] = dict(script_sha256=sha(__file__), matplotlib=matplotlib.__version__, numpy=np.__version__, p0a_included=bool(p0a), p0b_included=bool(p0b))
    (OUT / 'FIGURE_MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('figures written:', sorted(k for k in manifest if not k.startswith('_')))


if __name__ == '__main__':
    main()
