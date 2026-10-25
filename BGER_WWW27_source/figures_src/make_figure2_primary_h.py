#!/usr/bin/env python3

"""Accepted Figure 2: five matched primary-H arms, frozen input records only.

The fig02 function, style constants and vector exporter are copied exactly from
the accepted manuscript plotting source. rc_context confines this style to Figure
2; no other figure generator or data record is changed.
"""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"

ARMS = ['ce', 'aux', 'bger', 'cam_bce', 'abn_box']

NAME = dict(ce='CE', aux='Aux-BCE', bger='BGER', cam_bce='CAM-BCE', abn_box='ABN-box')

COL = dict(ce='#0072B2', bger='#D55E00', cam_bce='#009E73', aux='#595959', abn_box='#969696')

MK = dict(ce='o', aux='s', bger='D', cam_bce='^', abn_box='v')

INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'

COLW, TEXTW = 3.33, 7.0

ACCEPTED_RC = {'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'font.family': 'DejaVu Sans', 'svg.hashsalt': 'BGER-WWW27-V2'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(name, optional=False):
    p = DATA / name
    if not p.exists():
        if optional: return None
        raise FileNotFoundError(p)
    return json.loads(p.read_text())

def savefig(fig, name, manifest, inputs):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300 if ext == 'png' else None, bbox_inches=None, metadata={'CreationDate': None, 'ModDate': None} if ext == 'pdf' else ({'Date': None} if ext == 'svg' else None))
    plt.close(fig)
    manifest[name] = dict(inputs={i: sha(DATA / i) for i in inputs}, files=[f'{name}.{e}' for e in ['pdf', 'svg', 'png']])

def fig02(core, ra, p0a, manifest):
    """Five matched arms at primary H; p0a retained as a call-compatible argument.

    BGER-F is deliberately excluded here and remains in Table 3 / Section 6.1.
    All coordinates and the centre-prior reference come from the frozen inputs.
    """
    fig, ax = plt.subplots(figsize=(COLW, 2.75))
    fig.subplots_adjust(left=.20, right=.975, top=.97, bottom=.31)
    ms = core['mean_sd']
    for arm in ARMS:
        xs = np.array(ms[arm]['R1']['per_seed']) * 100
        ys = np.array(ms[arm]['Loc_q']['per_seed']) * 100
        # Smaller hollow seeds and larger filled means; no coordinate offsets.
        ax.scatter(xs, ys, marker=MK[arm], s=9, facecolors='none',
                   edgecolors=COL[arm], linewidths=.65, alpha=.75, zorder=6)
        ax.scatter([xs.mean()], [ys.mean()], marker=MK[arm], s=27,
                   facecolors=COL[arm], edgecolors=INK, linewidths=.55, zorder=5)
    prior = ra['centre_prior_own_threshold']['test_loc_at_val_threshold'] * 100
    ax.axhline(prior, color=INK2, ls=(0,(3,2)), lw=.8, zorder=1)
    ax.set_xlabel('Recall@1 (%)')
    ax.set_ylabel('Ranking-score localization (%)')
    ax.set_xlim(0,8.3); ax.set_ylim(0,22)
    ax.grid(True,color=GRID,lw=.4); ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=COL[arm], marker=MK[arm],
                         mfc=COL[arm], mec=INK, mew=.55, ls='', ms=5.2,
                         label=NAME[arm]) for arm in ARMS]
    handles.append(plt.Line2D([], [], color=INK2, ls=(0,(3,2)), lw=.8,
                              label='centre prior'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52,.008),
               ncol=3, frameon=False, fontsize=8, handlelength=1.2,
               handletextpad=.45, columnspacing=1.0, labelspacing=.5)
    savefig(fig,'fig02_retrieval_vs_localization',manifest,
            ['osld_core_results.json','osld_readout_analysis.json'])

def draw_figure2(core, readout, manifest, source_root=None, output_dir=None):
    global DATA, OUT
    previous_data, previous_out = DATA, OUT
    DATA = Path(source_root) / 'data' if source_root is not None else ROOT / 'data'
    OUT = Path(output_dir) if output_dir is not None else (Path(source_root) if source_root is not None else ROOT) / 'figures'
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        with plt.rc_context(ACCEPTED_RC):
            fig02(core, readout, None, manifest)
        manifest['fig02_retrieval_vs_localization']['generator'] = 'figures_src/make_figure2_primary_h.py'
    finally:
        DATA, OUT = previous_data, previous_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.source_root
    manifest_path = source / 'figures' / 'FIGURE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    core = json.loads((source / 'data' / 'osld_core_results.json').read_text())
    readout = json.loads((source / 'data' / 'osld_readout_analysis.json').read_text())
    draw_figure2(core, readout, manifest, source_root=source)
    manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
    print('Regenerated accepted primary-H Figure 2 only; no model computation.')


if __name__ == '__main__':
    main()
