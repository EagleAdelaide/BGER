#!/usr/bin/env python3
"""OSLD result tables, macros and the numeric-map figure from saved core outputs.

Inputs (data/): osld_core_results.json (analysis/core_tables.py),
osld_descriptive.json (analysis/descriptive_extra.py), osld_resources.json,
model_accounting.json. Refuses non-core identities unless --plumbing-test is
given with an output directory outside the paper (fixtures never enter the paper).
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CHECK = '$\\checkmark$'
NAMES = dict(ce='CE', aux='Aux-BCE', bger='BGER', cam_bce='CAM-BCE', abn_box='ABN-box')


def pct(x, d=1):
    return '--' if x is None else f'{100 * x:.{d}f}'


def pm(m, d=1):
    return f"{pct(m['mean'], d)}\\,$\\pm$\\,{pct(m['sd'], d)}"


def signed(x, d=1):
    if x is None: return '--'
    v = round(100 * x, d)
    if v == 0: return f'{0:.{d}f}'
    return f"{'+' if v > 0 else '$-$'}{abs(v):.{d}f}"


def ci(lo, hi, d=1):
    return f"[{signed(lo, d)}, {signed(hi, d)}]"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=str(HERE / 'data'))
    ap.add_argument('--out', default=str(HERE)); ap.add_argument('--plumbing-test', action='store_true')
    a = ap.parse_args(); D = Path(a.data); O = Path(a.out)
    core = json.loads((D / 'osld_core_results.json').read_text())
    desc = json.loads((D / 'osld_descriptive.json').read_text())
    if core['identity'] != 'OSLD_E1_CORE' or desc['identity'] != 'OSLD_E1_CORE' or not core['scientific_inference']:
        if not a.plumbing_test or O.resolve() == HERE.resolve():
            raise SystemExit('Refusing non-core identity for the paper: ' + str(core['identity']))
    res = json.loads((D / 'osld_resources.json').read_text()) if (D / 'osld_resources.json').exists() else None
    acct = json.loads((D / 'model_accounting.json').read_text())
    (O / 'tables').mkdir(parents=True, exist_ok=True); (O / 'figures').mkdir(parents=True, exist_ok=True)
    arms = core['arms_tested']; ms = core['mean_sd']
    # --- macros
    M = ''
    tag = dict(ce='CE', aux='Aux', bger='Bger', cam_bce='Cam', abn_box='Abn')
    for arm in arms:
        t = tag[arm]
        for k, key in [('Rone', 'R1'), ('Rten', 'R10'), ('mAP', 'mAP'), ('Loc', 'Loc_q'), ('Jq', 'Jq1'), ('EIB', 'EIB_q'), ('Deg', 'degenerate_rate'), ('LocSec', 'Loc_q_secondary')]:
            M += macro(f'os{k}{t}', pct(ms[arm][key]['mean'])) + macro(f'os{k}{t}SD', pct(ms[arm][key]['sd']))
    M += macro('osRandRef', f"{ms[arms[0]]['random_R1_reference']['mean']:.1e}".replace('e-0', '\\times10^{-').replace('e-', '\\times10^{-') + '}')
    M += macro('osRandRefPct', f"{100 * ms[arms[0]]['random_R1_reference']['mean']:.3f}")
    conf = core['confirmatory']['contrasts']
    for k, name in [('R', 'HOne'), ('Q', 'HTwo')]:
        c = conf[k]
        M += macro(f'os{name}Point', signed(c['point'])) + macro(f'os{name}CI', ci(*c['basic_two_sided_ci']))
        M += macro(f'os{name}Lower', signed(c['basic_one_sided_lower'])) + macro(f'os{name}P', f"{c['p_holm']:.4f}")
        M += macro(f'os{name}Reject', 'rejected' if c['reject_holm'] else 'not rejected')
        M += macro(f'os{name}Seeds', ', '.join(signed(x) for x in c['per_seed_point']))
    M += macro('osSelected', ', '.join(f"{NAMES[k]}~$\\lambda={v:g}$" for k, v in core['selected_lambdas'].items() if k != 'ce'))
    M += macro('osNClusters', str(core['confirmatory']['n_clusters'])) + macro('osNQueries', str(core['confirmatory']['n_queries']))
    if res:
        for k, v in res.get('macros', {}).items(): M += macro(k, v)
    # all whitelisted paired contrasts (shared draws; descriptive except the two confirmatory keys)
    allc = json.loads((D / 'osld_contrasts_all.json').read_text())
    cname = dict(ce='CE', aux='Aux', bger='Bger', cam_bce='Cam', abn_box='Abn')
    mname = {'R1': 'Rone', 'R10': 'Rten', 'mAP': 'mAP', 'primary/Q': 'Loc', 'primary/J': 'Jq', 'secondary/Q': 'LocSec', 'primary/query/EIB': 'EIB'}
    for key, c in allc.items():
        pair, metric = key.split('/', 1); x, y = pair.split('-')
        n = f'osC{cname[x]}{cname[y]}{mname[metric]}'
        M += macro(n, signed(c['point'])) + macro(n + 'CI', ci(*c['basic_ci']))
    tagA = dict(ce='CE', aux='Aux', bger='Bger', cam_bce='Cam', abn_box='Abn')
    for r in core['a5']:
        t = tagA[r['arm']] + ('Sec' if r['readout'] == 'secondary' else '')
        M += macro(f'osAfiveVerdict{t}', pct(r['verdict_change_rate'])) + macro(f'osAfiveLone{t}', f"{r['mean_l1_distance']:.2f}")
        M += macro(f'osAfiveCorr{t}', f"{r['mean_correlation_defined']:.2f}") + macro(f'osAfivePrior{t}', pct(r['centre_prior_loc_rate']))
        M += macro(f'osAfivePos{t}', pct(r['loc_rate_pos'])) + macro(f'osAfiveNeg{t}', pct(r['loc_rate_neg']))
        M += macro(f'osAfiveUndef{t}', str(r['correlation_undefined_total'])) + macro(f'osAfiveDs{t}', f"{r['mean_score_change']:.3f}".replace('-', '$-$'))
    for arm in arms:
        sl = desc['small_logo'][arm]
        for b, r in enumerate(sl):
            M += macro(f'osSize{tagA[arm]}Loc{"ABCD"[b]}', pct(r['Q'])) + macro(f'osSize{tagA[arm]}R{"ABCD"[b]}', pct(r['R']))
        M += macro(f'osTopCorrect{tagA[arm]}', pct(desc['by_top1'][arm]['correct']['Q'])) + macro(f'osTopWrong{tagA[arm]}', pct(desc['by_top1'][arm]['wrong']['Q']))
    e = desc['area_quartile_edges']
    M += macro('osSizeEdgeOne', f'{100 * e[1]:.1f}') + macro('osSizeEdgeTwo', f'{100 * e[2]:.1f}') + macro('osSizeEdgeThree', f'{100 * e[3]:.1f}')
    a5x = json.loads((D / 'osld_a5_extra.json').read_text())['negative_outscores_positive_fraction']
    M += macro('osNegWinsMin', f"{100 * min(a5x.values()):.0f}") + macro('osNegWinsMax', f"{100 * max(a5x.values()):.0f}")
    pa = acct['arms']
    M += macro('osParamsGate', f"{pa['ce']['total_parameters'] / 1e6:.1f}") + macro('osParamsAbn', f"{pa['abn_box']['total_parameters'] / 1e6:.1f}")
    M += macro('osParamsGateRet', f"{pa['ce']['retrieval_parameters'] / 1e6:.1f}") + macro('osParamsAbnRet', f"{pa['abn_box']['retrieval_parameters'] / 1e6:.1f}")
    (O / 'tables' / 'osld_results_numbers.tex').write_text('% generated by scripts/make_osld_results.py\n' + M)
    # --- main table (compact; best within the five matched arms in bold per column)
    cols = [('R1', 'R@1'), ('mAP', 'mAP'), ('Loc_q', '$\\bar L^{\\mathrm q}$'), ('Jq1', '$Jq@1$')]
    best = {k: max(ms[a][k]['mean'] for a in arms) for k, _ in cols}
    rows = []
    for arm in arms:
        cells = [(f"\\textbf{{{pm(ms[arm][k])}}}" if ms[arm][k]['mean'] == best[k] else pm(ms[arm][k])) for k, _ in cols]
        rows.append(f"{NAMES[arm]} & " + ' & '.join(cells) + "\\\\")
    T = ("\\begin{table}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
         "\\caption{OSLD test, five matched arms (\\%, mean\\,$\\pm$\\,sample SD over seeds 0--2; \\osNQueries{} single-logo queries of \\osldTestIds{} unseen identities vs "
         f"{core['per_seed'][0]['gallery_size']:,} templates). $\\bar L^{{\\mathrm q}}$: own-top-1 query-logo localization (primary Grad-CAM, IoU$\\ge$.5 at the "
         "validation-locked threshold; wrong templates included); $Jq@1$: correct top-1 and localized. BGER's primary map is read at the pre-gate $H$ (Section~\\ref{sec:p0a} "
         "reads it at $M{\\odot}H$). Bold: best of the five arms per column. Uniform-random Recall@1 is \\osRandRefPct\\%. EIB, degeneracy and per-seed values: Appendix~\\ref{app:osld-results}.}\n"
         "\\label{tab:osld-main}\n\\begin{tabular}{lcccc}\n\\toprule\nArm & " + ' & '.join(l for _, l in cols) + "\\\\\n\\midrule\n"
         + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (O / 'tables' / 'osld_main.tex').write_text(T)
    # --- contrasts: main (R@1 and localization) and full appendix table
    allc = json.loads((D / 'osld_contrasts_all.json').read_text())
    seeds_of = {}
    for x in core['descriptive_contrasts']: seeds_of[(x['contrast'], x['metric'])] = x['per_seed']
    def seedstr(v): return '/'.join(signed(x) for x in v)
    ps_mean = lambda arm, m: core['mean_sd'][arm][m]['per_seed']
    lines = []
    for k, label in [('R', 'R@1'), ('Q', '$\\bar L^{\\mathrm q}$')]:
        c = conf[k]
        lines.append(f"BGER$-$CE$^\\ast$ & {label} & {signed(c['point'])} & {ci(*c['basic_two_sided_ci'])} & {seedstr(c['per_seed_point'])} & {c['p_holm']:.4f}\\\\")
    lines.append('\\midrule')
    pairs_main = [('bger', 'aux'), ('bger', 'cam_bce'), ('bger', 'abn_box'), ('cam_bce', 'ce')]
    for x, y in pairs_main:
        for metric, key, label in [('R1', 'R1', 'R@1'), ('Loc_q', 'primary/Q', '$\\bar L^{\\mathrm q}$')]:
            c = allc[f'{x}-{y}/{key}']
            per = [a - b for a, b in zip(ps_mean(x, metric), ps_mean(y, metric))]
            SHORT = dict(ce='CE', aux='Aux', bger='BGER', cam_bce='CAM', abn_box='ABN')
            lines.append(f"{SHORT[x]}$-${SHORT[y]} & {label} & {signed(c['point'])} & {ci(*c['basic_ci'])} & {seedstr(per)} & --\\\\")
    T = ("\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{2.6pt}\n"
         "\\caption{Paired contrasts on the OSLD test set (percentage points): query-weighted, seed-averaged estimate; basic 95\\% interval from "
         "10,000 shared query-identity cluster draws (unadjusted, not simultaneous); per-seed differences. $^\\ast$Confirmatory, with Holm-adjusted "
         "one-sided $p$ (H1-logo boundary $-1$~pp, H2-logo boundary 0); all other rows descriptive. Aux, CAM, ABN: Aux-BCE, CAM-BCE, ABN-box. All contrasts and metrics: Table~\\ref{tab:osld-contrasts-all}.}\n"
         "\\label{tab:osld-contrasts}\n\\begin{tabular}{llrccc}\n\\toprule\nContrast & & Est. & 95\\% interval & Seeds 0/1/2 & $p_{\\mathrm{Holm}}$\\\\\n\\midrule\n"
         + '\n'.join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (O / 'tables' / 'osld_contrasts.tex').write_text(T)
    order = ['bger-ce', 'bger-aux', 'bger-cam_bce', 'bger-abn_box', 'aux-ce', 'cam_bce-ce', 'abn_box-ce']
    mets = [('R1', 'R@1'), ('R10', 'R@10'), ('mAP', 'mAP'), ('primary/Q', '$\\bar L^{\\mathrm q}$'), ('primary/J', '$Jq@1$'),
            ('primary/query/EIB', 'EIB'), ('secondary/Q', 'Sec.\\ $\\bar L^{\\mathrm q}$')]
    rows = []
    for pr in order:
        x, y = pr.split('-')
        cells = [f"{signed(allc[pr + '/' + m]['point'])} {ci(*allc[pr + '/' + m]['basic_ci'])}" for m, _ in mets]
        rows.append(f"{NAMES[x]}$-${NAMES[y]} & " + ' & '.join(cells) + '\\\\')
    T = ("\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
         "\\caption{All paired contrasts of the OSLD test set (percentage points; estimate and basic 95\\% interval from the shared draws; "
         "descriptive except BGER$-$CE R@1 and $\\bar L^{\\mathrm q}$, the prespecified hypotheses shown in Figure~\\ref{fig:effects}).}\n\\label{tab:osld-contrasts-all}\n"
         "\\begin{tabular}{l" + 'c' * len(mets) + "}\n\\toprule\nContrast & " + ' & '.join(l for _, l in mets) + "\\\\\n\\midrule\n"
         + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    (O / 'tables' / 'osld_contrasts_all.tex').write_text(T)
    # --- A5 split: change between candidates vs localization (both readouts)
    ra = json.loads((D / 'osld_readout_analysis.json').read_text())
    sp = ra['a5_split']
    lines = []
    for ro, tag in [('primary', 'Primary (Grad-CAM)'), ('secondary', 'Secondary (decomposition)')]:
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{tag}}}}}\\\\")
        for arm in arms:
            v = sp[arm][ro]
            lines.append(f"{NAMES[arm]} & {v['median_abs_log_rawmax_ratio']:.2f} & {v['mean_l1']:.2f} & {v['mean_corr']:.2f} ({v['corr_undefined']}) & "
                         f"{pct(v['verdict_change'])} [{pct(v['verdict_change_lower_bound'])}, {pct(v['verdict_change_upper_bound'])}] & "
                         f"{pct(v['loc_pos'])}/{pct(v['loc_neg'])} & {pct(v['centre_prior_at_arm_threshold'])}\\\\")
    T = ("\\begin{table}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{2.4pt}\n"
         "\\caption{Fixed-candidate diagnostic (query side; seed-averaged; same CE-seed-0 $d^+$ and $d^-$ for all arms). \\emph{Change between "
         "candidates}: raw magnitude (median $|\\ln|$ ratio of the maps' maximum absolute values), normalized shape (L1 distance of ReLU-normalized "
         "maps, 0--2; mean defined correlation, undefined count) and the thresholded verdict (rate of differing localization verdicts, \\%, with the "
         "bounds implied by the two hit rates). \\emph{Localization}: hit rate for $d^+/d^-$ and a centre prior evaluated at each arm's locked threshold "
         "(\\%; not tuned for the prior). No single column ranks overall candidate dependence; descriptive, not a faithfulness measure.}\n"
         "\\label{tab:osld-a5}\n\\begin{tabular}{lcccccc}\n\\toprule\n& \\multicolumn{4}{c}{Change between candidates} & \\multicolumn{2}{c}{Localization}\\\\\n"
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\nArm & Raw & L1 & Corr.\\ (undef.) & Verdict [bounds] & $d^+/d^-$ & Prior\\\\\n\\midrule\n"
         + '\n'.join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (O / 'tables' / 'osld_a5.tex').write_text(T)
    # --- readout-identity and prior macros
    idr = ra['identity_at_F']
    M2 = macro('osIdMaxT', f"{max(r['max_abs_diff_recovered_T_across_views'] for r in idr):.1e}".replace('e-0', '\\times10^{-') + '}')
    shares = {a: sum(r['median_norm_ratio_C_over_D'] for r in idr if r['arm'] == a) / 3 for a in ['ce', 'aux', 'cam_bce', 'abn_box']}
    p90 = {a: sum(r['p90_norm_ratio_C_over_D'] for r in idr if r['arm'] == a) / 3 for a in shares}
    cosv = {a: sum(r['median_cosine_D_vs_T'] for r in idr if r['arm'] == a) / 3 for a in shares}
    M2 += macro('osIdShareMin', pct(min(shares.values()))) + macro('osIdShareMax', pct(max(shares.values())))
    M2 += macro('osIdShareNinetyMax', pct(max(p90.values()))) + macro('osIdCosMin', f"{min(cosv.values()):.3f}")
    M2 += macro('osIdModels', str(len(idr)))
    sec_raw = [sp[a]['secondary']['median_abs_log_rawmax_ratio'] for a in arms]; pri_raw = [sp[a]['primary']['median_abs_log_rawmax_ratio'] for a in arms]
    M2 += macro('osSecRawMin', f"{min(sec_raw):.2f}") + macro('osSecRawMax', f"{max(sec_raw):.2f}")
    M2 += macro('osPriRawMin', f"{min(pri_raw):.2f}") + macro('osPriRawMax', f"{max(pri_raw):.2f}")
    M2 += macro('osCamVCLow', pct(sp['cam_bce']['primary']['verdict_change_lower_bound'])) + macro('osCamVCHigh', pct(sp['cam_bce']['primary']['verdict_change_upper_bound']))
    pr = ra['centre_prior_own_threshold']
    M2 += macro('osPriorOwnThr', f"{pr['val_selected_threshold']:.2f}") + macro('osPriorOwnLoc', pct(pr['test_loc_at_val_threshold'])) + macro('osPriorOwnVal', pct(pr['val_hit_rate']))
    # ---- P0 revision results (post-hoc exploratory): macros + controls table + frozen-reference table
    pa = json.loads((D / 'p0a_results.json').read_text()); pb = json.loads((D / 'p0b_results.json').read_text()); pc = json.loads((D / 'p0c_results.json').read_text())
    pp = json.loads((D / 'p0c_proxy_anchor_partial.json').read_text())
    M3 = ''
    for k, name in [('Loc_F', 'LocF'), ('Loc_H', 'LocH'), ('Jq_F', 'JqF'), ('Jq_H', 'JqH'), ('EIB_F', 'EIBF'), ('EIB_H', 'EIBH'), ('centre_prior_at_F_threshold', 'PriorF'), ('Loc_F_correct_top1', 'LocFCorrect'), ('Loc_F_wrong_top1', 'LocFWrong')]:
        M3 += macro(f'pa{name}', pct(pa['summary'][k]['mean'])) + macro(f'pa{name}SD', pct(pa['summary'][k]['sd']))
    M3 += macro('paLocFSeeds', '/'.join(pct(x) for x in pa['seeds_Loc_F'])) + macro('paThrF', '/'.join(f"{pa['seeds'][str(s)]['threshold_F']:.2f}" for s in range(3)))
    for k, name in [('BGER_F-BGER_H/Loc', 'FvsH'), ('BGER_F-ce/Loc', 'FvsCE'), ('BGER_F-aux/Loc', 'FvsAux'), ('BGER_F-cam_bce/Loc', 'FvsCam'), ('BGER_F-abn_box/Loc', 'FvsAbn')]:
        c = pa['contrasts'][k]; M3 += macro(f'paC{name}', signed(c['point'])) + macro(f'paC{name}CI', ci(*c['basic_ci'])) + macro(f'paC{name}Seeds', '/'.join(signed(x) for x in c['per_seed']))
    fx = pa['fixed_F_mean']; M3 += macro('paFixPos', pct(fx['loc_pos'])) + macro('paFixNeg', pct(fx['loc_neg'])) + macro('paFixVerdict', pct(fx['verdict_change'])) + macro('paFixLone', f"{fx['mean_l1']:.2f}") + macro('paFixCorr', f"{fx['mean_corr']:.2f}")
    M3 += macro('paIdRes', f"{pa['checks']['max_identity_residual']:.1e}".replace('e-0', '\\times10^{-') + '}')
    for arm, tag in [('gate_only', 'Gate'), ('shifted_box', 'Shift')]:
        for k, kn in [('R1', 'Rone'), ('R10', 'Rten'), ('mAP', 'mAP'), ('Loc', 'Loc'), ('Jq', 'Jq')]:
            M3 += macro(f'pb{kn}{tag}', pct(pb['mean_sd'][arm][k]['mean'])) + macro(f'pb{kn}{tag}SD', pct(pb['mean_sd'][arm][k]['sd']))
        M3 += macro(f'pbRoneSeeds{tag}', '/'.join(pct(r['R1']) for r in pb['per_seed'][arm])) + macro(f'pbLocSeeds{tag}', '/'.join(pct(r['Loc']) for r in pb['per_seed'][arm]))
    for k, name in [('gate_only-ce/R1', 'GateCERone'), ('gate_only-ce/Loc', 'GateCELoc'), ('bger-gate_only/R1', 'BgerGateRone'), ('bger-gate_only/Loc', 'BgerGateLoc'),
                    ('shifted_box-ce/R1', 'ShiftCERone'), ('shifted_box-ce/Loc', 'ShiftCELoc'), ('bger-shifted_box/R1', 'BgerShiftRone'), ('bger-shifted_box/Loc', 'BgerShiftLoc'),
                    ('gate_only-aux/R1', 'GateAuxRone'), ('gate_only-aux/Loc', 'GateAuxLoc'), ('shifted_box-gate_only/R1', 'ShiftGateRone'), ('shifted_box-gate_only/Loc', 'ShiftGateLoc')]:
        c = pb['contrasts'][k]; M3 += macro(f'pbC{name}', signed(c['point'])) + macro(f'pbC{name}CI', ci(*c['basic_ci'])) + macro(f'pbC{name}Seeds', '/'.join(signed(x) for x in c['per_seed']))
    st = pb['shifted_box_stats']; M3 += macro('pbShiftIoU', f"{st['mean_iou_with_true']:.3f}") + macro('pbShiftBelow', pct(st['frac_iou_below_0p1'], 1))
    for tag, name in [('openclip_vitb32__study', 'ClipS'), ('openclip_vitb32__native', 'ClipN'), ('dinov2_vitb14__study', 'DinoS'), ('dinov2_vitb14__native', 'DinoN')]:
        r = pc['results'][tag]
        for k, kn in [('recall1', 'Rone'), ('recall10', 'Rten'), ('mAP', 'mAP')]: M3 += macro(f'pc{kn}{name}', pct(r[k]))
        M3 += macro(f'pcQone{name}', pct(r['R1_by_box_area_quartile'][0])) + macro(f'pcQfour{name}', pct(r['R1_by_box_area_quartile'][3])) + macro(f'pcMs{name}', f"{r['ms_per_image']:.1f}")
        c = r['contrast_vs_BGER_R1']; M3 += macro(f'pcC{name}Bger', signed(c['point'])) + macro(f'pcC{name}BgerCI', ci(*c['basic_ci']))
    M3 += macro('pcClipParams', f"{pc['params']['openclip_vitb32'] / 1e6:.0f}") + macro('pcDinoParams', f"{pc['params']['dinov2_vitb14'] / 1e6:.0f}")
    M3 += macro('ppEpochs', str(pp['epochs_completed'])) + macro('ppValTen', pct(pp['val_recall1']['10'])) + macro('ppValTwenty', pct(pp['val_recall1']['20']))
    with open(O / 'tables' / 'osld_results_numbers.tex', 'a') as f: f.write(M2 + M3)
    # Table: controls and readout hooks (exploratory) — organized by question
    core_ms = core['mean_sd']
    rows = []
    for arm, label in [('ce','CE'),('aux','Aux-BCE (off path)'),('bger','BGER-H (true boxes)')]:
        rows.append(f"{label} & {pm(core_ms[arm]['R1'])} & {pm(core_ms[arm]['Loc_q'])} & {pm(core_ms[arm]['Jq1'])}\\\\")
    rows.append(r'\midrule')
    rows.append(f"BGER-F (same models) & {pm(core_ms['bger']['R1'])} & {pm(pa['summary']['Loc_F'])} & {pm(pa['summary']['Jq_F'])}\\\\")
    for arm, label in [('gate_only','Gate-only (no boxes)'),('shifted_box','Gate (shifted boxes)')]:
        rows.append(f"{label} & {pm(pb['mean_sd'][arm]['R1'])} & {pm(pb['mean_sd'][arm]['Loc'])} & {pm(pb['mean_sd'][arm]['Jq'])}\\\\")
    T = (r"\begin{table}[t]"+'\n'+r"\centering\footnotesize\setlength{\tabcolsep}{3pt}"+'\n'
         r"\caption{Gate controls and readout hooks (\%, mean$\pm$SD, seeds 0--2). Above the rule: matched-study references at $H$. Below: exploratory pooled-$F$ readout of the same BGER checkpoints, then two gate controls trained under the frozen recipe. All gates except Aux-BCE are on the pooling path. Shifted boxes retain size but use random train-only positions (mean true-box IoU \pbShiftIoU). Localization thresholds are validation-locked.}"+'\n'
         r"\label{tab:controls}\begin{tabular}{lccc}\toprule"+'\n'
         r"Arm / readout & R@1 & $\bar L^{\mathrm q}$ & $Jq@1$\\\midrule"+'\n'
         +'\n'.join(rows)+'\n'+r"\bottomrule\end{tabular}\end{table}"+'\n')
    (O / 'tables' / 'osld_controls.tex').write_text(T)
    rows = []
    for tag, label in [('openclip_vitb32__study','OpenCLIP whole'),('openclip_vitb32__native','OpenCLIP crop'),('dinov2_vitb14__study','DINOv2 whole'),('dinov2_vitb14__native','DINOv2 crop')]:
        r = pc['results'][tag]
        rows.append(f"{label} & {pct(r['recall1'])} & {pct(r['recall10'])} & {pct(r['mAP'])} & {pct(r['R1_by_box_area_quartile'][0])} & {pct(r['R1_by_box_area_quartile'][3])}\\\\")
    rows += [r'\midrule', f"BGER & {pct(core_ms['bger']['R1']['mean'])} & {pct(core_ms['bger']['R10']['mean'])} & {pct(core_ms['bger']['mAP']['mean'])} & \\osSizeBgerRA & \\osSizeBgerRD\\\\"]
    T = (r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}"+'\n'
         r"\caption{Frozen retrieval references (\%), identical queries, gallery and relevance rule; without OSLD-specific fine-tuning or box supervision. Q1/Q4: R@1 in the smallest/largest box-area quartile. Whole: bilinear whole-image resize; crop: implemented bicubic resize/centre crop, labelled \texttt{native} in records (Section~\ref{sec:frozen}). Both variants reported. Architecture, objective and pretraining differ from the matched arms; pretraining overlap is unknown. Score localization was not evaluated for these references.}"+'\n'
         r"\label{tab:frozen}\begin{tabular}{lccccc}\toprule"+'\n'
         r"Model / input & R@1 & R@10 & mAP & Q1 & Q4\\\midrule"+'\n'
         +'\n'.join(rows)+'\n'+r"\bottomrule\end{tabular}\end{table}"+'\n')
    (O / 'tables' / 'osld_frozen.tex').write_text(T)
    # All exploratory contrasts, including rows omitted from the main forest.
    rows = []
    labels = {'BGER_F-BGER_H':'BGER-F $-$ BGER-H','BGER_F-ce':'BGER-F $-$ CE','BGER_F-aux':'BGER-F $-$ Aux-BCE','BGER_F-cam_bce':'BGER-F $-$ CAM-BCE','BGER_F-abn_box':'BGER-F $-$ ABN-box',
              'gate_only-ce':'Gate-only $-$ CE','bger-gate_only':'BGER-H $-$ gate-only','shifted_box-ce':'Shifted-box $-$ CE','bger-shifted_box':'BGER-H $-$ shifted-box','gate_only-aux':'Gate-only $-$ Aux-BCE','shifted_box-gate_only':'Shifted-box $-$ gate-only'}
    for group in [pa, pb]:
        for key, c in group['contrasts'].items():
            pair, metric = key.split('/')
            rows.append(f"{labels[pair]} & {metric} & {signed(c['point'])} & {ci(*c['basic_ci'])} & {'/'.join(signed(x) for x in c['per_seed'])}\\\\")
    T = (r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}"+'\n'
         r"\caption{Complete exploratory hook/control contrasts (pp). Basic 95\% intervals from 10,000 shared query-identity cluster draws; conditional on trained models and fixed gallery, unadjusted and not simultaneous. No confirmatory $p$-values. R1: Recall@1; Loc: query-logo localization. Seeds list paired differences.}"+'\n'
         r"\label{tab:posthoc-all}\begin{tabular}{llrcc}\toprule Contrast & Metric & Estimate & 95\% interval & Seeds 0/1/2\\\midrule"+'\n'
         +'\n'.join(rows)+'\n'+r"\bottomrule\end{tabular}\end{table*}"+'\n')
    (O / 'tables' / 'osld_posthoc_all.tex').write_text(T)
    # --- per-seed (appendix)
    lines = []
    for r in core['per_seed']:
        lines.append(f"{NAMES[r['arm']]} & {r['seed']} & {pct(r['R1'])} & {pct(r['R10'])} & {pct(r['mAP'])} & {pct(r['Loc_q'])} & {pct(r['Jq1'])} & "
                     f"{pct(r['EIB_q'])} & {pct(r['degenerate_rate'])} & {pct(r['Loc_q_secondary'])} & {r['threshold_primary']:.2f}\\\\")
    T = ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\caption{OSLD per-seed test values (\\%). Sec.: secondary-readout query localization; "
         "$t$: development-locked primary threshold.}\n\\label{tab:osld-perseed}\n"
         "\\begin{tabular}{lcccccccccc}\n\\toprule\nArm & Seed & R@1 & R@10 & mAP & $\\bar L^{\\mathrm q}$ & $Jq@1$ & EIB & Degen. & Sec. & $t$\\\\\n\\midrule\n"
         + '\n'.join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    (O / 'tables' / 'osld_perseed.tex').write_text(T)
    # --- development grid (appendix)
    lines = [f"{NAMES[r['arm']]} & {r['seed']} & {r['lam']:g} & {pct(r['R1'])} & {pct(r['R10'])} & {pct(r['mAP'])} & {pct(r['Loc_q'])} & {pct(r['Jq1'])} & "
             f"{r['threshold_primary']:.2f} & {CHECK if r['selected'] else ''}\\\\" for r in core['dev_grid']]
    T = ("\\begin{table}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\n\\caption{Complete OSLD development record (official validation: "
         "1,295 queries, 7,272 train+validation templates; \\%). Seed-0 rows select $\\lambda$ within each arm (Recall@1 within 1~pp of the arm's "
         "best, then highest $\\bar L^{\\mathrm q}$, ties to the smaller $\\lambda$); seeds 1--2 lock thresholds.}\n\\label{tab:osld-dev}\n"
         "\\begin{tabular}{lccccccccc}\n\\toprule\nArm & Seed & $\\lambda$ & R@1 & R@10 & mAP & $\\bar L^{\\mathrm q}$ & $Jq@1$ & $t$ & Sel.\\\\\n\\midrule\n"
         + '\n'.join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (O / 'tables' / 'osld_dev.tex').write_text(T)
    # --- small-logo and correct/wrong (descriptive)
    edges = desc['area_quartile_edges']
    hdr = ' & '.join(f"Q{b + 1}" for b in range(4))
    lines = []
    for arm in arms:
        sl = desc['small_logo'][arm]; bt = desc['by_top1'][arm]
        lines.append(f"{NAMES[arm]} & " + ' & '.join(pct(r['Q']) for r in sl) + f" & {pct(bt['correct']['Q'])} & {pct(bt['wrong']['Q'])}\\\\")
    lines.append('\\midrule')
    for arm in arms:
        lines.append(f"{NAMES[arm]} R@1 & " + ' & '.join(pct(r['R']) for r in desc['small_logo'][arm]) + ' & & \\\\')
    T = ("\\begin{table}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
         f"\\caption{{Query localization (top block) and Recall@1 (bottom block) by box-area quartile of the test queries (Q1 smallest; edges "
         f"{', '.join(f'{100 * e:.2f}' for e in edges)}\\% of the image; {', '.join(str(n) for n in desc['bin_counts'])} queries), and localization "
         "for correct and wrong own top-1 templates (\\%, seed-averaged). Pre-specified descriptive breakdown.}\n\\label{tab:osld-size}\n"
         f"\\begin{{tabular}}{{lcccccc}}\n\\toprule\nArm & {hdr} & Correct & Wrong\\\\\n\\midrule\n" + '\n'.join(lines) +
         "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (O / 'tables' / 'osld_size.tex').write_text(T)
    # --- numeric map figure (saved maps, grid values; no photographs)
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42; matplotlib.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ex = desc['examples']
    if ex:
        fig, axes = plt.subplots(len(ex), len(arms), figsize=(0.95 * len(arms) + .3, 1.0 * len(ex) + .25), squeeze=False)
        for i, e in enumerate(ex):
            for j, arm in enumerate(arms):
                ax = axes[i][j]; m = np.array(e['arms'][arm]['map'], dtype=float); h, w = m.shape
                v = np.abs(m).max() or 1.0
                ax.imshow(m, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
                b = e['box']; sx, sy = w / e['width'], h / e['height']
                ax.add_patch(Rectangle((b[0] * sx - .5, b[1] * sy - .5), (b[2] - b[0]) * sx, (b[3] - b[1]) * sy, fill=False, lw=1.0, ec='k'))
                info = e['arms'][arm]
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(('R' if info['top1_relevant'] else 'W') + ('/L' if info['localized'] else '/--'), fontsize=10, labelpad=1)
                if i == 0: ax.set_title(NAMES[arm], fontsize=10.5, pad=2)
                if j == 0: ax.set_ylabel(e['label'].replace('_', ' '), fontsize=10)
        fig.tight_layout(pad=.2)
        fig.savefig(O / 'figures' / 'osld_maps.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    print('wrote OSLD result tables to', O)


if __name__ == '__main__':
    main()
