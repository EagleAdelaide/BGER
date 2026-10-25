#!/usr/bin/env python3
"""Exact Figure 2 coordinate/export check; no inference or statistical reanalysis."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'BGER_WWW27_source'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED_INPUTS={'osld_core_results.json': '782d160ddccf2a385fc753cc30e81f526c3f4af26fb8b5f55ac517bf63315198', 'osld_readout_analysis.json': '99806e6a2312b9a61c543412f0b1d0100e27ca911e403e22e7f43854a90f8e8d'}
EXPECTED_ASSETS={'fig02_retrieval_vs_localization.pdf': 'f4b99094bdbfe9fbee785d3b30f5a2163408a31a5e51e96fca81bdecd6266f2e', 'fig02_retrieval_vs_localization.svg': '75fd81efdcc4c125c99d65cda7d514313de7dff68c37f13ae4b01face31ea581', 'fig02_retrieval_vs_localization.png': 'a35a49589219791ceea744f544c775e38bc0ff906c4d74d4ab86faf9152086a6'}
EXPECTED_COORDINATE_SHA256='93e930fb2f7ef4d021bda8fdbdb8a768fdac3b03747b0b93d864aa2c7305de8c'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'scope':'Frozen-record Figure 2 plotting only; no bootstrap, statistical reanalysis, GPU, model computation or data downloads', 'readout':'Prespecified H; ABN-box corresponding final perception features', 'layers':{}}
    inputs={name:sha(SOURCE/'data'/name) for name in EXPECTED_INPUTS}
    assert inputs==EXPECTED_INPUTS, 'Frozen input hash mismatch'
    report['inputs_sha256']=inputs
    module=SOURCE/'figures_src/make_figure2_primary_h.py'
    spec=importlib.util.spec_from_file_location('accepted_figure2',module)
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    core=json.loads((SOURCE/'data/osld_core_results.json').read_text())
    readout=json.loads((SOURCE/'data/osld_readout_analysis.json').read_text())
    trace={};original=p.savefig
    def capture(fig,name,manifest,used_inputs):
        ax=fig.axes[0];assert len(fig.axes)==1 and len(ax.collections)==10
        arrays=[np.asarray(c.get_offsets()).tolist() for c in ax.collections]
        for i,arm in enumerate(p.ARMS):
            xs=np.array(core['mean_sd'][arm]['R1']['per_seed'])*100
            ys=np.array(core['mean_sd'][arm]['Loc_q']['per_seed'])*100
            assert len(xs)==len(ys)==3
            assert arrays[i*2]==np.column_stack([xs,ys]).tolist()
            assert arrays[i*2+1]==[[xs.mean(),ys.mean()]]
            assert len(ax.collections[i*2].get_facecolors())==0
            assert len(ax.collections[i*2+1].get_facecolors())>0
            assert ax.collections[i*2].get_sizes()[0]<ax.collections[i*2+1].get_sizes()[0]
        digest=hashlib.sha256(json.dumps(arrays,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert digest==EXPECTED_COORDINATE_SHA256
        assert ax.get_xlim()==(0,8.3) and ax.get_ylim()==(0,22)
        assert len(ax.lines)==1 and list(ax.lines[0].get_xdata())==[0,1]
        prior=readout['centre_prior_own_threshold']['test_loc_at_val_threshold']*100
        assert list(ax.lines[0].get_ydata())==[prior,prior]
        assert not ax.texts
        assert [ax.get_xlabel(),ax.get_ylabel()]==['Recall@1 (%)','Ranking-score localization (%)']
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        assert all(not leg.get_window_extent(renderer).overlaps(ax.bbox) for leg in fig.legends)
        labels=[t.get_text() for leg in fig.legends for t in leg.get_texts()]
        assert labels==['CE','Aux-BCE','BGER','CAM-BCE','ABN-box','centre prior']
        shapes=[np.asarray(ax.collections[i*2].get_paths()[0].vertices).tolist() for i in range(5)]
        assert len({json.dumps(s) for s in shapes})==5
        trace.update({'xy_percent':arrays,'coordinate_sha256':digest,'prior_percent':prior,'axes_limits':[[0,8.3],[0,22]],'model_points':20,'coordinate_scalars':40,'max_abs_difference':0.0,'legend_outside_data':True,'in_plot_model_annotations':0,'distinct_marker_shapes':5,'seed_style':'smaller hollow; above filled means','mean_style':'larger filled with dark outline'})
        original(fig,name,manifest,used_inputs)
    p.savefig=capture
    try:p.draw_figure2(core,readout,{},source_root=SOURCE,output_dir=out/'figures')
    finally:p.savefig=original
    hashes={name:sha(out/'figures'/name) for name in EXPECTED_ASSETS}
    assert hashes==EXPECTED_ASSETS, 'Accepted vector/preview bytes differ'
    # Check the existing delegation interface preserves the ambient style and
    # yields the same accepted bytes, without running other figure generators.
    sys.path.insert(0,str(SOURCE/'figures_src'))
    import make_publication_figures as legacy
    import make_figure2_primary_h as isolated
    before=dict(legacy.plt.rcParams);previous_data,previous_out=legacy.DATA,legacy.OUT
    legacy.OUT=out/'legacy_entry';legacy.OUT.mkdir()
    try:legacy.fig02(core,readout,None,{})
    finally:legacy.DATA,legacy.OUT=previous_data,previous_out
    assert dict(legacy.plt.rcParams)==before
    assert {name:sha(out/'legacy_entry'/name) for name in EXPECTED_ASSETS}==EXPECTED_ASSETS
    report['layers']={'frozen_inputs':'PASS','20_points_40_scalars':'PASS','prior_and_axes':'PASS','marker_and_legend_encoding':'PASS','exact_PDF_SVG_PNG_regeneration':'PASS','legacy_Figure2_entry_and_style_restoration':'PASS','fresh_training_checkpoint_inference':'NOT_RUN_NOT_AUTHORIZED','broader_saved_record_statistical_checks':'NOT_RERUN_HISTORICAL_REPORT_RETAINED'}
    report['coordinates']=trace;report['generated_sha256']=hashes
    report['generator_sha256']=sha(module)
    report['visual_limitation']='Known fixed-coordinate CE-mean / ABN-box-seed partial overlap retained in the visually accepted candidate; zero-marker-overlap is not certified.'
    report['status']='PASS_FIGURE2_ONLY'
    (out/'FIGURE2_PRIMARY_H_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 20 primary-H points, exact reference and axes, 3 accepted exports, isolated legacy delegation. No statistical reanalysis.')
if __name__=='__main__':main()
