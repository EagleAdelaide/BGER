#!/usr/bin/env python3
"""Present unchanged numerical tables without floats or a near-empty cover.

Only table wrappers change. Main-paper layout and numerical table sources
remain unchanged. The companion uses the same ACM body font and margins.
"""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tables'
names=['osld_posthoc_all','osld_contrasts_artifact','osld_dev','osld_a5','cub_osld_comparison','grid','all_seeds']
manifest=[]
for i,name in enumerate(names,1):
    p=T/(name+'.tex');body=p.read_text()
    body,n=re.subn(r'\\begin\{table\*?\}\[t\]',r'\\begingroup\n\\captionsetup{type=table}',body,count=1)
    assert n==1,name
    body,n=re.subn(r'\\end\{table\*?\}',r'\\par\n\\endgroup',body,count=1)
    assert n==1,name
    out=T/f'expanded_table_{i:02}.tex';out.write_text(body)
    manifest.append({'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'output':str(out.relative_to(ROOT)),'changes':'non-floating local wrapper only; all caption, numeric, label and tabular content unchanged'})
(T/'EXPANDED_TABLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Seven expanded-table wrappers regenerated without numerical changes')
