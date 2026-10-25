#!/usr/bin/env python3
"""Generate original small PNGs and annotation files; not DeepFashion data."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from e1_common import save,sha

def make(root):
    root=Path(root)
    if root.exists():raise FileExistsError('Refuse to overwrite fixture')
    for d in ['Img','Anno','Eval']:(root/d).mkdir(parents=True,exist_ok=True)
    rows=[];rng=np.random.default_rng(270127)
    for p in range(14):
        for v in range(2):rows.append((f'Img/t{p:02d}_{v}.png',f't{p:02d}','train'))
    for p in range(3):
        for v in range(p+1):rows.append((f'Img/q{p}_{v}.png',f'p{p}','query'))
        for v in range(2):rows.append((f'Img/g{p}_{v}.png',f'p{p}','gallery'))
    boxes=[]
    for idx,(name,pid,split) in enumerate(rows):
        # Independent generated pixels, arbitrary product IDs: no efficacy meaning.
        a=rng.integers(0,256,(32,40,3),dtype=np.uint8)
        Image.fromarray(a).save(root/name)
        box=[1,1,2,2] if split=='train' else [3,2,36,29]
        boxes.append(name+' 1 1 '+' '.join(map(str,box)))
    (root/'Eval/list_eval_partition.txt').write_text(str(len(rows))+'\nimage_name item_id evaluation_status\n'+'\n'.join(' '.join(r) for r in rows)+'\n')
    (root/'Anno/list_bbox_inshop.txt').write_text(str(len(rows))+'\nimage_name clothes_type pose x_1 y_1 x_2 y_2\n'+'\n'.join(boxes)+'\n')
    save(root/'SYNTHETIC_INTEGRATION.json',dict(identity='SYNTHETIC_INTEGRATION',generator='make_cpu_fixture.py',
        description='Generated random PNGs, artificial IDs/half-open boxes, not DeepFashion/B0',
        bbox_convention='zero_half_open',files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    return root
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);make(a.parse_args().out)
