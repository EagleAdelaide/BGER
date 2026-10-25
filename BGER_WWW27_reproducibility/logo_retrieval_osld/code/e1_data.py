"""File-backed loader. No GT crops; original half-open evaluation coordinates."""
from pathlib import Path
import json, math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
import deepfashion_integrity_check as I
from e1_common import sha, save, digest

class ImageDataset(Dataset):
    def __init__(self,root,records,labels=None):
        self.root=Path(root);self.records=sorted(records,key=lambda r:r['image_id']);self.labels=labels or {}
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index];p=I._image_path(self.root,r['image_id'])
        if sha(p)!=r['sha256']:raise ValueError('Image changed after scan: '+r['image_id'])
        with Image.open(p) as im:
            im=im.convert('RGB');w,h=im.size
            if (w,h)!=(r['width'],r['height']):raise ValueError('Image geometry changed')
            x=TF.pil_to_tensor(im).float()/255
        b=list(r['box_xyxy_half_open'])
        if not(0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError('Invalid loader coordinates')
        x=TF.resize(x,[224,224],interpolation=InterpolationMode.BILINEAR,antialias=True)
        x=TF.normalize(x,[.485,.456,.406],[.229,.224,.225])
        if not torch.isfinite(x).all():raise ValueError('Nonfinite loader input')
        scaled=[b[0]*224/w,b[1]*224/h,b[2]*224/w,b[3]*224/h]
        x0,y0=math.floor(scaled[0]),math.floor(scaled[1]);x1,y1=math.ceil(scaled[2]),math.ceil(scaled[3])
        mask=torch.zeros(224,224);mask[y0:y1,x0:x1]=1
        return {'x':x,'label':self.labels.get(r['product_id'],-1),'mask':mask,
                'has_box':True,'index':index,'id':r['image_id'],'product':r['product_id'],
                'box':torch.tensor(b),'size':torch.tensor([h,w])}

def prepare(root,out,convention,identity):
    report=I.scan(root,convention);report['identity']=identity
    save(out/'technical_integrity.json',report)
    if report['status']!='TECHNICAL_CHECKS_PASS':raise ValueError('Technical data gate failed: '+str(report['errors']))
    split=I.development_manifests(report,out/'manifests')
    records={r['image_id']:r for r in report['image_records']}
    sets={}
    for key,f in [('train','train_after_dev.txt'),('dev_query','dev_query.txt'),('dev_gallery','dev_gallery.txt')]:
        sets[key]=[records[line.split()[0]] for line in (out/'manifests'/f).read_text().splitlines()]
    for key in ['query','gallery']:sets[key]=[r for r in records.values() if r['split']==key]
    # Newly created train/dev split also cannot contain pixel-identical leakage.
    tr={r['decoded_rgb_sha256'] for r in sets['train']}
    if tr & {r['decoded_rgb_sha256'] for r in sets['dev_query']+sets['dev_gallery']}:
        raise ValueError('Exact decoded training/development leakage; no silent exclusions')
    labels=json.loads((out/'manifests/train_label_map.json').read_text())
    split.update(identity=identity,records_digest=digest(report['image_records']),labels=labels)
    save(out/'split.json',split)
    return sets,labels,report
