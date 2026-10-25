"""Ranking -> actual-top1 / fixed-pair -> both readouts/sides -> all endpoints."""
from collections import OrderedDict
from pathlib import Path
import json,os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import web_eval as W
from e1_common import GRID,RECIPE,evaluation,save,digest,sha,now,native,append_jsonl


def score_row(z, gallery):
    """FP32, raw embeddings, chunk-64 multiply/reduce (not normalized GEMM)."""
    if z.dtype!=torch.float32 or gallery.dtype!=torch.float32:raise ValueError('Ranking requires FP32')
    chunks=[]
    for g in gallery.split(RECIPE['score_chunk']):
        chunks.append(((g*z).sum(1)/(g.norm(dim=1)*z.norm()+1e-12)).detach().cpu().numpy())
    scores=np.concatenate(chunks)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite ranking score; entire run failed')
    return scores

LOADER_WORKERS=0  # set by the runner; parallel I/O only, batch size stays 1, order preserved

def embed(model,dataset,device,log):
    values=[]
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=LOADER_WORKERS,
                                       collate_fn=lambda b:b[0])
    with evaluation(model,log),torch.no_grad():
        for item in loader:
            x=item['x'].unsqueeze(0).to(device)
            _,_,z=model(x)
            if not torch.isfinite(z).all():raise ValueError('Nonfinite pooled embedding')
            values.append(z[0].detach().cpu())
    return torch.stack(values)

def rank(qz,gz,qr,gr):
    gid=np.array([r['image_id'] for r in gr]);gp=np.array([r['product_id'] for r in gr])
    result=[]
    if len(set(gid))!=len(gid) or len({r['image_id'] for r in qr})!=len(qr):raise ValueError('Duplicate ranking image IDs')
    for i,r in enumerate(qr):
        scores=score_row(qz[i],gz)
        m=W.retrieval_metrics(scores[None],np.array([r['product_id']]),gp,
                              query_ids=[r['image_id']],gallery_ids=gid)
        top=int(m['top1_index'][0]);valid=np.flatnonzero(gid!=r['image_id'])
        order=valid[np.lexsort((gid[valid],-scores[valid]))]
        pos=order[gp[order]==r['product_id']];neg=order[gp[order]!=r['product_id']]
        result.append(dict(query_id=r['image_id'],product_id=r['product_id'],gallery_id=str(gid[top]),
                           top_index=top,score=float(scores[top]),correct=int(m['top1_correct'][0]),
                           recall10=m['recall@10'],AP=m['mAP'],
                           positive=str(gid[pos[0]]),negative=str(gid[neg[0]]) if len(neg) else None))
    return result

def largest_box(a,threshold):
    lab,n=ndimage.label(a>=threshold,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    if n==0:return None
    counts=np.bincount(lab.ravel());counts[0]=0
    # Label numbering and argmax preserve row-major first component on ties.
    y,x=np.where(lab==counts.argmax())
    return (int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1))

def map_metrics(signed,record):
    positive=signed.detach().clamp_min(0)[None,None]
    resized=F.interpolate(positive,size=(record['height'],record['width']),mode='bilinear',align_corners=False)[0,0].cpu().double().numpy()
    cam,degen=W.normalize_map(resized)
    box=record['box_xyxy_half_open'];ious=np.zeros(len(GRID))
    if not degen:ious=np.array([W.iou(largest_box(cam,float(t)),box) for t in GRID])
    hits=np.stack([ious>=v for v in (.3,.5,.7)],axis=1).astype(int)
    # Continuous half-open box -> pixel index set: floor lower, ceil upper (OSLD boxes are
    # stored as floats; integer boxes are unchanged by this).
    x0,y0=int(np.floor(box[0])),int(np.floor(box[1]));x1,y1=int(np.ceil(box[2])),int(np.ceil(box[3]))
    # EIB uses exactly the nonnegative registered min-max localization map.
    energy=float(cam.sum());eib=0.0 if degen or energy==0 else float(cam[y0:y1,x0:x1].sum()/energy)
    return dict(hits=hits.tolist(),EIB=eib,degenerate=bool(degen),nonfinite=bool(not np.isfinite(resized).all()))

class PairReader:
    def __init__(self,model,qd,gd,qz,gz,device,log,maps_dir=None,identity=None):
        self.model=model;self.qd=qd;self.gd=gd;self.qz=qz;self.gz=gz;self.device=device;self.log=log
        self.maps_dir=maps_dir;self.cache=OrderedDict();self.score_cache=OrderedDict();self.identity=identity
        self.qindex={r['image_id']:i for i,r in enumerate(qd.records)}
        self.gindex={r['image_id']:i for i,r in enumerate(gd.records)}
    def features(self,ds,i):
        key=ds.records[i]['image_id']
        if key in self.cache:return self.cache[key]
        with torch.no_grad():h=self.model.features(ds[i]['x'][None].to(self.device))[0].detach()
        self.cache[key]=h
        if len(self.cache)>8:self.cache.popitem(last=False)
        return h
    def pair(self,qid,gid,label):
        i,j=self.qindex[qid],self.gindex[gid]
        hq,hg=self.features(self.qd,i),self.features(self.gd,j)
        zq,zg=self.qz[i].to(self.device),self.gz[j].to(self.device)
        # Read the chosen pair from the FULL gallery ranking row. This checks
        # the deployed chunked ranking reduction against single-pair autograd.
        if i not in self.score_cache:
            self.score_cache[i]=score_row(self.qz[i],self.gz)
            if len(self.score_cache)>4:self.score_cache.popitem(last=False)
        ranking=float(self.score_cache[i][j])
        result=dict(query_id=qid,gallery_id=gid,view=label,score=ranking)
        maps={}
        for side,h,other,record in [('query',hq,zg,self.qd.records[i]),('gallery',hg,zq,self.gd.records[j])]:
            primary,_,diag=W.retrieval_gradcam_audited(h,lambda f:self.model.pool(f[None])[0],other)
            with torch.no_grad():secondary,ss=W.score_decomposition(self.model.pooled_features(h[None])[0],other)
            error=abs(diag['score']-ranking);sumerror=abs(float(secondary.sum())-ss)
            if not np.isfinite([diag['score'],ss]).all():raise ValueError('Nonfinite readout score')
            if error>RECIPE['score_atol'] or abs(ss-ranking)>RECIPE['score_atol']:raise ValueError('Ranking/readout score mismatch')
            if torch.isfinite(secondary).all() and sumerror>RECIPE['secondary_sum_atol']:raise ValueError('Secondary additivity exceeds FP32 tolerance')
            # Nonfinite MAPS are retained as localization misses; finite scores
            # remain in retrieval. JSON uses null for undefined diagnostics.
            diag={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in diag.items()}
            if not np.isfinite(sumerror):sumerror=None
            # OSLD: canonical templates have no product-region ground truth; their
            # maps are kept descriptively but never scored for localization.
            scored=record.get('box_xyxy_half_open') is not None
            result[side]={'primary':map_metrics(primary,record) if scored else None,
                          'secondary':map_metrics(secondary,record) if scored else None,
                          'diagnostic':dict(diag,ranking_score_error=error,secondary_score_error=abs(ss-ranking),
                                            secondary_additivity_error=sumerror)}
            maps[side+'_primary_signed']=primary.detach().cpu().numpy()
            maps[side+'_secondary_signed']=secondary.detach().cpu().numpy()
        if self.maps_dir is not None:
            self.maps_dir.mkdir(parents=True,exist_ok=True)
            name=digest([qid,gid,label])+'.npz';tmp=self.maps_dir/(name+'.tmp.npz')
            np.savez_compressed(tmp,identity=np.array(self.identity),**maps);tmp.replace(self.maps_dir/name)
            result['signed_maps_file']=name
        return result

def endpoints(pairs,ranks,thresholds):
    """OSLD-SingleLogo-Retrieval-v1: query-side localization only; Jq@1=R*Lq."""
    result={};correct=np.array([r['correct'] for r in ranks])
    for readout in ['primary','secondary']:
        idx=int(np.argmin(abs(GRID-thresholds[readout])))
        q=np.array([p['query'][readout]['hits'] for p in pairs])
        hq=q[:,idx,1]
        jq=correct*hq
        result[readout]=dict(Jq_at_1=float(jq.mean()),marginal_top1=float(correct.mean()),marginal_loc_query=float(hq.mean()),
            threshold=thresholds[readout],
            EIB_query=float(np.mean([p['query'][readout]['EIB'] for p in pairs])),
            V2_query=float(q.mean(0).max(0).mean()),
            degenerate_query=int(sum(p['query'][readout]['degenerate'] for p in pairs)),
            nonfinite_query=int(sum(p['query'][readout]['nonfinite'] for p in pairs)),
            per_query=dict(R=correct.tolist(),Q=hq.tolist(),J=jq.tolist()))
    result.update(recall1=float(correct.mean()),recall10=float(np.mean([r['recall10'] for r in ranks])),
                  mAP=float(np.mean([r['AP'] for r in ranks])),n_queries=len(ranks),incorrect_top1_included=int((1-correct).sum()))
    return result

PAIR_CHUNK=100   # readout pairs per atomic progress file (resume granularity)
EVAL_OUTPUTS=('ranking.json','own_top1.json','fixed_pairs.json','summary.json')
EMBED_KEYS=('identity','stage','checkpoint_sha256','source_digest')

def records_digest(records):
    return digest([(r['image_id'],r.get('sha256'),r['product_id'],r.get('box_xyxy_half_open'),r['width'],r['height']) for r in records])

def cached_embeddings(model,qd,gd,device,out,log,binding):
    """Embeddings are computed once per (model checkpoint, inputs, implementation) and
    read back on resume, so resumed rankings/readouts use exactly the stored scores."""
    out.mkdir(parents=True,exist_ok=True);path=out/'embeddings.pt'
    key=digest(dict({k:binding.get(k) for k in EMBED_KEYS},query_records=records_digest(qd.records),gallery_records=records_digest(gd.records)))
    if path.exists():
        c=torch.load(path,map_location='cpu',weights_only=False)
        if c.get('key')!=key:raise ValueError('Stored embeddings have a different identity; kept, not overwritten: '+str(path))
        log('embeddings_reused',out=str(out));return c['query'],c['gallery']
    qz,gz=embed(model,qd,device,log),embed(model,gd,device,log)
    tmp=path.with_name(path.name+'.tmp');torch.save(dict(key=key,query=qz,gallery=gz),tmp);tmp.replace(path)
    return qz,gz

def verify_complete(out,key,qd,fixed_pairs):
    """A completed evaluation is read back only if identity, file hashes and the expected
    queries / readouts / fixed signs / map files all check; it is never recomputed."""
    m=json.loads((out/'EVALUATION_COMPLETE.json').read_text())
    if m['evaluation_key']!=digest(key):raise ValueError('Completed evaluation identity conflict; kept: '+str(out))
    if set(m['files'])!=set(EVAL_OUTPUTS):raise ValueError('Completed evaluation manifest lacks outputs: '+str(out))
    for f,h in m['files'].items():
        if not (out/f).is_file() or sha(out/f)!=h:raise ValueError('Completed evaluation file missing/changed: '+str(out/f))
    summary=json.loads((out/'summary.json').read_text());own=json.loads((out/'own_top1.json').read_text())['pairs']
    fixed=json.loads((out/'fixed_pairs.json').read_text())['pairs'];ids=[r['image_id'] for r in qd.records]
    signs=[] if fixed_pairs is None else ['ce_seed0_reference_positive','ce_seed0_reference_negative']*len(fixed_pairs)
    ok=(summary['query_ids']==ids and [p['query_id'] for p in own]==ids and sorted(p['view'] for p in fixed)==sorted(signs)
        and all(ro in summary for ro in ('primary','secondary')) and (not signs or 'fixed_candidate' in summary)
        and all(p['query'][ro] is not None for p in own+fixed for ro in ('primary','secondary'))
        and all((out/'signed_maps'/p['signed_maps_file']).is_file() for p in own+fixed))
    if not ok:raise ValueError('Completed evaluation fails completeness check: '+str(out))
    return summary

def evaluate(model,qd,gd,device,out,log,identity,thresholds=None,fixed_pairs=None,embeddings=None,binding=None):
    """Resumable evaluation. EVALUATION_IDENTITY.json binds run/checkpoint/inputs/gallery/
    implementation/thresholds/fixed-pair manifest; readouts are written in atomic chunks;
    a verified EVALUATION_COMPLETE.json is read back. Identity conflicts stop, never overwrite."""
    out.mkdir(parents=True,exist_ok=True);base=dict(binding or {},identity=identity)
    key=native(dict(base,query_records=records_digest(qd.records),gallery_records=records_digest(gd.records),
                    n_queries=len(qd.records),n_gallery=len(gd.records),thresholds=thresholds,
                    fixed_pairs=None if fixed_pairs is None else digest(fixed_pairs),pair_chunk=PAIR_CHUNK))
    idpath=out/'EVALUATION_IDENTITY.json';done=out/'EVALUATION_COMPLETE.json';pdir=out/'partial'
    if idpath.exists():
        if json.loads(idpath.read_text())!=key:raise ValueError('Evaluation identity conflict; existing outputs kept: '+str(out))
    else:
        if any((out/f).exists() for f in EVAL_OUTPUTS) or pdir.exists() or done.exists():raise ValueError('Unbound evaluation outputs present; not overwritten: '+str(out))
        save(idpath,key)
    if done.exists():
        summary=verify_complete(out,key,qd,fixed_pairs)
        append_jsonl(out/'evaluation_attempts.jsonl',dict(utc=now(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),state='reused_complete'))
        log('evaluation_reused',out=str(out),queries=summary['n_queries']);return summary
    kd=digest(key);resumed=(out/'ranking.json').exists() or pdir.exists()
    qz,gz=embeddings if embeddings is not None else cached_embeddings(model,qd,gd,device,out,log,base)
    if (out/'ranking.json').exists():
        stored=json.loads((out/'ranking.json').read_text())
        if stored.get('evaluation_key')!=kd:raise ValueError('Stored ranking identity conflict: '+str(out))
        ranks=stored['rows']
    else:
        ranks=rank(qz,gz,qd.records,gd.records);save(out/'ranking.json',dict(identity=identity,evaluation_key=kd,rows=ranks))
    jobs=[[r['query_id'],r['gallery_id'],'own_top1'] for r in ranks]
    for r in fixed_pairs or []:
        for key_ in ['positive','negative']:
            if r[key_] is None:raise ValueError('Fixed reference lacks negative candidate')
            jobs.append([r['query_id'],r[key_],'ce_seed0_reference_'+key_])
    pdir.mkdir(exist_ok=True);got=[];computed=reused=0
    with evaluation(model,log):
        reader=PairReader(model,qd,gd,qz,gz,device,log,out/'signed_maps',identity)
        for c0 in range(0,len(jobs),PAIR_CHUNK):
            part=jobs[c0:c0+PAIR_CHUNK];cpath=pdir/f'chunk_{c0//PAIR_CHUNK:05d}.json'
            if cpath.exists():
                stored=json.loads(cpath.read_text())
                if stored['evaluation_key']!=kd or stored['jobs']!=part:raise ValueError('Partial readout identity conflict: '+str(cpath))
                got+=stored['pairs'];reused+=1;continue
            pairs_=[reader.pair(*j) for j in part]
            save(cpath,dict(evaluation_key=kd,jobs=part,pairs=pairs_));got+=pairs_;computed+=1
    pairs=got[:len(ranks)];fixed=got[len(ranks):]
    for p,r in zip(pairs,ranks):
        if (p['query_id'],p['gallery_id'])!=(r['query_id'],r['gallery_id']) or p['score']!=r['score']:raise ValueError('Own-top1 ranking score changed between callers')
    if thresholds is None:
        thresholds={key_:float(GRID[np.array([p['query'][key_]['hits'] for p in pairs])[:,:,1].mean(0).argmax()]) for key_ in ['primary','secondary']}
    summary=endpoints(pairs,ranks,thresholds)
    gal_counts={}
    for r in gd.records:gal_counts[r['product_id']]=gal_counts.get(r['product_id'],0)+1
    summary['random_R1_reference']=float(np.mean([gal_counts.get(r['product_id'],0)/len(gd.records) for r in qd.records]))
    summary['gallery_size']=len(gd.records)
    summary.update(identity=identity,thresholds=thresholds,query_ids=[r['query_id'] for r in ranks],
                   query_products=[r['product_id'] for r in ranks],max_ranking_readout_error=max(p[s]['diagnostic']['ranking_score_error'] for p in pairs+fixed for s in ['query','gallery']),
                   max_secondary_additivity_error=max((p[s]['diagnostic']['secondary_additivity_error'] for p in pairs+fixed for s in ['query','gallery'] if p[s]['diagnostic']['secondary_additivity_error'] is not None),default=None))
    save(out/'own_top1.json',dict(identity=identity,pairs=pairs));save(out/'fixed_pairs.json',dict(identity=identity,pairs=fixed))
    # Fixed pairs have relevance by construction; no retrieval endpoint is redefined.
    if fixed:
        summary['fixed_candidate']={}
        for key_ in ['positive','negative']:
            subset=[p for p in fixed if p['view'].endswith('_'+key_)]
            fake=[{'correct':int(key_=='positive'),'recall10':int(key_=='positive'),'AP':int(key_=='positive')} for p in subset]
            stats=endpoints(subset,fake,thresholds)
            summary['fixed_candidate'][key_]={ro:{k:v for k,v in stats[ro].items() if k not in ['Jq_at_1','marginal_top1','per_query']} for ro in ['primary','secondary']}
    save(out/'summary.json',summary)
    # Completion marker last: summary.json alone never means complete.
    save(done,dict(evaluation_key=kd,files={f:sha(out/f) for f in EVAL_OUTPUTS},n_queries=len(ranks),n_own_top1=len(pairs),
                   n_fixed=len(fixed),chunks=len(range(0,len(jobs),PAIR_CHUNK)),completed_utc=now()))
    append_jsonl(out/'evaluation_attempts.jsonl',dict(utc=now(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),state='completed',
                 resumed=resumed,chunks_reused=reused,chunks_computed=computed))
    summary=json.loads((out/'summary.json').read_text())   # identical return on fresh and resumed paths
    log('evaluation_complete',out=str(out),queries=len(ranks),wrong_top1=summary['incorrect_top1_included'],fixed_pairs=len(fixed),
        score_error=summary['max_ranking_readout_error'],chunks_reused=reused,chunks_computed=computed)
    return summary

def choose(candidates):
    best=max(c['summary']['recall1'] for c in candidates)
    eligible=[c for c in candidates if c['summary']['recall1']>=best-.01]
    return min(eligible,key=lambda c:(-c['summary']['primary']['marginal_loc_query'],c['lambda']))
