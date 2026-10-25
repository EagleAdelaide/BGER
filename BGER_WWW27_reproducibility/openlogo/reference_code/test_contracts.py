#!/usr/bin/env python3
"""Data-independent contracts; distinct from the real production target gate."""
import unittest,tempfile,copy
from pathlib import Path
import numpy as np
import torch
from b_common import *
from cpu_metrics import evaluate_map,largest_box
from analyze_results import mixture_series,SharedBootstrap
from e1_evaluate import map_metrics

class Contracts(unittest.TestCase):
    def test_source_metric_equivalence_two_conventions(self):
        m=np.random.default_rng(4).normal(size=(7,7)).astype(np.float32)
        r={'width':47,'height':33,'raw_box':[3,4,40,30]}
        pred,ious,eib,degen,non=evaluate_map(m,r,list(range(19)))
        for k in [0,1]:
            source=map_metrics(torch.from_numpy(m),dict(r,box_xyxy_half_open=box(r['raw_box'],k)))
            self.assertEqual((ious[:,k]>=.5).astype(int).tolist(),[x[1] for x in source['hits']])
            self.assertEqual(eib[k],source['EIB']);self.assertEqual(degen,source['degenerate'])
    def test_degenerate_maps_are_misses(self):
        # Empty positive maps remain exactly empty through FP32 interpolation.
        # Positive constants can pick up rounding at resize, as in the source
        # evaluator; do not impose a stronger degeneracy policy on this run.
        for m in [np.zeros((7,7),np.float32),-np.ones((7,7),np.float32)]:
            p,ious,_,deg,_=evaluate_map(m,{'width':37,'height':41,'raw_box':[1,1,20,21]},[0,18])
            self.assertTrue(deg);self.assertFalse((ious>=.5).any())
    def test_actual_loss_target_transform(self):
        m=torch.zeros((1,224,224))
        m[:,:32,:32]=1.
        m[:,32:64,32:64]=.5
        m[:,64:96,64:96]=.5001
        t=effective_target(m)
        self.assertEqual(tuple(t.shape),(1,7,7))
        self.assertEqual(float(t[0,0,0]),1.)
        self.assertEqual(float(t[0,1,1]),0.)
        self.assertEqual(float(t[0,2,2]),1.)
        self.assertEqual(float(t.sum()),2.)
    def test_mixture_bounds_are_per_image(self):
        delta=np.array([[1,-1],[-1,1]],float);lo,hi=mixture_series(delta)
        self.assertEqual(lo.mean(),-1);self.assertEqual(hi.mean(),1)
        self.assertEqual(delta.mean(0).min(),0)
    def test_paired_score_has_normalization_derivative(self):
        h=torch.tensor([[[1.,3.]],[[2.,5.]]]);other=torch.tensor([2.,-1.])
        cam,_,diag=W.retrieval_gradcam_audited(h,lambda x:x.mean((1,2)),other)
        x=h.clone().requires_grad_();z=x.mean((1,2));s=(z*other).sum()/(z.norm()*other.norm()+1e-12)
        g=torch.autograd.grad(s,x)[0]
        self.assertTrue(torch.allclose(cam,(g.mean((1,2))[:,None,None]*h).sum(0)))
    def test_bootstrap_draws_shared_and_clustered(self):
        b=SharedBootstrap(['a','a','b'],['x','x','y']);x=np.array([0.,1.,1.])
        p,r=b.point_boot(x);p2,r2=b.point_boot(x.copy());self.assertTrue(np.array_equal(r,r2));self.assertEqual(p,2/3)
    def test_budget_cap_and_concurrency(self):
        import resources
        with tempfile.TemporaryDirectory() as t:
            old=resources.LEDGER;resources.LEDGER=Path(t)/'ledger.json'
            try:
                d={'allocation':'BGER_OPENLOGO_ONE_SHOT_20261022','cap_gpu_seconds':86400,'charged_gpu_seconds':86000,'pending_reserved_gpu_seconds':0,'jobs':[]}
                save(resources.LEDGER,d)
                with self.assertRaises(AssertionError):resources.reserve('too_much',401)
                resources.reserve('fits',400)
                with self.assertRaises(AssertionError):resources.reserve('overlap',1)
            finally:resources.LEDGER=old
if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
