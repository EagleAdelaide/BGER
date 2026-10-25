"""Synthetic tensor/metric contracts only: no trained models, images or weights."""
import sys,unittest,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'BGER_WWW27_reproducibility'
sys.path.insert(0,str(R/'openlogo/reference_code'))
import test_contracts as archived
import torch
import numpy as np

class AnalyticContracts(unittest.TestCase):
    def test_stabilized_cosine_identity_at_pooled_F(self):
        torch.manual_seed(4)
        f=torch.randn(5,7,dtype=torch.float64,requires_grad=True)
        other=torch.randn(5,dtype=torch.float64)
        z=f.mean(1);a=z.norm();b=other.norm();den=a*b+1e-12
        score=(z*other).sum()/den
        grad=torch.autograd.grad(score,f)[0]
        cam=(grad.mean(1)[:,None]*f).sum(0)
        decomposition=(f*other[:,None]).sum(0)/(7*den)
        query=(f*z[:,None]).sum(0)/(7*a*a)
        self.assertTrue(torch.allclose(cam,decomposition-score*(a*b/den)*query,atol=1e-12,rtol=0))
        self.assertAlmostEqual(float(decomposition.sum()),float(score),places=12)
        self.assertAlmostEqual(float(query.sum()),1.0,places=12)

    def test_osld_batch_balanced_loss_matches_paper(self):
        from train_web_arms import balanced_bce
        x=torch.tensor([[-1.,.2,.6],[.8,-.3,.1]],dtype=torch.float64)
        y=torch.tensor([[0.,1.,0.],[1.,1.,0.]],dtype=torch.float64)
        pos=y.sum().clamp_min(1);neg=(1-y).sum().clamp_min(1)
        weight=torch.where(y>.5,(pos+neg)/(2*pos),(pos+neg)/(2*neg))
        expected=-(weight*(y*torch.nn.functional.logsigmoid(x)+(1-y)*torch.nn.functional.logsigmoid(-x))).mean()
        self.assertTrue(torch.allclose(balanced_bce(x,y),expected,atol=1e-12,rtol=0))

    def test_cub_matched_loss_matches_paper(self):
        sys.path.insert(0,str(R/'cub_training'))
        name=R/'cub_training/scripts/cvpr2027/run_matched_cub.py'
        spec=importlib.util.spec_from_file_location('matched_cub_fixture',name)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        pred=torch.tensor([[[[.2,.4],[.6,.8]]],[[[.3,.4],[.7,.9]]]],dtype=torch.float64)
        target=torch.tensor([[[[0.,1.],[0.,0.]]],[[[1.,1.],[0.,0.]]]],dtype=torch.float64)
        fraction=target.mean((1,2,3)).clamp(.02,.98)
        per=-(target*torch.log(pred+1e-6)).mean((1,2,3))/(2*fraction)
        per-=((1-target)*torch.log(1-pred+1e-6)).mean((1,2,3))/(2*(1-fraction))
        self.assertTrue(torch.allclose(module.balanced_bce(pred,target),per.mean(),atol=1e-12,rtol=0))

    @unittest.skip('Historical GPU allocation/orchestration is disabled and outside CPU reproduction')
    def test_archived_gpu_resource_orchestration(self):
        pass

if __name__=='__main__':
    torch.set_num_threads(1)
    suite=unittest.TestSuite()
    for name in ['test_source_metric_equivalence_two_conventions','test_degenerate_maps_are_misses',
                 'test_actual_loss_target_transform','test_mixture_bounds_are_per_image',
                 'test_paired_score_has_normalization_derivative','test_bootstrap_draws_shared_and_clustered']:
        suite.addTest(archived.Contracts(name))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(AnalyticContracts))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
