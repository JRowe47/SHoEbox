"""Native recurrence derivatives, temporal connectivity and model equivalence."""
import copy
import os
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from experiments.hat_minkron.kron import MinKron,compact_chunk
from . import CorticalModel
from .fused_kron import fused_final,native_status
from .runtime import minkron_last
from .test_wide import small_wide

torch.set_num_threads(1)


class FusedKronChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status=native_status()
        if not status['available']:
            if os.environ.get('SHOEBOX_REQUIRE_FUSED')=='1':raise RuntimeError(str(status))
            raise unittest.SkipTest(str(status))

    def inputs(self,length=5,dtype=torch.float64):
        k=F.normalize(torch.randn(2,length,2,3,dtype=dtype),dim=-1)
        v=F.normalize(torch.randn(2,length,2,5,dtype=dtype),dim=-1)
        g=-torch.rand_like(k)*.2
        beta=torch.rand(2,length,2,dtype=dtype)
        write=torch.rand_like(beta)
        state=torch.randn(2,2,3,5,dtype=dtype)
        return tuple(x.requires_grad_() for x in (k,v,g,beta,write,state))

    def reference(self,args):
        k,v,g,beta,write,state=args
        # Original independent, eager step equation is the derivative oracle.
        for t in range(k.shape[1]):
            _,state=compact_chunk(k[:,t:t+1],k[:,t:t+1],v[:,t:t+1],g[:,t:t+1],
                                  beta[:,t:t+1],write[:,t:t+1],state)
        return state

    def test_scan_values_derivatives_and_clamped_directions(self):
        torch.manual_seed(811)
        for length,scale in [(1,1.),(5,1.),(33,1.),(3,0.),(3,1e-10)]:
            args=self.inputs(length)
            with torch.no_grad():args[1].mul_(scale)
            other=tuple(x.detach().clone().requires_grad_() for x in args)
            a=self.reference(args);b=fused_final(*other)
            torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-11)
            upstream=torch.randn_like(a)
            ga=torch.autograd.grad(a,args,upstream)
            gb=torch.autograd.grad(b,other,upstream)
            for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-9,rtol=2e-10)

    def test_numerical_gradcheck(self):
        torch.manual_seed(812)
        args=tuple(x[:1,:3,:1].detach().requires_grad_() for x in self.inputs(3)[:-1])
        state=torch.randn(1,1,3,5,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(fused_final,(*args,state),eps=1e-6,atol=1e-7,rtol=1e-5,fast_mode=True))

    def test_full_minkron_output_and_parameter_gradients(self):
        torch.manual_seed(813)
        for length in (1,9,65):
            reference=MinKron(12,2,3,5,chunk_size=8,safe_zero=True).double()
            native=copy.deepcopy(reference)
            x=torch.randn(2,length,12,dtype=torch.float64,requires_grad=True)
            x2=x.detach().clone().requires_grad_()
            state=torch.randn(2,2,3,5,dtype=torch.float64,requires_grad=True)
            state2=state.detach().clone().requires_grad_()
            all_reads,a=reference(x,state)
            last,b=minkron_last(native,x2,state2,fused=True)
            torch.testing.assert_close(last,all_reads[:,-1:],atol=1e-11,rtol=1e-9)
            torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
            do,ds=torch.randn_like(last),torch.randn_like(a)
            ga=torch.autograd.grad((all_reads[:,-1:],a),(*reference.parameters(),x,state),(do,ds))
            gb=torch.autograd.grad((last,b),(*native.parameters(),x2,state2),(do,ds))
            for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=2e-9,rtol=2e-8)

    def test_multiple_calls_preserve_history_and_compact_state(self):
        torch.manual_seed(814)
        args=self.inputs(17)
        other=tuple(x.detach().clone().requires_grad_() for x in args)
        full=fused_final(*args)
        prefix=fused_final(*(x[:,:7] for x in other[:-1]),other[-1])
        split=fused_final(*(x[:,7:] for x in other[:-1]),prefix)
        torch.testing.assert_close(full,split,atol=0,rtol=0)
        grad=torch.randn_like(full)
        ga=torch.autograd.grad(full,args,grad)
        gb=torch.autograd.grad(split,other,grad)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=0,rtol=0)
        self.assertEqual(split.detach().untyped_storage().nbytes(),split.numel()*split.element_size())
        with torch.no_grad():
            no_tape=fused_final(*args)
        torch.testing.assert_close(no_tape,full,atol=0,rtol=0)

    def test_heterogeneous_cortical_model_and_fallback(self):
        torch.manual_seed(815)
        reference=CorticalModel(small_wide()).double().set_execution_backend('optimized')
        fused=copy.deepcopy(reference).set_execution_backend('fused')
        x=torch.tensor([[65,66,67,68],[71,72,73,74]])
        a=reference(x,targets=x.roll(-1,1));b=fused(x,targets=x.roll(-1,1))
        torch.testing.assert_close(a['logits'],b['logits'],atol=2e-9,rtol=2e-8)
        torch.testing.assert_close(a['loss'],b['loss'],atol=2e-10,rtol=2e-9)
        a['loss'].backward();b['loss'].backward()
        for (name,p),(_,q) in zip(reference.named_parameters(),fused.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-6,msg=name)
        with patch.dict(os.environ,{'SHOEBOX_FUSED':'0'}),torch.no_grad():
            c=fused(x)
        torch.testing.assert_close(a['logits'],c['logits'],atol=1e-12,rtol=1e-11)

    def test_invalid_native_shapes_are_rejected(self):
        args=self.inputs(2)
        with self.assertRaisesRegex(RuntimeError,'dimensions'):
            fused_final(*args[:-1],args[-1][:,:,:2])


if __name__=='__main__':unittest.main()
