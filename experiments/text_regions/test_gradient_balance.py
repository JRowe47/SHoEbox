"""Gradient isolation, scale control, and exact optimizer/stream continuation."""
from pathlib import Path
import tempfile
import unittest
import sys
import torch
from .test_multifact import fixture
from .gradient_balance import combine, norm, statistics, update
from .joint_stability import load, primary_conditions
from .joint_text import optimizer_for, selection
from .data import OrderedByteStreams
from .multifact import table_batches
from .multifact_data import make_batch
from .retained_composition import teacher_inputs

torch.set_num_threads(1)


class GradientBalance(unittest.TestCase):
    def test_group_cap_prevents_core_spike_from_scaling_decoder_or_producer(self):
        p = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
        a = [torch.tensor([10000.,0.]), torch.tensor([.3,.4]), None]
        b = [torch.tensor([0.,2.]), torch.tensor([0.,.2]), torch.tensor([.1,0.])]
        row = combine(p,a,b,dict(core=[0],readout=[1],producer=[2]),'group_clip')
        torch.testing.assert_close(p[0].grad,torch.tensor([1.,1.]))
        torch.testing.assert_close(p[1].grad,torch.tensor([.3,.6]))
        torch.testing.assert_close(p[2].grad,b[2])
        self.assertAlmostEqual(row['core']['text_scale'],.0001)
        self.assertEqual(row['readout']['text_scale'],1.)
        torch.testing.assert_close(a[0],torch.tensor([10000.,0.]))

    def test_sum_matches_mathematical_task_sum_and_keeps_absence(self):
        p = [torch.nn.Parameter(torch.zeros(2)) for _ in range(4)]
        a = [torch.tensor([1.,2.]),None,torch.tensor([4.,3.]),None]
        b = [torch.tensor([-3.,2.]),torch.tensor([2.,1.]),None,None]
        combine(p,a,b,dict(all=list(range(4))),'sum')
        torch.testing.assert_close(p[0].grad,a[0]+b[0],rtol=0,atol=0)
        torch.testing.assert_close(p[1].grad,b[1],rtol=0,atol=0)
        torch.testing.assert_close(p[2].grad,a[2],rtol=0,atol=0)
        self.assertIsNone(p[3].grad)

    def test_nonfinite_rejection_happens_before_any_gradient_mutation(self):
        for bad in (float('nan'),float('inf')):
            p = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
            for v in p:v.grad=torch.tensor([7.])
            with self.assertRaises(FloatingPointError):
                combine(p,[torch.ones(1),torch.tensor([bad])],[None,None],dict(a=[0],b=[1]))
            for v in p:self.assertEqual(float(v.grad),7.)

    def test_groups_must_cover_parameters_once_and_cap_is_valid(self):
        p = [torch.nn.Parameter(torch.zeros(1))]
        for groups in ({},dict(a=[0],b=[0]),dict(a=[1])):
            with self.assertRaises(ValueError):combine(p,[None],[None],groups)
        for cap in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):combine(p,[None],[None],dict(a=[0]),cap=cap)

    def test_zero_and_large_finite_gradients_have_finite_statistics(self):
        huge=torch.tensor([1e30,1e30])
        self.assertGreater(norm([huge]),1e30)
        row=statistics([huge],[-huge])
        self.assertAlmostEqual(row['cosine'],-1.)
        self.assertIsNone(statistics([None],[torch.zeros(2)])['cosine'])

    def test_guard_covers_all_withheld_conditions(self):
        conditions=primary_conditions()
        self.assertEqual(len(conditions),9)
        self.assertEqual(sum(c[2]=='withheld' for c in conditions),4)
        rows=[dict(fields=[dict(accuracy=.95) for _ in range(3)]) for _ in conditions]
        self.assertEqual(selection(dict(nats=3),rows)[0],1)
        rows[-1]['fields'][-1]['accuracy']=.89
        self.assertEqual(selection(dict(nats=3),rows)[0],0)

    def test_fresh_text_evaluation_never_reuses_previous_scored_positions(self):
        scripts=str(Path(__file__).resolve().parents[2]/'scripts')
        sys.path.insert(0,scripts)
        try:
            from evaluate_joint_stability import fresh_text
            raw=bytes(i%256 for i in range(55770))
            text,layout=fresh_text(raw)
            old={len(raw)*i//8+j for i in range(8) for j in range(1,513)}
            new={start+j for start in layout['test_relative_starts'] for j in range(1,513)}
            self.assertFalse(old & new)
            self.assertEqual(len(new),4096)
            self.assertGreaterEqual(min(new),len(raw)//2)
            for i,start in enumerate(layout['test_relative_starts']):
                self.assertEqual(text[1024*i:1024*(i+1)],raw[start:start+1024])
            with self.assertRaises(ValueError):fresh_text(b'ab'*100)
        finally:
            sys.path.remove(scripts)

    def test_clipped_update_resume_preserves_weights_moments_state_and_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            model,_=fixture(folder)
            opt=optimizer_for(model)
            streams=OrderedByteStreams.from_bytes(b'Romeo says hello. '*12,2)
            ref=teacher_inputs(model,torch.arange(65,69)).detach()
            b=make_batch('train',0,2,912,0)
            state,_=update(model,opt,streams.next_chunk(2),None,b,table_batches(b,'train',0,912),ref)
            saved=dict(schema='shoebox-joint-stability-v1',config=model.cfg.to_dict(),weights=model.state_dict(),
                optimizer=opt.state_dict(),protocol=dict(core_lr=.0003,readout_lr=.002,producer_lr=.0001),
                graph_state=state.to_dict(),streams=streams.state_dict(),torch_rng=torch.get_rng_state())
            path=Path(folder)/'checkpoint.pt';torch.save(saved,path)
            streams2=OrderedByteStreams.from_bytes(b'Romeo says hello. '*12,2)
            other,opt2,state2,_=load(path,streams2)
            b=make_batch('train',2,2,912,0,'shared_left');tb=table_batches(b,'train',2,912)
            state,a=update(model,opt,streams.next_chunk(2),state,b,tb,ref)
            state2,z=update(other,opt2,streams2.next_chunk(2),state2,b,tb,ref)
            self.assertEqual(a,z);self.assertEqual(streams.state_dict(),streams2.state_dict())
            for x,y in zip(model.state_dict().values(),other.state_dict().values()):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            for x,y in zip(state.tensors(),state2.tensors()):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            for x,y in zip(opt.state.values(),opt2.state.values()):
                for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
            self.assertTrue(all(v.grad_fn is None for v in state.tensors()))


if __name__=='__main__':unittest.main()
