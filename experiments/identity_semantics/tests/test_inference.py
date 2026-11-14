"""Action 7: frozen values, episodic policy and explicit eligibility transitions."""
from pathlib import Path
import ctypes as ct
import json
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core

class InferenceTests(unittest.TestCase):
    def make(self,**kw):
        m=fresh_model(seed=7,**kw);self.addCleanup(m.close);return m
    def own(self,m):self.addCleanup(m.close);return m
    def data(self,n=1025):
        raw=b'The river crosses the valley. A lantern shines above the stone bridge. '
        return np.frombuffer((raw*(n//len(raw)+1))[:n],np.uint8)
    def train(self,m,n=259):
        x=self.data(n+1);m.run(x[:-1],x[1:],mode='periodic',lr=.04,lp=.04/3)
    def eval(self,m,x):return m.run(x[:-1],x[1:],logits=True)[1]
    def no_derivatives(self,m):
        a=m.inspect_inference()
        self.assertTrue(a['forward_only'])
        for k in ('derivative_slab_bytes','derivative_vector_bytes','neural_archives','context_archives',
                  'neural_records','context_records','temporal_archives','reverse_roots','pending_bank_job',
                  'pending_bank_scopes','slow_y_samples','recycled_archive_bytes'):
            self.assertEqual(a[k],0,k)
    def same_forward(self,a,b):
        np.testing.assert_array_equal(a.publications(),b.publications())
        np.testing.assert_array_equal(a.paired_memory_state(),b.paired_memory_state())
        np.testing.assert_array_equal(a.pattern_events(),b.pattern_events())
        aa,bb=a.content_events(),b.content_events()
        for r in aa:
            for k in aa[r]:np.testing.assert_array_equal(aa[r][k],bb[r][k],err_msg=f'{r}.{k}')
    def test_profiles_and_irregular_continuation(self):
        profiles=[{},dict(profile='event'),dict(profile='event',event=dict(context_heads=2)),
            dict(profile='epochs',pipeline_workers=2,bank_epochs=dict(epoch_bytes=16,lag_epochs=1),
                 prediction=dict(compact=True,slow_stride=8)),
            dict(profile='event',frames='lightweight-frames-v1',prediction=dict(compact=True,slow_stride=4)),
            dict(profile='event',event=dict(learning='ancestry'))]
        for kw in profiles:
            with self.subTest(kw=kw):
                a=self.make(**kw);self.train(a);b=self.own(a.snapshot_copy());a.finish_training();b.enter_inference()
                x=self.data(1026);expected=self.eval(a,x);parts=[]
                for lo,hi in ((0,1),(1,17),(17,64),(64,81),(81,257),(257,1025)):
                    parts.append(b.run(x[lo:hi],x[lo+1:hi+1],logits=True)[1])
                np.testing.assert_array_equal(expected,np.concatenate(parts));self.same_forward(a,b);self.no_derivatives(b)
                for m in (a,b):m.finish_document()
                self.same_forward(a,b)
                a.close();b.close()
    def test_all_four_memory_policies(self):
        source=self.make(profile='event',prediction=dict(compact=True,slow_stride=8));self.train(source,1025)
        before=source.fingerprint()
        for policy in ('cold_adaptive','cold_frozen','warm_adaptive','warm_frozen'):
            with self.subTest(policy=policy):
                a=self.own(source.evaluation_copy(memory=policy));b=self.own(source.inference_copy(memory=policy))
                prior=b.evaluation_memory_info();np.testing.assert_array_equal(self.eval(a,self.data()),self.eval(b,self.data()))
                self.same_forward(a,b);after=b.evaluation_memory_info()
                if policy.endswith('frozen'):self.assertEqual(prior['store_digest'],after['store_digest'])
                else:self.assertNotEqual(prior['store_digest'],after['store_digest'])
                self.no_derivatives(b);a.close();b.close()
        self.assertEqual(before,source.fingerprint())
    def test_autoregressive_generation_and_save_load(self):
        for policy in ('warm_adaptive','warm_frozen'):
            source=self.make(profile='epochs',bank_epochs=dict(epoch_bytes=4,lag_epochs=1),prediction=dict(compact=True,slow_stride=8))
            self.train(source,1025);a=self.own(source.evaluation_copy(memory=policy));b=self.own(source.inference_copy(memory=policy))
            self.eval(a,self.data(33));self.eval(b,self.data(33));ra=np.random.default_rng(81);rb=np.random.default_rng(81)
            with tempfile.TemporaryDirectory() as d:
                for i in range(129):
                    aa=a.draw(ra,temperature=.8);bb=b.draw(rb,temperature=.8);self.assertEqual(aa,bb)
                    np.testing.assert_array_equal(a.run([aa],logits=True)[1],b.run([bb],logits=True)[1])
                    if i in (0,16,63):
                        path=Path(d)/'inference.npz';before=b.fingerprint();b.save(path);b.close()
                        b=self.own(Model.load(path));self.assertTrue(b.forward_only);self.assertEqual(before,b.fingerprint())
            self.same_forward(a,b);self.no_derivatives(b);source.close();a.close();b.close()
    def test_long_stream_drops_archives_and_retains_forward(self):
        a=self.make(profile='event',frames='lightweight-frames-v1');self.train(a,1025)
        b=self.own(a.snapshot_copy());a.finish_training();b.enter_inference()
        x=self.data(16386)
        for lo,hi in ((0,8193),(8193,16385)):
            np.testing.assert_array_equal(a.run(x[lo:hi],x[lo+1:hi+1],logits=True)[1],b.run(x[lo:hi],x[lo+1:hi+1],logits=True)[1])
            self.same_forward(a,b);self.no_derivatives(b)
        self.assertGreater(a.inspect_inference()['neural_archives'],0)
    def test_train_rejected_until_explicit_reset(self):
        m=self.make(profile='epochs',pipeline_workers=2,bank_epochs=dict(epoch_bytes=4,lag_epochs=1),prediction=dict(compact=True,slow_stride=8))
        self.train(m);m.enter_inference();self.eval(m,self.data(193));before=m.fingerprint()
        for mode in core.MODES:
            if mode=='eval':continue
            with self.assertRaisesRegex(ValueError,'Forward-only'):m.run([65,66],mode=mode)
            with self.assertRaisesRegex(ValueError,'Forward-only'):m.finish_document(mode=mode)
        with self.assertRaisesRegex(ValueError,'reset_context'):m.resume_training()
        self.assertEqual(before,m.fingerprint())
        # The native entry points reject training too, even without Python guards.
        x=np.array([65],np.uint8);out=np.zeros(256,np.float32)
        self.assertTrue(np.isnan(m.lib.run(m.handle,x.ctypes.data_as(ct.POINTER(ct.c_ubyte)),x.ctypes.data_as(ct.POINTER(ct.c_ubyte)),1,6,.04,.01,.002,1,out.ctypes.data_as(ct.POINTER(ct.c_float)))))
        self.assertEqual(before,m.fingerprint())
        n=self.own(m.snapshot_copy());episodes=m.evaluation_memory_info()['store_digest']
        for a in (m,n):a.resume_training(reset_context=True);self.assertFalse(a.forward_only)
        self.assertEqual(episodes,m.evaluation_memory_info()['store_digest'])
        np.testing.assert_array_equal(m.run(self.data(129),mode='periodic',logits=True)[1],n.run(self.data(129),mode='periodic',logits=True)[1])
        self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_targets_do_not_change_inference_and_parameters_are_frozen(self):
        m=self.make(profile='event',inference=True);self.eval(m,self.data(65));n=self.own(m.snapshot_copy())
        flat=m.flat.copy();extra=m.extra_flat.copy();fitted=[m.composition_state(True),m.pattern_state(True),m.event_state(True),m.prediction_state(True)]
        x=self.data(513);np.testing.assert_array_equal(m.run(x,np.zeros_like(x),logits=True)[1],n.run(x,np.full_like(x,255),logits=True)[1])
        self.same_forward(m,n);np.testing.assert_array_equal(flat,m.flat);np.testing.assert_array_equal(extra,m.extra_flat)
        for a,b in zip(fitted,[m.composition_state(True),m.pattern_state(True),m.event_state(True),m.prediction_state(True)]):np.testing.assert_array_equal(a,b)
    def test_reset_checkpoint_and_resume_policy_is_mandatory(self):
        m=self.make(profile='inference');self.eval(m,self.data(129));m.reset_context();self.no_derivatives(m)
        m.reset();self.assertTrue(m.forward_only);self.no_derivatives(m);self.eval(m,self.data(65))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'infer.npz';m.save(path)
            with np.load(path) as z:original={k:z[k].copy() for k in z.files}
            for key in ('inference_policy','inference_state','paired_memory_state','pattern_fitted'):
                arrays=original.copy();arrays.pop(key);np.savez(path,**arrays)
                with self.assertRaises(ValueError):Model.load(path)
            arrays=original.copy();arrays['inference_policy']=np.array(json.dumps(dict(runtime='training')));np.savez(path,**arrays)
            with self.assertRaises(ValueError):Model.load(path)
        state=m.inference_state();before=m.fingerprint()
        for bad in (state[:-1],np.zeros_like(state)):
            with self.assertRaises(ValueError):m.import_inference(bad)
            self.assertEqual(before,m.fingerprint())
    def test_continue_copy_does_not_commit_original_jobs(self):
        m=self.make(profile='epochs',pipeline_workers=2,bank_epochs=dict(epoch_bytes=8,lag_epochs=1));self.train(m)
        before=m.fingerprint();n=self.own(m.inference_copy(memory='continue'));self.assertEqual(before,m.fingerprint())
        a=self.own(m.snapshot_copy());a.finish_training();np.testing.assert_array_equal(self.eval(a,self.data()),self.eval(n,self.data()))
        caps=n.capabilities();self.assertEqual(caps['checkpoint_version'],47)
        self.assertFalse(caps['features']['lace_credit']);self.assertFalse(caps['bank_pipeline']['enabled'])

if __name__=='__main__':unittest.main(verbosity=2)
