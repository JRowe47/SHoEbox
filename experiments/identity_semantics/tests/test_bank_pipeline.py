"""Stage-6 serial schedule oracle, worker races, causal versions and lifecycle."""
from pathlib import Path
import ctypes as ct
import sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'));sys.path.insert(0,str(ROOT/'src'))
from runtime import fresh_model,Model,core
from reference_acceptance import collect,check_integrity

class BankPipelineTests(unittest.TestCase):
    def make(self,workers=1,fused=True,**event):
        m=fresh_model(seed=7,profile='pipeline',event=event)
        m.configure_pipeline_workers(workers,fused=fused);self.addCleanup(m.close);return m
    def data(self,n=513):
        raw=b'The river runs through the valley. The king crossed the bridge. '
        return np.frombuffer((raw*(n//len(raw)+1))[:n],np.uint8)
    def train(self,m,x,a=0,b=None):
        b=len(x)-1 if b is None else b
        return m.run(x[a:b],x[a+1:b+1],mode='periodic',lr=.04,lp=.04/3,logits=True)
    def test_serial_threaded_and_unfused_have_identical_full_state(self):
        x=self.data();models=[self.make(1,False),self.make(),self.make(2)]
        expected=None
        for m in models:
            _,actual=self.train(m,x)
            if expected is None:expected=actual;fingerprint=m.fingerprint()
            np.testing.assert_array_equal(expected,actual);self.assertEqual(fingerprint,m.fingerprint())
            check_integrity(collect(m))
        self.assertGreater(models[-1].inspect_pipeline()['pending_weight_tiles'],0)
        self.assertIsNone(models[-1].pipeline_metrics()['compute_ns'])
        self.assertEqual(models[-1].pipeline_metrics()['committed_version'],511)
    def test_single_byte_irregular_chunks_worker_switch_save_load(self):
        x=self.data(194);a=self.make();b=self.make(2)
        _,expected=self.train(a,x);parts=[]
        with tempfile.TemporaryDirectory() as tmp:
            for lo,hi in ((0,1),(1,2),(2,65),(65,66),(66,131),(131,193)):
                _,v=self.train(b,x,lo,hi);parts.append(v)
                self.assertEqual(b.pipeline_metrics()['pending'],1)
                path=Path(tmp)/'state.npz';b.save(path);before=b.fingerprint();b.close()
                b=Model.load(path);self.addCleanup(b.close);self.assertEqual(before,b.fingerprint())
                b.configure_pipeline_workers(2 if hi%2 else 1)
        np.testing.assert_array_equal(expected,np.concatenate(parts));self.assertEqual(a.fingerprint(),b.fingerprint())
    def test_snapshot_keeps_uncommitted_version(self):
        m=self.make(2);x=self.data(220);self.train(m,x,0,151)
        n=m.snapshot_copy();self.addCleanup(n.close);self.assertEqual(m.fingerprint(),n.fingerprint())
        _,a=self.train(m,x,151);_,b=self.train(n,x,151)
        np.testing.assert_array_equal(a,b);self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_eval_targets_cannot_change_logits_or_pending_job(self):
        m=self.make(2);x=self.data(192);self.train(m,x)
        a=m.snapshot_copy();b=m.snapshot_copy();self.addCleanup(a.close);self.addCleanup(b.close)
        flat=a.flat.copy();queue=a.pipeline_state();event=a.event_state(fitted=True)
        _,v=a.run(x[:-1],x[1:],mode='eval',logits=True)
        _,w=b.run(x[:-1],(x[1:]+31).astype(np.uint8),mode='eval',logits=True)
        np.testing.assert_array_equal(v,w);np.testing.assert_array_equal(flat,a.flat)
        np.testing.assert_array_equal(queue,a.pipeline_state());np.testing.assert_array_equal(event,a.event_state(fitted=True))
    def test_one_byte_delay_and_explicit_drain(self):
        m=self.make();x=self.data(66);self.train(m,x)
        before=m.flat.copy();state=m.pipeline_state();header=state[:96].view(np.uint64)
        self.assertEqual(tuple(header[3:9]),(64,65,1,63,64,65))
        m.finish_pipeline();self.assertEqual(m.pipeline_metrics()['pending'],0)
        self.assertGreater(float(np.max(abs(m.flat-before))),0)
        n=m.snapshot_copy();self.addCleanup(n.close);self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_document_and_reset_drain_once(self):
        x=self.data(196);a=self.make();b=self.make(2)
        for m in (a,b):
            self.train(m,x);m.finish_document(mode='periodic',lr=.04,lp=.04/3,reset=False)
            self.assertEqual(m.pipeline_metrics()['pending'],0)
        self.assertEqual(a.fingerprint(),b.fingerprint())
        a.reset_context();b.reset_context();self.assertEqual(a.fingerprint(),b.fingerprint())
        self.train(a,x);self.train(b,x);self.assertEqual(a.fingerprint(),b.fingerprint())
    def test_malformed_extension_is_rejected_atomically(self):
        m=self.make(2);self.train(m,self.data(130));state=m.pipeline_state();before=m.fingerprint()
        cases=[state[:-1],np.zeros(0,np.uint8)]
        for index,value in ((0,3),(1,0),(3,999),(9,999999),(10,1),(12,0)):
            bad=state.copy();bad[index*8:(index+1)*8]=np.array([value],np.uint64).view(np.uint8);cases.append(bad)
        bad=state.copy();bad[104:108]=np.array([np.nan],np.float32).view(np.uint8);cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):m.import_pipeline(bad)
            self.assertEqual(before,m.fingerprint())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.npz';m.save(path)
            with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
            arrays.pop('bank_pipeline_state');np.savez(path,**arrays)
            with self.assertRaises(ValueError):Model.load(path)
    def test_two_heads_zero_replay_and_frozen_bank_coverage(self):
        x=self.data(258)
        for policy in ({'context_heads':2},{'replay_budget':0}):
            a=self.make(1,**policy);b=self.make(2,**policy)
            frozen=np.zeros(18,np.int32);frozen[3]=1
            for m in (a,b):m.configure_training_policy(frozen,1.);self.train(m,x)
            self.assertEqual(a.fingerprint(),b.fingerprint())
    def test_reference_and_stage5_policies_stay_unchanged(self):
        x=self.data(100)
        for profile in (None,'event'):
            m=fresh_model(seed=7,profile=profile);self.addCleanup(m.close);self.train(m,x)
            n=m.snapshot_copy();self.addCleanup(n.close);self.assertFalse(n.bank_pipeline)
            self.assertEqual(m.fingerprint(),n.fingerprint())
    def test_configuration_rejects_incompatible_paths(self):
        m=fresh_model();self.addCleanup(m.close)
        with self.assertRaises(ValueError):m.configure_bank_pipeline(True)
        n=self.make()
        with self.assertRaises(ValueError):n.configure_event(dict(context_heads=0))
        with self.assertRaises(ValueError):n.configure_reverse(dict(epoch_bytes=4))
        for count in (0,3,True):
            with self.assertRaises(ValueError):n.configure_pipeline_workers(count)
        self.train(n,self.data(20))
        with self.assertRaises(ValueError):n.configure_bank_pipeline(False)
    def test_destroy_with_pending_worker_and_repeat_construction(self):
        x=self.data(33)
        for _ in range(4):
            m=self.make(2);self.train(m,x);m.close()
    def test_all_upper_regions_keep_task_credit(self):
        m=self.make(2);self.train(m,self.data())
        if m.inline_diagnostics_enabled():
            self.assertTrue(all(m.event_metrics()[name]['task_exact'] for name in core.NAMES[2:]))
        else:
            from memory_inspection import read_memory_index
            self.assertTrue(all(read_memory_index(m)[name]['event_ordinal'] for name in core.NAMES[2:]))
        self.assertTrue(all(v['missing']==0 for v in m.event_metrics().values()))
    def test_native_mode_switches_keep_all_captured_updates_in_checkpoint(self):
        x=self.data(97);a=self.make();b=self.make(2)
        for mode in ('periodic','readout','local','dfa','hybrid','readout'):
            expected=None
            for m in (a,b):
                sequence=m.pipeline_metrics()['sequence']
                # The Python comparator restricts modes to eval/dfa/periodic;
                # exercise the additional supported native entry modes here.
                logits=np.empty((96,256),np.float32);byteptr=ct.POINTER(ct.c_ubyte)
                m.lib.run(m.handle,x[:-1].ctypes.data_as(byteptr),x[1:].ctypes.data_as(byteptr),96,
                          core.MODES[mode],.04,.04/3,.002,1,logits.ctypes.data_as(ct.POINTER(ct.c_float)))
                self.assertEqual(m.pipeline_metrics()['sequence'],sequence+96)
                if expected is None:expected=logits
                else:np.testing.assert_array_equal(expected,logits)
            self.assertEqual(a.fingerprint(),b.fingerprint())
            n=b.snapshot_copy();self.addCleanup(n.close);self.assertEqual(b.fingerprint(),n.fingerprint())

if __name__=='__main__':unittest.main()
