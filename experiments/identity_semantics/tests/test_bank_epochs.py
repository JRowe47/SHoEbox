"""Action 6: declared schedules, partial immutable jobs and sparse versions."""
from pathlib import Path
import sys,tempfile,unittest,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'src')]
from runtime import fresh_model,Model,core
from memory_inspection import capture_memory,compare_memory,read_memory_index

class BankEpochTests(unittest.TestCase):
    def make(self,epoch=4,lag=1,workers=1,storage='sparse',**kw):
        m=fresh_model(seed=7,profile='epochs',pipeline_workers=workers,
                      bank_epochs=dict(epoch_bytes=epoch,lag_epochs=lag,storage=storage),**kw)
        self.addCleanup(m.close);return m
    def data(self,n=259):
        raw=b'The river crosses the valley. A lantern shines above the stone bridge. '
        return np.frombuffer((raw*(n//len(raw)+1))[:n],np.uint8)
    def train(self,m,x,lo=0,hi=None):
        hi=len(x)-1 if hi is None else hi
        return m.run(x[lo:hi],x[lo+1:hi+1],mode='periodic',lr=.04,lp=.04/3,logits=True)[1]
    def test_all_schedules_worker_and_irregular_chunk_parity(self):
        x=self.data()
        for epoch in (1,4,8,16):
            for lag in (0,1):
                with self.subTest(epoch=epoch,lag=lag):
                    a=self.make(epoch,lag);b=self.make(epoch,lag,2)
                    expected=self.train(a,x);parts=[]
                    for lo,hi in ((0,1),(1,3),(3,31),(31,32),(32,137),(137,258)):
                        parts.append(self.train(b,x,lo,hi));b.inspect_bank_epochs()
                    np.testing.assert_array_equal(expected,np.concatenate(parts))
                    self.assertTrue(compare_memory(capture_memory(a),capture_memory(b))['equal'])
                    a.close();b.close()
    def test_partial_epoch_checkpoint_restart_worker_switch(self):
        x=self.data();a=self.make(8,1);b=self.make(8,1,2)
        expected=self.train(a,x);parts=[]
        with tempfile.TemporaryDirectory() as d:
            for lo,hi in ((0,1),(1,7),(7,8),(8,25),(25,193),(193,258)):
                parts.append(self.train(b,x,lo,hi));before=b.fingerprint()
                path=Path(d)/'epoch.npz';b.save(path);b.close();b=Model.load(path);self.addCleanup(b.close)
                self.assertEqual(before,b.fingerprint());b.configure_pipeline_workers(1 if hi%2 else 2)
        np.testing.assert_array_equal(expected,np.concatenate(parts));self.assertEqual(a.fingerprint(),b.fingerprint())
    def test_sparse_dense_and_unfused_are_arithmetic_identical(self):
        x=self.data(516);models=[self.make(8,1),self.make(8,1,2,'dense'),self.make(8,1)]
        models[-1].configure_pipeline_workers(1,fused=False)
        expected=self.train(models[0],x)
        for m in models[1:]:
            np.testing.assert_array_equal(expected,self.train(m,x));np.testing.assert_array_equal(models[0].flat,m.flat)
        sparse,dense=[m.inspect_pipeline() for m in models[:2]]
        self.assertLess(sparse['pending_weight_tiles'],dense['pending_weight_tiles'])
        self.assertEqual(dense['pending_weight_tiles'],16*core.E)
        for m in models:m.finish_pipeline()
        for m in models[1:]:np.testing.assert_array_equal(models[0].flat,m.flat)
    def test_epoch_one_lag_one_preserves_historical_pipeline(self):
        x=self.data();a=self.make(1,1);b=fresh_model(seed=7,profile='pipeline');self.addCleanup(b.close)
        np.testing.assert_array_equal(self.train(a,x),self.train(b,x))
        left,right=capture_memory(a),capture_memory(b)
        for name in left.arrays:np.testing.assert_array_equal(left.arrays[name],right.arrays[name])
        for bank in left.pipeline['banks']:np.testing.assert_array_equal(left.pipeline['banks'][bank],right.pipeline['banks'][bank])
    def test_job_versions_and_visibility_follow_epoch_lag(self):
        x=self.data()
        for lag in (0,1):
            m=self.make(4,lag,2);base=m.flat[2*core.E*4096:18*core.E*4096].copy()
            self.train(m,x,0,3)
            np.testing.assert_array_equal(base,m.flat[2*core.E*4096:18*core.E*4096])
            self.assertEqual(m.inspect_bank_epochs()['position'],3)
            self.train(m,x,3,8);p=m.inspect_pipeline();e=m.inspect_bank_epochs()
            self.assertEqual(p['sequence'],2);self.assertEqual(p['committed_version'],2-lag)
            if lag:
                self.assertEqual((p['read_version'],p['base_version']),(0,1))
                self.assertEqual((e['pending_first_step'],e['pending_last_step']),(4,8))
                self.assertGreater(e['pending_objectives'],0)
            m.finish_pipeline();self.assertEqual(m.pipeline_metrics()['committed_version'],2)
    def test_eval_targets_do_not_change_banks_or_partial_jobs(self):
        a=self.make(16,1,2);x=self.data(99);self.train(a,x)
        b=a.snapshot_copy();self.addCleanup(b.close);flat=a.flat.copy();pending=a.pipeline_state();fitted=a.event_state(True)
        _,v=a.run(x[:-1],x[1:],mode='eval',logits=True)
        _,w=b.run(x[:-1],(x[1:]+71).astype(np.uint8),mode='eval',logits=True)
        np.testing.assert_array_equal(v,w);np.testing.assert_array_equal(flat,a.flat)
        np.testing.assert_array_equal(pending,a.pipeline_state());np.testing.assert_array_equal(fitted,a.event_state(True))
    def test_document_reset_and_explicit_partial_drains(self):
        x=self.data(134);a=self.make(16,1);b=self.make(16,1,2)
        for m in (a,b):
            self.train(m,x);m.finish_document(mode='periodic',lr=.04,lp=.04/3,reset=False)
            self.assertEqual(m.inspect_bank_epochs()['position'],0);self.assertEqual(m.pipeline_metrics()['pending'],0)
        self.assertEqual(a.fingerprint(),b.fingerprint())
        for m in (a,b):m.reset_context();self.train(m,x);m.finish_pipeline()
        self.assertEqual(a.fingerprint(),b.fingerprint())
        for row in read_memory_index(a).values():
            self.assertEqual(row['invalid_retained_spans'],0);self.assertEqual(row['missing_content_records'],0)
    def test_ancestry_prediction_frames_and_two_heads_continue(self):
        x=self.data(194)
        for kw in ({'event':dict(learning='ancestry')},{'event':dict(context_heads=2)},
                   {'prediction':dict(compact=True,slow_stride=8),'frames':'lightweight-frames-v1'}):
            a=self.make(4,1,**kw);b=self.make(4,1,2,**kw)
            np.testing.assert_array_equal(self.train(a,x),self.train(b,x));self.assertEqual(a.fingerprint(),b.fingerprint())
            c=a.snapshot_copy();self.addCleanup(c.close)
            np.testing.assert_array_equal(self.train(a,x),self.train(c,x));self.assertEqual(a.fingerprint(),c.fingerprint())
    def test_pending_state_rejections_are_atomic(self):
        m=self.make(8,1,2);self.train(m,self.data(94));state=m.pipeline_state();before=m.fingerprint()
        cases=[state[:-1],state[:160]]
        for i,value in ((0,2),(1,999),(2,16),(3,0),(4,1),(5,8),(7,0),(8,2**63),(9,2**63),(10,2**63),(12,4),(15,4),(16,1)):
            bad=state.copy();bad[i*8:(i+1)*8]=np.array([value],np.uint64).view(np.uint8);cases.append(bad)
        h=state[:160].view(np.uint64);start=160+int(h[7])+24
        self.assertGreater(h[8],0)
        bad=state.copy();bad[start:start+4]=np.array([np.nan],np.float32).view(np.uint8);cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):m.import_pipeline(bad)
            self.assertEqual(before,m.fingerprint())
        m.import_pipeline(state);self.assertEqual(before,m.fingerprint())
    def test_checkpoint_policy_is_required_and_validated(self):
        m=self.make();self.train(m,self.data(94))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'epoch.npz';m.save(path)
            with np.load(path) as z:original={k:z[k].copy() for k in z.files}
            self.assertEqual(original['version'][0],46)
            for key in ('bank_epochs_json','bank_pipeline_state','prediction_state','frame_policy'):
                arrays=original.copy();arrays.pop(key);np.savez(path,**arrays)
                with self.assertRaises(ValueError):Model.load(path)
            arrays=original.copy();arrays['bank_epochs_json']=np.array(json.dumps(dict(epoch_bytes=16)))
            np.savez(path,**arrays)
            with self.assertRaises(ValueError):Model.load(path)
    def test_invalid_combinations_and_live_reconfiguration(self):
        for policy in (dict(epoch_bytes=2),dict(lag_epochs=2),dict(epoch_bytes=True),dict(storage='lowrank'),dict(unknown=1)):
            with self.assertRaises(ValueError):fresh_model(profile='epochs',bank_epochs=policy)
        with self.assertRaises(ValueError):fresh_model(profile='epochs',event=dict(context_heads=0))
        m=self.make();self.train(m,self.data(18))
        with self.assertRaises(ValueError):m.configure_bank_epochs(dict(epoch_bytes=8))
        with self.assertRaises(ValueError):m.configure_reverse(dict(epoch_bytes=4))
    def test_long_wrap_collection_pending_restart_and_task_coverage(self):
        x=self.data(8194);a=self.make(8,1,prediction=dict(compact=True,slow_stride=8))
        before=a.flat[:18*core.E*4096].copy().reshape(18,-1)
        frozen=np.zeros(18,np.int32);frozen[3]=1;a.configure_training_policy(frozen)
        self.train(a,x,0,4099);b=a.snapshot_copy();self.addCleanup(b.close);b.configure_pipeline_workers(2)
        expected=self.train(a,x,4099);got=[]
        for lo,hi in ((4099,4100),(4100,6001),(6001,8193)):got.append(self.train(b,x,lo,hi))
        np.testing.assert_array_equal(expected,np.concatenate(got));self.assertEqual(a.fingerprint(),b.fingerprint())
        a.finish_pipeline();after=a.flat[:18*core.E*4096].reshape(18,-1)
        np.testing.assert_array_equal(before[3],after[3])
        for r in range(2,18):
            if r!=3:self.assertTrue(np.any(before[r]!=after[r]),core.NAMES[r])
        for row in read_memory_index(a).values():
            self.assertEqual(row['invalid_retained_spans'],0);self.assertEqual(row['missing_content_records'],0)
    def test_reset_without_document_drain_commits_final_replay(self):
        x=self.data(240)
        for profile in ('pipeline','epochs'):
            for reset in ('reset','reset_context'):
                a=fresh_model(seed=7,profile=profile);self.addCleanup(a.close);self.train(a,x)
                getattr(a,reset)();inventory=a.inspect_bank_epochs()
                self.assertEqual(inventory['building_scopes'],0)
                self.assertEqual(a.pipeline_metrics()['pending'],0)
                b=a.snapshot_copy();self.addCleanup(b.close);self.assertEqual(a.fingerprint(),b.fingerprint())
                np.testing.assert_array_equal(self.train(a,x),self.train(b,x));a.close();b.close()

if __name__=='__main__':unittest.main()
