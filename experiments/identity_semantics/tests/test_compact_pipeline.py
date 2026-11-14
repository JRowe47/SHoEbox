"""Production correctness using detached memory, without inline telemetry."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core
from memory_inspection import capture_memory,compare_memory,read_memory_index,decode_pending,diagnostic_ranges

class CompactPipelineTests(unittest.TestCase):
    def make(self,workers=1,fused=True):
        m=fresh_model(seed=7,profile='pipeline');m.configure_pipeline_workers(workers,fused=fused)
        self.addCleanup(m.close);return m
    def data(self,n=513):
        text=b'The river runs through the valley. The king crossed the bridge. '
        return np.frombuffer((text*(n//len(text)+1))[:n],np.uint8)
    def train(self,m,x):return m.run(x[:-1],x[1:],mode='periodic',lr=.04,lp=.04/3,logits=True)[1]
    def test_serial_threaded_unfused_operational_memory_is_exact(self):
        x=self.data();models=[self.make(1,False),self.make(),self.make(2)]
        values=[self.train(m,x) for m in models];snapshots=[capture_memory(m) for m in models]
        for v,s in zip(values[1:],snapshots[1:]):
            np.testing.assert_array_equal(values[0],v);self.assertTrue(compare_memory(snapshots[0],s)['equal'])
    def test_inspection_does_not_commit_or_mutate_training(self):
        m=self.make(2);x=self.data(194);self.train(m,x)
        fingerprint=m.fingerprint();pending=m.pipeline_state();snap=capture_memory(m)
        for _ in range(3):
            info=m.inspect_pipeline();read_memory_index(m);capture_memory(m)
            self.assertEqual(m.fingerprint(),fingerprint);np.testing.assert_array_equal(m.pipeline_state(),pending)
        self.assertTrue(compare_memory(snap,capture_memory(m))['equal'])
        self.assertEqual(info['sequence'],info['committed_version']+1)
    def test_compact_router_tiles_and_shared_inputs(self):
        m=self.make();x=self.data(1025);infos=[]
        for i in range(0,1024,64):
            self.train(m,x[i:i+65]);infos.append(m.inspect_pipeline())
        info=max(infos,key=lambda v:v['input_bytes_avoided'])
        self.assertGreater(info['pending_banks'],info['pending_weight_tiles'])
        self.assertGreater(info['pending_factors'],info['unique_inputs'])
        self.assertEqual(info['router_weight_bytes_avoided'],(info['pending_banks']-info['pending_weight_tiles'])*16384)
        self.assertEqual(info['pending_state_bytes'],96+info['pending_banks']*528+info['pending_weight_tiles']*16384)
        self.assertGreater(info['reserved_bytes'],info['active_payload_bytes'])
        self.assertTrue(any(not b['has_weights'] and b['updates_router'] for b in info['banks']))
    def test_two_job_objects_reused_across_byte_calls(self):
        m=self.make(2);x=self.data(131);self.train(m,x[:97]);addresses=set()
        for i in range(96,130):
            self.train(m,x[i:i+2]);info=m.inspect_pipeline()
            addresses.update((info['building_buffer'],info['pending_buffer']))
        self.assertEqual(len(addresses),2)
    def test_inspected_byte_chunks_match_single_call(self):
        x=self.data(194);a=self.make();b=self.make(2);expected=self.train(a,x);got=[]
        for i in range(len(x)-1):
            got.append(self.train(b,x[i:i+2]));b.inspect_pipeline();read_memory_index(b)
        np.testing.assert_array_equal(expected,np.concatenate(got));self.assertTrue(compare_memory(capture_memory(a),capture_memory(b))['equal'])
    def test_compact_checkpoint_and_pending_worker_resume(self):
        a=self.make(2);x=self.data();self.train(a,x[:257])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'saved.npz';a.save(path);b=Model.load(path);self.addCleanup(b.close)
            b.configure_pipeline_workers(2);self.assertTrue(compare_memory(capture_memory(a),capture_memory(b))['equal'])
            np.testing.assert_array_equal(self.train(a,x[256:]),self.train(b,x[256:]))
            self.assertTrue(compare_memory(capture_memory(a),capture_memory(b))['equal'])
    def test_legacy_pending_extension_resumes_exactly(self):
        a=self.make();x=self.data();self.train(a,x[:257]);b=a.snapshot_copy();self.addCleanup(b.close)
        state=a.pipeline_state();decoded=decode_pending(state,a.flat,core.E);h=state[:96].view(np.uint64).copy();h[0]=1;h[10]=0
        parts=[h.view(np.uint8)]
        for bank,v in decoded['banks'].items():parts += [np.array([bank],np.uint64).view(np.uint8),v.view(np.uint8)]
        b.import_pipeline(np.concatenate(parts));self.assertTrue(compare_memory(capture_memory(a),capture_memory(b))['equal'])
        np.testing.assert_array_equal(self.train(a,x[256:]),self.train(b,x[256:]))
        self.assertTrue(compare_memory(capture_memory(a),capture_memory(b))['equal'])
    def test_invalid_compact_entries_are_atomic(self):
        m=self.make(2);self.train(m,self.data(194));state=m.pipeline_state();fingerprint=m.fingerprint()
        cases=[]
        for offset,value in [(0,3),(80,999999),(96,0),(104,4)]:
            bad=state.copy();bad[offset:offset+8]=np.array([value],np.uint64).view(np.uint8);cases.append(bad)
        bad=state.copy();bad[112:116]=np.array([np.nan],np.float32).view(np.uint8);cases.extend([bad,state[:-1]])
        for bad in cases:
            with self.assertRaises(ValueError):m.import_pipeline(bad)
            self.assertEqual(fingerprint,m.fingerprint())
    def test_detached_comparison_detects_changes(self):
        m=self.make();self.train(m,self.data(97));a=capture_memory(m);b=capture_memory(m)
        b.arrays['flat'][0]+=1;result=compare_memory(a,b)
        self.assertFalse(result['equal']);self.assertIn('flat',result['changes']);self.assertEqual(a.arrays['flat'][0],m.flat[0])
    def test_all_upper_banks_receive_task_learning_without_observers(self):
        m=self.make();before=m.flat[:18*core.E*4096].copy().reshape(18,-1);x=self.data()
        m.run(x[:-1],x[1:],mode='periodic',lr=.04,lp=0);m.finish_pipeline()
        after=m.flat[:18*core.E*4096].reshape(18,-1)
        self.assertTrue(np.all(np.any(before[2:]!=after[2:],axis=1)))
        for v in read_memory_index(m).values():
            self.assertEqual(v['invalid_retained_spans'],0);self.assertEqual(v['missing_content_records'],0)
    def test_timers_are_unavailable_and_release_is_explicit(self):
        m=self.make();self.train(m,self.data(65))
        self.assertIsNone(m.pipeline_metrics()['compute_ns']);self.assertIsNone(m.pipeline_metrics()['wait_ns'])
        self.assertIsInstance(m.capabilities()['inline_diagnostics'],bool)

if __name__=='__main__':unittest.main()
