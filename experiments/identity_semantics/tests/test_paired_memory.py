"""Joint episodic retrieval, asynchronous causality and checkpoint identity."""
import ctypes as ct
from pathlib import Path
import os,subprocess,sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core,fresh_model

class PairedMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'results'/f'paired_fixture_e{core.E}_{os.getpid()}.so';path.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-O2','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}',
                        '-shared','-fPIC',str(ROOT/'tests/paired_memory_fixture.cpp'),'-o',str(path)],check=True)
        cls.fx=ct.CDLL(str(path));p,i=ct.c_void_p,ct.c_int
        for name,args in dict(paired_fixture_step=[p,i,i,i,i],paired_fixture_commit=[p],
                             paired_fixture_returns=[p,p,p],paired_fixture_entries=[p,p,p],
                             paired_fixture_credit=[p,i,i,ct.c_uint64,p,p]).items():
            getattr(cls.fx,name).argtypes=args
        cls.fx.paired_fixture_entries.restype=ct.c_size_t
        cls.fx.paired_fixture_priority_offset.restype=ct.c_size_t
        cls.data=np.frombuffer(b'The river runs through the valley. The king crossed the bridge. '*16,np.uint8)

    def make(self,fixture=True,ways=4):
        m=Model(config=dict(key=0,key_mix=0,center_value=0,threshold=.2,buckets=4,ways=ways)) if fixture else fresh_model(seed=7)
        self.addCleanup(m.close);return m
    def step(self,m,mask=3,c=0,x=1,commit=True):
        self.fx.paired_fixture_step(m.handle,mask,c,x,commit)
    def entries(self,m):
        meta=np.zeros((8192,10),np.uint64);values=np.zeros((8192,2,64),np.float32)
        n=self.fx.paired_fixture_entries(m.handle,meta.ctypes.data,values.ctypes.data)
        return meta[:n],values[:n]
    def returns(self,m):
        ids=np.zeros(4,np.uint64);values=np.zeros((2,64),np.float32)
        self.fx.paired_fixture_returns(m.handle,ids.ctypes.data,values.ctypes.data);return ids,values
    def credit(self,m,r,slot,id,values):
        out=np.zeros(2,np.float32);a=np.ascontiguousarray(values,np.float32)
        self.fx.paired_fixture_credit(m.handle,r,slot,int(id),a.ctypes.data,out.ctypes.data);return out

    def test_asynchronous_pairing_waits_for_both_sides_and_tracks_held_member(self):
        for first in (1,2):
            m=self.make();self.step(m,first);self.assertEqual(m.paired_memory_metrics()['occupied'],0)
            self.step(m,3-first);meta,_=self.entries(m);self.assertEqual(len(meta),1)
            self.assertEqual(tuple(meta[0,[6,9]]),(1,2) if first==1 else (2,1))
            self.step(m,1,c=7);meta,_=self.entries(m)
            last=meta[meta[:,1].argmax()];self.assertEqual(tuple(last[[6,9]]),(3,2) if first==1 else (3,1))

    def test_same_step_coalesces_once_and_reads_before_writes(self):
        m=self.make();self.step(m);self.assertEqual(self.returns(m)[0][0],0)
        self.assertEqual(m.paired_memory_metrics()['candidate_pairs'],1)
        state=m.paired_memory_state().copy();self.fx.paired_fixture_commit(m.handle)
        np.testing.assert_array_equal(state,m.paired_memory_state())
        self.step(m);self.assertGreater(self.returns(m)[0][0],0)
        self.assertEqual(m.paired_memory_metrics()['reinforcements'],1)

    def test_identical_content_is_disambiguated_by_context(self):
        m=self.make();self.step(m,x=1);first=self.entries(m)[0][0,1]
        self.step(m,x=9);meta,values=self.entries(m);self.assertEqual(len(meta),2)
        second=meta[meta[:,1].argmax(),1];self.assertNotEqual(first,second)
        for context,expected in ((1,first),(9,second),(1,first)):
            self.step(m,2,x=context,commit=False);ids,returned=self.returns(m)
            self.assertEqual(tuple(ids[:3]),(expected,expected,expected))
            row=int(np.where(meta[:,1]==expected)[0][0]);np.testing.assert_allclose(returned,values[row],atol=1e-6)

    def test_first_request_locks_both_ports_despite_later_context_change(self):
        m=self.make();self.step(m,x=1);original=self.entries(m)[0][0,1]
        self.step(m,x=9);ids,values=self.returns(m)
        self.assertEqual(tuple(ids[:3]),(original,original,original))
        self.step(m,1,x=9,commit=False);next_ids,_=self.returns(m)
        self.assertNotEqual(next_ids[0],original);self.assertEqual(next_ids[0],next_ids[1])

    def test_reinforcement_preserves_immutable_values_and_identity(self):
        m=self.make();self.step(m);before,values=self.entries(m)
        for _ in range(5):self.step(m)
        after,now=self.entries(m)
        np.testing.assert_array_equal(before[:,[0,1,2,4,5,6,7,8,9]],after[:,[0,1,2,4,5,6,7,8,9]])
        np.testing.assert_array_equal(values,now);self.assertGreater(after[0,3],before[0,3])

    def test_replacement_retires_utility_handle_but_saved_return_still_trains_port(self):
        m=self.make(ways=1);self.step(m);self.step(m,commit=False)
        ids,values=self.returns(m);slot=int(ids[3])-1;episode=int(ids[0]);self.assertGreater(episode,0)
        for token in range(2,302):
            self.step(m,c=token,x=token+20)
            meta,_=self.entries(m)
            if episode not in meta[:,1]:break
        self.assertNotIn(episode,meta[:,1]);self.assertGreater(m.paired_memory_metrics()['replacements'],0)
        for r in (15,16):
            before=m.paired_memory_metrics()['retired_utility_handles']
            change,utility=self.credit(m,r,slot,episode,values[r-15])
            self.assertGreater(change,0);self.assertEqual(utility,0)
            self.assertEqual(m.paired_memory_metrics()['retired_utility_handles'],before+1)

    def test_live_episode_accepts_credit_from_both_ec_ports_only(self):
        m=self.make();self.step(m);self.step(m,commit=False);ids,values=self.returns(m)
        for r in (15,16):
            change,_=self.credit(m,r,int(ids[3])-1,ids[0],values[r-15]);self.assertGreater(change,0)
        self.assertEqual(m.paired_memory_metrics()['utility_credit'],2)
        self.credit(m,14,int(ids[3])-1,ids[0],values[0])
        self.assertEqual(m.paired_memory_metrics()['utility_credit'],2)

    def test_document_reset_prevents_cross_document_pairs_and_clear_never_reuses_id(self):
        m=self.make();self.step(m);old=m.paired_memory_metrics();m.lib.reset_context(m.handle)
        self.assertEqual(m.paired_memory_metrics()['occupied'],1)
        self.step(m,1,c=22);self.assertEqual(m.paired_memory_metrics()['occupied'],1)
        self.step(m,2,x=23);meta,_=self.entries(m);new=meta[meta[:,1].argmax()]
        self.assertGreater(new[4],old['stream_epoch']);self.assertEqual(new[4],new[7])
        serial=m.paired_memory_metrics()['serial'];m.clear_memory();self.step(m)
        self.assertGreater(m.paired_memory_metrics()['serial'],serial)

    def test_suppressed_and_wrong_memory_returns_are_still_paired(self):
        m=self.make();self.step(m);self.step(m,c=6,x=7)
        m.ablate(3);self.step(m,commit=False);self.assertFalse(self.returns(m)[1].any())
        m.ablate(4);self.step(m,commit=False);ids,_=self.returns(m)
        self.assertEqual(ids[0],ids[1]);self.assertGreater(ids[0],0)
        m.ablate(0);m.block_before(1000);self.step(m,commit=False);self.assertFalse(self.returns(m)[1].any())

    def test_sidecar_rejects_corruption_without_mutating_live_state(self):
        m=self.make();self.step(m);good=m.paired_memory_state();corrupt=[]
        for index,value in ((0,99),(1,0),(2,8193)):
            bad=good.copy();bad[:32].view(np.uint64)[index]=value;corrupt.append(bad)
        bad=good.copy();offset=self.fx.paired_fixture_priority_offset();bad[offset:offset+4]=np.array([np.nan],np.float32).view(np.uint8);corrupt.append(bad)
        for bad in [good[:-1],*corrupt]:
            with self.assertRaises(ValueError):m.import_paired_memory(bad)
            np.testing.assert_array_equal(good,m.paired_memory_state())

    def test_mid_pair_and_cached_return_checkpoint_resume_exactly(self):
        for mask in (1,3):
            m=self.make();self.step(m,mask)
            if mask==3:self.step(m,commit=False)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'paired.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
                self.assertEqual(m.fingerprint(),n.fingerprint())
                for model in (m,n):self.step(model,2,x=8);self.step(model)
                self.assertEqual(m.fingerprint(),n.fingerprint())
                np.testing.assert_array_equal(self.returns(m)[1],self.returns(n)[1])

    def test_real_stream_exact_chunks_snapshot_and_frozen_memory(self):
        m=self.make(False);m.run(self.data[:321],mode='periodic');self.assertGreater(m.paired_memory_metrics()['occupied'],0)
        n=m.snapshot_copy();self.addCleanup(n.close)
        a=m.run(self.data[321:449],mode='periodic',logits=True)[1]
        b=np.concatenate([n.run(self.data[i:i+32],mode='periodic',logits=True)[1] for i in range(321,449,32)])
        np.testing.assert_array_equal(a,b);self.assertEqual(m.fingerprint(),n.fingerprint())
        for policy in ('cold_frozen','warm_frozen','cold_adaptive','warm_adaptive'):
            copy=m.evaluation_copy(memory=policy);self.addCleanup(copy.close);before=copy.evaluation_memory_info()
            self.assertEqual(bool(before['occupied'][0]),policy.startswith('warm'))
            copy.run(self.data[:128]);after=copy.evaluation_memory_info()
            if policy.endswith('frozen'):self.assertEqual(before['store_digest'],after['store_digest'])
            else:self.assertNotEqual(before['store_digest'],after['store_digest'])

    def test_no_target_leakage_and_saved_policy_is_mandatory(self):
        m=self.make(False);n=self.make(False);x=self.data[:161]
        m.run(x,np.zeros_like(x));n.run(x,np.full_like(x,255))
        np.testing.assert_array_equal(m.paired_memory_state(),n.paired_memory_state())
        with self.assertRaises(ValueError):m.configure_paired_memory(False)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.npz';m.save(path)
            with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
            del arrays['paired_memory_policy'];np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'paired-memory policy'):Model.load(path)

    def test_legacy_neural_returns_require_a_reset_before_policy_switch(self):
        m=Model(composition={'enabled':False});self.addCleanup(m.close)
        m.run(self.data[:32]);m.clear_memory()
        with self.assertRaises(ValueError):m.configure_paired_memory(False)
        with self.assertRaises(ValueError):m.configure_composition({'enabled':True})
        m.reset_context();m.configure_paired_memory(False);m.configure_composition({'enabled':True})

    def test_disabled_composition_checkpoint_keeps_its_active_memory_contract(self):
        m=Model(composition={'enabled':False});self.addCleanup(m.close);m.run(self.data[:32])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy_path.npz';m.save(path);n=Model.load(path);self.addCleanup(n.close)
            self.assertFalse(n.composition['enabled']);self.assertEqual(m.fingerprint(),n.fingerprint())
            np.testing.assert_array_equal(m.run(self.data[32:64],logits=True)[1],n.run(self.data[32:64],logits=True)[1])

if __name__=='__main__':unittest.main(verbosity=2)
