"""A9 enabled parity; independent inherited gradients and masked API contracts."""
import hashlib,unittest
import numpy as np
from .backend import Model,header,source,HEADER_CHANGES
from .tasks import cases
from experiments.balanced_record_context import backend as a8
from experiments.joint_cold_gate.adapter import prefix_labels

def as_a8(raw):
    payload=(0x3841353050584553).to_bytes(8,'little')+raw[40:]
    return hashlib.sha256(payload).digest()+payload

class ObjectiveTests(unittest.TestCase):
    def test_exact_source_isolation(self):
        text=header('a9').replace('0x3941353050584553ULL','0x3841353050584553ULL')
        for before,after in HEADER_CHANGES.items():
            self.assertEqual(text.count(after),1);text=text.replace(after,before)
        self.assertEqual(text,a8.header('a8'))
        self.assertTrue(source('a9').startswith(a8.source('a8')))
    def test_enabled_exact_every_byte_local_native_inference_saved_state(self):
        with Model('a9',seed=7) as candidate,a8.Model('a8',seed=7) as control:
            self.assertEqual(as_a8(candidate.dumps()),control.dumps())
            for c in cases('train',2,condition='overwrite16'):
                candidate.reset();control.reset();ep=c.ep;labels=prefix_labels(ep)
                for i in range(len(ep.data)-1):
                    args=(ep.data[i:i+1],ep.data[i+1:i+2],None,labels[i:i+1])
                    self.assertEqual(candidate.run(*args),control.run(*args))
                    self.assertEqual(as_a8(candidate.dumps()),control.dumps())
                    np.testing.assert_array_equal(candidate.log_probs(),control.log_probs())
            data=bytes(range(256))*2
            self.assertEqual(candidate.run(data[:-1],data[1:]),control.run(data[:-1],data[1:]))
            self.assertEqual(as_a8(candidate.dumps()),control.dumps())
            for byte in data:
                candidate.observe(byte);control.observe(byte)
                np.testing.assert_array_equal(candidate.log_probs(),control.log_probs())
            self.assertEqual(as_a8(candidate.dumps()),control.dumps())
    def test_generic_ignores_credit_and_freezes_local(self):
        with Model('a9',seed=19) as a,Model('a9',seed=19) as b:
            initial=a.probe()[:7605].copy();data=bytes(range(256))
            self.assertEqual(a.run(data[:-1],data[1:],allow_record_key_update=False),b.run(data[:-1],data[1:]))
            self.assertEqual(a.dumps(),b.dumps());np.testing.assert_array_equal(a.probe()[:7605],initial)
    def test_byte_flags_equal_chunk_flags_and_actual_row_counts(self):
        from .credit import Counter
        ep=cases('train',1,condition='compete16')[0].ep;n=len(ep.data)-1;labels=prefix_labels(ep)[:n]
        flags=np.array([(i//11)%2 for i in range(n)],dtype=bool)
        with Model('a9',seed=1) as a,Model('a9',seed=1) as b:
            initial=a.inventory();counter=Counter()
            a.run(ep.data[:-1],ep.data[1:],labels=labels,allow_record_key_update=flags)
            counter.run(ep.data[:-1],ep.data[1:],labels=labels,allow_record_key_update=flags)
            for i in range(n):b.run(ep.data[i:i+1],ep.data[i+1:i+2],labels=labels[i:i+1],allow_record_key_update=bool(flags[i]))
            self.assertEqual(a.dumps(),b.dumps())
            for name in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):
                self.assertEqual(a.inventory()[name]-initial[name],counter.total[name])
            self.assertGreater(counter.total['record_key_omitted'],0)
    def test_invalid_flags_atomic_and_no_inference_flag(self):
        with Model('a9') as a:
            before=a.dumps()
            for flag in ([0,1],[2],[-1]):
                with self.assertRaises(ValueError):a.run(b'a',b'b',allow_record_key_update=flag)
                self.assertEqual(a.dumps(),before)
            with self.assertRaises(TypeError):a.log_probs(allow_record_key_update=False)
            self.assertEqual(a.dumps(),before)
    def test_independent_gradients_all_256_and_finite_differences(self):
        import experiments.record_presence_cost.test_objective as ref
        from .backend import library
        old=ref.library
        try:
            ref.library=lambda _:library('a9')
            ref.ObjectiveTests().test_independent_logistic_gradient_and_finite_differences()
            ref.ObjectiveTests().test_all_256_hybrid_reference_and_codebook_unchanged()
        finally:ref.library=old

if __name__=='__main__':unittest.main()
