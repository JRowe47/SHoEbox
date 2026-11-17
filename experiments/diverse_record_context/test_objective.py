"""A7 inherits independent A6/A5 references and proves complete step equality."""
import hashlib, unittest
import numpy as np
from .backend import Model, header, source
from .tasks import cases
from experiments.record_presence_cost import backend as a6
from experiments.joint_cold_gate.adapter import prefix_labels

def as_a6(raw):
    payload=(0x3641353050584553).to_bytes(8,'little')+raw[40:]
    return hashlib.sha256(payload).digest()+payload

class ObjectiveTests(unittest.TestCase):
    def test_only_magic_differs(self):
        self.assertEqual(header('a7').replace('0x3741353050584553ULL','0x3641353050584553ULL'),a6.header('a6'))
        self.assertEqual(source('a7'),a6.source('a6'))

    def test_identical_all_local_and_native_updates(self):
        with Model('a7',seed=7) as candidate, a6.Model('a6',seed=7) as control:
            self.assertEqual(as_a6(candidate.dumps()),control.dumps())
            for c in cases('train',4,condition='overwrite16'):
                candidate.reset();control.reset();ep=c.ep
                labels=prefix_labels(ep)
                self.assertEqual(candidate.run(ep.data[:-1],ep.data[1:],labels=labels),
                                 control.run(ep.data[:-1],ep.data[1:],labels=labels))
                self.assertEqual(as_a6(candidate.dumps()),control.dumps())
            data=bytes(range(256))*2
            self.assertEqual(candidate.run(data[:-1],data[1:]),control.run(data[:-1],data[1:]))
            self.assertEqual(as_a6(candidate.dumps()),control.dumps())
            for byte in data:
                candidate.observe(byte);control.observe(byte)
                np.testing.assert_array_equal(candidate.log_probs(),control.log_probs())
            self.assertEqual(as_a6(candidate.dumps()),control.dumps())

# Independent loss/gradient, finite differences and all-256 checks are executed
# against A7 by rebinding only the test module's backend references temporarily.
class IndependentObjectives(unittest.TestCase):
    def test_independent_references_on_a7(self):
        import experiments.record_presence_cost.test_objective as ref
        old_lib=ref.library
        from .backend import library
        try:
            ref.library=lambda policy:library('a7')
            ref.ObjectiveTests().test_independent_logistic_gradient_and_finite_differences()
            ref.ObjectiveTests().test_all_256_hybrid_reference_and_codebook_unchanged()
        finally:
            ref.library=old_lib

if __name__=='__main__':unittest.main()
