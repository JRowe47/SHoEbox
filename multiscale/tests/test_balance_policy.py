"""Coverage policy survives save/load and preserves explicit historical D."""
import unittest
from shoe_multiscale import Model

class BalancePolicyTests(unittest.TestCase):
    def test_default_and_historical_checkpoint_policies(self):
        data=b"Distinct contexts require reusable specialists. "*53
        for options,version,policy in [({},'e8-public-1','warmup'),
                ({'policy':'v19'},19,'warmup'),
                ({'core':'full'},10,'warmup'),
                ({'routing_arm':'D'},9,'legacy'),
                ({'routing_arm':'C'},9,'legacy'),
                ({'routing_arm':'C','balance_policy':'warmup'},10,'warmup'),
                ({'balance_policy':'legacy'},9,'legacy')]:
            with Model(**options) as m:
                self.assertEqual(m.stats()['checkpoint_version'],version)
                self.assertEqual(m.stats()['balance_policy'],policy)
                m.train(data[:514])
                with Model.loads(m.dumps()) as q:
                    self.assertEqual(q.stats()['balance_policy'],policy)
                    self.assertEqual(m.train(data[513:]),q.train(data[513:]))
                    self.assertEqual(m.dumps(),q.dumps())

    def test_explicit_migration_preserves_pending_state_and_weights(self):
        with Model(routing_arm='D') as m:
            m.train(b"An unfinished block has pending learning credit.")
            before=m.dumps()[56:]
            m.set_balance_policy('warmup')
            self.assertEqual(m.dumps()[56:],before)
            with Model.loads(m.dumps()) as q:
                self.assertEqual(q.stats()['checkpoint_version'],10)
                self.assertEqual(m.train(b"Continuation is exact."),q.train(b"Continuation is exact."))
                self.assertEqual(m.dumps(),q.dumps())

    def test_invalid_or_inapplicable_policy_is_rejected(self):
        for options in ({'balance_policy':'none'}, {'routing_arm':'B','balance_policy':'warmup'},
                        {'memory_profile':'legacy','balance_policy':'warmup'}):
            with self.assertRaises(ValueError):Model(**options)
        for arm in ('A','B'):
            with Model(routing_arm=arm) as m:
                self.assertEqual(m.routing_stats()['balance_policy'],'none')
        with Model(routing_arm='A') as m:
            with self.assertRaises(RuntimeError):m.set_balance_policy('warmup')
        with Model() as m, m.inference_copy() as frozen:
            with self.assertRaises(RuntimeError):frozen.set_balance_policy('legacy')

if __name__=='__main__':unittest.main()
