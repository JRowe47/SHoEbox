"""Inference-policy interventions preserve learned parameters and saved defaults."""
import unittest
import torch
from .config import GraphConfig
from .view_score import apply_publication_policy
from .view_train import build_learner, tensor_fingerprint


class PublicationPolicyChecks(unittest.TestCase):
    def test_policy_overrides_preserve_weights_and_allow_both_directions(self):
        torch.manual_seed(17)
        model=build_learner(GraphConfig.tiny(evidence_gated_ports=True),view_enabled=True)
        fingerprint=tensor_fingerprint(model)
        for policy,expected in (('trained',True),('original',False),('trained',False),
                                ('evidence',True),('evidence',True)):
            apply_publication_policy(model,policy)
            self.assertEqual(model.cfg.evidence_gated_ports,expected)
            self.assertEqual(tensor_fingerprint(model),fingerprint)
            self.assertEqual(model.get_config()['model']['evidence_gated_ports'],expected)

    def test_unknown_policy_is_rejected_before_configuration_changes(self):
        model=build_learner(GraphConfig.tiny())
        config=model.cfg
        with self.assertRaises(ValueError):apply_publication_policy(model,'unknown')
        self.assertIs(model.cfg,config)


if __name__=='__main__':unittest.main()
