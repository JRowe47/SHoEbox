"""Compatible learned-weight initialization and provenance across resume."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from .ports import tensor_list
from .task_memory import TaskMemoryConfig, TaskMemoryModel
from .task_credit import CompetitiveWriteEligibility
from .task_stream import TaskByteStreams, TaskStreamConfig, make_documents
from . import task_memory_train as train


torch.set_num_threads(1)


def tiny_config(**changes):
    options = dict(width=32, owners=2, experts_per_block=32, shared_per_block=16,
                   groups=4, address_dim=8, laces=16, hat_symbols=16)
    options.update(changes)
    return TaskMemoryConfig(**options)


class TaskInitializationChecks(unittest.TestCase):
    def same(self, left, right):
        a, b = tensor_list(left), tensor_list(right)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_legacy_initialization_restores_shared_weights_and_keeps_new_projection_zero(self):
        torch.manual_seed(357)
        parent = TaskMemoryModel(tiny_config())
        target = TaskMemoryModel(tiny_config(write_mode='competitive', memory_aware_write=True,
                                             bootstrap_empty=True))
        self.assertFalse(torch.equal(parent.decoder.weight, target.decoder.weight))
        with TemporaryDirectory() as folder:
            path = Path(folder)/'learned_parent.pt'
            # Legacy metadata lacks the newly introduced configuration fields.
            legacy_cfg = {key: value for key, value in parent.cfg.to_dict().items()
                          if key not in ('write_mode', 'memory_aware_write', 'bootstrap_empty')}
            torch.save(dict(weights=parent.state_dict(), config=legacy_cfg, step=600,
                            optimizer={'must_not_be_restored': True},
                            state={'must_not_be_restored': True}), path)
            result = train.initialize_weights(target, path)
            for key, value in parent.state_dict().items():
                self.same(value, target.state_dict()[key])
            self.assertEqual(int(torch.count_nonzero(target.memory_policy_features.weight)), 0)
            self.assertTrue(all(p.requires_grad for p in target.parameters()))
            self.assertEqual(result['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result['parent_step'], 600)
            self.assertEqual(result['new_parameters'], ['memory_policy_features.weight'])
            self.assertEqual(result['optimizer'], 'fresh AdamW')
            self.assertEqual(result['runtime_state'], 'fresh independent streams')
            self.assertEqual(len(torch.optim.AdamW(target.parameters()).state), 0)
            self.assertTrue(bool(target.initial_state(2).events_seen.eq(0).all()))

    def test_missing_unexpected_and_wrong_shaped_parent_weights_are_rejected(self):
        parent = TaskMemoryModel(tiny_config())
        original = parent.state_dict()
        invalid = []
        missing = dict(original); missing.pop('decoder.weight')
        invalid.append(('missing', missing, ValueError))
        extra = dict(original); extra['foreign.weight'] = torch.zeros(1)
        invalid.append(('unexpected', extra, ValueError))
        shape = dict(original); shape['decoder.bias'] = shape['decoder.bias'][:-1]
        invalid.append(('shape', shape, RuntimeError))
        with TemporaryDirectory() as folder:
            for name, weights, exception in invalid:
                with self.subTest(name=name):
                    path = Path(folder)/(name+'.pt')
                    torch.save(dict(weights=weights, step=1), path)
                    target = TaskMemoryModel(tiny_config(memory_aware_write=True))
                    with self.assertRaises(exception):
                        train.initialize_weights(target, path)

    def test_checkpoint_preserves_initialization_and_serializes_path_arguments(self):
        parent = TaskMemoryModel(tiny_config())
        model = TaskMemoryModel(tiny_config(write_mode='competitive', memory_aware_write=True))
        cfg = TaskStreamConfig()
        streams = TaskByteStreams(make_documents(cfg, count=2), batch_size=2)
        with TemporaryDirectory() as folder:
            folder = Path(folder); parent_path = folder/'parent.pt'; destination = folder/'last.pt'
            torch.save(dict(weights=parent.state_dict(), step=23), parent_path)
            initialization = train.initialize_weights(model, parent_path)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
            trace = CompetitiveWriteEligibility.zeros(2, model.owner_names, model.memory_dim,
                                                       model.cfg.policy_features)
            train.checkpoint(destination, model, optimizer, model.initial_state(2), trace, streams,
                SimpleNamespace(out=folder, init_from=parent_path, resume=None), cfg, 0, 8.,
                initialization=initialization)
            saved = torch.load(destination, weights_only=True)
            self.assertEqual(saved['initialization'], initialization)
            self.assertEqual(saved['args']['out'], str(folder))
            self.assertEqual(saved['args']['init_from'], str(parent_path))
            json.dumps(saved['args'])
            self.assertFalse(destination.with_suffix('.pt.tmp').exists())

    def test_cli_initialization_provenance_survives_checkpoint_and_resume_invocation(self):
        # Exercise actual argument parsing, training, checkpoint and invocation
        # paths. Mock only evaluation and corpus size to keep this a short test.
        parent = TaskMemoryModel(TaskMemoryConfig(width=32, owners=2))
        real_documents = make_documents
        def short_documents(config, *, seed, count):
            return real_documents(config, seed=seed, count=4)
        with TemporaryDirectory() as folder:
            folder = Path(folder); parent_path = folder/'parent.pt'; out = folder/'run'
            torch.save(dict(weights=parent.state_dict(), step=600), parent_path)
            common = ['task_memory_train', '--out', str(out), '--batch', '1', '--context', '4',
                      '--width', '32', '--owners', '2', '--eval-count', '1', '--eval-every', '1',
                      '--seed', '11', '--data-seed', '52', '--write-mode', 'competitive',
                      '--memory-aware-write', '--bootstrap-empty']
            evaluation = ({'answer_nll': 2., 'accuracy': 0.}, [])
            with patch.object(train, 'evaluate', return_value=evaluation), \
                 patch.object(train, 'make_documents', side_effect=short_documents), \
                 redirect_stdout(io.StringIO()):
                with patch('sys.argv', common+['--steps', '1', '--init-from', str(parent_path)]):
                    train.main()
                first = json.loads((out/'invocation.json').read_text())
                saved = torch.load(out/'last.pt', weights_only=True)
                self.assertEqual(first['initialization'], saved['initialization'])
                self.assertEqual(first['initialization']['parent_step'], 600)
                self.assertEqual(first['initialization']['sha256'], hashlib.sha256(parent_path.read_bytes()).hexdigest())
                self.assertEqual(first['init_from'], str(parent_path))
                with patch('sys.argv', common+['--steps', '2', '--resume', str(out/'last.pt')]):
                    train.main()
            resumed_invocation = json.loads((out/'invocation.json').read_text())
            resumed_checkpoint = torch.load(out/'last.pt', weights_only=True)
            self.assertEqual(resumed_invocation['initialization'], first['initialization'])
            self.assertEqual(resumed_checkpoint['initialization'], first['initialization'])
            self.assertIsNone(resumed_invocation['init_from'])
            self.assertEqual(resumed_invocation['resume'], str(out/'last.pt'))
            self.assertEqual(resumed_checkpoint['step'], 2)


if __name__ == '__main__':
    unittest.main()
