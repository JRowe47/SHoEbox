"""Verify actual trainer supervision, independent documents and exact updates."""
import tempfile
import unittest
import hashlib
from pathlib import Path
import torch
from .hierarchy import HierarchyModel
from .test_hierarchy import tiny
from .train_hierarchy import train_update, evaluate, count_baselines, load_data, initialize_stream
from .train_fineweb import optimizer_for
from .checkpoint import save, load
from scripts.prepare_hierarchy_data import prepare
from scripts.prepare_hierarchy_phase import prepare as prepare_phase


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_optimizer_resume(self):
        torch.manual_seed(89)
        model = HierarchyModel(tiny()).set_execution_backend('optimized')
        optimizer = optimizer_for(model, .001, .01)
        raw = torch.randint(256, (2, 33))
        state, _ = train_update(model, optimizer, raw[:, :7], raw[:, 1:8], None, raw[:, 1:23])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pt'
            save(path, model, state, optimizer, step=1)
            other, restored, _, record = load(path)
            other.set_execution_backend('optimized')
            opt = optimizer_for(other, .001, .01); opt.load_state_dict(record['optimizer'])
            state, a = train_update(model, optimizer, raw[:, 7:24], raw[:, 8:25], state, raw[:, 8:])
            restored, b = train_update(other, opt, raw[:, 7:24], raw[:, 8:25], restored, raw[:, 8:])
            self.assertEqual(a['nll'], b['nll'])
            for p, q in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            torch.testing.assert_close(state.latest['L3'].value, restored.latest['L3'].value, atol=0, rtol=0)

    def test_document_evaluation_matches_separate_runs_and_limit(self):
        model = HierarchyModel(tiny(local_only=True)).set_execution_backend('optimized')
        docs = [torch.randint(256, (n,)) for n in (7, 25, 18)]
        separate = [evaluate(model, [d], context=8) for d in docs]
        combined = evaluate(model, docs, context=8)
        expected = sum(r['nll']*r['targets'] for r in separate)/sum(r['targets'] for r in separate)
        self.assertAlmostEqual(combined['nll'], expected, places=5)
        self.assertEqual(combined['targets'], 47)
        limited = evaluate(model, docs, context=8, limit=13)
        self.assertEqual(limited['targets'], 13)
        self.assertTrue(model.training)

    def test_document_split_and_hash_validation(self):
        rows = [dict(row_idx=i, row=dict(id=str(i), text=('document '+str(i)+' ')*10)) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'corpus'
            prepare(rows, path, train_targets=256, batch=2, validation_targets=300, document_cap=50)
            train, docs, manifest = load_data(path)
            self.assertEqual(tuple(train.shape), (2, 129))
            self.assertEqual(sum(d.numel()-1 for d in docs), 300)
            self.assertEqual(len(docs), 6)
            (path/'train.bin').write_bytes(b'x'*258)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_data(path)

    def test_baseline_counts_only_training_pairs(self):
        counts = torch.zeros(256, 256, dtype=torch.float64)
        counts[3, 4] = 1000
        scores = count_baselines(counts, [torch.tensor([3, 4])])
        self.assertEqual(scores['bigram']['targets'], 1)
        self.assertAlmostEqual(scores['bigram']['nll'], -__import__('math').log(1001/1256))

    def test_continuity_documents_are_reserved(self):
        rows = [dict(row_idx=i, row=dict(id=str(i), text=('document '+str(i)+' ')*10)) for i in range(20)]
        reserved = [dict(id='2', source_text_sha256=hashlib.sha256(rows[2]['row']['text'].encode()).hexdigest())]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare(rows, Path(tmp)/'data', train_targets=512, batch=2,
                               validation_targets=300, document_cap=50, reserved=reserved)
            for split in manifest['files'].values():
                self.assertNotIn('2', {r['id'] for r in split['documents']})

    def test_new_stream_preserves_progress_without_rescaling_prior_exposures(self):
        protocol = dict(train_sha256='old', validation_sha256='fixed', batch=8, context=128,
                        lr=.001, local_only=False)
        counts = torch.zeros(256, 256)
        checkpoint = dict(step=256, metadata=dict(protocol=protocol, initial_updates=256,
                                                  cursor=16896, epoch=2, counts=counts))
        changed = dict(protocol, train_sha256='fresh', context=256)
        old_state = object()
        with self.assertRaisesRegex(ValueError, 'new-stream'):
            initialize_stream(checkpoint, old_state, changed)
        fork = initialize_stream(checkpoint, old_state, changed, new_stream=True)
        self.assertIsNone(fork['state'])
        self.assertEqual((fork['cursor'], fork['epoch']), (0, 0))
        self.assertIs(fork['counts'], counts)
        self.assertEqual((fork['initial_updates'], fork['initial_targets']), (512, 524288))
        # A second transition must use the accumulated mixed-horizon history.
        later = dict(step=256, metadata=dict(protocol=changed, initial_updates=512,
                      initial_targets=fork['initial_targets'], cursor=65536, epoch=0, counts=counts))
        second = initialize_stream(later, old_state, changed, new_stream=True)
        self.assertEqual((second['initial_updates'], second['initial_targets']), (768, 1048576))
        with self.assertRaisesRegex(ValueError, 'validation_sha256'):
            initialize_stream(checkpoint, old_state, dict(changed, validation_sha256='changed'), new_stream=True)

    def test_same_stream_initialization_keeps_state_and_cursor(self):
        protocol = dict(train_sha256='same', validation_sha256='fixed', batch=2, context=16,
                        lr=.001, local_only=False)
        state = object()
        checkpoint = dict(step=7, metadata=dict(protocol=protocol, cursor=112, epoch=0, counts=torch.zeros(1)))
        fork = initialize_stream(checkpoint, state, protocol)
        self.assertIs(fork['state'], state)
        self.assertEqual(fork['cursor'], 112)
        self.assertEqual(fork['initial_targets'], 224)

    def test_fresh_corpus_excludes_all_prior_documents_and_preserves_validation(self):
        rows = [dict(row_idx=i, row=dict(id=str(i), text=('document '+str(i)+' ')*10)) for i in range(30)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = prepare(rows[:20], root/'old', train_targets=256, batch=2, validation_targets=100, document_cap=50)
            # Different ID with identical text is still an excluded duplicate.
            duplicate = dict(row_idx=0, row=dict(id='alias', text=rows[0]['row']['text']))
            current = prepare_phase([duplicate]+rows, root/'old', root/'new', train_targets=256, batch=2)
            old_ids = {r['id'] for split in previous['files'].values() for r in split['documents']}
            new_ids = {r['id'] for r in current['files']['train']['documents']}
            self.assertFalse(old_ids & new_ids)
            self.assertNotIn('alias', new_ids)
            self.assertEqual((root/'new/validation.bin').read_bytes(), (root/'old/validation.bin').read_bytes())
            self.assertEqual(current['files']['validation'], previous['files']['validation'])
            load_data(root/'new')


if __name__ == '__main__':
    unittest.main()
