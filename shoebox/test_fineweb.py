"""Budget, source separation and production-runner resume checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from . import CorticalModel, cortical_10m
from .train_fineweb import load_stream
from scripts.prepare_cortical_fineweb import prepare


class FineWebChecks(unittest.TestCase):
    def test_capacity_and_disjoint_documents(self):
        torch.set_num_threads(1)
        model = CorticalModel(cortical_10m())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9982290)
        self.assertEqual(len(model.regions), 18)
        rows = [{'row_idx': i, 'row': {'id': str(i), 'text': str(i)*20}, 'truncated_cells': []}
                for i in range(4)]
        payloads, records = prepare(rows, train_targets=16, batch=2,
                                   validation_targets=8, validation_batch=2)
        self.assertEqual(len(payloads['train']), 18)
        self.assertEqual(len(payloads['validation']), 10)
        self.assertEqual(records['train']['documents'][0]['id'], '0')
        self.assertEqual(records['validation']['documents'][0]['id'], '1')
        rows[0]['truncated_cells'] = ['text']
        with self.assertRaises(ValueError):
            prepare(rows, train_targets=16, batch=2, validation_targets=8, validation_batch=2)

    def test_runner_exact_resume_and_budget(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/'data'; data.mkdir()
            train, val = b'abcdeghijk', b'uvwxyz'
            (data/'train.bin').write_bytes(train)
            (data/'validation.bin').write_bytes(val)
            manifest = {'batch': 2, 'train_targets': 8, 'validation_batch': 2, 'validation_targets': 4,
                        'files': {'train': {'sha256': hashlib.sha256(train).hexdigest()},
                                  'validation': {'sha256': hashlib.sha256(val).hexdigest()}}}
            (data/'manifest.json').write_text(json.dumps(manifest))
            stream = load_stream(data/'train.bin', 2, 8, manifest['files']['train']['sha256'])
            self.assertEqual(stream[:, :-1].numel(), 8)
            with self.assertRaises(ValueError):
                load_stream(data/'train.bin', 2, 7, manifest['files']['train']['sha256'])
            common = [sys.executable, '-m', 'shoebox.train_fineweb', '--data', str(data),
                      '--preset', 'skeleton', '--threads', '1', '--sample-bytes', '0']
            for name, extra in [('whole', []), ('part', ['--max-updates', '1']),
                                ('resumed', ['--resume', str(root/'part/last.pt')])]:
                result = subprocess.run(common+['--out', str(root/name)]+extra,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            a = torch.load(root/'whole/last.pt', weights_only=True)
            b = torch.load(root/'resumed/last.pt', weights_only=True)
            self.assertEqual(a['step'], 2)
            self.assertEqual(a['metadata']['trained_targets'], 8)
            self.assertEqual(b['metadata']['trained_targets'], 8)
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], atol=0, rtol=0)
            self.assertEqual(a['metadata']['cursor'], b['metadata']['cursor'])
            self.assertEqual(a['state']['event'], b['state']['event'])
            for left, right in zip(a['metadata']['history'], b['metadata']['history']):
                self.assertEqual(left['nll'], right['nll'])


if __name__ == '__main__':
    unittest.main()
