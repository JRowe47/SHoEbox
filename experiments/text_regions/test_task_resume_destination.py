"""Resuming a run must not reuse an unrelated output's best checkpoint."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .task_memory_train import validate_resume_destination


class ResumeDestinationChecks(unittest.TestCase):
    def test_other_nonempty_destination_is_rejected_without_mutation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root/'source', root/'other'
            source.mkdir(); target.mkdir()
            checkpoint = source/'last.pt'
            checkpoint.write_bytes(b'source checkpoint')
            stale = target/'best.pt'
            stale.write_bytes(b'unrelated checkpoint')
            with self.assertRaisesRegex(ValueError, 'empty output directory'):
                validate_resume_destination(target, checkpoint)
            self.assertEqual(stale.read_bytes(), b'unrelated checkpoint')
            self.assertEqual(checkpoint.read_bytes(), b'source checkpoint')

    def test_new_empty_and_in_place_destinations_are_allowed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source'
            source.mkdir()
            checkpoint = source/'last.pt'
            checkpoint.write_bytes(b'checkpoint')
            validate_resume_destination(source, checkpoint)
            target = root/'new'
            validate_resume_destination(target, checkpoint)
            self.assertFalse(target.exists())
            target.mkdir()
            validate_resume_destination(target, checkpoint)
            alias = root/'source_alias'
            alias.symlink_to(source, target_is_directory=True)
            validate_resume_destination(alias, checkpoint)

    def test_history_or_other_files_alone_make_destination_nonempty(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root/'source', root/'other'
            source.mkdir(); target.mkdir()
            (target/'invocation.json').write_text('{}')
            with self.assertRaises(ValueError):
                validate_resume_destination(target, source/'last.pt')
            validate_resume_destination(target, None)


if __name__ == '__main__':
    unittest.main()
