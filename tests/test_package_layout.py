"""Check package boundaries and result path protection."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PackageLayoutTest(unittest.TestCase):
    def test_classical_imports_do_not_load_training_dependencies(self):
        result = subprocess.run([sys.executable, '-c', (
            'import sys; '
            'import hawk_eye.models.ball_detector.classical; '
            'import hawk_eye.models.court_detector.classical; '
            'assert not {"torch", "catboost", "torchvision"}.intersection(sys.modules)'
        )], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_output_paths_and_existing_directory_protection(self):
        from hawk_eye.experiments.outputs import start_run, finish_run
        from hawk_eye.paths import ROOT

        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'run'
            self.assertEqual(start_run(target, 'court', 'v3', 'test', [], {}), target)
            finish_run(target, {'checked': True})
            with self.assertRaises(FileExistsError):
                start_run(target, 'court', 'v3', 'test', [], {})


if __name__ == '__main__':
    unittest.main()
