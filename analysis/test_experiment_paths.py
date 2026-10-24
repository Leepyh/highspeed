"""Regression checks for downloaded projects and missing checkpoints. No training."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = (ROOT / '_tmp_check').resolve()
assert TEST_TMP.is_relative_to(ROOT)
TEST_TMP.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from experiment_paths import format_command, manifest_entry_path, resolve_input_path
from experiment_matrix import make_plan
from evaluate_matrix import commands


class ExperimentPathsTest(unittest.TestCase):
    def test_copyable_powershell_command(self):
        command = format_command(["C:/a b/python.exe", "it's.py", '--check-only'], powershell=True)
        self.assertEqual(command, "& 'C:/a b/python.exe' 'it''s.py' '--check-only'")

    def test_seed_alias_and_project_root(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            target = root / 'runs' / 'seed_00' / 'best_validation.pth'
            target.parent.mkdir(parents=True)
            target.touch()
            with patch('experiment_paths.PROJECT_ROOT', root):
                self.assertEqual(resolve_input_path('runs/seed_0/best_validation.pth',
                                                    checkpoint=True), target.resolve())

    def test_existing_exact_path_is_preferred(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            for seed in ('seed_0', 'seed_00'):
                target = root / seed / 'best_validation.pth'
                target.parent.mkdir()
                target.touch()
            exact = root / 'seed_0' / 'best_validation.pth'
            self.assertEqual(resolve_input_path(exact, checkpoint=True), exact.resolve())

    def test_portable_plan_and_legacy_manifest(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            jobs = make_plan(root / 'old location', ['main'], 700, 1)
            self.assertTrue(all(not Path(job['config']).is_absolute() for job in jobs))
            self.assertTrue(all(not Path(job['output']).is_absolute() for job in jobs))
            new = root / 'new location'
            (new / 'configs').mkdir(parents=True)
            config = {'ca': {'warm_up_steps': 150, 'max_steps': 400,
                             'n_cars_range': [350, 850]},
                      'env': {'decision_interval': 5, 'n_segments': 5}}
            (new / 'configs' / 'main_mixed.json').write_text(json.dumps(config))
            manifest = new / 'manifest.json'
            manifest.write_text(json.dumps(jobs))
            batch = list(commands(manifest, root / 'results', 1000, 10))
            self.assertEqual(len(batch), 1)
            self.assertIn(str(new / 'runs' / 'main_mixed' / 'maddpg' /
                              'seed_00' / 'best_validation.pth'), batch[0][2])
            old = root / 'missing old location' / 'configs' / 'main_mixed.json'
            self.assertEqual(manifest_entry_path(old, manifest),
                             new / 'configs' / 'main_mixed.json')

    def test_cli_reports_all_missing_before_creating_output(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            output = root / 'results'
            result = subprocess.run(
                [sys.executable, '-X', 'utf8', str(ROOT / 'paper_experiments.py'), '--check-only',
                 '--model', str(root / 'absent_maddpg.pth'),
                 '--idqn-model', str(root / 'absent_idqn.pth'), '--out', str(output)],
                cwd=root, encoding='utf-8', capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('absent_maddpg.pth', result.stderr)
            self.assertIn('absent_idqn.pth', result.stderr)
            self.assertIn('No training or evaluation has started', result.stderr)
            self.assertFalse(output.exists())

    def test_rules_check_only_runs_no_simulation(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            output = Path(directory) / 'results'
            result = subprocess.run(
                [sys.executable, '-X', 'utf8', str(ROOT / 'paper_experiments.py'),
                 '--check-only', '--out', str(output)],
                encoding='utf-8', capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('three rule policies only', result.stdout)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
