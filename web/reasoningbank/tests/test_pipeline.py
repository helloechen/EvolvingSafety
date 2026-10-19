"""Coordinator regression checks without browsers or model calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'pipeline.py'


class PipelineTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('coordinator', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.args = SimpleNamespace(
            website='reddit', is_wo='False', faithfulness_experiment='False',
            use_bank='False', start_index=0, end_index=None, restart='True',
            embeddings_path=None, retrieve_k=2, max_memory_items=4,
            temperature=0.4, reflection_temperature=0.6, model_name='gpt-4o',
            fewshot_modification_type='None', insights_modification_type='None')
        self.directory = tempfile.TemporaryDirectory()
        self.previous = os.getcwd()
        os.chdir(self.directory.name)
        Path('config_files').mkdir()
        Path('config_files/0.json').write_text(json.dumps({'task_id': 0, 'sites': ['reddit']}))
        self.calls = []

    def tearDown(self):
        os.chdir(self.previous)
        self.directory.cleanup()

    def run_coordinator(self, code=0):
        def launch(command, **kwargs):
            self.calls.append((command, kwargs))
            return SimpleNamespace(wait=lambda **unused: code, returncode=code)
        def bounded_range(*args):
            return range(1) if args == (812,) else range(*args)
        with patch.object(self.module, 'Popen', side_effect=launch), \
             patch.object(self.module, 'range', bounded_range, create=True), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only', 'OPENAI_BASE_URL': 'https://example.invalid/v1'}, clear=True):
            self.module.main()

    def test_feedback_inherits_configured_credentials(self):
        self.run_coordinator()
        evaluation_calls = [kw for cmd, kw in self.calls if 'autoeval.evaluate_trajectory' in cmd]
        self.assertTrue(evaluation_calls)
        for kw in evaluation_calls:
            self.assertEqual(kw['env']['OPENAI_API_KEY'], 'test-only')
            self.assertEqual(kw['env']['OPENAI_BASE_URL'], 'https://example.invalid/v1')

    def test_retry_keeps_memory_disabled(self):
        self.run_coordinator()
        runs = [cmd for cmd, _ in self.calls if 'run.py' in cmd]
        self.assertEqual(len(runs), 4)
        for cmd in runs:
            self.assertIn('--use_bank', cmd)
            self.assertEqual(cmd[cmd.index('--use_bank') + 1], 'False')

    def test_start_beyond_task_range_runs_nothing(self):
        self.module.args.start_index = 100
        self.run_coordinator()
        self.assertFalse(self.calls, "No subprocess should run outside the selected range")

    def test_failed_rollout_does_not_produce_feedback_or_memory(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_coordinator(code=1)
        self.assertEqual(len(self.calls), 1)

    def test_resume_rejects_feedback_without_committed_memory(self):
        self.module.record_run_configuration('results/nobank', 'bank/nobank/reddit.json', self.module.args)
        output = Path('results/nobank/webarena.0')
        output.mkdir(parents=True)
        (output / 'gpt-3.5-turbo_autoeval.json').write_text('[{"rm": true}]')
        with self.assertRaises(RuntimeError):
            self.run_coordinator()
        self.assertFalse(self.calls)

    def test_resume_accepts_completed_task(self):
        self.module.record_run_configuration('results/nobank', 'bank/nobank/reddit.json', self.module.args)
        output = Path('results/nobank/webarena.0')
        output.mkdir(parents=True)
        (output / 'gpt-3.5-turbo_autoeval.json').write_text('[{"rm": true}]')
        Path('bank/nobank').mkdir(parents=True)
        Path('bank/nobank/reddit.json').write_text('[{"task_id": "0", "memory_item": "saved"}]')
        self.run_coordinator()
        self.assertFalse(self.calls)

    def test_lowercase_false_uses_no_memory_output_directory(self):
        self.module.args.use_bank = 'false'
        self.run_coordinator()
        runs = [cmd for cmd, _ in self.calls if 'run.py' in cmd]
        for cmd in runs:
            self.assertEqual(cmd[cmd.index('--result_dir') + 1], 'results/nobank')


    def test_configuration_change_cannot_reuse_results(self):
        self.module.args.restart = 'False'
        self.run_coordinator()
        self.calls.clear()
        self.module.args.model_name = 'gpt-4o-mini'
        with self.assertRaises(ValueError):
            self.run_coordinator()
        self.assertFalse(self.calls)



if __name__ == '__main__':
    unittest.main()
