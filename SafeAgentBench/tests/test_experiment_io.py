import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import time
from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'methods' / 'experiment_io.py'


class ExperimentIOTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('experiment_io', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'bank.json'

    def test_sequential_next_task_sees_committed_memory(self):
        records = []
        def process(task):
            seen = json.loads(self.output.read_text()) if self.output.exists() else []
            self.assertEqual(len(seen), task['idx'])
            return task
        self.module.collect_memory([{'idx': 0}, {'idx': 1}], process, records, str(self.output), 1)
        self.assertEqual(len(records), 2)

    def test_sequential_failure_stops_before_next_task(self):
        seen = []
        def process(task):
            seen.append(task['idx'])
            raise ValueError('model unavailable')
        with self.assertRaises(ValueError):
            self.module.collect_memory([{'idx': 0}, {'idx': 1}], process, [], str(self.output), 1)
        self.assertEqual(seen, [0])

    def test_disk_error_leaves_previous_records_intact(self):
        records = [{'idx': 0}]
        self.output.write_text(json.dumps(records))
        with patch.object(self.module.os, 'replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                self.module.append_record(str(self.output), records, {'idx': 1})
        self.assertEqual(records, [{'idx': 0}])
        self.assertEqual(json.loads(self.output.read_text()), records)

    def test_evaluation_denominator_independent_of_memory_bank(self):
        dataset = Path(self.temp.name) / 'tasks.json'
        dataset.write_text(json.dumps([{'idx': 0}, {'idx': 1}, {'idx': 2}]))
        self.output.write_text('[{"idx": 0, "memory_item": "experience"}]')
        tasks = self.module.load_evaluation_tasks(str(dataset), str(self.output), 1)
        self.assertEqual([t['idx'] for t in tasks], [0, 1, 2])
        self.assertEqual(tasks[0]['memory_item'], 'experience')

    def test_missing_bank_prefix_is_rejected(self):
        dataset = Path(self.temp.name) / 'tasks.json'
        dataset.write_text('[{"idx": 0}, {"idx": 1}]')
        self.output.write_text('[{"idx": 1, "memory_item": "later"}]')
        with self.assertRaises(ValueError):
            self.module.load_evaluation_tasks(str(dataset), str(self.output), 2)

    def test_resume_cannot_mix_model_configurations(self):
        self.module.ensure_run_config(str(self.output), {'model': 'gpt-4o'})
        self.output.write_text('[]')
        self.module.ensure_run_config(str(self.output), {'model': 'gpt-4o'})
        with self.assertRaises(ValueError):
            self.module.ensure_run_config(str(self.output), {'model': 'gpt-4o-mini'})

    def test_existing_untracked_results_not_adopted_silently(self):
        self.output.write_text('[{"idx": 0}]')
        with self.assertRaises(ValueError):
            self.module.ensure_run_config(str(self.output), {'model': 'gpt-4o'})

    def test_sequential_resume_rejects_gap_in_history(self):
        with self.assertRaises(ValueError):
            self.module.collect_memory([{'idx': 1}], lambda x: x, [{'idx': 2}], str(self.output), 1)

    def run_entrypoint(self, filename, use_bank=True):
        root = Path(self.temp.name)
        (root / 'dataset').mkdir(exist_ok=True)
        (root / 'methods/bank').mkdir(parents=True, exist_ok=True)
        tasks = [{'idx': i, 'scene_name': f'scene{i}',
                  'safe_task': {'instruction': f'benign{i}'},
                  'unsafe_task': {'instruction': f'harmful{i}'}} for i in range(3)]
        (root / 'dataset/combined_data.json').write_text(json.dumps(tasks))
        if 'pipeline' in filename:
            (root / 'methods/bank/detailed_demo_bank.json').write_text(json.dumps([dict(tasks[0], memory_item='memory')]))
        seen = []
        def process(task):
            if filename == 'generate_memory.py':
                bank = Path('bank/detailed_demo_bank.json')
                previous = json.loads(bank.read_text()) if bank.exists() else []
                self.assertEqual(len(previous), task['idx'])
            seen.append(task['idx'])
            return task
        ns = dict(json=json, os=os, sys=sys, time=time, model_tag='demo', process_task=process,
                  ThreadPoolExecutor=ThreadPoolExecutor, as_completed=as_completed,
                  args=SimpleNamespace(use_bank=use_bank, bank_count=1, workers=1, framework='reasoningbank',
                                       long_sys=False, workflow=False, retrival_num=3))
        for name in ['append_record', 'ensure_run_config', 'load_evaluation_tasks', 'collect_memory']:
            ns[name] = getattr(self.module, name)
        source = SOURCE.parent / filename
        tree = ast.parse(source.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                    and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')
        previous = os.getcwd()
        try:
            os.chdir(root / 'methods')
            exec(compile(ast.Module(body=main.body, type_ignores=[]), str(source), 'exec'), ns)
        finally:
            os.chdir(previous)
        return seen

    def test_actual_harmful_entrypoint_evaluates_full_dataset(self):
        self.assertEqual(self.run_entrypoint('pipeline_detailed.py'), [0, 1, 2])

    def test_actual_benign_entrypoint_evaluates_full_dataset(self):
        self.assertEqual(self.run_entrypoint('pipeline_benign.py'), [0, 1, 2])

    def test_actual_collection_entrypoint_commits_before_next_task(self):
        self.assertEqual(self.run_entrypoint('generate_memory.py'), [0, 1, 2])
