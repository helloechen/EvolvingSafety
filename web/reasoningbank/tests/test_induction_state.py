"""Validate memory persistence without importing model clients."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / 'induce_prompt.py'


class InductionStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = os.getcwd()
        os.chdir(self.temp.name)
        Path('results/webarena.0').mkdir(parents=True)
        Path('config_files').mkdir()
        Path('config_files/0.json').write_text(json.dumps({'intent': 'test', 'sites': ['reddit']}))
        self.record = Path('results/webarena.0/gpt-3.5-turbo_autoeval.json')
        self.bank = Path('bank/reddit.json')
        self.bank.parent.mkdir()
        self.generate = Mock(return_value=('memory', 'trajectory'))
        namespace = dict(os=os, json=json, args=SimpleNamespace(result_dir='results', task_id='0', bank_path=str(self.bank)),
                         extract_think_and_action=lambda _: (['thought'], ['action']), llm_generate=self.generate)
        tree = ast.parse(SOURCE.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.main = namespace['main']

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()

    def test_missing_feedback_is_failure(self):
        with self.assertRaises(FileNotFoundError):
            self.main()
        self.generate.assert_not_called()

    def test_error_feedback_cannot_be_learned_as_failure(self):
        self.record.write_text(json.dumps({'rm': None}))
        with self.assertRaises(ValueError):
            self.main()
        self.generate.assert_not_called()

    def test_corrupt_bank_is_preserved_and_no_model_called(self):
        self.record.write_text(json.dumps([{'rm': True}]))
        self.bank.write_text('{broken')
        with self.assertRaises(json.JSONDecodeError):
            self.main()
        self.assertEqual(self.bank.read_text(), '{broken')
        self.generate.assert_not_called()

    def test_success_preserves_other_tasks_and_replaces_current(self):
        self.record.write_text(json.dumps([{'rm': False}]))
        self.bank.write_text(json.dumps([{'task_id': '1', 'memory_item': 'keep'}, {'task_id': '0', 'memory_item': 'old'}]))
        self.main()
        rows = json.loads(self.bank.read_text())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['memory_item'], 'keep')
        self.assertEqual(rows[1]['memory_item'], 'memory')
        self.assertIs(rows[1]['success'], False)
