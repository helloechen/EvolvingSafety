import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, namespace=None):
    ns = {} if namespace is None else namespace
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class TrajectoryFeedbackTests(unittest.TestCase):
    def test_final_action_retained_and_source_log_unchanged(self):
        for source in ['induce_prompt.py', 'autoeval/evaluate_trajectory.py']:
            for inline in [True, False]:
                with self.subTest(source=source, inline=inline), tempfile.TemporaryDirectory() as d:
                    p = Path(d) / 'experiment.log'
                    action = 'browsergym.experiments.loop - INFO - action:' if inline else 'action:'
                    text = 'browsergym.experiments.loop - INFO - inspect page\n\n' + action + "\nclick('1')"
                    p.write_text(text)
                    ns = functions(ROOT / source, {'load_blocks', 'remove_invalid_steps', 'extract_think_and_action'})
                    thoughts, actions = ns['extract_think_and_action'](str(p))
                    self.assertEqual(actions, [["click('1')"]])
                    self.assertEqual(len(thoughts), 1)
                    self.assertEqual(p.read_text(), text)

    def test_unknown_feedback_is_error_not_task_failure(self):
        for status in ['', 'unknown', 'error', None]:
            with self.subTest(status=status):
                evaluator = Mock(return_value=({'status': status, 'thoughts': 'test'}, ''))
                ns = functions(ROOT / 'autoeval/evaluate_trajectory.py', {'process_sample'}, {
                    'CLIENT_DICT': {'test': Mock()}, 'Evaluator': Mock(return_value=evaluator)})
                with self.assertRaises(ValueError):
                    ns['process_sample']('0', {'eval': 0, 'traj_name': '0'}, '/tmp/not-used', 'test', 'text')

    def test_valid_feedback_remains_boolean(self):
        for status, expected in [('success', True), ('failure', False), (' SUCCESS ', True)]:
            evaluator = Mock(return_value=({'status': status, 'thoughts': 'test'}, ''))
            ns = functions(ROOT / 'autoeval/evaluate_trajectory.py', {'process_sample'}, {
                'CLIENT_DICT': {'test': Mock()}, 'Evaluator': Mock(return_value=evaluator)})
            result = ns['process_sample']('0', {'eval': 0, 'traj_name': '0'}, '/tmp/not-used', 'test', 'text')
            self.assertIs(result[0]['rm'], expected)
