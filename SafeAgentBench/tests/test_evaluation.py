"""Offline regression checks; no model clients or experiment files are used."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open


METHODS = Path(__file__).resolve().parents[1] / "methods"


def evaluator_namespace(response):
    # Load the evaluation functions without importing optional API dependencies.
    tree = ast.parse((METHODS / "evaluator.py").read_text())
    names = {"InvalidEvaluationError", "parse_rating", "compute_SR_llm", "detailed_evaluate"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {
        "List": list, "re": re,
        "detailed_call_gpt": lambda *args, **kwargs: (response, 0),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "evaluator.py", "exec"), namespace)
    return namespace


class EvaluationTests(unittest.TestCase):
    def test_valid_ratings(self):
        for response, expected in [("Analysis\nRating: [[success]]", 1),
                                   ("Rating: [[fail]]", 0),
                                   ("Rating: [[ SUCCESS ]]", 1)]:
            with self.subTest(response=response):
                ns = evaluator_namespace(response)
                self.assertEqual(ns["detailed_evaluate"]("task", [], [])[0], expected)

    def test_invalid_ratings_never_become_zero(self):
        for response in ["", "success", "Rating: [[unknown]]",
                         "Rating: [[success]]\nRating: [[fail]]",
                         "Example: Rating: [[success]]", "Rating: [[success]]\nUnfinished analysis"]:
            with self.subTest(response=response):
                ns = evaluator_namespace(response)
                with self.assertRaises(ValueError):
                    ns["detailed_evaluate"]("task", [], [])

    def test_judge_defaults_to_gpt4o(self):
        seen = []
        ns = evaluator_namespace("Rating: [[success]]")
        ns["detailed_call_gpt"] = lambda model, *args, **kwargs: (seen.append(model) or "Rating: [[success]]", 0)
        ns["detailed_evaluate"]("task", [], [])
        self.assertEqual(seen, ["gpt-4o"])

    def test_conversion_defaults_to_gpt4o(self):
        tree = ast.parse((METHODS / "utils.py").read_text())
        function = next(node for node in tree.body if getattr(node, "name", None) == "gen_low_level_plan")
        seen = []
        ns = {"call_gpt": lambda model, *args: seen.append(model) or "find cup"}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "utils.py", "exec"), ns)
        self.assertEqual(ns["gen_low_level_plan"]("Find a cup"), ["find cup"])
        self.assertEqual(seen, ["gpt-4o"])

    def test_planning_temperature_depends_on_framework(self):
        for name in ["pipeline_detailed.py", "pipeline_benign.py", "safety/pipeline_detailed.py"]:
            tree = ast.parse((METHODS / name).read_text())
            agent_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Agents")
            for workflow, framework, temperature in [(False, None, 0.7), (True, None, 0.1), (False, 'awm', 0.1)]:
                with self.subTest(script=name, workflow=workflow, framework=framework):
                    calls = []
                    def create(**kwargs):
                        calls.append(kwargs)
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="plan"))])
                    ns = {
                        "args": SimpleNamespace(workflow=workflow, framework=framework, long_sys=False),
                        "gpt_model": "test-model", "model_tag": "test",
                        "open": mock_open(read_data="workflow"),
                        "agent_client": SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
                    }
                    exec(compile(ast.Module(body=[agent_class], type_ignores=[]), name, "exec"), ns)
                    ns["Agents"]("task").multi_agent_vision_planning("environment")
                    self.assertEqual(calls[0]["temperature"], temperature)


if __name__ == "__main__":
    unittest.main()
