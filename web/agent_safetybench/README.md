# Agent-SafetyBench evaluation component

This directory contains an Agent-SafetyBench-based tool-environment evaluator.

- `data/released_data_origin.json`: 2,000 original tasks.
- `data/released_data.json`: the 657-task filtered subset.
- `environments/`: tool descriptions and environment implementations.
- `evaluation/model_api/`: model adapters.
- `evaluation/eval.py`: task execution without an added experience block.
- `evaluation/eval_injection.py`: task execution with a supplied text block.

Install dependencies from `requirements.txt` and configure the chosen model adapter before running commands from `evaluation/`. The injection entry expects `prompt_injection.txt` in this directory; this input must be supplied for an experience-injection run.

The evaluator generates interaction records. The complete downstream scoring pipeline is not bundled. Previously scored outputs are available under [saved results](../results/README.md).

The original Agent-SafetyBench license is retained in [LICENSE](LICENSE). Please acknowledge the original benchmark in work using its data or code.
