# Household experiments

Household planning experiments for [On Safety Risks in Experience-Driven Self-Evolving Agents](https://aclanthology.org/2026.findings-acl.2091/), using SafeAgentBench task data.

## Contents

- `dataset/combined_data.json`: 269 paired benign and harmful household tasks.
- `methods/generate_memory.py`: benign trajectory collection and experience generation.
- `methods/extract_workflow.py`: offline workflow extraction.
- `methods/pipeline_detailed.py`: harmful-task evaluation.
- `methods/pipeline_benign.py`: benign-task evaluation.
- `methods/safety/`: experiments with different experience compositions.

The main evaluation scripts assess text-based plans: they generate a plan, convert it into low-level actions, and use an LLM judge to assess task completion.

## Setup

Install the Python dependencies used by these entry points:

```bash
pip install openai numpy pandas scikit-learn pillow jsonlines requests
```

Set `OPENAI_API_KEY` for GPT-4o access. `OPENAI_BASE_URL` optionally selects an OpenAI-compatible endpoint. For locally served Qwen models, set `EVAL_BASE_URL`; the main scripts use the same `OPENAI_API_KEY` for authentication. The scripts under `methods/safety/` additionally accept `EVAL_API_KEY`.

Run the following commands from this directory. Model API calls incur usage costs.

## Example commands

```bash
cd methods
mkdir -p results/detailed

# ReasoningBank baseline: no retrieved experience, temperature 0.7.
python pipeline_detailed.py --model_name gpt-4o-2024-11-20 --use_bank false

# Collect benign experience, then evaluate a memory-bank prefix.
python generate_memory.py --model_name gpt-4o-2024-11-20 --workers 1
python pipeline_detailed.py --model_name gpt-4o-2024-11-20 --bank_count 20

# Collect offline trajectories and extract a fixed workflow.
python pipeline_detailed.py --model_name gpt-4o-2024-11-20 --framework awm --use_bank false
python generate_memory.py --model_name gpt-4o-2024-11-20 --framework awm --workers 1
python extract_workflow.py --model_name gpt-4o-2024-11-20
python pipeline_detailed.py --model_name gpt-4o-2024-11-20 --workflow true --use_bank false
```

Use `pipeline_benign.py` for the corresponding benign-task evaluation. Sequential collection (`--workers 1`) preserves task order. Generated banks and results are stored under `methods/bank/` and `methods/results/`. AWM baseline filenames include an `-awm` suffix to keep them separate from ReasoningBank baselines.

## Settings and outputs

The defaults use GPT-4o for action conversion and judging, temperature 0.1 for offline AWM collection/extraction/evaluation, and 0.7 for ReasoningBank collection and memory induction. Conversion and judging use temperature 0.2. Workflow extraction uses all collected offline trajectories by default; `--sample_size` selects a subset using `--seed` (default 0).

The evaluator accepts a final `success` or `fail` rating. Complete or resume unsuccessful runs before calculating metrics. For harmful tasks, task completion is counted as attack success; for benign tasks, it is counted as task success.

Resume uses existing output records. Use a fresh output location for each configuration.

Model-specific trajectory banks must be generated before memory-based evaluation. The composition experiments also require prepared base banks under `methods/safety/bank/base/`, which are not included. The commands above can regenerate the workflow files.

## Offline checks

From this directory:

```bash
python -B -m unittest discover -s tests -v
```

These checks do not call model APIs or run the benchmark.
