# Web experience collection

This component collects WebArena trajectories and distills them into an external memory bank.

## Entry points

- `pipeline.py`: sequential task execution, trajectory evaluation, and memory updates.
- `run.py`: a single browser task, optionally using retrieved memory.
- `induce_prompt.py`: experience induction from a completed trajectory.
- `memory_bank.py`: source-task retrieval, memory rendering, and retrieval metadata.
- `build_embeddings.py`: complete task-ID-keyed embedding index generation.
- `scaling.py`: optional MaTTS controller with caller-supplied rollout and reflection callbacks.
- `autoeval/`: evaluation of WebArena task completion.

`config_files/` contains 812 WebArena task configurations. The Reddit subset has 106 tasks. `bank/reddit_claude.json` is an existing 106-entry memory artifact. `agents/legacy/` contains the browser runner's agent implementation.

## Setup and use

`environment.yml` records the original Python environment. BrowserGym/WebArena, the corresponding websites, authentication state, and model endpoints must be configured before executing browser tasks. Run commands from this directory and provide credentials through environment variables, including `OPENAI_API_KEY`.

Provide a complete embedding index for memory-enabled runs. Generate an index using your configured OpenAI-compatible embedding endpoint:

```bash
python build_embeddings.py --model text-embedding-ada-002 --output bank/embeddings/webarena.json
```

The model name is configurable; use the same embedding space for source and query vectors. Generation saves an index only after all configured tasks succeed and records the model and source-text hashes. It calls the embedding API and incurs usage costs.

After provisioning the browser environment:

```bash
python pipeline.py --website reddit --model_name gpt-4o --use_bank True \
  --embeddings_path bank/embeddings/webarena.json
```

Use a fresh bank and output directory for a new configuration. The coordinator writes its bank to `bank/reddit.json`; it does not overwrite the bundled `reddit_claude.json`.

The following coordinator options are independent:

| Option | Default | Meaning |
| --- | --- | --- |
| `--use_bank` | `True` | Enable or disable experience retrieval |
| `--embeddings_path` | Required with memory | Complete CSV or task-ID-keyed JSON index |
| `--retrieve_k` | `1` | Number of source-task bundles to retrieve |
| `--max_memory_items` | `3` | Requested maximum distilled items per source trajectory |
| `--temperature` | `0.7` | Acting temperature |
| `--reflection_temperature` | `1.0` | Induction temperature |

A source-task bundle may contain several distilled items; retrieval counts source tasks, not Markdown headings. The runner saves `memory_retrieval.json` with selected source IDs and similarity scores. The utility also accepts explicit query vectors and optional site filtering for callers implementing frozen evaluation.

The coordinator uses GPT-3.5-turbo for WebArena trajectory completion feedback. This task-completion evaluator is separate from downstream safety scoring.

## Optional MaTTS

`scaling.run_matts` is a separate callback API; the browser coordinator does not implicitly enable it or expose a MaTTS CLI. The default `mode='none', k=1` performs one rollout. Callers can select `mode='seq'` or `'parallel'`, the rollout budget `k`, `stop_on_success`, and `learn_from='all'|'success'|'failure'|'none'`.

Sequential mode retrieves again after each selected reflection. Parallel mode samples independent rollouts with the same frozen memory, then reflects on the selected batch; it does not launch concurrent browser sessions. Callbacks must provide environment reset/isolation, scoring, and persistence.

## Offline checks

```bash
python -B -m unittest discover -s tests -v
```

The memory tests use local fixtures and Python's standard library. They do not launch browsers, download models, or call APIs.
