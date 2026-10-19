# Experimental results

Saved predictions, checkpoint summaries, and case-study records for the web experiments.

| Directory | Contents |
| --- | --- |
| `offline/browserart/` | Per-sample baseline and experience-condition classifier outputs for seven models; task categories |
| `offline/agent_safetybench/` | Per-sample baseline and experience-condition outputs with recorded Shield labels for seven models |
| `online/browserart/` | Checkpoint summary |
| `online/agent_safetybench/` | Checkpoint summary |
| `online/experience_composition/` | Execution/refusal/mixed-condition summaries and checkpoint statistics |
| `case_studies/browserart/` | Predictions, matched memory records, and derived case analyses |

BrowserART outputs use `clf_prediction`; Agent-SafetyBench outputs use `pred_label`. A label of 1 denotes an unsafe/attack-success outcome, and 0 denotes a safe outcome. Case-study analyses are derived from the recorded predictions and memory.

[index.csv](index.csv) provides file paths, sizes, and SHA-256 digests. [counts.csv](counts.csv) summarizes unsafe-label counts in the offline prediction files.
