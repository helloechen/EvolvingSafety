# Web experiments

Code and saved results for the web setting in [On Safety Risks in Experience-Driven Self-Evolving Agents](https://aclanthology.org/2026.findings-acl.2091/).

| Directory | Contents |
| --- | --- |
| [reasoningbank](reasoningbank/README.md) | WebArena experience collection and memory retrieval |
| [agent_safetybench](agent_safetybench/README.md) | Tool-environment task evaluation |
| [results](results/README.md) | Saved predictions, checkpoint summaries, and case-study records |

Experience collection runs a task, evaluates its trajectory, and updates an external memory bank. Downstream safety evaluation is a separate component. BrowserART execution and the complete safety-scoring pipeline are not included here.

Memory retrieval settings and the optional MaTTS callback API are described in the [experience collection documentation](reasoningbank/README.md).
