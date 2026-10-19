"""Compact memory retrieval adapted from Evolving-Agent-V6."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Iterable

_REACT_MEMORY_TAG_RE = re.compile(r"<\s*(/?)\s*(thought|action)\s*>", re.IGNORECASE)

def sanitize_memory_text(value: object) -> str:
    """Neutralize ReAct delimiters before memory reaches a future prompt."""
    text = str(value or "").strip()

    def replacement(match: re.Match) -> str:
        closing, name = match.groups()
        suffix = " end" if closing else ""
        return f"[{name.lower()} tag{suffix}]"

    return _REACT_MEMORY_TAG_RE.sub(replacement, text)

def _site_compatible_memory(
    item: dict,
    candidate_sites: Iterable[str] | None,
) -> bool:
    if candidate_sites is None:
        return True
    target = {str(value) for value in candidate_sites}
    source_value = item.get("sites")
    source = (
        {str(value) for value in source_value}
        if isinstance(source_value, (list, tuple, set))
        else set()
    )
    if not source or not target:
        return False
    if len(target) == 1:
        return source == target
    # A cross-site task may reuse either a prior exact cross-site lesson or a
    # lesson from one of its component sites, but never an unrelated site.
    return source.issubset(target)

def _source_task_text(item, benchmark: str) -> str:
    task_text = str(item.get("task", ""))
    if benchmark == "alfworld":
        task_text = task_text.split("___")[0]
    return task_text.strip()

def render_memory_block(item, benchmark: str, *, include_source_outcome: bool = False) -> str:
    """Format a retrieved bank entry without source-evaluation metadata."""
    memory = sanitize_memory_text(item.get("memory_item", ""))
    if not memory:
        return ""
    task_text = _source_task_text(item, benchmark)

    lines = []
    if task_text:
        lines.extend(["[Source task instruction]", task_text, ""])
    if include_source_outcome:
        outcome = str(item.get("source_outcome", "")).strip()
        if outcome:
            lines.extend(["[Source task outcome]", outcome, ""])
    lines.extend(["[Memory items]", memory])
    return "\n".join(lines)


def _vector(values):
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Embeddings must be nonempty finite vectors")
    if math.hypot(*vector) == 0:
        raise ValueError("A zero embedding has undefined cosine similarity")
    return vector


def retrieve_memory(bank_path, embeddings_path, *, query_id=None,
                    query_embedding=None, exclude_self=True, top_k=1,
                    candidate_sites=None):
    """Retrieve source-task bundles; explicit query vectors support frozen evaluation.

    Accepts historical task_id/query and V6 task_idx/task bank records.
    IDs in CSV embeddings are row positions; JSON embeddings are keyed by ID.
    Does not mutate the bank or the embedding index.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    metadata = {"selected_task_ids": [], "similarities": [], "candidate_task_count": 0}
    path = Path(bank_path)
    if not path.exists():
        raise FileNotFoundError(path)
    bank = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(bank, list):
        raise ValueError("Memory bank must be a JSON list")
    candidates = {}
    for entry in bank:
        if not isinstance(entry, dict):
            raise ValueError("Memory bank entries must be objects")
        task_id = int(entry.get("task_idx", entry.get("task_id")))
        if exclude_self and query_id is not None and task_id == int(query_id):
            continue
        if not _site_compatible_memory(entry, candidate_sites):
            continue
        if not str(entry.get("memory_item", "")).strip():
            continue
        if task_id in candidates:
            raise ValueError("Duplicate source-task ID; select a single bank snapshot")
        candidates[task_id] = dict(entry, task=entry.get("task", entry.get("query", "")))
    metadata["candidate_task_count"] = len(candidates)
    if not candidates:
        return "", metadata
    path = Path(embeddings_path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8') as stream:
            vectors = {index: row for index, row in enumerate(csv.reader(stream))}
    else:
        vectors = {int(key): value for key, value in json.loads(path.read_text()).items()}
    if query_embedding is None:
        if query_id is None or int(query_id) not in vectors:
            raise ValueError("Supply a query vector or a valid in-dataset query ID")
        query_embedding = vectors[int(query_id)]
    query = _vector(query_embedding)
    query_norm = math.hypot(*query)
    ranked = []
    for task_id in sorted(candidates):
        if task_id not in vectors:
            raise ValueError(f"Missing embedding for source task {task_id}")
        vector = _vector(vectors[task_id])
        if len(query) != len(vector):
            raise ValueError("Query and source embeddings have different dimensions")
        norm = math.hypot(*vector)
        score = math.fsum((a / query_norm) * (b / norm) for a, b in zip(query, vector))
        ranked.append((task_id, score))
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    chosen = ranked[:top_k]
    metadata['selected_task_ids'] = [task_id for task_id, _ in chosen]
    metadata['similarities'] = [round(score, 8) for _, score in chosen]
    blocks = [render_memory_block(candidates[task_id], 'webarena') for task_id, _ in chosen]
    return '\n\n'.join(blocks), metadata
