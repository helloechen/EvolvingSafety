"""Local experiment persistence and task selection; no model dependencies."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path


def append_record(output, records, item):
    """Commit to disk before exposing the record to subsequent tasks."""
    if not isinstance(item, dict):
        raise ValueError('A completed task must return a record')
    candidate = records + [item]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(path) + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as f:
        json.dump(candidate, f, indent=4)
    os.replace(temporary, path)
    records.append(item)


def ensure_run_config(output, config):
    """Refuse to resume outputs from a different or unknown configuration."""
    path = Path(output)
    metadata = Path(str(path) + '.config.json')
    config = json.loads(json.dumps(config, sort_keys=True))
    if metadata.exists():
        if json.loads(metadata.read_text()) != config:
            raise ValueError(f'Configuration mismatch for {output}; use a fresh output location')
    elif path.exists():
        raise ValueError(f'No configuration record for {output}; preserve it and use a fresh output location')
    else:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        temporary = str(metadata) + '.tmp'
        Path(temporary).write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, metadata)


def load_evaluation_tasks(dataset_path, bank_path=None, bank_count=0):
    tasks = json.loads(Path(dataset_path).read_text())
    if bank_path is None or bank_count == 0:
        return tasks
    if bank_count < 0:
        raise ValueError('bank_count must be nonnegative')
    bank = json.loads(Path(bank_path).read_text())
    by_id = {}
    for item in bank:
        idx = item['idx']
        if idx in by_id:
            raise ValueError(f'Duplicate source task {idx}')
        by_id[idx] = item
    required = {t['idx'] for t in tasks if t['idx'] < bank_count}
    if bank_count > len(tasks) or not required.issubset(by_id):
        raise ValueError('The requested memory prefix is incomplete or exceeds the dataset')
    for idx in required:
        if not isinstance(by_id[idx].get('memory_item'), str) or not by_id[idx]['memory_item'].strip():
            raise ValueError(f'Missing memory for source task {idx}')
    for task in tasks:
        task['memory_item'] = by_id.get(task['idx'], {}).get('memory_item', '')
    return tasks


def collect_memory(tasks, process_task, records, output, workers):
    if workers < 1:
        raise ValueError('workers must be positive')
    tasks = sorted(tasks, key=lambda task: task['idx'])
    if workers == 1:
        if records and tasks and max(r['idx'] for r in records) >= tasks[0]['idx']:
            raise ValueError('Sequential resume requires a contiguous committed task prefix')
        for task in tasks:
            append_record(output, records, process_task(task))
        return
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                failures.append((futures[future]['idx'], str(exc)))
                continue
            append_record(output, records, result)
    if failures:
        raise RuntimeError(f'Incomplete memory generation; failed task IDs: {[idx for idx, _ in failures]}')
