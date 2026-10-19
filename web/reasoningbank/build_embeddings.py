"""Build an ID-keyed index; publish only after every configured task succeeds."""
import argparse
import hashlib
import json
from pathlib import Path

from memory_bank import _vector


def build_index(config_dir, embed):
    configs = sorted((p for p in Path(config_dir).glob('*.json') if p.stem.isdigit()),
                     key=lambda p: int(p.stem))
    if not configs:
        raise ValueError('No numbered task configurations found')
    vectors, hashes = {}, {}
    dimension = None
    for path in configs:
        task = json.loads(path.read_text())
        task_id = str(int(task['task_id']))
        text = task['intent']
        if task_id in vectors or not isinstance(text, str) or not text.strip():
            raise ValueError('Duplicate task ID or empty instruction')
        vector = _vector(embed(text))
        if dimension is not None and len(vector) != dimension:
            raise ValueError('Embedding dimensions changed during generation')
        dimension = len(vector)
        vectors[task_id] = vector
        hashes[task_id] = hashlib.sha256(text.encode()).hexdigest()
    return vectors, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_dir', default='config_files')
    parser.add_argument('--model', required=True, help='Embedding model served by your endpoint')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(args.output.suffix + '.meta.json')
    if args.output.exists() or metadata_path.exists():
        parser.error('Choose a new output path; existing indexes are not overwritten')
    from openai import OpenAI
    client = OpenAI()
    def embed(text):
        response = client.embeddings.create(model=args.model, input=text)
        if len(response.data) != 1:
            raise ValueError('Expected one embedding for one instruction')
        return response.data[0].embedding
    vectors, hashes = build_index(args.config_dir, embed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(vectors) + '\n'
    metadata = {'model': args.model, 'task_count': len(vectors),
                'source_text_sha256': hashes,
                'index_sha256': hashlib.sha256(payload.encode()).hexdigest()}
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(payload)
    temporary.replace(args.output)
    print(f'Wrote {len(vectors)} source-task vectors to {args.output}')


if __name__ == '__main__':
    main()
