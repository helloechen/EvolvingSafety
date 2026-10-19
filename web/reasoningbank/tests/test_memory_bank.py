import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_bank import retrieve_memory, sanitize_memory_text


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bank = Path(self.tmp.name) / 'bank.json'
        self.vectors = Path(self.tmp.name) / 'vectors.json'
        self.bank.write_text(json.dumps([
            {'task_id': '0', 'query': 'source zero', 'sites': ['reddit'], 'memory_item': 'first'},
            {'task_idx': 1, 'task': 'source one', 'sites': ['gitlab'], 'memory_item': 'second'},
            {'task_idx': 2, 'task': 'source two', 'sites': ['reddit'], 'memory_item': 'third'},
        ]))
        self.vectors.write_text(json.dumps({'0': [1, 0], '1': [0, 1], '2': [0.8, 0.2]}))

    def retrieve(self, **kwargs):
        return retrieve_memory(self.bank, self.vectors, **kwargs)

    def test_online_self_exclusion_and_ranking(self):
        text, info = self.retrieve(query_id=0, top_k=2)
        self.assertEqual(info['selected_task_ids'], [2, 1])
        self.assertNotIn('first', text)

    def test_frozen_query_does_not_reuse_source_vector(self):
        before = self.vectors.read_bytes()
        _, info = self.retrieve(query_id=0, query_embedding=[0, 1], exclude_self=False)
        self.assertEqual(info['selected_task_ids'], [1])
        self.assertEqual(before, self.vectors.read_bytes())

    def test_site_filter_precedes_ranking(self):
        text, info = self.retrieve(query_embedding=[0, 1], candidate_sites=['reddit'], top_k=3)
        self.assertEqual(set(info['selected_task_ids']), {0, 2})
        self.assertNotIn('second', text)

    def test_empty_site_pool_has_no_fallback(self):
        text, info = self.retrieve(query_embedding=[1, 0], candidate_sites=['shopping'])
        self.assertEqual(text, '')
        self.assertEqual(info['selected_task_ids'], [])

    def test_missing_source_embedding_is_error(self):
        self.vectors.write_text(json.dumps({'0': [1, 0]}))
        with self.assertRaises(ValueError):
            self.retrieve(query_id=0)

    def test_nonfinite_vector_is_error(self):
        with self.assertRaises(ValueError):
            self.retrieve(query_embedding=[float('nan'), 0])

    def test_memory_delimiters_are_not_executable_tags(self):
        self.assertEqual(sanitize_memory_text('<action>click</action>'), '[action tag]click[action tag end]')

    def test_csv_source_compatibility(self):
        self.vectors = self.vectors.with_suffix('.csv')
        self.vectors.write_text('1,0\n0,1\n0.8,0.2\n')
        _, info = self.retrieve(query_id=0)
        self.assertEqual(info['selected_task_ids'], [2])

    def test_index_builder_keeps_noncontiguous_task_ids(self):
        from build_embeddings import build_index
        directory = Path(self.tmp.name) / 'configs'
        directory.mkdir()
        for task_id in [27, 580]:
            (directory / f'{task_id}.json').write_text(json.dumps({'task_id': task_id, 'intent': str(task_id)}))
        vectors, hashes = build_index(directory, lambda _: [1, 0])
        self.assertEqual(set(vectors), {'27', '580'})
        self.assertEqual(set(hashes), set(vectors))

    def test_index_builder_rejects_dimension_drift(self):
        from build_embeddings import build_index
        directory = Path(self.tmp.name) / 'configs'
        directory.mkdir()
        for task_id in [0, 1]:
            (directory / f'{task_id}.json').write_text(json.dumps({'task_id': task_id, 'intent': str(task_id)}))
        with self.assertRaises(ValueError):
            build_index(directory, lambda text: [1, 0] if text == '0' else [1, 0, 0])


if __name__ == '__main__':
    unittest.main()
