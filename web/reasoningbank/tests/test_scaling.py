import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaling import run_matts


class ScalingTests(unittest.TestCase):
    def test_default_is_one_rollout_without_scaling(self):
        calls = []
        result = run_matts(rollout=lambda memory, index: calls.append(index) or {'success': True})
        self.assertEqual(calls, [1])
        self.assertEqual(result['mode'], 'none')

    def test_sequential_updates_are_available_before_next_retrieval(self):
        bank = []
        seen = []
        def rollout(memory, index):
            seen.append(memory)
            return {'success': index == 2}
        result = run_matts(mode='seq', k=5, rollout=rollout,
                           retrieve=lambda include_current: list(bank),
                           reflect=lambda records: bank.append('lesson'), learn_from='failure')
        self.assertEqual(seen, [[], ['lesson']])
        self.assertTrue(result['success'])

    def test_parallel_uses_frozen_memory_and_one_batch_reflection(self):
        seen, updates = [], []
        result = run_matts(mode='parallel', k=3,
            rollout=lambda memory, index: seen.append(memory) or {'success': index == 1},
            retrieve=lambda include_current: 'frozen', reflect=lambda rows: updates.append(rows))
        self.assertEqual(seen, ['frozen'] * 3)
        self.assertEqual(len(updates), 1)
        self.assertEqual(len(updates[0]), 3)
        self.assertEqual(len(result['rollouts']), 3)

    def test_stop_and_learning_policy_are_optional(self):
        updates = []
        result = run_matts(mode='seq', k=3, stop_on_success=False, learn_from='success',
            rollout=lambda memory, index: {'success': index == 1},
            reflect=lambda rows: updates.extend(rows))
        self.assertEqual(len(result['rollouts']), 3)
        self.assertEqual(len(updates), 1)

    def test_error_is_not_a_failed_task(self):
        with self.assertRaises(ValueError):
            run_matts(rollout=lambda memory, index: {'success': False, 'evaluation_status': 'error'})

    def test_disabled_scaling_rejects_accidental_extra_rollouts(self):
        with self.assertRaises(ValueError):
            run_matts(mode='none', k=3, rollout=lambda *_: {'success': True})


if __name__ == '__main__':
    unittest.main()
