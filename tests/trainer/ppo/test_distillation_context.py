import unittest
from verl.trainer.ppo.distillation_context import find_over_context_rows


class RRCurrentContextBudgetTest(unittest.TestCase):
    def test_privileged_extension_within_rollout_context_is_allowed(self):
        self.assertEqual(find_over_context_rows([1116], [4096], 6144), ())

    def test_only_sequences_beyond_rollout_context_are_reported(self):
        self.assertEqual(find_over_context_rows([2048, 2049], [4096, 4096], 6144), (1,))


if __name__ == "__main__":
    unittest.main()
