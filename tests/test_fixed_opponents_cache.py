import os
import unittest

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king.config import config as king_config


def _collect_lineup_ids(env: LudoEnv, step: int, max_c: int):
    ids = []
    for c in range(0, max_c + 1):
        env._reset_count = c
        lineup = env._get_lineup(num_opponents=3)
        ids.append(id(lineup))
    return ids


class TestFixedOpponentsCache(unittest.TestCase):
    def setUp(self):
        # Keep pool simple and deterministic
        self._prev_opponents = os.environ.get("OPPONENTS")
        os.environ["OPPONENTS"] = "random,killer,defensive"

    def tearDown(self):
        if self._prev_opponents is None:
            os.environ.pop("OPPONENTS", None)
        else:
            os.environ["OPPONENTS"] = self._prev_opponents

    def test_fixed_cache_updates_every_k_steps(self):
        original_fixed_steps = king_config.FIXED_OPPONENTS_STEPS
        try:
            # Monkeypatch equivalent
            object.__setattr__(king_config, "FIXED_OPPONENTS_STEPS", 3)

            env = LudoEnv(use_fixed_opponents=True)
            # Make curriculum very large so it doesn't add candidates and affect behavior
            env.curriculum_interval_resets = 10_000_000

            ids = _collect_lineup_ids(env, step=3, max_c=10)

            # 0,1,2 same id; 3 new id; 3,4,5 same; 6 new id; etc.
            self.assertEqual(ids[0], ids[1])
            self.assertEqual(ids[1], ids[2])
            self.assertNotEqual(ids[3], ids[2])
            self.assertEqual(ids[3], ids[4])
            self.assertEqual(ids[4], ids[5])
            self.assertNotEqual(ids[6], ids[5])
            self.assertEqual(ids[6], ids[7])
            self.assertEqual(ids[7], ids[8])
            self.assertNotEqual(ids[9], ids[8])
            self.assertEqual(ids[9], ids[10])
        finally:
            object.__setattr__(king_config, "FIXED_OPPONENTS_STEPS", original_fixed_steps)

    def test_fixed_cache_updates_every_step_when_k_is_one(self):
        original_fixed_steps = king_config.FIXED_OPPONENTS_STEPS
        try:
            object.__setattr__(king_config, "FIXED_OPPONENTS_STEPS", 1)

            env = LudoEnv(use_fixed_opponents=True)
            env.curriculum_interval_resets = 10_000_000

            ids = _collect_lineup_ids(env, step=1, max_c=6)

            # Every reset recomputes, thus different ids each time after first
            self.assertNotEqual(ids[0], ids[1])
            self.assertNotEqual(ids[1], ids[2])
            self.assertNotEqual(ids[2], ids[3])
            self.assertNotEqual(ids[3], ids[4])
            self.assertNotEqual(ids[4], ids[5])
            self.assertNotEqual(ids[5], ids[6])
        finally:
            object.__setattr__(king_config, "FIXED_OPPONENTS_STEPS", original_fixed_steps)

    def test_fixed_cache_does_not_update_between_boundaries(self):
        original_fixed_steps = king_config.FIXED_OPPONENTS_STEPS
        try:
            object.__setattr__(king_config, "FIXED_OPPONENTS_STEPS", 4)

            env = LudoEnv(use_fixed_opponents=True)
            env.curriculum_interval_resets = 10_000_000

            ids = _collect_lineup_ids(env, step=4, max_c=7)

            # 0..3 same id; 4 new id; 5..7 share with 4
            self.assertEqual(len(set(ids[0:4])), 1)
            self.assertNotEqual(ids[4], ids[3])
            self.assertEqual(len(set(ids[4:8])), 1)
        finally:
            object.__setattr__(king_config, "FIXED_OPPONENTS_STEPS", original_fixed_steps)


if __name__ == "__main__":
    unittest.main()
