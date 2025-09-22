import unittest
import numpy as np
from unittest.mock import Mock, patch

from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.config import EnvConfig
from ludo_engine.core import LudoGame
from ludo_engine.models import PlayerColor, ALL_COLORS


class TestObservationBuilder(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.game = LudoGame(ALL_COLORS)
        self.agent_color = PlayerColor.RED
        self.builder = ObservationBuilder(self.cfg, self.game, self.agent_color)

    def test_compute_size(self):
        size = self.builder.compute_size()
        # 4 agent + 12 opp + 4 progress + 6 dice + 1 turn = 27
        expected = 4 + 12 + 4 + 6 + 1
        self.assertEqual(size, expected)

    def test_normalize_pos_home(self):
        result = self.builder.normalize_pos(-1)  # HOME_POSITION
        self.assertEqual(result, -1.0)

    def test_normalize_pos_main_board(self):
        result = self.builder.normalize_pos(0)
        self.assertIsInstance(result, float)

    def test_token_progress_home(self):
        result = self.builder.token_progress(-1, 0)
        self.assertEqual(result, 0.0)

    def test_build(self):
        obs = self.builder.build(turn_counter=5, dice=3)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(len(obs), self.builder.size)


if __name__ == '__main__':
    unittest.main()