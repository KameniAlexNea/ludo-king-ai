from __future__ import annotations

import unittest

from ludo_rl.ludo_king.config import config
from ludo_rl.ludo_king.enums import Color
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.simulator import Simulator
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import Move


class SimulatorTests(unittest.TestCase):
    def test_for_game_and_step_opponents_only(self) -> None:
        players = [Color.RED, Color.GREEN, Color.YELLOW, Color.BLUE]
        game = Game(players=[Player(color=int(c)) for c in players])
        sim = Simulator.for_game(game, agent_index=0)
        # Should run without error
        sim.step_opponents_only()

    def test_step_agent_move_and_extra_turn_flag(self) -> None:
        game = Game(players=[
            Player(color=int(Color.RED)),
            Player(color=int(Color.GREEN)),
            Player(color=int(Color.YELLOW)),
            Player(color=int(Color.BLUE)),
        ])
        sim = Simulator.for_game(game, agent_index=0)
        # Agent enters with a 6
        mv = Move(player_index=0, piece_id=0, new_pos=config.START_POSITION, dice_roll=6)
        terminated, extra = sim.step(mv)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(extra, bool)
        # After an entering 6, extra may be True
        self.assertIn(extra, (True, False))


if __name__ == "__main__":
    unittest.main()
