"""Functional tests for reward calculation in realistic game scenarios."""

import unittest
from unittest.mock import Mock

from ludo_engine.models import PlayerColor

from src.models.configs.config import EnvConfig
from src.models.envs.reward import AdvancedRewardCalculator


class TestRewardFunctionality(unittest.TestCase):
    """Functional tests for reward system in realistic scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = AdvancedRewardCalculator()
        self.cfg = EnvConfig()
        self.agent_color = PlayerColor.RED

    def test_win_scenario_rewards(self):
        """Test reward accumulation in a winning game scenario."""
        game = Mock()
        winner = Mock()
        winner.color = self.agent_color
        game.winner = winner

        # Simulate a winning game with multiple moves
        rewards = []

        # Move 1: Progress move
        move_result1 = Mock()
        move_result1.success = True
        move_result1.old_position = 0
        move_result1.new_position = 6
        move_result1.captured_tokens = []
        move_result1.finished_token = False

        reward1, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=move_result1,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=False,
        )
        rewards.append(reward1)

        # Move 2: Capture move
        move_result2 = Mock()
        move_result2.success = True
        move_result2.old_position = 6
        move_result2.new_position = 12
        move_result2.captured_tokens = [Mock()]
        move_result2.finished_token = False

        reward2, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=move_result2,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=False,
        )
        rewards.append(reward2)

        # Move 3: Finish token
        move_result3 = Mock()
        move_result3.success = True
        move_result3.old_position = 50
        move_result3.new_position = 99
        move_result3.captured_tokens = []
        move_result3.finished_token = True

        reward3, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=move_result3,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=False,
        )
        rewards.append(reward3)

        # Final terminal reward
        reward_terminal, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=None,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=True,
        )
        rewards.append(reward_terminal)

        # Verify rewards are positive and accumulate properly
        total_reward = sum(rewards)
        self.assertGreater(
            total_reward, 0, "Winning scenario should yield positive total reward"
        )
        self.assertGreater(reward_terminal, 0, "Terminal win reward should be positive")

    def test_loss_scenario_rewards(self):
        """Test reward accumulation in a losing game scenario."""
        game = Mock()
        winner = Mock()
        winner.color = PlayerColor.BLUE  # Different from agent
        game.winner = winner

        # Simulate a losing game with some progress but ultimate defeat
        rewards = []

        # Move 1: Good progress
        move_result1 = Mock()
        move_result1.success = True
        move_result1.old_position = 0
        move_result1.new_position = 8
        move_result1.captured_tokens = []
        move_result1.finished_token = False

        reward1, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=move_result1,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=False,
        )
        rewards.append(reward1)

        # Move 2: Got captured
        reward2, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=None,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=1,
            terminated=False,
        )
        rewards.append(reward2)

        # Move 3: Illegal move
        reward3, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=None,
            cfg=self.cfg,
            is_illegal=True,
            opponent_captures=0,
            terminated=False,
        )
        rewards.append(reward3)

        # Terminal loss reward
        reward_terminal, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=None,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=True,
        )
        rewards.append(reward_terminal)

        # Verify loss scenario yields negative total reward
        total_reward = sum(rewards)
        self.assertLess(
            total_reward, 0, "Losing scenario should yield negative total reward"
        )
        self.assertLess(reward_terminal, 0, "Terminal loss reward should be negative")

    def test_capture_battle_scenario(self):
        """Test reward dynamics in a capture battle scenario."""
        game = Mock()
        game.winner = None

        rewards = []

        # Agent captures opponent
        move_result1 = Mock()
        move_result1.success = True
        move_result1.old_position = 10
        move_result1.new_position = 16
        move_result1.captured_tokens = [Mock()]
        move_result1.finished_token = False

        reward1, breakdown1 = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=move_result1,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=False,
        )
        rewards.append(reward1)

        # Opponent captures agent back
        reward2, breakdown2 = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=None,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=1,
            terminated=False,
        )
        rewards.append(reward2)

        # Agent captures again
        move_result3 = Mock()
        move_result3.success = True
        move_result3.old_position = 5
        move_result3.new_position = 11
        move_result3.captured_tokens = [Mock(), Mock()]  # Double capture
        move_result3.finished_token = False

        reward3, breakdown3 = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=move_result3,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=False,
        )
        rewards.append(reward3)

        # Verify capture rewards balance out appropriately
        self.assertGreater(reward1, 0, "Agent capture should give positive reward")
        self.assertLess(reward2, 0, "Getting captured should give negative reward")
        self.assertGreater(
            reward3, reward1, "Double capture should give higher reward than single"
        )

        # Verify breakdown components
        self.assertEqual(breakdown1["capture"], self.cfg.reward.capture)
        self.assertEqual(breakdown2["got_captured"], self.cfg.reward.got_captured)
        self.assertEqual(breakdown3["capture"], 2 * self.cfg.reward.capture)

    def test_progress_optimization_scenario(self):
        """Test that progress rewards encourage optimal play."""
        game = Mock()
        game.winner = None

        # Compare different progress moves
        scenarios = [
            (0, 6, "Exit home"),
            (6, 12, "Normal progress"),
            (45, 51, "Home stretch progress"),
            (51, 57, "Home column progress"),
        ]

        rewards = []
        for old_pos, new_pos, description in scenarios:
            move_result = Mock()
            move_result.success = True
            move_result.old_position = old_pos
            move_result.new_position = new_pos
            move_result.captured_tokens = []
            move_result.finished_token = False

            reward, breakdown = self.calculator.compute(
                game=game,
                agent_color=self.agent_color,
                move_result=move_result,
                cfg=self.cfg,
                is_illegal=False,
                opponent_captures=0,
                terminated=False,
            )
            rewards.append((reward, description, breakdown["progress"]))

        # All should have time penalty + some progress reward
        for reward, desc, progress_reward in rewards:
            # At minimum, should not be worse than just time penalty
            self.assertGreaterEqual(
                reward,
                self.cfg.reward.time_penalty,
                f"{desc} should not be worse than time penalty",
            )
            # Progress reward should be non-negative
            self.assertGreaterEqual(
                progress_reward, 0, f"{desc} should have non-negative progress reward"
            )

    def test_penalty_avoidance_scenario(self):
        """Test that penalties discourage illegal moves and time wasting."""
        game = Mock()
        game.winner = None

        # Compare legal vs illegal moves
        legal_move = Mock()
        legal_move.success = True
        legal_move.old_position = 10
        legal_move.new_position = 16
        legal_move.captured_tokens = []
        legal_move.finished_token = False

        legal_reward, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=legal_move,
            cfg=self.cfg,
            is_illegal=False,
            opponent_captures=0,
            terminated=False,
        )

        illegal_reward, _ = self.calculator.compute(
            game=game,
            agent_color=self.agent_color,
            move_result=None,
            cfg=self.cfg,
            is_illegal=True,
            opponent_captures=0,
            terminated=False,
        )

        # Illegal move should be heavily penalized
        self.assertLess(
            illegal_reward,
            legal_reward,
            "Illegal moves should be penalized more than legal moves",
        )
        self.assertLess(
            illegal_reward,
            self.cfg.reward.time_penalty,
            "Illegal moves should have large negative reward",
        )

    def test_reward_consistency_across_episodes(self):
        """Test that reward calculation is consistent across episodes."""
        # Run the same scenario multiple times
        game = Mock()
        game.winner = None

        move_result = Mock()
        move_result.success = True
        move_result.old_position = 0
        move_result.new_position = 6
        move_result.captured_tokens = [Mock()]
        move_result.finished_token = False

        rewards = []
        for _ in range(5):
            self.calculator.reset_for_new_episode()
            reward, breakdown = self.calculator.compute(
                game=game,
                agent_color=self.agent_color,
                move_result=move_result,
                cfg=self.cfg,
                is_illegal=False,
                opponent_captures=0,
                terminated=False,
            )
            rewards.append((reward, breakdown))

        # All rewards should be identical
        first_reward, first_breakdown = rewards[0]
        for reward, breakdown in rewards[1:]:
            self.assertEqual(
                reward, first_reward, "Rewards should be consistent across episodes"
            )
            self.assertEqual(
                breakdown,
                first_breakdown,
                "Breakdowns should be consistent across episodes",
            )


if __name__ == "__main__":
    unittest.main()
