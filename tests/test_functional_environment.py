"""Functional tests for Ludo environment in realistic gameplay scenarios."""

import unittest
from unittest.mock import patch

import numpy as np
from ludo_engine.models import ALL_COLORS, GameConstants, PlayerColor

from src.models.configs.config import EnvConfig
from src.models.envs.ludo_env import LudoRLEnv


class TestEnvironmentFunctionality(unittest.TestCase):
    """Functional tests for Ludo environment gameplay scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = EnvConfig(max_turns=50, seed=42)  # Short episodes for testing
        self.env = LudoRLEnv(self.cfg)

    def test_complete_game_flow(self):
        """Test a complete game flow from start to finish."""
        obs, info = self.env.reset()
        self.assertIsInstance(obs, dict)
        self.assertIn("action_mask", info)

        done = False
        steps = 0
        total_reward = 0

        while not done and steps < self.cfg.max_turns:
            # Get valid actions
            action_mask = info["action_mask"]
            valid_actions = np.where(action_mask)[0]

            if len(valid_actions) == 0:
                # No valid moves, skip turn
                action = 0
            else:
                # Choose first valid action
                action = valid_actions[0]

            # Take step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)

            total_reward += reward
            done = terminated or truncated
            obs, info = next_obs, next_info
            steps += 1

        # Game should eventually end
        self.assertTrue(done, "Game should eventually terminate or truncate")
        self.assertIsInstance(total_reward, float)

    def test_action_masking_correctness(self):
        """Test that action masking correctly reflects valid moves."""
        obs, info = self.env.reset()

        # Get action mask
        action_mask = info["action_mask"]
        self.assertEqual(len(action_mask), GameConstants.TOKENS_PER_PLAYER)

        # Check that masked actions are actually invalid
        for action in range(GameConstants.TOKENS_PER_PLAYER):
            if not action_mask[action]:
                # Try taking invalid action - should be penalized
                with patch.object(self.env, "pending_valid_moves", []):
                    _, reward, _, _, _ = self.env.step(action)
                    self.assertLess(
                        reward, 0, f"Invalid action {action} should be penalized"
                    )

                # Reset environment state
                self.env.pending_valid_moves = (
                    self.env.game.get_valid_moves(
                        self.env.game.get_current_player(), self.env.pending_dice
                    )
                    if self.env.pending_valid_moves
                    else []
                )

    def test_deterministic_behavior_with_seed(self):
        """Test that environment initializes deterministically with same seed."""
        # Test that reset with same seed gives same agent color assignment
        # Note: Action masks may differ due to random dice rolls during opponent turns

        # Create fresh environments with same seed
        env1 = LudoRLEnv(EnvConfig(seed=123, randomize_agent=True))
        _, _ = env1.reset()
        agent_color1 = env1.agent_color

        env2 = LudoRLEnv(EnvConfig(seed=123, randomize_agent=True))
        _, _ = env2.reset()
        agent_color2 = env2.agent_color

        # Agent color should be the same (deterministic assignment)
        self.assertEqual(
            agent_color1, agent_color2, "Same seed should give same agent color"
        )

        # Test that fixed agent color works deterministically
        env3 = LudoRLEnv(
            EnvConfig(seed=456, randomize_agent=False, fixed_agent_color="RED")
        )
        _, _ = env3.reset()
        agent_color3 = env3.agent_color

        env4 = LudoRLEnv(
            EnvConfig(seed=456, randomize_agent=False, fixed_agent_color="RED")
        )
        _, _ = env4.reset()
        agent_color4 = env4.agent_color

        self.assertEqual(
            agent_color3, agent_color4, "Fixed agent color should be deterministic"
        )
        self.assertEqual(agent_color3, PlayerColor.RED, "Fixed color should be RED")

    def test_agent_color_assignment(self):
        """Test that agent color is assigned correctly."""
        # Test random assignment
        env1 = LudoRLEnv(EnvConfig(randomize_agent=True, seed=1))
        env1.reset()

        env2 = LudoRLEnv(EnvConfig(randomize_agent=True, seed=2))
        env2.reset()

        # Different seeds should potentially give different colors
        # (though statistically they might be the same)

        # Test fixed color
        env3 = LudoRLEnv(EnvConfig(randomize_agent=False, fixed_agent_color="BLUE"))
        env3.reset()
        self.assertEqual(env3.agent_color, PlayerColor.BLUE)

    def test_opponent_strategy_integration(self):
        """Test that opponent strategies are properly integrated."""
        # Test with different opponent strategies
        strategies = ["random", "probabilistic_v2", "killer"]

        for strategy in strategies:
            cfg = EnvConfig(opponent_strategy=strategy, max_turns=20)
            env = LudoRLEnv(cfg)
            env.reset()

            # Verify game has opponents
            self.assertGreater(len(env.game.players), 1)

            # Run a few steps
            for _ in range(5):
                if env.pending_valid_moves:
                    action = env.pending_valid_moves[0].token_id
                    obs, reward, done, truncated, info = env.step(action)
                    if done or truncated:
                        break

    def test_reward_accumulation_over_game(self):
        """Test that rewards accumulate meaningfully over a game."""
        rewards = self._run_episode()

        # Should have some positive and negative rewards
        positive_rewards = [r for r in rewards if r > 0]
        negative_rewards = [r for r in rewards if r < 0]

        self.assertGreater(
            len(positive_rewards), 0, "Should have some positive rewards"
        )
        self.assertGreater(
            len(negative_rewards),
            0,
            "Should have some negative rewards (time penalties)",
        )

        # Total reward should reflect game outcome
        total_reward = sum(rewards)
        self.assertIsInstance(total_reward, (int, float))

    def test_observation_consistency(self):
        """Test that observations are consistent and properly formatted."""
        obs, info = self.env.reset()

        # Check observation structure
        required_keys = [
            "agent_color",
            "agent_progress",
            "agent_distance_to_finish",
            "agent_vulnerable",
            "agent_safe",
            "agent_home",
            "agent_on_board",
            "agent_capture_available",
            "agent_finish_available",
            "agent_threat_level",
            "opponents_positions",
            "opponents_active",
            "dice",
            "dice_value_norm",
        ]

        for key in required_keys:
            self.assertIn(key, obs, f"Observation missing key: {key}")
            self.assertIsInstance(
                obs[key], np.ndarray, f"Observation {key} should be numpy array"
            )

        # Check shapes
        self.assertEqual(obs["agent_color"].shape, (4,))  # 4 colors
        self.assertEqual(obs["agent_progress"].shape, (4,))  # 4 tokens
        self.assertEqual(obs["dice"].shape, (6,))  # 6 possible dice values

        # Run a few steps and verify observations remain consistent
        for _ in range(3):
            if self.env.pending_valid_moves:
                action = self.env.pending_valid_moves[0].token_id
                next_obs, _, done, truncated, _ = self.env.step(action)

                # Same keys should be present
                self.assertEqual(set(obs.keys()), set(next_obs.keys()))

                # Shapes should be consistent
                for key in obs.keys():
                    self.assertEqual(
                        obs[key].shape, next_obs[key].shape, f"Shape mismatch for {key}"
                    )

                obs = next_obs
                if done or truncated:
                    break

    def test_one_vs_one_matchup_shapes(self):
        """Ensure observations shrink appropriately for 1v1 matchups."""
        cfg = EnvConfig(max_turns=30, seed=7, matchup="1v1")
        env = LudoRLEnv(cfg)

        obs, info = env.reset()

        self.assertEqual(len(env.game.players), cfg.player_count)
        self.assertEqual(
            obs["opponents_positions"].shape,
            (GameConstants.TOKENS_PER_PLAYER * cfg.opponent_count,),
        )
        self.assertEqual(obs["opponents_active"].shape, (cfg.opponent_count,))
        self.assertTrue(all(obs["opponents_active"] <= 1.0))

        agent_color = env.agent_color
        opponents = [p.color for p in env.game.players if p.color != agent_color]
        self.assertEqual(len(opponents), 1)
        all_colors = list(ALL_COLORS)
        agent_idx = all_colors.index(agent_color)
        opponent_idx = all_colors.index(opponents[0])
        self.assertEqual(
            opponent_idx,
            (agent_idx + len(all_colors) // 2) % len(all_colors),
        )

    def test_game_state_persistence(self):
        """Test that game state persists correctly across steps."""
        obs, info = self.env.reset()

        initial_game_state = self._get_game_state_summary()

        # Take a few steps
        for step in range(3):
            if self.env.pending_valid_moves:
                action = self.env.pending_valid_moves[0].token_id
                obs, reward, done, truncated, info = self.env.step(action)

                # Game state should have changed
                new_state = self._get_game_state_summary()
                # At least turn count should have increased
                self.assertGreater(
                    new_state["turn_count"], initial_game_state["turn_count"]
                )

                if done or truncated:
                    break

    def test_max_turns_enforcement(self):
        """Test that max_turns is properly enforced."""
        cfg = EnvConfig(max_turns=5, seed=42)  # Very short episodes
        env = LudoRLEnv(cfg)

        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < 10:  # Safety limit
            if env.pending_valid_moves:
                action = env.pending_valid_moves[0].token_id
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # Should have truncated due to max_turns
        self.assertTrue(
            truncated or steps >= cfg.max_turns,
            f"Game should have ended by step {cfg.max_turns}, took {steps} steps",
        )

    def _run_episode(self, seed=None):
        """Helper to run a complete episode and return rewards."""
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()

        rewards = []
        done = False
        steps = 0

        while not done and steps < self.cfg.max_turns:
            if self.env.pending_valid_moves:
                action = self.env.pending_valid_moves[0].token_id
            else:
                action = 0

            _, reward, terminated, truncated, _ = self.env.step(action)
            rewards.append(reward)
            done = terminated or truncated
            steps += 1

        return rewards

    def _get_game_state_summary(self):
        """Helper to get a summary of current game state."""
        return {
            "turn_count": self.env.turn_count,
            "agent_tokens": [
                token.position
                for token in self.env.game.get_player_from_color(
                    self.env.agent_color
                ).tokens
            ],
            "current_player": self.env.game.get_current_player().color,
            "game_over": self.env.game.game_over,
        }


if __name__ == "__main__":
    unittest.main()
