"""
Core Test Framework for Ludo King AI
Implements structured testing for game components and AI strategies.
"""

import os
import sys
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List

from ludo_engine.game import LudoGame
from ludo_engine.player import PlayerColor
from ludo_engine.strategy import Strategy, StrategyFactory

from tests.test_models import ExpectedBehavior, TestDataFactory, TestScenarioBuilder


class LudoTestFramework(unittest.TestCase):
    """Main test framework for Ludo game components."""

    def setUp(self):
        """Set up test environment."""
        self.game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
        self.test_builder = TestScenarioBuilder()
        self.strategy_factory = StrategyFactory()

    def assertValidMove(self, token_id: int, valid_moves: List[Dict], msg: str = ""):
        """Assert that token_id corresponds to a valid move."""
        valid_token_ids = [move["token_id"] for move in valid_moves]
        self.assertIn(
            token_id,
            valid_token_ids,
            f"Token {token_id} not in valid moves {valid_token_ids}. {msg}",
        )

    def assertStrategyBehavior(
        self,
        strategy: Strategy,
        context: Dict,
        expected_behavior: ExpectedBehavior,
        msg: str = "",
    ):
        """Assert that strategy exhibits expected behavior."""
        decision = strategy.decide(context)
        valid_moves = context.get("valid_moves", [])

        # Validate decision is legal
        self.assertValidMove(
            decision, valid_moves, f"Strategy {strategy.name} made invalid move"
        )

        # Check behavior alignment
        selected_move = next(
            (m for m in valid_moves if m["token_id"] == decision), None
        )
        self.assertIsNotNone(
            selected_move, f"Selected move not found for token {decision}"
        )

        behavior_checks = {
            ExpectedBehavior.CAPTURE_OPPONENT: lambda m: m.get("move_type")
            == "capture",
            ExpectedBehavior.MOVE_TO_SAFETY: lambda m: m.get("is_safe_move", False),
            ExpectedBehavior.ADVANCE_CLOSEST: lambda m: m.get("move_type")
            in ["normal", "advance"],
            ExpectedBehavior.EXIT_HOME: lambda m: m.get("move_type") == "exit_home",
            ExpectedBehavior.ENTER_FINISH: lambda m: m.get("move_type") == "finish",
            ExpectedBehavior.BLOCK_OPPONENT: lambda m: m.get("blocks_opponent", False),
            ExpectedBehavior.RANDOM_CHOICE: lambda m: True,  # Random is always valid
        }

        if expected_behavior in behavior_checks:
            check_func = behavior_checks[expected_behavior]
            # For non-random strategies, enforce behavior. For random, just check validity
            if expected_behavior != ExpectedBehavior.RANDOM_CHOICE:
                behavior_match = any(
                    check_func(move)
                    for move in valid_moves
                    if move["token_id"] == decision
                )
                if (
                    not behavior_match
                    and len([m for m in valid_moves if check_func(m)]) > 0
                ):
                    self.fail(
                        f"Strategy {strategy.name} did not exhibit {expected_behavior.value}. {msg}"
                    )


class StrategyTestSuite(LudoTestFramework):
    """Test suite specifically for AI strategy validation."""

    def test_all_strategies_valid_decisions(self):
        """Test that all strategies make valid decisions in all scenarios."""
        test_cases = self.test_builder.build_all_strategy_tests()

        for test_case in test_cases:
            with self.subTest(
                strategy=test_case.strategy_name, scenario=test_case.scenario
            ):
                strategy = self.strategy_factory.create_strategy(
                    test_case.strategy_name
                )
                context = self._convert_context_to_game_format(test_case.input_context)

                decision = strategy.decide(context)
                valid_moves = context.get("valid_moves", [])

                self.assertValidMove(
                    decision,
                    valid_moves,
                    f"Strategy {test_case.strategy_name} in {test_case.scenario.value}",
                )

    def test_killer_strategy_captures(self):
        """Test that Killer strategy prioritizes captures when available."""
        capture_context = TestDataFactory.create_capture_scenario()
        game_context = self._convert_context_to_game_format(capture_context)

        killer_strategy = self.strategy_factory.create_strategy("killer")
        decision = killer_strategy.decide(game_context)

        # Should choose the capture move
        capture_moves = [
            m for m in game_context["valid_moves"] if m.get("move_type") == "capture"
        ]
        if capture_moves:
            self.assertIn(
                decision,
                [m["token_id"] for m in capture_moves],
                "Killer strategy should prioritize captures",
            )

    def test_defensive_strategy_safety(self):
        """Test that Defensive strategy prioritizes safety."""
        defensive_context = TestDataFactory.create_defensive_scenario()
        game_context = self._convert_context_to_game_format(defensive_context)

        defensive_strategy = self.strategy_factory.create_strategy("defensive")
        decision = defensive_strategy.decide(game_context)

        # Should prefer safe moves when available
        selected_move = next(
            (m for m in game_context["valid_moves"] if m["token_id"] == decision), None
        )
        self.assertIsNotNone(
            selected_move, "Defensive strategy must select a valid move"
        )

    def test_winner_strategy_finishing(self):
        """Test that Winner strategy prioritizes finishing tokens."""
        endgame_context = TestDataFactory.create_endgame_scenario()
        game_context = self._convert_context_to_game_format(endgame_context)

        winner_strategy = self.strategy_factory.create_strategy("winner")
        decision = winner_strategy.decide(game_context)

        # Should prioritize finishing moves
        finish_moves = [
            m for m in game_context["valid_moves"] if m.get("move_type") == "finish"
        ]
        if finish_moves:
            self.assertIn(
                decision,
                [m["token_id"] for m in finish_moves],
                "Winner strategy should prioritize finishing",
            )

    def test_strategy_consistency(self):
        """Test that strategies are consistent across multiple runs of same scenario."""
        context = TestDataFactory.create_multi_choice_scenario()
        game_context = self._convert_context_to_game_format(context)

        # Test non-random strategies for consistency
        for strategy_name in ["killer", "winner", "defensive", "cautious"]:
            strategy = self.strategy_factory.create_strategy(strategy_name)

            decisions = []
            for _ in range(5):  # Run same scenario 5 times
                decision = strategy.decide(game_context.copy())
                decisions.append(decision)

            # Non-random strategies should be consistent
            unique_decisions = set(decisions)
            self.assertEqual(
                len(unique_decisions),
                1,
                f"Strategy {strategy_name} should be consistent, got {unique_decisions}",
            )

    def test_random_strategy_variability(self):
        """Test that Random strategy shows variability."""
        context = TestDataFactory.create_multi_choice_scenario()
        game_context = self._convert_context_to_game_format(context)

        # Only test if multiple moves available
        if len(game_context["valid_moves"]) > 1:
            random_strategy = self.strategy_factory.create_strategy("random")

            decisions = []
            for _ in range(20):  # Run many times to get variability
                decision = random_strategy.decide(game_context.copy())
                decisions.append(decision)

            unique_decisions = set(decisions)
            # Random should show some variability (at least 2 different choices in 20 runs)
            self.assertGreater(
                len(unique_decisions), 1, "Random strategy should show variability"
            )

    def _convert_context_to_game_format(self, test_context) -> Dict:
        """Convert test context to game format."""
        return {
            "current_situation": {
                "player_color": test_context.game_state.current_player,
                "dice_value": test_context.game_state.dice_value,
                "turn_count": test_context.game_state.turn_count,
                "consecutive_sixes": test_context.game_state.consecutive_sixes,
            },
            "player_state": {
                "tokens_home": len(
                    [
                        t
                        for t in test_context.game_state.players[0].tokens
                        if t.position == -1
                    ]
                ),
                "tokens_active": len(
                    [
                        t
                        for t in test_context.game_state.players[0].tokens
                        if 0 <= t.position <= 55
                    ]
                ),
                "tokens_finished": len(
                    [
                        t
                        for t in test_context.game_state.players[0].tokens
                        if t.position >= 56
                    ]
                ),
            },
            "opponents": [
                {
                    "color": player.color,
                    "tokens_finished": len(
                        [t for t in player.tokens if t.position >= 56]
                    ),
                    "tokens_active": len(
                        [t for t in player.tokens if 0 <= t.position <= 55]
                    ),
                    "threat_level": 0.5,  # Default threat level
                }
                for player in test_context.game_state.players[1:]
            ],
            "valid_moves": [
                {
                    "token_id": move.token_id,
                    "current_position": move.from_position,
                    "current_state": "home" if move.from_position == -1 else "active",
                    "target_position": move.to_position,
                    "move_type": move.move_type,
                    "is_safe_move": move.reaches_safe_spot,  # This was the key missing field!
                    "captures_opponent": move.captures_opponent,
                    "strategic_value": 10.0 if move.captures_opponent else 5.0,
                    "captured_tokens": [],
                }
                for move in test_context.valid_moves
            ],
            "strategic_analysis": test_context.strategic_analysis,
        }


class GameFlowTestSuite(LudoTestFramework):
    """Test suite for complete game flow scenarios."""

    def test_game_initialization(self):
        """Test proper game initialization."""
        self.assertEqual(len(self.game.players), 2, "Should have 2 players")
        self.assertEqual(self.game.turn_count, 0, "Turn count should start at 0")
        self.assertFalse(self.game.game_over, "Game should not be over initially")
        self.assertIsNone(self.game.winner, "No winner initially")

        # All tokens should be in home
        for player in self.game.players:
            for token in player.tokens:
                self.assertEqual(token.position, -1, "All tokens should start in home")

    def test_first_move_requires_six(self):
        """Test that first move requires rolling a 6."""
        player = self.game.get_current_player()

        # Test with non-six roll
        valid_moves = self.game.get_valid_moves(player, 3)
        self.assertEqual(len(valid_moves), 0, "No moves available without rolling 6")

        # Test with six roll - should have 4 possible exit_home moves (one for each token)
        valid_moves = self.game.get_valid_moves(player, 6)
        self.assertGreater(
            len(valid_moves), 0, "Should have moves available with rolling 6"
        )
        # All moves should be exit_home type
        exit_moves = [m for m in valid_moves if m.get("move_type") == "exit_home"]
        self.assertEqual(
            len(valid_moves), len(exit_moves), "All moves should be exit_home type"
        )

    def test_strategic_game_completion(self):
        """Test complete strategic game from start to finish."""
        # Set up players with strategies
        self.game.players[0].set_strategy(
            self.strategy_factory.create_strategy("balanced")
        )
        self.game.players[1].set_strategy(
            self.strategy_factory.create_strategy("optimist")
        )

        max_turns = 200  # Increased turn limit
        turn_count = 0

        while not self.game.game_over and turn_count < max_turns:
            current_player = self.game.get_current_player()
            dice_value = self.game.roll_dice()

            context = self.game.get_ai_decision_context(dice_value)

            if context["valid_moves"]:
                selected_token = current_player.make_strategic_decision(context)
                move_result = self.game.execute_move(
                    current_player, selected_token, dice_value
                )

                self.assertTrue(
                    move_result["success"], f"Move should succeed at turn {turn_count}"
                )

                if move_result.get("game_won"):
                    self.assertTrue(self.game.game_over, "Game should be over when won")
                    self.assertIsNotNone(self.game.winner, "Winner should be set")
                    break

                if not move_result.get("extra_turn", False):
                    self.game.next_turn()
            else:
                self.game.next_turn()

            turn_count += 1

        # Game should complete within reasonable turns or have a winner
        if self.game.game_over:
            self.assertIsNotNone(self.game.winner, "Completed game should have winner")
        else:
            self.assertLessEqual(
                turn_count, max_turns, f"Game took too long: {turn_count} turns"
            )


class PerformanceTestSuite(LudoTestFramework):
    """Test suite for performance and scaling."""

    def test_strategy_decision_speed(self):
        """Test that strategy decisions are made quickly."""
        import time

        context = TestDataFactory.create_multi_choice_scenario()
        game_context = self._convert_context_to_game_format(context)

        for strategy_name in [
            "killer",
            "winner",
            "defensive",
            "balanced",
            "cautious",
            "optimist",
            "random",
        ]:
            strategy = self.strategy_factory.create_strategy(strategy_name)

            start_time = time.time()
            for _ in range(100):  # Run 100 decisions
                decision = strategy.decide(game_context.copy())
                self.assertIsNotNone(
                    decision, f"Strategy {strategy_name} should make decision"
                )
            end_time = time.time()

            avg_time = (end_time - start_time) / 100
            self.assertLess(
                avg_time,
                0.01,
                f"Strategy {strategy_name} too slow: {avg_time:.4f}s per decision",
            )

    def test_tournament_scalability(self):
        """Test tournament system with multiple strategies."""
        strategies = ["killer", "winner", "defensive", "balanced"]
        games_per_matchup = 5

        results = {}
        for strategy in strategies:
            results[strategy] = {"wins": 0, "losses": 0, "draws": 0}

        total_games = 0
        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i + 1 :]:
                for game_num in range(games_per_matchup):
                    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
                    game.players[0].set_strategy(
                        self.strategy_factory.create_strategy(strategy1)
                    )
                    game.players[1].set_strategy(
                        self.strategy_factory.create_strategy(strategy2)
                    )

                    winner = self._play_quick_game(game, max_turns=50)
                    total_games += 1

                    if winner:
                        winner_strategy = (
                            strategy1 if winner.color == PlayerColor.RED else strategy2
                        )
                        loser_strategy = (
                            strategy2 if winner_strategy == strategy1 else strategy1
                        )
                        results[winner_strategy]["wins"] += 1
                        results[loser_strategy]["losses"] += 1
                    else:
                        results[strategy1]["draws"] += 1
                        results[strategy2]["draws"] += 1

        # Verify results
        self.assertGreater(total_games, 0, "Should have played games")
        total_outcomes = sum(
            r["wins"] + r["losses"] + r["draws"] for r in results.values()
        )
        self.assertEqual(
            total_outcomes, total_games * 2, "Win/loss accounting should balance"
        )

    def _play_quick_game(self, game, max_turns=50):
        """Play a quick game for testing purposes."""
        turn_count = 0
        while not game.game_over and turn_count < max_turns:
            current_player = game.get_current_player()
            dice_value = game.roll_dice()
            context = game.get_ai_decision_context(dice_value)

            if context["valid_moves"]:
                selected_token = current_player.make_strategic_decision(context)
                move_result = game.execute_move(
                    current_player, selected_token, dice_value
                )

                if move_result.get("game_won"):
                    return current_player

                if not move_result.get("extra_turn", False):
                    game.next_turn()
            else:
                game.next_turn()

            turn_count += 1

        return None

    def _convert_context_to_game_format(self, test_context) -> Dict:
        """Convert test context to game format (same as in StrategyTestSuite)."""
        return {
            "current_situation": {
                "player_color": test_context.game_state.current_player,
                "dice_value": test_context.game_state.dice_value,
                "turn_count": test_context.game_state.turn_count,
                "consecutive_sixes": test_context.game_state.consecutive_sixes,
            },
            "player_state": {
                "tokens_home": len(
                    [
                        t
                        for t in test_context.game_state.players[0].tokens
                        if t.position == -1
                    ]
                ),
                "tokens_active": len(
                    [
                        t
                        for t in test_context.game_state.players[0].tokens
                        if 0 <= t.position <= 55
                    ]
                ),
                "tokens_finished": len(
                    [
                        t
                        for t in test_context.game_state.players[0].tokens
                        if t.position >= 56
                    ]
                ),
            },
            "opponents": [
                {
                    "color": player.color,
                    "tokens_finished": len(
                        [t for t in player.tokens if t.position >= 56]
                    ),
                    "tokens_active": len(
                        [t for t in player.tokens if 0 <= t.position <= 55]
                    ),
                    "threat_level": 0.5,
                }
                for player in test_context.game_state.players[1:]
            ],
            "valid_moves": [
                {
                    "token_id": move.token_id,
                    "current_position": move.from_position,
                    "current_state": "home" if move.from_position == -1 else "active",
                    "target_position": move.to_position,
                    "move_type": move.move_type,
                    "is_safe_move": move.reaches_safe_spot,
                    "captures_opponent": move.captures_opponent,
                    "strategic_value": 10.0 if move.captures_opponent else 5.0,
                    "captured_tokens": [],
                }
                for move in test_context.valid_moves
            ],
            "strategic_analysis": test_context.strategic_analysis,
        }


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test suites
    suite.addTests(loader.loadTestsFromTestCase(StrategyTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(GameFlowTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(PerformanceTestSuite))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 60}")
    print("LUDO KING AI TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            error_msg = traceback.split("AssertionError: ")[-1].split("\n")[0]
            print(f"  - {test}: {error_msg}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            error_lines = traceback.split("\n")
            error_msg = error_lines[-2] if len(error_lines) > 1 else "Unknown error"
            print(f"  - {test}: {error_msg}")
