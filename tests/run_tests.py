"""
Test Runner for Ludo King AI Test Suite
Provides different test execution modes and comprehensive reporting.
"""

import unittest
import sys
import os
import time
import argparse
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunner:
    """Enhanced test runner with detailed reporting."""
    
    def __init__(self):
        self.test_modules = {
            'unit': 'tests.test_unit',
            'framework': 'tests.test_framework',
            'integration': 'tests.test_unit.IntegrationTestCase',
            'performance': 'tests.test_framework.PerformanceTestSuite'
        }
    
    def run_tests(self, test_type='all', verbose=True):
        """Run specified test type."""
        print(f"{'='*60}")
        print(f"LUDO KING AI TEST EXECUTION - {test_type.upper()}")
        print(f"{'='*60}")
        
        if test_type == 'all':
            return self._run_all_tests(verbose)
        elif test_type in self.test_modules:
            return self._run_specific_tests(test_type, verbose)
        else:
            print(f"Unknown test type: {test_type}")
            return False
    
    def _run_all_tests(self, verbose):
        """Run all test suites."""
        overall_success = True
        overall_stats = {'total': 0, 'failures': 0, 'errors': 0, 'time': 0}
        
        for test_name in ['unit', 'framework']:
            print(f"\n{'-'*40}")
            print(f"Running {test_name.upper()} tests...")
            print(f"{'-'*40}")
            
            success, stats = self._run_specific_tests(test_name, verbose)
            overall_success = overall_success and success
            
            overall_stats['total'] += stats['total']
            overall_stats['failures'] += stats['failures']
            overall_stats['errors'] += stats['errors']
            overall_stats['time'] += stats['time']
        
        self._print_overall_summary(overall_stats, overall_success)
        return overall_success
    
    def _run_specific_tests(self, test_type, verbose):
        """Run specific test suite."""
        try:
            # Load test module
            if test_type == 'unit':
                from tests.test_unit import (
                    TokenTestCase, PlayerTestCase, BoardTestCase, 
                    GameLogicTestCase, IntegrationTestCase
                )
                test_classes = [TokenTestCase, PlayerTestCase, BoardTestCase, 
                               GameLogicTestCase, IntegrationTestCase]
            
            elif test_type == 'framework':
                from tests.test_framework import (
                    StrategyTestSuite, GameFlowTestSuite, PerformanceTestSuite
                )
                test_classes = [StrategyTestSuite, GameFlowTestSuite, PerformanceTestSuite]
            
            elif test_type == 'integration':
                from tests.test_unit import IntegrationTestCase
                test_classes = [IntegrationTestCase]
            
            elif test_type == 'performance':
                from tests.test_framework import PerformanceTestSuite
                test_classes = [PerformanceTestSuite]
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            for test_class in test_classes:
                suite.addTests(loader.loadTestsFromTestCase(test_class))
            
            # Run tests
            stream = StringIO() if not verbose else None
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=2 if verbose else 1
            )
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            # Collect statistics
            stats = {
                'total': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'time': end_time - start_time
            }
            
            success = stats['failures'] == 0 and stats['errors'] == 0
            
            # Print results
            self._print_test_results(test_type, stats, result, success)
            
            return success, stats
            
        except ImportError as e:
            print(f"Error loading {test_type} tests: {e}")
            return False, {'total': 0, 'failures': 1, 'errors': 0, 'time': 0}
        except Exception as e:
            print(f"Error running {test_type} tests: {e}")
            return False, {'total': 0, 'failures': 0, 'errors': 1, 'time': 0}
    
    def _print_test_results(self, test_type, stats, result, success):
        """Print detailed test results."""
        print(f"\n{test_type.upper()} TEST RESULTS:")
        print(f"  Tests run: {stats['total']}")
        print(f"  Failures: {stats['failures']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Time: {stats['time']:.2f}s")
        print(f"  Success rate: {((stats['total'] - stats['failures'] - stats['errors']) / max(stats['total'], 1) * 100):.1f}%")
        
        if stats['failures'] > 0:
            print(f"\n  FAILURES ({stats['failures']}):")
            for test, traceback in result.failures:
                error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
                print(f"    - {test}: {error_msg}")
        
        if stats['errors'] > 0:
            print(f"\n  ERRORS ({stats['errors']}):")
            for test, traceback in result.errors:
                error_msg = traceback.split('\n')[-2] if traceback.split('\n') else "Unknown error"
                print(f"    - {test}: {error_msg}")
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n  Status: {status}")
    
    def _print_overall_summary(self, stats, success):
        """Print overall test summary."""
        print(f"\n{'='*60}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests run: {stats['total']}")
        print(f"Total failures: {stats['failures']}")
        print(f"Total errors: {stats['errors']}")
        print(f"Total time: {stats['time']:.2f}s")
        print(f"Overall success rate: {((stats['total'] - stats['failures'] - stats['errors']) / max(stats['total'], 1) * 100):.1f}%")
        
        if success:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print("\n💥 SOME TESTS FAILED 💥")
    
    def run_strategy_analysis(self):
        """Run detailed strategy behavior analysis."""
        print(f"{'='*60}")
        print("STRATEGY BEHAVIOR ANALYSIS")
        print(f"{'='*60}")
        
        try:
            from ludo.strategy import StrategyFactory
            from tests.test_models import TestDataFactory
            
            # Test scenarios
            scenarios = [
                ("Game Start", TestDataFactory.create_game_start_scenario()),
                ("Token Capture", TestDataFactory.create_capture_scenario()),
                ("Multi-Choice", TestDataFactory.create_multi_choice_scenario()),
                ("Endgame", TestDataFactory.create_endgame_scenario()),
                ("Defensive Play", TestDataFactory.create_defensive_scenario())
            ]
            
            strategies = ["killer", "winner", "optimist", "defensive", "balanced", "cautious", "random"]
            
            print(f"\n{'Scenario':<15} {'Strategy':<12} {'Decision':<8} {'Move Type':<12} {'Reasoning'}")
            print("-" * 80)
            
            for scenario_name, context_data in scenarios:
                # Convert to game format
                game_context = self._convert_to_game_context(context_data)
                
                for strategy_name in strategies:
                    strategy = StrategyFactory.create_strategy(strategy_name)
                    decision = strategy.decide(game_context)
                    
                    # Find selected move
                    selected_move = next(
                        (m for m in game_context['valid_moves'] if m['token_id'] == decision), 
                        {'move_type': 'unknown'}
                    )
                    
                    move_type = selected_move.get('move_type', 'unknown')
                    
                    print(f"{scenario_name:<15} {strategy_name.upper():<12} {decision:<8} {move_type:<12} {self._get_move_reasoning(strategy_name, move_type)}")
            
            print(f"\n{'='*60}")
            
        except Exception as e:
            print(f"Error in strategy analysis: {e}")
    
    def _convert_to_game_context(self, context_data):
        """Convert test context to game format."""
        return {
            'current_situation': {
                'player_color': context_data.game_state.current_player,
                'dice_value': context_data.game_state.dice_value,
                'turn_count': context_data.game_state.turn_count
            },
            'player_state': {
                'tokens_home': len([t for t in context_data.game_state.players[0].tokens if t.position == -1]),
                'tokens_active': len([t for t in context_data.game_state.players[0].tokens if 0 <= t.position <= 55]),
                'tokens_finished': len([t for t in context_data.game_state.players[0].tokens if t.position >= 56])
            },
            'opponents': [
                {
                    'color': player.color,
                    'tokens_finished': len([t for t in player.tokens if t.position >= 56]),
                    'threat_level': 0.5
                }
                for player in context_data.game_state.players[1:]
            ],
            'valid_moves': [
                {
                    'token_id': move.token_id,
                    'move_type': move.move_type,
                    'captures_opponent': move.captures_opponent,
                    'reaches_safe_spot': move.reaches_safe_spot,
                    'enters_finish': move.enters_finish
                }
                for move in context_data.valid_moves
            ],
            'strategic_analysis': context_data.strategic_analysis
        }
    
    def _get_move_reasoning(self, strategy, move_type):
        """Get reasoning for strategy decision."""
        reasoning = {
            'killer': {
                'capture': 'Aggressive capture',
                'normal': 'Advance to attack',
                'exit_home': 'Deploy forces'
            },
            'defensive': {
                'normal': 'Safe advancement',
                'capture': 'Defensive capture',
                'exit_home': 'Cautious deployment'
            },
            'winner': {
                'finish': 'Focus on winning',
                'normal': 'Advance to finish',
                'capture': 'Strategic capture'
            }
        }
        
        return reasoning.get(strategy, {}).get(move_type, 'Strategic choice')


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Ludo King AI Test Runner')
    parser.add_argument('--type', choices=['all', 'unit', 'framework', 'integration', 'performance'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--analysis', action='store_true', help='Run strategy behavior analysis')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.analysis:
        runner.run_strategy_analysis()
        return
    
    success = runner.run_tests(args.type, args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
