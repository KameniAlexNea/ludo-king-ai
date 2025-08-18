#!/usr/bin/env python3
"""
Real Move Evaluation System for Ludo AI
Evaluates moves based on actual game state and board positions.
"""

from game_state_saver import GameStateSaver

class RealMoveEvaluator:
    """Evaluates moves based on actual game mechanics and board state"""
    
    def __init__(self):
        self.weights = {
            "safety": 3,      # High weight
            "progress": 3,    # High weight  
            "capture": 2,     # Medium weight
            "blocking": 2     # Medium weight
        }
    
    def evaluate_move_from_context(self, game_context, chosen_move):
        """
        Evaluate a move based on the actual game context
        
        Args:
            game_context: Full game state from get_ai_decision_context()
            chosen_move: The token_id that was chosen
            
        Returns:
            dict: Real evaluation scores based on game mechanics
        """
        # Find the actual move details
        chosen_move_details = None
        for move in game_context.get("valid_moves", []):
            if move["token_id"] == chosen_move:
                chosen_move_details = move
                break
        
        if not chosen_move_details:
            return {"error": "Move not found in valid moves"}
        
        # Extract game state components
        player_state = game_context.get("player_state", {})
        opponents = game_context.get("opponents", [])
        strategic_analysis = game_context.get("strategic_analysis", {})
        current_situation = game_context.get("current_situation", {})
        
        # Calculate real scores based on actual game mechanics
        scores = {
            "safety": self._evaluate_safety_real(chosen_move_details, strategic_analysis),
            "progress": self._evaluate_progress_real(chosen_move_details, player_state, current_situation),
            "capture": self._evaluate_capture_real(chosen_move_details),
            "blocking": self._evaluate_blocking_real(chosen_move_details, opponents, strategic_analysis),
        }
        
        # Calculate weighted total
        total = sum(scores[key] * self.weights[key] for key in scores)
        max_possible = sum(5 * weight for weight in self.weights.values())
        scores["total"] = total / max_possible
        
        # Add move context for analysis
        scores["move_details"] = {
            "move_type": chosen_move_details.get("move_type", "unknown"),
            "strategic_value": chosen_move_details.get("strategic_value", 0),
            "is_safe_move": chosen_move_details.get("is_safe_move", False),
            "captures_opponent": chosen_move_details.get("captures_opponent", False),
            "current_position": chosen_move_details.get("current_position", -1),
            "target_position": chosen_move_details.get("target_position", -1)
        }
        
        return scores
    
    def _evaluate_safety_real(self, move_details, strategic_analysis):
        """Evaluate safety based on actual move data"""
        # Check if move is explicitly marked as safe in game engine
        if move_details.get("is_safe_move", False):
            return 5  # Game engine says it's safe
        
        move_type = move_details.get("move_type", "")
        
        # Moves that are inherently safe
        if move_type in ["finish", "advance_home_column"]:
            return 5  # Cannot be captured in home column
        elif move_type == "exit_home":
            return 4  # Starting position is relatively safe
        
        # Check for risky moves from strategic analysis
        risky_moves = strategic_analysis.get("risky_moves", [])
        move_token_id = move_details.get("token_id", -1)
        
        # Check if this move is in the risky list
        for risky_move in risky_moves:
            if risky_move.get("token_id") == move_token_id:
                return 1  # High risk
        
        # Default to neutral if no clear safety info
        return 3
    
    def _evaluate_progress_real(self, move_details, player_state, current_situation):
        """Evaluate progress based on actual move advancement"""
        move_type = move_details.get("move_type", "")
        strategic_value = move_details.get("strategic_value", 0)
        
        # Excellent progress moves
        if move_type == "finish":
            return 5  # Finishing a token is always excellent progress
        elif move_type == "advance_home_column":
            return 5  # Moving toward finish is excellent
        
        # Exit home evaluation based on game state
        elif move_type == "exit_home":
            tokens_active = player_state.get("tokens_on_board", 0)
            tokens_home = player_state.get("tokens_in_home", 4)
            
            if tokens_active == 0:
                return 5  # Must get tokens out
            elif tokens_home > 2:
                return 4  # Good to get more tokens active
            elif tokens_active >= 3:
                return 2  # Maybe too many tokens active
            else:
                return 3  # Neutral
        
        # Normal board advancement - use strategic value
        else:
            # Convert strategic value to 1-5 scale
            if strategic_value >= 15:
                return 5
            elif strategic_value >= 10:
                return 4
            elif strategic_value >= 5:
                return 3
            elif strategic_value >= 1:
                return 2
            else:
                return 1
    
    def _evaluate_capture_real(self, move_details):
        """Evaluate capture opportunity based on actual capture data"""
        if move_details.get("captures_opponent", False):
            captured_tokens = move_details.get("captured_tokens", [])
            if len(captured_tokens) > 1:
                return 5  # Multiple captures
            else:
                return 5  # Single capture is still excellent
        else:
            return 1  # No capture opportunity
    
    def _evaluate_blocking_real(self, move_details, opponents, strategic_analysis):
        """Evaluate blocking value based on opponent threat levels"""
        # Check if we can block based on strategic analysis
        can_block = strategic_analysis.get("can_block_opponent", False)
        if not can_block:
            return 1  # No blocking opportunity
        
        # Get opponent threat levels
        threat_levels = [opp.get("threat_level", 0) for opp in opponents]
        max_threat = max(threat_levels) if threat_levels else 0
        
        if max_threat > 0.8:
            return 5  # Blocking a leader
        elif max_threat > 0.6:
            return 4  # Blocking a strong opponent
        elif max_threat > 0.4:
            return 3  # Blocking moderate threat
        else:
            return 2  # Blocking weak opponent


class GameStateAnalyzer:
    """Analyzes saved game states with real evaluation"""
    
    def __init__(self, save_dir="saved_states"):
        self.saver = GameStateSaver(save_dir)
        self.evaluator = RealMoveEvaluator()
    
    def analyze_decisions_with_context(self, strategy_name, max_samples=1000):
        """Analyze decisions using the actual game context that was saved"""
        states = self.saver.load_states(strategy_name)
        
        if not states:
            print(f"No saved data for {strategy_name}")
            return
        
        print(f"\n=== REAL ANALYSIS: {strategy_name.upper()} ===")
        
        # Sample if too many states
        if len(states) > max_samples:
            import random
            states = random.sample(states, max_samples)
            print(f"Analyzing {max_samples} random samples from {len(self.saver.load_states(strategy_name))} total decisions")
        else:
            print(f"Analyzing all {len(states)} decisions")
        
        evaluations = []
        evaluation_errors = 0
        context_errors = 0
        
        for state in states:
            try:
                # Check if we have the new rich context format
                game_context = state.get("game_context")
                if not game_context:
                    # Old format - skip or try to reconstruct minimal context
                    context_errors += 1
                    continue
                
                evaluation = self.evaluator.evaluate_move_from_context(
                    game_context, 
                    state["chosen_move"]
                )
                
                if "error" not in evaluation:
                    # Add outcome information
                    evaluation["outcome_quality"] = self._score_outcome(state["outcome"])
                    evaluations.append(evaluation)
                else:
                    evaluation_errors += 1
                    
            except Exception:
                evaluation_errors += 1
                continue
        
        if context_errors > 0:
            print(f"⚠️  Skipped {context_errors} decisions with old context format")
        if evaluation_errors > 0:
            print(f"⚠️  Could not evaluate {evaluation_errors} decisions due to errors")
        
        if not evaluations:
            print("❌ No decisions could be evaluated")
            if context_errors > 0:
                print("💡 Run new tournaments to generate data with rich context")
            return
        
        # Calculate statistics
        self._print_strategy_analysis(evaluations, strategy_name)
        
        return evaluations
    
    def _score_outcome(self, outcome):
        """Score the actual outcome"""
        score = 0
        
        if outcome.get("game_won", False):
            score += 10
        if outcome.get("token_finished", False):
            score += 5
        if outcome.get("captured_tokens"):
            score += len(outcome["captured_tokens"]) * 3
        if outcome.get("extra_turn", False):
            score += 2
            
        return min(10, score)
    
    def _print_strategy_analysis(self, evaluations, strategy_name):
        """Print detailed analysis"""
        # Calculate averages
        avg_scores = {}
        for criteria in ["safety", "progress", "capture", "blocking", "total", "outcome_quality"]:
            scores = [e[criteria] for e in evaluations if criteria in e]
            avg_scores[criteria] = sum(scores) / len(scores) if scores else 0
        
        print("\nSTRATEGY PERFORMANCE:")
        print(f"Safety:    {avg_scores['safety']:.2f}/5.0")
        print(f"Progress:  {avg_scores['progress']:.2f}/5.0")
        print(f"Capture:   {avg_scores['capture']:.2f}/5.0")
        print(f"Blocking:  {avg_scores['blocking']:.2f}/5.0")
        print(f"Overall:   {avg_scores['total']:.3f}/1.0")
        print(f"Outcomes:  {avg_scores['outcome_quality']:.2f}/10.0")
        
        # Find correlation between evaluation and outcome
        eval_scores = [e["total"] for e in evaluations]
        outcome_scores = [e["outcome_quality"] for e in evaluations]
        
        if len(eval_scores) > 10:
            correlation = self._simple_correlation(eval_scores, outcome_scores)
            print(f"Evaluation-Outcome Correlation: {correlation:.3f}")
            
            if correlation > 0.3:
                print("✅ Good moves tend to have good outcomes")
            elif correlation < -0.1:
                print("⚠️  Evaluation system may be backwards")
            else:
                print("❓ Weak correlation - evaluation needs improvement")
        
        # Move type analysis
        move_types = {}
        for evaluation in evaluations:
            move_details = evaluation.get("move_details", {})
            move_type = move_details.get("move_type", "unknown")
            if move_type not in move_types:
                move_types[move_type] = {"count": 0, "avg_outcome": 0, "avg_strategic_value": 0}
            move_types[move_type]["count"] += 1
            move_types[move_type]["avg_outcome"] += evaluation["outcome_quality"]
            move_types[move_type]["avg_strategic_value"] += move_details.get("strategic_value", 0)
        
        print("\nMOVE TYPE ANALYSIS:")
        for move_type, data in move_types.items():
            avg_outcome = data["avg_outcome"] / data["count"]
            avg_strategic = data["avg_strategic_value"] / data["count"]
            print(f"{move_type}: {data['count']} moves, avg outcome {avg_outcome:.2f}, avg strategic value {avg_strategic:.1f}")
            
        # Safety analysis
        safe_moves = [e for e in evaluations if e.get("move_details", {}).get("is_safe_move", False)]
        risky_moves = [e for e in evaluations if not e.get("move_details", {}).get("is_safe_move", False)]
        
        if safe_moves and risky_moves:
            safe_outcomes = sum(e["outcome_quality"] for e in safe_moves) / len(safe_moves)
            risky_outcomes = sum(e["outcome_quality"] for e in risky_moves) / len(risky_moves)
            print(f"\nSAFETY ANALYSIS:")
            print(f"Safe moves: {len(safe_moves)} moves, avg outcome {safe_outcomes:.2f}")
            print(f"Risky moves: {len(risky_moves)} moves, avg outcome {risky_outcomes:.2f}")
            
            if safe_outcomes > risky_outcomes:
                print("✅ Safe moves perform better")
            else:
                print("⚠️  Risky moves perform better - aggressive strategy?")
    
    def _simple_correlation(self, x, y):
        """Calculate simple correlation coefficient"""
        n = len(x)
        if n == 0:
            return 0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0
        
        return numerator / denominator


def main():
    """Analyze strategies with real evaluation"""
    analyzer = GameStateAnalyzer()
    
    # Get available strategies
    all_states = analyzer.saver.load_states()
    strategies = list(set(s['strategy'] for s in all_states))
    
    if not strategies:
        print("❌ No saved data found. Run tournaments first!")
        return
    
    print("🔍 REAL MOVE EVALUATION ANALYSIS")
    print("=" * 50)
    print(f"Available strategies: {', '.join(strategies)}")
    
    # Analyze each strategy
    for strategy in strategies[:3]:  # Limit to first 3 for demo
        analyzer.analyze_decisions_with_context(strategy)
    
    print("\n✅ ANALYSIS COMPLETE")
    print("💡 For better results, run fresh tournaments to generate rich context data.")


if __name__ == "__main__":
    main()
