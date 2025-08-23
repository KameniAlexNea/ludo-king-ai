"""
Model validation and interpretation utilities for the improved Ludo RL system.
"""

import json
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .rl_player import RLPlayer
from .states import LudoStateEncoder
from .trainer import LudoRLTrainer


class LudoRLValidator:
    """Comprehensive validation and interpretation tools for Ludo RL models."""

    def __init__(self, model_path: str, game_data_path: str = None):
        """
        Initialize the validator.

        Args:
            model_path: Path to trained model
            game_data_path: Path to game data for validation
        """
        self.model_path = model_path
        self.player = RLPlayer(model_path)
        self.encoder = LudoStateEncoder()

        if game_data_path:
            self.trainer = LudoRLTrainer(game_data_file=game_data_path)
        else:
            self.trainer = LudoRLTrainer()

    def validate_against_expert_moves(self, expert_data: List[Dict]) -> Dict:
        """
        Validate model against expert/strategic moves.

        Args:
            expert_data: List of game states with expert move choices

        Returns:
            Dict: Validation metrics
        """
        correct_predictions = 0
        total_predictions = 0
        agreement_scores = []
        strategic_alignment = []

        for game_state in expert_data:
            valid_moves = game_state.get("valid_moves", [])
            expert_choice = game_state.get("expert_move", 0)

            if not valid_moves:
                continue

            # Get model prediction
            model_choice = self.player.choose_move(game_state)

            # Check exact agreement
            if model_choice == expert_choice:
                correct_predictions += 1

            total_predictions += 1

            # Calculate strategic alignment
            if expert_choice < len(valid_moves) and model_choice < len(valid_moves):
                expert_strategic = valid_moves[expert_choice].get("strategic_value", 0)
                model_strategic = valid_moves[model_choice].get("strategic_value", 0)

                if expert_strategic > 0:
                    alignment = min(1.0, model_strategic / expert_strategic)
                    strategic_alignment.append(alignment)

            # Agreement score (how close the choices are strategically)
            agreement_score = self._calculate_agreement_score(
                valid_moves, expert_choice, model_choice
            )
            agreement_scores.append(agreement_score)

        return {
            "exact_accuracy": correct_predictions / max(total_predictions, 1),
            "strategic_alignment": (
                np.mean(strategic_alignment) if strategic_alignment else 0
            ),
            "agreement_score": np.mean(agreement_scores) if agreement_scores else 0,
            "total_comparisons": total_predictions,
        }

    def _calculate_agreement_score(
        self, valid_moves: List[Dict], expert_idx: int, model_idx: int
    ) -> float:
        """Calculate how strategically similar two move choices are."""
        if expert_idx >= len(valid_moves) or model_idx >= len(valid_moves):
            return 0.0

        expert_move = valid_moves[expert_idx]
        model_move = valid_moves[model_idx]

        # Compare strategic attributes
        score = 0.0

        # Move type similarity
        if expert_move.get("move_type") == model_move.get("move_type"):
            score += 0.3

        # Strategic value similarity
        expert_val = expert_move.get("strategic_value", 0)
        model_val = model_move.get("strategic_value", 0)
        if expert_val > 0:
            val_sim = min(1.0, model_val / expert_val)
            score += 0.4 * val_sim

        # Safety similarity
        expert_safe = expert_move.get("is_safe_move", True)
        model_safe = model_move.get("is_safe_move", True)
        if expert_safe == model_safe:
            score += 0.3

        return score

    def analyze_decision_patterns(self, game_states: List[Dict]) -> Dict:
        """
        Analyze the model's decision-making patterns.

        Args:
            game_states: List of game states to analyze

        Returns:
            Dict: Analysis results
        """
        move_type_preferences = {}
        strategic_value_distribution = []
        safety_preferences = {"safe": 0, "risky": 0}
        capture_opportunities = {"taken": 0, "missed": 0}
        game_phase_decisions = {"early": [], "mid": [], "late": []}

        for game_state in game_states:
            valid_moves = game_state.get("valid_moves", [])
            if not valid_moves:
                continue

            # Get model decision
            analysis = self.player.choose_move_with_analysis(game_state)
            chosen_idx = analysis["move_index"]

            if chosen_idx >= len(valid_moves):
                continue

            chosen_move = valid_moves[chosen_idx]

            # Analyze move type preferences
            move_type = chosen_move.get("move_type", "unknown")
            move_type_preferences[move_type] = (
                move_type_preferences.get(move_type, 0) + 1
            )

            # Strategic value distribution
            strategic_val = chosen_move.get("strategic_value", 0)
            strategic_value_distribution.append(strategic_val)

            # Safety preferences
            is_safe = chosen_move.get("is_safe_move", True)
            safety_preferences["safe" if is_safe else "risky"] += 1

            # Capture analysis
            captures_opponent = chosen_move.get("captures_opponent", False)
            has_capture_opportunity = any(
                m.get("captures_opponent", False) for m in valid_moves
            )

            if has_capture_opportunity:
                if captures_opponent:
                    capture_opportunities["taken"] += 1
                else:
                    capture_opportunities["missed"] += 1

            # Game phase analysis
            player_state = game_state.get("game_context", {}).get("player_state", {})
            finished_tokens = player_state.get("finished_tokens", 0)
            active_tokens = player_state.get("active_tokens", 0)

            total_progress = finished_tokens + active_tokens
            if total_progress <= 1:
                phase = "early"
            elif finished_tokens >= 2:
                phase = "late"
            else:
                phase = "mid"

            game_phase_decisions[phase].append(
                {
                    "strategic_value": strategic_val,
                    "move_type": move_type,
                    "is_safe": is_safe,
                }
            )

        return {
            "move_type_preferences": move_type_preferences,
            "strategic_value_stats": {
                "mean": (
                    np.mean(strategic_value_distribution)
                    if strategic_value_distribution
                    else 0
                ),
                "std": (
                    np.std(strategic_value_distribution)
                    if strategic_value_distribution
                    else 0
                ),
                "distribution": strategic_value_distribution,
            },
            "safety_preferences": safety_preferences,
            "capture_analysis": capture_opportunities,
            "game_phase_decisions": game_phase_decisions,
        }

    def visualize_model_behavior(
        self, analysis_results: Dict, save_path: str = "model_behavior.png"
    ):
        """Create visualizations of model behavior patterns."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Move type preferences
        move_types = list(analysis_results["move_type_preferences"].keys())
        move_counts = list(analysis_results["move_type_preferences"].values())

        axes[0, 0].bar(move_types, move_counts)
        axes[0, 0].set_title("Move Type Preferences")
        axes[0, 0].set_xlabel("Move Type")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Strategic value distribution
        strategic_values = analysis_results["strategic_value_stats"]["distribution"]
        if strategic_values:
            axes[0, 1].hist(strategic_values, bins=20, alpha=0.7)
            axes[0, 1].set_title("Strategic Value Distribution")
            axes[0, 1].set_xlabel("Strategic Value")
            axes[0, 1].set_ylabel("Frequency")

        # Safety preferences
        safety_data = analysis_results["safety_preferences"]
        axes[0, 2].pie(
            safety_data.values(), labels=safety_data.keys(), autopct="%1.1f%%"
        )
        axes[0, 2].set_title("Safety Preferences")

        # Capture analysis
        capture_data = analysis_results["capture_analysis"]
        axes[1, 0].bar(capture_data.keys(), capture_data.values())
        axes[1, 0].set_title("Capture Opportunities")
        axes[1, 0].set_ylabel("Count")

        # Game phase strategic values
        phase_data = analysis_results["game_phase_decisions"]
        phases = ["early", "mid", "late"]
        phase_strategic_means = []

        for phase in phases:
            if phase_data[phase]:
                phase_strategic_means.append(
                    np.mean([d["strategic_value"] for d in phase_data[phase]])
                )
            else:
                phase_strategic_means.append(0)

        axes[1, 1].bar(phases, phase_strategic_means)
        axes[1, 1].set_title("Strategic Value by Game Phase")
        axes[1, 1].set_xlabel("Game Phase")
        axes[1, 1].set_ylabel("Average Strategic Value")

        # Safety by game phase
        phase_safety = []
        for phase in phases:
            if phase_data[phase]:
                safe_count = sum(1 for d in phase_data[phase] if d["is_safe"])
                total_count = len(phase_data[phase])
                phase_safety.append(safe_count / total_count)
            else:
                phase_safety.append(0)

        axes[1, 2].bar(phases, phase_safety)
        axes[1, 2].set_title("Safety Preference by Game Phase")
        axes[1, 2].set_xlabel("Game Phase")
        axes[1, 2].set_ylabel("Proportion of Safe Moves")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Model behavior visualization saved to {save_path}")

    def generate_model_report(
        self, validation_data: List[Dict], save_path: str = "model_report.json"
    ) -> Dict:
        """Generate a comprehensive model evaluation report."""
        print("Generating comprehensive model report...")

        # Validate against expert moves if available
        expert_validation = self.validate_against_expert_moves(validation_data)

        # Analyze decision patterns
        decision_analysis = self.analyze_decision_patterns(validation_data)

        # Create visualizations
        self.visualize_model_behavior(decision_analysis)

        # Performance metrics
        sequences = self.trainer.create_improved_training_sequences()
        if sequences:
            test_sequences = sequences[-len(sequences) // 10 :]  # Last 10%
            performance_metrics = self.trainer.evaluate_model(test_sequences)
        else:
            performance_metrics = {}

        report = {
            "model_info": {
                "model_path": self.model_path,
                "state_dim": self.encoder.state_dim,
                "validation_samples": len(validation_data),
                "report_timestamp": datetime.now().isoformat(),
            },
            "expert_validation": expert_validation,
            "decision_analysis": decision_analysis,
            "performance_metrics": performance_metrics,
            "recommendations": self._generate_recommendations(
                expert_validation, decision_analysis, performance_metrics
            ),
        }

        # Save report
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Model report saved to {save_path}")
        return report

    def _generate_recommendations(
        self, expert_val: Dict, decision_analysis: Dict, performance: Dict
    ) -> List[str]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []

        # Expert agreement recommendations
        if expert_val.get("exact_accuracy", 0) < 0.6:
            recommendations.append(
                "Consider additional training with expert demonstrations"
            )

        if expert_val.get("strategic_alignment", 0) < 0.7:
            recommendations.append(
                "Increase weight of strategic value in reward function"
            )

        # Decision pattern recommendations
        capture_analysis = decision_analysis.get("capture_analysis", {})
        total_capture_ops = capture_analysis.get("taken", 0) + capture_analysis.get(
            "missed", 0
        )
        if total_capture_ops > 0:
            capture_rate = capture_analysis.get("taken", 0) / total_capture_ops
            if capture_rate < 0.7:
                recommendations.append(
                    "Model misses capture opportunities - increase capture rewards"
                )

        # Safety recommendations
        safety_prefs = decision_analysis.get("safety_preferences", {})
        total_safety = safety_prefs.get("safe", 0) + safety_prefs.get("risky", 0)
        if total_safety > 0:
            risk_rate = safety_prefs.get("risky", 0) / total_safety
            if risk_rate > 0.4:
                recommendations.append(
                    "Model takes too many risks - increase safety bonus"
                )

        # Performance recommendations
        if performance.get("accuracy", 0) < 0.5:
            recommendations.append(
                "Low accuracy - consider longer training or different network architecture"
            )

        if not recommendations:
            recommendations.append("Model performance looks good overall")

        return recommendations


class ModelComparator:
    """Compare multiple Ludo RL models."""

    def __init__(self, model_paths: List[str], model_names: List[str] = None):
        """
        Initialize model comparator.

        Args:
            model_paths: List of paths to trained models
            model_names: Optional names for models
        """
        self.model_paths = model_paths
        self.model_names = model_names or [
            f"Model_{i+1}" for i in range(len(model_paths))
        ]
        self.models = [RLPlayer(path) for path in model_paths]

    def compare_on_dataset(self, test_data: List[Dict]) -> Dict:
        """Compare models on a test dataset."""
        results = {}

        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            validator = LudoRLValidator(self.model_paths[i])

            # Get validation metrics
            expert_validation = validator.validate_against_expert_moves(test_data)
            decision_analysis = validator.analyze_decision_patterns(test_data)

            results[name] = {
                "expert_accuracy": expert_validation.get("exact_accuracy", 0),
                "strategic_alignment": expert_validation.get("strategic_alignment", 0),
                "capture_rate": self._calculate_capture_rate(decision_analysis),
                "safety_rate": self._calculate_safety_rate(decision_analysis),
                "avg_strategic_value": decision_analysis["strategic_value_stats"][
                    "mean"
                ],
            }

        return results

    def _calculate_capture_rate(self, analysis: Dict) -> float:
        """Calculate capture opportunity utilization rate."""
        capture_analysis = analysis.get("capture_analysis", {})
        total = capture_analysis.get("taken", 0) + capture_analysis.get("missed", 0)
        return capture_analysis.get("taken", 0) / max(total, 1)

    def _calculate_safety_rate(self, analysis: Dict) -> float:
        """Calculate safety preference rate."""
        safety_prefs = analysis.get("safety_preferences", {})
        total = safety_prefs.get("safe", 0) + safety_prefs.get("risky", 0)
        return safety_prefs.get("safe", 0) / max(total, 1)

    def visualize_comparison(
        self, comparison_results: Dict, save_path: str = "model_comparison.png"
    ):
        """Create comparison visualization."""
        metrics = [
            "expert_accuracy",
            "strategic_alignment",
            "capture_rate",
            "safety_rate",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            model_names = list(comparison_results.keys())
            values = [comparison_results[name][metric] for name in model_names]

            bars = axes[i].bar(model_names, values)
            axes[i].set_title(metric.replace("_", " ").title())
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis="x", rotation=45)

            # Color bars based on performance
            for j, bar in enumerate(bars):
                if values[j] > 0.8:
                    bar.set_color("green")
                elif values[j] > 0.6:
                    bar.set_color("orange")
                else:
                    bar.set_color("red")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Model comparison saved to {save_path}")
