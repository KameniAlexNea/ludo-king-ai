"""
Advanced state encoder for converting Ludo game states to numerical vectors
suitable for reinforcement learning training with enhanced robustness and features.
"""

from typing import Dict, List, Optional

import numpy as np


class LudoStateEncoder:
    """Advanced state encoder with enhanced robustness and feature engineering."""

    def __init__(self, add_noise: bool = False):
        """
        Initialize the advanced state encoder.

        Args:
            add_noise: Whether to add regularization noise during training

        State structure (64 features total):
        - Own tokens: 16 features (4 tokens × 4 features each)
        - Game context: 8 features
        - Opponent summary: 12 features (3 opponents × 4 features)
        - Strategic context: 16 features
        - Valid moves summary: 12 features
        """
        self.state_dim = 64
        self.board_size = 52
        self.max_tokens = 4
        self.num_players = 4
        self.add_noise = add_noise

        # Normalization constants for better training stability
        self.normalization_constants = {
            "position_scale": 52.0,
            "strategic_value_scale": 30.0,
            "turn_scale": 200.0,
            "threat_scale": 1.0,
        }

    def _calculate_flexible_state_dim(self):
        """Calculate state dimension for the improved compact representation."""
        own_tokens = 4 * 4  # position, is_active, is_in_home_column, safety
        game_context = 8  # dice, sixes, turn, phase, active_tokens, progress, etc.
        opponents = 3 * 4  # summarized opponent info per opponent
        strategic = 16  # strategic analysis features
        moves = 12  # valid moves summary features
        return own_tokens + game_context + opponents + strategic + moves

    def encode_state(self, game_data: Dict) -> np.ndarray:
        """
        Enhanced state encoding with better error handling and normalization.

        Args:
            game_data: Game state data with structure from GameStateSaver

        Returns:
            np.ndarray: Fixed-length state vector (64 features)
        """
        state = np.zeros(self.state_dim)

        context = game_data.get("game_context", {})
        if not context:
            return self._get_default_state()

        player_state = context.get("player_state", {})
        current_situation = context.get("current_situation", {})
        opponents = context.get("opponents", [])
        valid_moves = context.get("valid_moves", [])
        strategic_analysis = context.get("strategic_analysis", {})

        # Enhanced encoding with modular approach
        idx = self._encode_own_tokens_enhanced(state, 0, player_state, context)
        idx = self._encode_game_context_enhanced(
            state, idx, current_situation, player_state
        )
        idx = self._encode_opponents_enhanced(state, idx, opponents)
        idx = self._encode_strategic_context_enhanced(
            state, idx, strategic_analysis, valid_moves, current_situation
        )
        idx = self._encode_moves_summary_enhanced(
            state, idx, valid_moves, strategic_analysis
        )

        # Add state validation and normalization
        state = self._validate_and_normalize_state(state)

        return state

    def _encode_own_tokens_enhanced(self, state, start_idx, player_state, context):
        """Enhanced token encoding with relative positioning."""
        tokens = player_state.get("tokens", [])
        current_player_color = context.get("current_situation", {}).get(
            "player_color", "red"
        )
        start_position = self._get_start_position(current_player_color)

        for i in range(4):
            base_idx = start_idx + i * 4

            if i < len(tokens):
                token = tokens[i]
                pos = token.get("position", -1)

                # Enhanced position encoding (relative to player's start)
                if pos == -1:
                    state[base_idx] = -1.0  # Home
                elif token.get("is_finished", False):
                    state[base_idx] = 2.0  # Finished (distinct from board positions)
                else:
                    # Relative position from start
                    relative_pos = (pos - start_position) % 52
                    state[base_idx] = relative_pos / 51.0  # Normalize 0-1

                # Token state features
                state[base_idx + 1] = 1.0 if token.get("is_active", False) else 0.0
                state[base_idx + 2] = (
                    1.0 if token.get("is_in_home_column", False) else 0.0
                )

                # Enhanced safety calculation
                state[base_idx + 3] = self._calculate_enhanced_token_safety(
                    pos, context, token
                )

        return start_idx + 16

    def _encode_game_context_enhanced(
        self, state, start_idx, current_situation, player_state
    ):
        """Enhanced game context with momentum indicators."""
        # Basic game state
        state[start_idx] = current_situation.get("dice_value", 1) / 6.0
        state[start_idx + 1] = (
            min(current_situation.get("consecutive_sixes", 0), 3) / 3.0
        )

        # Normalized turn with logarithmic scaling for very long games
        turn_count = current_situation.get("turn_count", 0)
        state[start_idx + 2] = min(
            np.log1p(turn_count) / np.log1p(self.normalization_constants["turn_scale"]),
            1.0,
        )

        # Player progress metrics
        tokens = player_state.get("tokens", [])
        state[start_idx + 3] = (
            len([t for t in tokens if t.get("is_active", False)]) / 4.0
        )
        state[start_idx + 4] = (
            len([t for t in tokens if t.get("is_finished", False)]) / 4.0
        )
        state[start_idx + 5] = player_state.get("tokens_in_home", 0) / 4.0
        state[start_idx + 6] = 1.0 if player_state.get("has_won", False) else 0.0

        # Game phase and momentum
        state[start_idx + 7] = self._calculate_game_phase(
            player_state, current_situation
        )

        return start_idx + 8

    def _encode_opponents_enhanced(self, state, start_idx, opponents):
        """Enhanced opponent encoding with threat assessment."""
        for i in range(3):  # Always process 3 opponent slots
            base_idx = start_idx + i * 4

            if i < len(opponents):
                opp = opponents[i]
                state[base_idx] = opp.get("tokens_active", 0) / 4.0
                state[base_idx + 1] = opp.get("tokens_finished", 0) / 4.0
                state[base_idx + 2] = min(
                    opp.get("threat_level", 0.0), 1.0
                )  # Clamped threat
                state[base_idx + 3] = (
                    1.0 if opp.get("tokens_finished", 0) >= 4 else 0.0
                )  # Won
            # If no opponent data, features remain 0.0

        return start_idx + 12

    def _encode_strategic_context_enhanced(
        self, state, start_idx, strategic_analysis, valid_moves, current_situation
    ):
        """Enhanced strategic features with better tactical awareness."""
        # Basic strategic flags
        state[start_idx] = 1.0 if strategic_analysis.get("can_capture", False) else 0.0
        state[start_idx + 1] = (
            1.0 if strategic_analysis.get("can_finish_token", False) else 0.0
        )
        state[start_idx + 2] = (
            1.0 if strategic_analysis.get("can_exit_home", False) else 0.0
        )

        # Move quality metrics
        safe_moves = strategic_analysis.get("safe_moves", [])
        risky_moves = strategic_analysis.get("risky_moves", [])
        total_moves = len(valid_moves)

        state[start_idx + 3] = len(safe_moves) / max(total_moves, 1)
        state[start_idx + 4] = len(risky_moves) / max(total_moves, 1)

        # Strategic value analysis
        if valid_moves:
            strategic_values = [m.get("strategic_value", 0) for m in valid_moves]
            state[start_idx + 5] = (
                np.mean(strategic_values)
                / self.normalization_constants["strategic_value_scale"]
            )
            state[start_idx + 6] = (
                np.max(strategic_values)
                / self.normalization_constants["strategic_value_scale"]
            )
            state[start_idx + 7] = (
                np.std(strategic_values)
                / self.normalization_constants["strategic_value_scale"]
            )  # Value diversity

        # Tactical opportunities
        state[start_idx + 8] = (
            1.0 if any(m.get("captures_opponent", False) for m in valid_moves) else 0.0
        )
        state[start_idx + 9] = (
            1.0 if any(m.get("move_type") == "finish" for m in valid_moves) else 0.0
        )
        state[start_idx + 10] = (
            1.0 if current_situation.get("dice_value", 1) == 6 else 0.0
        )

        # Move distribution analysis
        if valid_moves:
            move_types = [m.get("move_type", "") for m in valid_moves]
            state[start_idx + 11] = move_types.count("exit_home") / len(valid_moves)
            state[start_idx + 12] = move_types.count("advance_main_board") / len(
                valid_moves
            )
            state[start_idx + 13] = move_types.count("enter_home_column") / len(
                valid_moves
            )

            # Token diversity in moves
            unique_tokens = len(set(m.get("token_id", -1) for m in valid_moves))
            state[start_idx + 14] = unique_tokens / 4.0

        # Decision pressure (fewer good options = higher pressure)
        good_moves = len([m for m in valid_moves if m.get("strategic_value", 0) > 10])
        state[start_idx + 15] = 1.0 - (good_moves / max(len(valid_moves), 1))

        return start_idx + 16

    def _encode_moves_summary_enhanced(
        self, state, start_idx, valid_moves, strategic_analysis
    ):
        """Enhanced move summary with comprehensive analysis."""
        state[start_idx] = len(valid_moves) / 4.0

        if valid_moves:
            # Strategic value statistics
            strategic_values = [m.get("strategic_value", 0) for m in valid_moves]
            state[start_idx + 1] = (
                np.mean(strategic_values)
                / self.normalization_constants["strategic_value_scale"]
            )
            state[start_idx + 2] = (
                np.max(strategic_values)
                / self.normalization_constants["strategic_value_scale"]
            )

            # Move type distribution
            state[start_idx + 3] = sum(
                1 for m in valid_moves if m.get("move_type") == "exit_home"
            ) / len(valid_moves)
            state[start_idx + 4] = sum(
                1 for m in valid_moves if m.get("captures_opponent", False)
            ) / len(valid_moves)
            state[start_idx + 5] = sum(
                1 for m in valid_moves if m.get("is_safe_move", True)
            ) / len(valid_moves)
            state[start_idx + 6] = sum(
                1 for m in valid_moves if m.get("move_type") == "finish"
            ) / len(valid_moves)

            # Best move indicator
            best_move = strategic_analysis.get("best_strategic_move", {})
            if best_move:
                best_token_id = best_move.get("token_id", -1)
                state[start_idx + 7] = (
                    1.0
                    if any(m.get("token_id") == best_token_id for m in valid_moves)
                    else 0.0
                )

            # Move diversity and risk analysis
            unique_tokens = len(set(m.get("token_id", -1) for m in valid_moves))
            state[start_idx + 8] = unique_tokens / 4.0

            risky_moves = sum(1 for m in valid_moves if not m.get("is_safe_move", True))
            state[start_idx + 9] = risky_moves / len(valid_moves)

            # Progress potential
            progress_potential = sum(
                m.get("strategic_value", 0) > 0 for m in valid_moves
            )
            state[start_idx + 10] = progress_potential / len(valid_moves)

            # Move quality variance (decision difficulty)
            if len(strategic_values) > 1:
                state[start_idx + 11] = np.var(strategic_values) / (
                    self.normalization_constants["strategic_value_scale"] ** 2
                )

        return start_idx + 12

    def _calculate_enhanced_token_safety(
        self, position: int, context: Dict, token: Dict
    ) -> float:
        """Enhanced safety calculation considering multiple factors."""
        if position == -1:  # Home
            return 1.0
        if token.get("is_finished", False):  # Finished
            return 1.0
        if token.get("is_in_home_column", False):  # Home column is safer
            return 0.9

        # Check safe positions (start positions)
        safe_positions = {1, 14, 27, 40}  # Start positions for 4 players
        if position in safe_positions:
            return 0.8

        # Calculate threat from opponents
        opponents = context.get("opponents", [])
        threat_score = 0.0

        for opp in opponents:
            # Simplified threat calculation - in real game you'd know opponent positions
            active_tokens = opp.get("tokens_active", 0)
            threat_level = opp.get("threat_level", 0.0)

            # Higher threat if opponents have many active tokens near our position
            if active_tokens > 0:
                threat_score += threat_level * (active_tokens / 4.0)

        # Normalize threat and convert to safety
        max_threat = len(opponents) * 1.0  # Maximum possible threat
        normalized_threat = min(threat_score / max(max_threat, 0.1), 1.0)
        safety = 1.0 - normalized_threat

        # Add position-based safety (middle positions are generally less safe)
        position_safety = 1.0 - abs(position - 26) / 26.0  # Safer near home or finish

        return safety * 0.7 + position_safety * 0.3

    def _calculate_game_phase(
        self, player_state: Dict, current_situation: Dict
    ) -> float:
        """Calculate current game phase (0=early, 1=endgame)."""
        finished = player_state.get("finished_tokens", 0)
        active = player_state.get("active_tokens", 0)
        home = player_state.get("tokens_in_home", 0)

        # Phase based on token distribution
        if finished >= 2:  # Endgame
            return 0.8 + (finished - 2) * 0.1
        elif active >= 3:  # Mid game
            return 0.4 + active * 0.1
        elif home == 4:  # Early game
            return 0.1
        else:  # Early-mid transition
            return 0.2 + active * 0.05

    def _validate_and_normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Validate and normalize state vector."""
        # Replace any NaN or inf values
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        # Ensure all values are in reasonable range
        state = np.clip(state, -2.0, 2.0)

        # Optional: add noise for regularization during training
        if self.add_noise:
            noise = np.random.normal(0, 0.001, state.shape)
            state += noise

        return state

    def _get_default_state(self) -> np.ndarray:
        """Return a safe default state for error cases."""
        return np.zeros(self.state_dim)

    def _get_start_position(self, color: str) -> int:
        """Get start position for a player color."""
        start_positions = {"red": 1, "green": 14, "yellow": 27, "blue": 40}
        return start_positions.get(color.lower(), 1)

    def encode_state_with_attention(self, game_data: Dict) -> np.ndarray:
        """Encode state with attention weights for important features."""
        base_state = self.encode_state(game_data)

        # Calculate attention weights based on game situation
        attention_weights = self._calculate_attention_weights(game_data)

        # Apply attention (element-wise multiplication)
        attended_state = base_state * attention_weights

        # Concatenate original and attended features for richer representation
        return np.concatenate([base_state, attended_state])

    def _calculate_attention_weights(self, game_data: Dict) -> np.ndarray:
        """Calculate attention weights based on game situation."""
        weights = np.ones(self.state_dim)

        context = game_data.get("game_context", {})
        current_situation = context.get("current_situation", {})

        # Increase attention on strategic features during critical moments
        if current_situation.get("dice_value") == 6:
            weights[36:52] *= 1.5  # Boost strategic context features

        # Increase attention on opponent features when they're close to winning
        opponents = context.get("opponents", [])
        max_finished = max(
            (opp.get("tokens_finished", 0) for opp in opponents), default=0
        )
        if max_finished >= 3:  # Opponent close to winning
            weights[24:36] *= 1.3  # Boost opponent features

        # Increase attention on move features when many options available
        valid_moves = context.get("valid_moves", [])
        if len(valid_moves) >= 3:
            weights[52:64] *= 1.2  # Boost move analysis features

        return weights

    def debug_state_encoding(self, game_data: Dict) -> Dict:
        """Debug helper to understand state encoding."""
        state = self.encode_state(game_data)

        debug_info = {
            "state_shape": state.shape,
            "state_range": (float(state.min()), float(state.max())),
            "non_zero_features": int(np.count_nonzero(state)),
            "feature_breakdown": {
                "own_tokens": state[0:16].tolist(),
                "game_context": state[16:24].tolist(),
                "opponents": state[24:36].tolist(),
                "strategic": state[36:52].tolist(),
                "moves": state[52:64].tolist(),
            },
            "feature_statistics": {
                "mean": float(state.mean()),
                "std": float(state.std()),
                "zero_features": int(np.sum(state == 0)),
                "negative_features": int(np.sum(state < 0)),
                "positive_features": int(np.sum(state > 0)),
            },
        }

        return debug_info

    def visualize_state(self, state: np.ndarray, save_path: Optional[str] = None):
        """Visualize state encoding for debugging."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            # Plot different feature groups
            feature_groups = [
                ("Own Tokens", state[0:16]),
                ("Game Context", state[16:24]),
                ("Opponents", state[24:36]),
                ("Strategic", state[36:52]),
                ("Moves", state[52:64]),
                ("Full State", state),
            ]

            for idx, (title, features) in enumerate(feature_groups):
                ax = axes[idx]
                bars = ax.bar(range(len(features)), features)
                ax.set_title(title)
                ax.set_ylim(-2, 2)
                ax.grid(True, alpha=0.3)

                # Color code bars
                for i, bar in enumerate(bars):
                    if features[i] > 0:
                        bar.set_color("green")
                    elif features[i] < 0:
                        bar.set_color("red")
                    else:
                        bar.set_color("gray")

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"State visualization saved to {save_path}")
            plt.show()

        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Error in visualization: {e}")

    def encode_action(self, chosen_move: int, valid_moves: List[Dict]) -> int:
        """
        Convert chosen move to action index.

        Args:
            chosen_move: Index of chosen move in valid_moves list
            valid_moves: List of valid moves

        Returns:
            int: Action index (move index, not token ID)
        """
        if 0 <= chosen_move < len(valid_moves):
            return chosen_move
        return 0  # Default to first move

    def decode_action(self, action_idx: int, valid_moves: List[Dict]) -> Dict:
        """
        Convert action index back to move information.

        Args:
            action_idx: Action index
            valid_moves: List of valid moves

        Returns:
            Dict: Move information
        """
        if 0 <= action_idx < len(valid_moves):
            return valid_moves[action_idx]
        return valid_moves[0] if valid_moves else {}
