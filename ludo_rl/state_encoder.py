"""
Improved state encoder for converting Ludo game states to numerical vectors
suitable for reinforcement learning training.
"""

from typing import Dict, List

import numpy as np


class LudoStateEncoder:
    """Converts Ludo game states to compact, fixed-length numerical vectors."""

    def __init__(self):
        """
        Initialize the improved state encoder with compact representation.

        State structure (64 features total):
        - Own tokens: 16 features (4 tokens × 4 features each)
        - Game context: 8 features
        - Opponent summary: 12 features (3 opponents × 4 features)
        - Strategic context: 16 features
        - Valid moves summary: 12 features
        """
        self.state_dim = 64  # Fixed, compact size
        self.board_size = 52
        self.max_tokens = 4
        self.num_players = 4

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
        Convert game state to compact numerical vector.

        Args:
            game_data: Game state data with structure from GameStateSaver

        Returns:
            np.ndarray: Fixed-length state vector (64 features)
        """
        state = np.zeros(self.state_dim)

        context = game_data["game_context"]
        player_state = context["player_state"]
        current_situation = context["current_situation"]
        opponents = context.get("opponents", [])
        valid_moves = context.get("valid_moves", [])
        strategic_analysis = context.get("strategic_analysis", {})

        # Own tokens (16 features: 4 tokens × 4 features each)
        tokens = player_state.get("tokens", [])
        for i in range(4):  # Always process 4 token slots
            base_idx = i * 4

            if i < len(tokens):
                token = tokens[i]
                pos = token.get("position", -1)

                # Normalized position
                if pos == -1:
                    state[base_idx] = -1.0  # Home
                elif token.get("is_finished", False):
                    state[base_idx] = 1.0  # Finished
                else:
                    state[base_idx] = pos / 52.0  # Board position (normalized)

                state[base_idx + 1] = 1.0 if token.get("is_active", False) else 0.0
                state[base_idx + 2] = (
                    1.0 if token.get("is_in_home_column", False) else 0.0
                )
                state[base_idx + 3] = self._calculate_token_safety(pos, context)
            # If no token data, features remain 0.0

        # Game context (8 features)
        state[16] = current_situation.get("dice_value", 1) / 6.0
        state[17] = min(current_situation.get("consecutive_sixes", 0), 3) / 3.0
        state[18] = min(current_situation.get("turn_count", 0), 100) / 100.0

        # Player progress metrics
        active_tokens = len([t for t in tokens if t.get("is_active", False)])
        finished_tokens = len([t for t in tokens if t.get("is_finished", False)])
        state[19] = active_tokens / 4.0
        state[20] = finished_tokens / 4.0
        state[21] = player_state.get("tokens_in_home", 0) / 4.0
        state[22] = 1.0 if player_state.get("has_won", False) else 0.0

        # Game phase indicator
        total_progress = (
            active_tokens + finished_tokens * 2
        ) / 8.0  # Weighted progress
        state[23] = total_progress

        # Opponent summary (12 features: 3 opponents × 4 features)
        for i in range(3):  # Always process 3 opponent slots
            base_idx = 24 + i * 4

            if i < len(opponents):
                opp = opponents[i]
                state[base_idx] = opp.get("tokens_active", 0) / 4.0
                state[base_idx + 1] = opp.get("tokens_finished", 0) / 4.0
                state[base_idx + 2] = opp.get("threat_level", 0.0)
                state[base_idx + 3] = 1.0 if opp.get("tokens_finished", 0) == 4 else 0.0
            # If no opponent data, features remain 0.0

        # Strategic context (16 features)
        state[36] = 1.0 if strategic_analysis.get("can_capture", False) else 0.0
        state[37] = 1.0 if strategic_analysis.get("can_finish_token", False) else 0.0
        state[38] = 1.0 if strategic_analysis.get("can_exit_home", False) else 0.0
        state[39] = len(strategic_analysis.get("safe_moves", [])) / 4.0
        state[40] = len(strategic_analysis.get("risky_moves", [])) / 4.0
        state[41] = strategic_analysis.get("best_strategic_value", 0.0) / 30.0

        # Tactical features
        state[42] = (
            1.0 if any(m.get("captures_opponent", False) for m in valid_moves) else 0.0
        )
        state[43] = (
            1.0 if any(m.get("move_type") == "finish" for m in valid_moves) else 0.0
        )
        state[44] = 1.0 if current_situation.get("dice_value", 1) == 6 else 0.0
        state[45] = len([m for m in valid_moves if m.get("is_safe_move", True)]) / max(
            len(valid_moves), 1
        )

        # Position analysis
        home_tokens = len([t for t in tokens if t.get("position", -1) == -1])
        board_tokens = len(
            [
                t
                for t in tokens
                if t.get("position", -1) >= 0 and not t.get("is_finished", False)
            ]
        )
        state[46] = home_tokens / 4.0
        state[47] = board_tokens / 4.0

        # Risk assessment
        vulnerable_tokens = self._count_vulnerable_tokens(tokens, context)
        state[48] = vulnerable_tokens / 4.0

        # Strategic positioning
        advanced_tokens = len(
            [t for t in tokens if t.get("position", -1) > 26]
        )  # Past halfway
        state[49] = advanced_tokens / 4.0

        # Home column advantage
        home_column_tokens = len(
            [t for t in tokens if t.get("is_in_home_column", False)]
        )
        state[50] = home_column_tokens / 4.0

        # Relative position to opponents
        state[51] = self._calculate_relative_position_advantage(tokens, opponents)

        # Valid moves summary (12 features)
        state[52] = len(valid_moves) / 4.0

        if valid_moves:
            # Strategic value statistics
            strategic_values = [m.get("strategic_value", 0) for m in valid_moves]
            state[53] = np.mean(strategic_values) / 30.0
            state[54] = np.max(strategic_values) / 30.0

            # Move type distribution
            state[55] = sum(
                1 for m in valid_moves if m.get("move_type") == "exit_home"
            ) / len(valid_moves)
            state[56] = sum(
                1 for m in valid_moves if m.get("captures_opponent", False)
            ) / len(valid_moves)
            state[57] = sum(
                1 for m in valid_moves if m.get("is_safe_move", True)
            ) / len(valid_moves)
            state[58] = sum(
                1 for m in valid_moves if m.get("move_type") == "finish"
            ) / len(valid_moves)

            # Best move indicator
            best_move = strategic_analysis.get("best_strategic_move", {})
            if best_move:
                best_token_id = best_move.get("token_id", -1)
                state[59] = (
                    1.0
                    if any(m.get("token_id") == best_token_id for m in valid_moves)
                    else 0.0
                )

            # Move diversity
            unique_tokens = len(set(m.get("token_id", -1) for m in valid_moves))
            state[60] = unique_tokens / 4.0

            # Risk distribution
            risky_moves = sum(1 for m in valid_moves if not m.get("is_safe_move", True))
            state[61] = risky_moves / len(valid_moves)

            # Progress potential
            progress_potential = sum(
                m.get("strategic_value", 0) > 0 for m in valid_moves
            )
            state[62] = progress_potential / len(valid_moves)

        # Overall game state
        state[63] = self._calculate_overall_game_state_score(
            player_state, opponents, current_situation
        )

        return state

    def _calculate_token_safety(self, position: int, context: Dict) -> float:
        """Calculate safety score for a token position."""
        if position == -1 or position >= 52:  # Home or finished
            return 1.0

        # Check if position is in safe zone (start positions and some special positions)
        safe_positions = {0, 8, 13, 21, 26, 34, 39, 47}  # Common safe positions in Ludo
        if position in safe_positions:
            return 0.8

        # Check if opponents can capture (simplified heuristic)
        opponents = context.get("opponents", [])
        threat_level = sum(opp.get("threat_level", 0.0) for opp in opponents)
        safety = max(0.0, 1.0 - threat_level / 3.0)  # Normalize threat level

        return safety

    def _count_vulnerable_tokens(self, tokens: List[Dict], context: Dict) -> int:
        """Count tokens that are vulnerable to capture."""
        vulnerable = 0
        for token in tokens:
            pos = token.get("position", -1)
            if pos >= 0 and pos < 52:  # On board
                safety = self._calculate_token_safety(pos, context)
                if safety < 0.5:  # Consider unsafe if safety < 0.5
                    vulnerable += 1
        return vulnerable

    def _calculate_relative_position_advantage(
        self, own_tokens: List[Dict], opponents: List[Dict]
    ) -> float:
        """Calculate relative positional advantage compared to opponents."""
        # Simple heuristic: compare average progress
        own_progress = 0
        active_own = 0

        for token in own_tokens:
            if token.get("is_finished", False):
                own_progress += 52
                active_own += 1
            elif token.get("position", -1) >= 0:
                own_progress += token.get("position", 0)
                active_own += 1

        own_avg = own_progress / max(active_own, 1)

        # Estimate opponent progress
        opp_progress = 0
        total_opp_active = 0

        for opp in opponents:
            active = opp.get("tokens_active", 0)
            finished = opp.get("tokens_finished", 0)
            # Estimate average position (crude heuristic)
            estimated_progress = finished * 52 + active * 26  # Assume average position
            opp_progress += estimated_progress
            total_opp_active += active + finished

        opp_avg = opp_progress / max(total_opp_active, 1)

        # Return normalized advantage (-1 to 1)
        if opp_avg == 0:
            return 1.0 if own_avg > 0 else 0.0

        advantage = (own_avg - opp_avg) / 52.0
        return max(-1.0, min(1.0, advantage))

    def _calculate_overall_game_state_score(
        self, player_state: Dict, opponents: List[Dict], situation: Dict
    ) -> float:
        """Calculate overall game state score."""
        score = 0.0

        # Own progress weight
        finished = player_state.get("finished_tokens", 0)
        active = player_state.get("active_tokens", 0)
        score += finished * 0.4 + active * 0.1

        # Opponent threat assessment
        max_opp_finished = max(
            (opp.get("tokens_finished", 0) for opp in opponents), default=0
        )
        score -= max_opp_finished * 0.3

        # Turn momentum (consecutive sixes give advantage)
        sixes = situation.get("consecutive_sixes", 0)
        score += min(sixes, 2) * 0.1

        # Normalize to 0-1 range
        return max(0.0, min(1.0, score))

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
