"""Simple online Ludo environment for single-agent RL training.

The learning agent controls one color; opponents use a simple random (or priority) policy.
Only the learning agent's decisions generate experiences. Opponent turns are simulated
between the agent's turns.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from ludo import StrategyFactory
from ludo.game import LudoGame
from ludo.player import PlayerColor

from .states import LudoStateEncoder
from .config import REWARDS


class OnlineLudoEnv:
    def __init__(
        self,
        agent_color: str = "red",
        opponent_colors: Optional[List[str]] = None,
        opponent_strategies: Optional[List[str]] = None,
        add_noise: bool = False,
        max_turns: int = 500,
    ):
        if opponent_colors is None:
            opponent_colors = [
                c for c in ["green", "yellow", "blue"] if c != agent_color
            ]
        colors = [agent_color] + opponent_colors
        # Map to PlayerColor enum
        enum_colors = [PlayerColor(c) for c in colors]
        self.game = LudoGame(enum_colors)
        self.agent_color = agent_color
        self.encoder = LudoStateEncoder(add_noise=add_noise)
        self.max_turns = max_turns
        self.turns_elapsed = 0
        # Choose opponent strategies (exclude any LLM strategies for speed)
        allowed = [
            s
            for s in StrategyFactory.get_available_strategies()
            if "llm" not in s.lower()
        ]
        if opponent_strategies:
            self.opponent_strategy_names = [
                s for s in opponent_strategies if s in allowed
            ]
        else:
            # Randomly assign (with replacement) strategies to each opponent
            self.opponent_strategy_names = [
                random.choice(allowed) for _ in opponent_colors
            ]
        # Keep original opponent colors for potential reshuffling logic
        self.opponent_colors = opponent_colors
        self._assign_opponent_strategies()

    def reset(self) -> np.ndarray:
        # Re-create the game to fully reset
        enum_colors = [PlayerColor(self.agent_color)] + [
            PlayerColor(c) for c in ["green", "yellow", "blue"] if c != self.agent_color
        ]
        self.game = LudoGame(enum_colors)
        self.turns_elapsed = 0
        # Shuffle strategy assignment order each new game for variability
        if (
            hasattr(self, "opponent_strategy_names")
            and len(self.opponent_strategy_names) > 1
        ):
            random.shuffle(self.opponent_strategy_names)
        self._assign_opponent_strategies()
        return self._get_state()

    def _assign_opponent_strategies(self):
        """Assign chosen strategies to all opponent players (agent stays with RL)."""
        non_agent_players = [
            p for p in self.game.players if p.color.value != self.agent_color
        ]
        for idx, p in enumerate(non_agent_players):
            if not self.opponent_strategy_names:
                continue
            strat_name = self.opponent_strategy_names[
                min(idx, len(self.opponent_strategy_names) - 1)
            ]
            if "llm" in strat_name.lower():  # Safety filter
                continue
            impl = StrategyFactory.create_strategy(strat_name)
            p.set_strategy(impl)
            p.strategy_name = strat_name

    def _build_encoder_input(self, dice_value: int) -> Dict:
        # current_player = self.game.get_current_player()
        context = self.game.get_ai_decision_context(dice_value)
        # Adapt to encoder expected structure
        game_context = {
            "current_situation": context["current_situation"],
            "player_state": context["player_state"],
            "opponents": context["opponents"],
            "valid_moves": context["valid_moves"],
            "strategic_analysis": context["strategic_analysis"],
        }
        return {"game_context": game_context}

    def _get_state(self) -> np.ndarray:
        # Roll a dice only to obtain context? Dice is rolled each step externally
        # During state request we simulate a dice to present available moves.
        # For consistency we roll but store it for next action selection.
        return (
            self.last_state
            if hasattr(self, "last_state")
            else np.zeros(self.encoder.state_dim)
        )

    def get_current_valid_moves(self) -> List[Dict]:
        dice = self.current_dice if hasattr(self, "current_dice") else 1
        current_player = self.game.get_current_player()
        return self.game.get_valid_moves(current_player, dice)

    def agent_turn_prepare(self):
        # Roll dice for agent
        dice = self.game.roll_dice()
        self.current_dice = dice
        enc_input = self._build_encoder_input(dice)
        state = self.encoder.encode_state(enc_input)
        self.last_state = state
        return state, self.get_current_valid_moves()

    def _simulate_opponents_until_agent(self):
        # Advance turns until it's agent's color or game over
        safety_counter = 0
        while (
            self.game.get_current_player().color.value != self.agent_color
            and not self.game.game_over
            and safety_counter < 1000
        ):
            dice = self.game.roll_dice()
            current_player = self.game.get_current_player()
            context = self.game.get_ai_decision_context(dice)
            moves = context["valid_moves"]
            if moves:
                if hasattr(current_player, "make_strategic_decision"):
                    chosen_token = current_player.make_strategic_decision(context)
                else:
                    chosen_token = random.choice(moves)["token_id"]
                move_result = self.game.execute_move(current_player, chosen_token, dice)
                extra = move_result.get("extra_turn", False)
            else:
                extra = False
            if not extra:
                self.game.next_turn()
            safety_counter += 1
        if safety_counter >= 1000:
            self.game.game_over = True

    def step(self, action_move_index: int) -> Tuple[np.ndarray, float, bool]:
        if self.game.game_over:
            return self.last_state, 0.0, True
        current_player = self.game.get_current_player()
        if current_player.color.value != self.agent_color:
            # Shouldn't happen; simulate opponents
            self._simulate_opponents_until_agent()
        # Ensure dice and state prepared
        if not hasattr(self, "current_dice"):
            self.agent_turn_prepare()
        valid_moves = self.get_current_valid_moves()
        reward = 0.0
        done = False
        if not valid_moves:
            reward = -1.0
            self.game.next_turn()
            self._simulate_opponents_until_agent()
        else:
            # Clamp index
            idx = max(0, min(action_move_index, len(valid_moves) - 1))
            chosen = valid_moves[idx]
            token_id = chosen["token_id"]
            move_result = self.game.execute_move(
                current_player, token_id, self.current_dice
            )
            reward = self._compute_reward(move_result)
            # Extra turn: prepare next dice without switching players
            if move_result.get("extra_turn") and not self.game.game_over:
                # Prepare next state for same player
                next_state, _ = self.agent_turn_prepare()
                self.turns_elapsed += 1
                done = self.game.game_over or self.turns_elapsed >= self.max_turns
                return next_state, reward, done
            else:
                if not self.game.game_over:
                    self.game.next_turn()
                    self._simulate_opponents_until_agent()
        # Prepare next agent state if game not over
        self.turns_elapsed += 1
        if self.game.game_over or self.turns_elapsed >= self.max_turns:
            done = True
            next_state = np.zeros(self.encoder.state_dim)
        else:
            next_state, _ = self.agent_turn_prepare()
        return next_state, reward, done

    def _compute_reward(self, move_result: Dict) -> float:
        """Compute scaled reward using REWARDS config (divided by 10 for stability)."""
        SCALE = 0.1
        if not move_result.get("success", False):
            return REWARDS.FAILS * SCALE
        r = REWARDS.SUCCESS * SCALE
        if move_result.get("captured_tokens"):
            r += REWARDS.CAPTURE * SCALE * len(move_result["captured_tokens"])
        if move_result.get("token_finished"):
            r += REWARDS.TOKEN_FINISHED * SCALE
        if move_result.get("extra_turn"):
            r += REWARDS.EXTRA_TURN * SCALE
        # Progress reward (distance advanced along track)
        old_pos = move_result.get("old_position", -1)
        new_pos = move_result.get("new_position", -1)
        if old_pos != -1 and new_pos != -1 and new_pos != old_pos:
            delta = new_pos - old_pos
            if delta < 0:
                delta += 52
            # Scale by configured progress weight
            r += (delta / 52.0) * REWARDS.PROGRESS_WEIGHT * SCALE
        if move_result.get("game_won"):
            r += REWARDS.WON * SCALE
        return r
