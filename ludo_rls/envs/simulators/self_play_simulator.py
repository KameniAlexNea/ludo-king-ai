"""Opponent simulation utilities for LudoGymEnv."""

import random
from typing import Any, List, Optional

from ludo.constants import GameConstants
from ludo.game import LudoGame

from ..builders.observation_builder import ObservationBuilder
from ..model import EnvConfig


class SelfPlaySimulator:
    """Handles simulation of opponent turns."""

    def __init__(
        self,
        cfg: EnvConfig,
        game: LudoGame,
        agent_color: str,
        roll_dice_func,
        make_strategy_context_func,
        model: Any,
    ):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self._roll_dice = roll_dice_func
        self._make_strategy_context = make_strategy_context_func
        self.model = model

    def _simulate_single_player_turn(
        self, reward_components: Optional[List[float]] = None
    ):
        """Simulate exactly one player turn including any extra turns.

        All players use the same model for self-play.
        If reward_components provided, apply capture penalties immediately.
        """
        current_player = self.game.get_current_player()
        if current_player.color.value == self.agent_color:
            return
        max_chain = GameConstants.MAX_OPPONENT_CHAIN_LENGTH  # generous safety cap
        chain_count = 0
        while not self.game.game_over and chain_count < max_chain:
            dice_value = self._roll_dice()
            valid_moves = self.game.get_valid_moves(current_player, dice_value)
            if not valid_moves:
                # forfeited turn (e.g., 3 sixes or no moves)
                self.game.next_turn()
                return
            try:
                # All players use the same model for self-play
                opp_obs_builder = ObservationBuilder(self.cfg, self.game, current_player.color.value)
                obs = opp_obs_builder._build_observation(0, dice_value)
                action, _ = self.model.predict(obs, deterministic=False)
                valid_token_ids = [m["token_id"] for m in valid_moves]
                if action in valid_token_ids:
                    token_choice = action
                else:
                    token_choice = random.choice(valid_moves)["token_id"]
            except Exception:
                token_choice = valid_moves[0]["token_id"]
            move_res = self.game.execute_move(current_player, token_choice, dice_value)
            # Immediate capture penalty if this move captured agent tokens
            if reward_components and move_res.get("captured_tokens"):
                for ct in move_res["captured_tokens"]:
                    if ct.get("player_color") == self.agent_color:
                        reward_components.append(self.cfg.reward_cfg.got_captured)
            if not move_res.get("extra_turn") or self.game.game_over:
                if not self.game.game_over:
                    self.game.next_turn()
                return
            chain_count += 1
        # Safety: force turn end if chain exceeded
        if not self.game.game_over and self.game.get_current_player() is current_player:
            self.game.next_turn()

    def _simulate_other_players(self, reward_components: List[float]):
        # Simulate other players in order until agent's turn or game over.
        # Capture penalties are handled inside _simulate_single_player_turn now.
        while (
            not self.game.game_over
            and self.game.get_current_player().color.value != self.agent_color
        ):
            self._simulate_single_player_turn(reward_components)

    def _ensure_agent_turn(self):
        # Simulate other players until agent color is current player or game over
        while (
            not self.game.game_over
            and self.game.get_current_player().color.value != self.agent_color
        ):
            self._simulate_single_player_turn()
