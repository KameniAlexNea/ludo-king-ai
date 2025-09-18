"""Opponent simulation utilities for LudoGymEnv (single-seat training)."""

from typing import Any, List, Optional

from ludo_engine.core import LudoGame
from ludo_engine.models import GameConstants

from ..model import EnvConfig
from rl.rl_base.envs.builders.observation_builder import ObservationBuilder


class OpponentSimulator:
    """Handles simulation of opponent turns using frozen policy."""

    def __init__(
        self,
        cfg: EnvConfig,
        game: LudoGame,
        agent_color: str,
        frozen_policy: Optional[Any],
        obs_builder: ObservationBuilder,  # Reference to observation builder for temp obs
        policy_action_func,  # Reference to _policy_action method
        rng,  # Random number generator
    ):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self._frozen_policy = frozen_policy
        self.obs_builder = obs_builder
        self._policy_action = policy_action_func
        self.rng = rng

    def _simulate_single_opponent_turn(
        self, reward_components: Optional[List[float]] = None, turns: int = 0
    ):
        """Simulate exactly one opponent player sequence including any extra turns.

        If reward_components provided, apply capture penalties immediately.
        """
        current_player = self.game.get_current_player()
        if current_player.color.value == self.agent_color:
            return
        max_chain = GameConstants.MAX_OPPONENT_CHAIN_LENGTH  # Safety cap
        chain_count = 0
        while not self.game.game_over and chain_count < max_chain:
            dice = self.game.roll_dice()
            valid_moves = self.game.get_valid_moves(current_player, dice)
            if not valid_moves:
                self.game.next_turn()
                return
            # Build temp observation from agent's perspective
            temp_obs = self.obs_builder._build_observation(turns, dice)
            action = self._policy_action(self._frozen_policy, temp_obs, valid_moves)
            valid_ids = [m.token_id for m in valid_moves]
            if action not in valid_ids:
                action = valid_ids[0]
            move_res = self.game.execute_move(current_player, action, dice)
            # Check for agent captures and add penalties
            if reward_components and move_res.captured_tokens:
                agent_captured = False
                for ct in move_res.captured_tokens:
                    if ct.player_color == self.agent_color:
                        agent_captured = True
                        reward_components.append(self.cfg.reward_cfg.got_captured)
                if agent_captured:
                    player = self.game.get_player_from_color(self.agent_color)
                    all_captured = all(
                        t.position == GameConstants.HOME_POSITION for t in player.tokens
                    )
                    if all_captured:
                        reward_components.append(self.cfg.reward_cfg.all_tokens_killed)
            if not move_res.extra_turn or self.game.game_over:
                if not self.game.game_over:
                    self.game.next_turn()
                return
            chain_count += 1
        # Safety: force turn end if chain exceeded
        if not self.game.game_over and self.game.get_current_player() is current_player:
            self.game.next_turn()

    def _simulate_until_agent_turn(
        self, reward_components: Optional[List[float]] = None, turns: int = 0
    ):
        """Simulate opponents until it's the agent's turn or game over."""
        safety_counter = 0
        while (
            not self.game.game_over
            and self.game.get_current_player().color.value != self.agent_color
        ):
            safety_counter += 1
            if safety_counter > 5000:  # Hard safety
                break
            self._simulate_single_opponent_turn(reward_components, turns)
