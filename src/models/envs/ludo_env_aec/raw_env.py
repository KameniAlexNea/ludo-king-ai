"""PettingZoo AEC environment wrapper for multi-agent Ludo training."""

from __future__ import annotations

import copy
import functools
from typing import Dict, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle
from ludo_engine.core import LudoGame, Player
from ludo_engine.models import (
    ALL_COLORS,
    GameConstants,
    MoveResult,
    PlayerColor,
    ValidMove,
)
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector

from ...configs.config import EnvConfig
from ..observation.factory import make_observation_builder
from ..reward import AdvancedRewardCalculator
from ..spaces import get_flat_space_config


def _make_mask(valid_moves: Optional[list[ValidMove]]) -> np.ndarray:
    """Create action mask from valid moves.

    Returns int8 array for compatibility with observation space definition.
    In Ludo, when there are no valid moves (e.g., didn't roll 6 to start),
    all actions are marked as valid since any action results in passing the turn.
    """
    mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
    if valid_moves:
        for move in valid_moves:
            mask[move.token_id] = 1
    else:
        mask[:] = 1
    return mask


def env(cfg: Optional[EnvConfig] = None):
    """Factory function for creating wrapped environment (standard name)."""
    aec_env = raw_env(cfg)
    aec_env = wrappers.TerminateIllegalWrapper(aec_env, illegal_reward=-1)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


class raw_env(AECEnv, EzPickle):
    """PettingZoo AEC environment for 4-player Ludo with turn-based play."""

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "LudoMultiAgent-v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(
        self, cfg: Optional[EnvConfig] = None, render_mode: Optional[str] = None
    ):
        EzPickle.__init__(self, cfg, render_mode)
        super().__init__()
        self.cfg = cfg or EnvConfig()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.possible_agents = ["player_0", "player_1", "player_2", "player_3"]

        self.game: Optional[LudoGame] = None
        self._agent_color_map: Dict[str, PlayerColor] = {}
        self._color_agent_map: Dict[PlayerColor, str] = {}
        self._obs_builders: Dict[str, object] = {}
        self.turn_count = 0

        self._pending_dice: Dict[str, int] = {}
        self._pending_valid_moves: Dict[str, list[ValidMove]] = {}
        self._action_masks: Dict[str, np.ndarray] = {}

        self.reward_calc = AdvancedRewardCalculator()
        self._last_move_results: Dict[str, MoveResult] = {}
        self._opponent_captures: Dict[str, int] = {}

        tokens = GameConstants.TOKENS_PER_PLAYER
        observation_space = get_flat_space_config()
        action_space = spaces.Discrete(tokens)

        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        self.action_spaces = {agent: action_space for agent in self.possible_agents}

    # -----------------------------
    # Public accessors for wrapper-safe state queries
    # -----------------------------
    def pending_dice(self, agent: str) -> int:
        """Return the current pending dice value for the given agent.

        This is a safe accessor to internal state so external wrappers do not
        need to reach into private attributes (which PettingZoo wrappers block).
        """
        return int(self._pending_dice.get(agent, 0))

    def valid_move_tokens(self, agent: str) -> list[int]:
        """Return list of token ids that have a valid move for the agent.

        When there are no valid moves, returns an empty list (the environment
        will internally handle no-move/pass turns).
        """
        return [m.token_id for m in self._pending_valid_moves.get(agent, [])]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = copy.deepcopy(self.possible_agents)
        if len(self.agents) < 1:
            print("Warning: No agents defined in environment.")
        self.turn_count = 0
        self._pending_dice = {}
        self._pending_valid_moves = {}
        self._action_masks = {}
        self._last_move_results = {}
        self._opponent_captures = {agent: 0 for agent in self.agents}

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._create_game()
        self.reward_calc.reset_for_new_episode()

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self._roll_dice_for_current()

    def _create_game(self) -> None:
        colors = list(ALL_COLORS)
        self.game = LudoGame(colors)

        for idx, agent in enumerate(self.agents):
            color = colors[idx]
            self._agent_color_map[agent] = color
            self._color_agent_map[color] = agent

            self._obs_builders[agent] = make_observation_builder(
                self.cfg, self.game, color
            )

    def _roll_dice_for_current(self) -> None:
        agent = self.agent_selection
        if agent not in self.agents:
            return

        color = self._agent_color_map[agent]
        player = self.game.get_player_from_color(color)

        dice = self.game.roll_dice()
        valid_moves = self.game.get_valid_moves(player, dice)

        self._pending_dice[agent] = dice
        self._pending_valid_moves[agent] = valid_moves
        self._action_masks[agent] = _make_mask(valid_moves)

    def step(self, action: int):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection
        color = self._agent_color_map[agent]
        player = self.game.get_player_from_color(color)

        action = int(action)

        dice = self._pending_dice.get(agent, 0)
        valid_moves = self._pending_valid_moves.get(agent, [])

        is_illegal = False
        if not valid_moves:
            move_result = self._no_move_result(player, dice)
        else:
            # Consider action illegal if chosen token id is not one of the
            # currently valid move token ids.
            valid_token_ids = {m.token_id for m in valid_moves}
            is_illegal = int(action) not in valid_token_ids
            move_result = self.game.execute_move(player, action, dice)

        self._last_move_results[agent] = move_result
        self.turn_count += 1

        for captured_token in move_result.captured_tokens:
            captured_agent = self._color_agent_map[captured_token.player_color]
            self._opponent_captures[captured_agent] += 1

        game_over = self.game.game_over or (self.game.winner is not None)
        truncated = not game_over and self.turn_count >= self.cfg.max_turns

        if game_over or truncated:
            self._set_game_result(game_over, truncated, agent, is_illegal)
        else:
            self._update_agent_reward(agent, color, move_result, is_illegal)

        self._accumulate_rewards()

        if not (game_over or truncated):
            if move_result.extra_turn:
                self._roll_dice_for_current()
            else:
                self.game.next_turn()
                self.agent_selection = self._agent_selector.next()
                self._roll_dice_for_current()
        else:
            self.agents = []

        if self.render_mode == "human":
            self.render()

    def _set_game_result(
        self, game_over: bool, truncated: bool, current_agent: str, is_illegal: bool
    ) -> None:
        for ag in self.agents:
            ag_color = self._agent_color_map[ag]
            ag_result = self._last_move_results.get(
                ag,
                self._no_move_result(self.game.get_player_from_color(ag_color), 0),
            )

            reward, breakdown = self.reward_calc.compute(
                game=self.game,
                agent_color=ag_color,
                move_result=ag_result,
                cfg=self.cfg,
                is_illegal=(ag == current_agent and is_illegal),
                opponent_captures=self._opponent_captures.get(ag, 0),
                terminated=game_over,
                turn_count=self.turn_count,
            )

            self.rewards[ag] = reward
            self.terminations[ag] = game_over
            self.truncations[ag] = truncated
            # Expose both keys for compatibility:
            # - "illegal_move" is used by PettingZoo's TerminateIllegalWrapper
            # - "illegal_action" is used by our local tests and logging
            illegal_flag = ag == current_agent and is_illegal
            self.infos[ag] = {
                "reward_breakdown": breakdown,
                "illegal_action": illegal_flag,
                "illegal_move": illegal_flag,
            }

    def _update_agent_reward(
        self, agent: str, color: PlayerColor, move_result: MoveResult, is_illegal: bool
    ) -> None:
        reward, breakdown = self.reward_calc.compute(
            game=self.game,
            agent_color=color,
            move_result=move_result,
            cfg=self.cfg,
            is_illegal=is_illegal,
            opponent_captures=self._opponent_captures.get(agent, 0),
            terminated=False,
            turn_count=self.turn_count,
        )

        self.rewards[agent] = reward
        # Expose both keys for compatibility with wrappers and tests
        self.infos[agent] = {
            "reward_breakdown": breakdown,
            "illegal_action": is_illegal,
            "illegal_move": is_illegal,
        }

        self._opponent_captures[agent] = 0

    def observe(self, agent: str):
        dice_val = self._pending_dice.get(agent, 0)
        obs = self._obs_builders[agent].build(dice_val)

        if agent == self.agent_selection:
            action_mask = self._action_masks.get(
                agent, np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
            )
        else:
            action_mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)

        return {
            "observation": obs,
            "action_mask": action_mask,
        }

    def action_mask(self, agent: str) -> np.ndarray:
        if agent == self.agent_selection:
            return self._action_masks.get(
                agent, np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
            )
        return np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)

    def _no_move_result(self, player: Player, dice: int) -> MoveResult:
        return MoveResult(
            success=True,
            player_color=player.color,
            token_id=0,
            dice_value=dice,
            old_position=-1,
            new_position=-1,
            captured_tokens=[],
            finished_token=False,
            extra_turn=False,
            error=None,
            game_won=False,
        )

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            print(self._render_ansi())
            return None
        raise ValueError(
            f"{self.render_mode} is not a valid render mode. "
            f"Available modes are: {self.metadata['render_modes']}"
        )

    def _render_ansi(self) -> str:
        if not self.game:
            return "Game not initialized"

        lines = []
        lines.append(f"=== Ludo Game - Turn {self.turn_count} ===")
        lines.append(f"Current player: {self.agent_selection}")

        if self.agent_selection:
            dice = self._pending_dice.get(self.agent_selection, 0)
            lines.append(f"Dice roll: {dice}")

        lines.append("\nPlayer positions:")
        for agent in self.possible_agents:
            if agent not in self._agent_color_map:
                continue
            color = self._agent_color_map[agent]
            player = self.game.get_player_from_color(color)
            token_positions = [t.position for t in player.tokens]
            lines.append(f"  {agent} ({color.name}): {token_positions}")

        if self.game.winner:
            lines.append(f"\n*** Winner: {self._color_agent_map[self.game.winner]} ***")

        return "\n".join(lines)

    def close(self):
        self.game.board.reset_token_positions()
