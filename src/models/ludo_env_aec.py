"""PettingZoo AEC environment wrapper for multi-agent Ludo training."""

from __future__ import annotations

import functools
from typing import Dict, Optional

import numpy as np
from gymnasium import spaces
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
from pettingzoo.utils.agent_selector import agent_selector

from .config import EnvConfig
from .observation import make_observation_builder
from .reward import AdvancedRewardCalculator
from .spaces import get_space_config


def _make_mask(valid_moves: Optional[list[ValidMove]]) -> np.ndarray:
    """Create action mask from valid moves."""
    mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=bool)
    if valid_moves:
        for move in valid_moves:
            mask[move.token_id] = True
    return mask


def make_aec_env(cfg: Optional[EnvConfig] = None):
    """Factory function for creating wrapped environment."""
    env = raw_env(cfg)
    # Add standard wrappers
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """PettingZoo AEC environment for 4-player Ludo with turn-based play.

    This environment enables multi-agent reinforcement learning where:
    - All 4 players can be learning agents (or mix of agents/bots)
    - Turn-based gameplay with proper agent cycling
    - Action masking for invalid moves
    - Per-agent observations and rewards
    """

    metadata = {
        "render_modes": ["human"],
        "name": "LudoMultiAgent-v0",
        "is_parallelizable": False,  # Turn-based game
    }

    def __init__(self, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.render_mode = "human"  # Required by PettingZoo wrappers

        # Define agents (4 players)
        self.possible_agents = ["player_0", "player_1", "player_2", "player_3"]

        # Game state
        self.game: Optional[LudoGame] = None
        self._agent_color_map: Dict[str, PlayerColor] = {}
        self._color_agent_map: Dict[PlayerColor, str] = {}
        self._obs_builders: Dict[str, object] = {}
        self.turn_count = 0

        # Dice and move tracking
        self._pending_dice: Dict[str, int] = {}
        self._pending_valid_moves: Dict[str, list[ValidMove]] = {}
        self._action_masks: Dict[str, np.ndarray] = {}

        # Reward tracking
        self.reward_calc = AdvancedRewardCalculator()
        self._last_move_results: Dict[str, MoveResult] = {}
        self._opponent_captures: Dict[str, int] = {}

        # Define observation and action spaces
        tokens = GameConstants.TOKENS_PER_PLAYER

        # Shared observation space for all agents
        observation_space = get_space_config()

        action_space = spaces.Discrete(tokens)

        # Set observation and action spaces for all agents
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        self.action_spaces = {agent: action_space for agent in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return observation space for agent."""
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return action space for agent."""
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed, options=options)
        if seed is not None:
            np.random.seed(seed)

        # Reset game state
        self.agents = self.possible_agents[:]
        self.turn_count = 0
        self._pending_dice = {}
        self._pending_valid_moves = {}
        self._action_masks = {}
        self._last_move_results = {}
        self._opponent_captures = {agent: 0 for agent in self.agents}

        # Initialize rewards, terminations, truncations, infos
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Create game
        self._create_game()
        self.reward_calc.reset_for_new_episode()

        # Set up agent selector for turn order
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Roll dice for first agent
        self._roll_dice_for_current()

    def _create_game(self) -> None:
        """Initialize a new Ludo game with 4 players."""
        colors = list(ALL_COLORS)
        self.game = LudoGame(colors)

        # Map agents to colors
        for idx, agent in enumerate(self.agents):
            color = colors[idx]
            self._agent_color_map[agent] = color
            self._color_agent_map[color] = agent

            # Create observation builder for each agent
            self._obs_builders[agent] = make_observation_builder(
                self.cfg, self.game, color
            )

    def _roll_dice_for_current(self) -> None:
        """Roll dice for the current agent."""
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
        """Execute action for current agent and advance to next turn."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection
        color = self._agent_color_map[agent]
        player = self.game.get_player_from_color(color)

        # PettingZoo requires per-step rewards to be reset before assigning new ones
        for name in self.agents:
            self.rewards[name] = 0.0

        # Execute move
        dice = self._pending_dice.get(agent, 0)
        valid_moves = self._pending_valid_moves.get(agent, [])

        is_illegal = False
        if not valid_moves:
            # No valid moves, skip turn
            move_result = self._no_move_result(player, dice)
        else:
            valid_tokens = {m.token_id for m in valid_moves}
            chosen = int(action)

            if chosen not in valid_tokens:
                # Illegal move - penalize and choose random valid move
                is_illegal = True
                chosen = np.random.choice(list(valid_tokens))

            move_result = self.game.execute_move(player, chosen, dice)

        self._last_move_results[agent] = move_result
        self.turn_count += 1

        # Track captures
        for captured_token in move_result.captured_tokens:
            captured_agent = self._color_agent_map[captured_token.player_color]
            self._opponent_captures[captured_agent] += 1

        # Check for game termination
        game_over = self.game.game_over or (self.game.winner is not None)
        truncated = not game_over and self.turn_count >= self.cfg.max_turns

        # Calculate rewards for all agents if game is over
        if game_over or truncated:
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
                    is_illegal=(ag == agent and is_illegal),
                    opponent_captures=self._opponent_captures.get(ag, 0),
                    terminated=game_over,
                )

                self.rewards[ag] = reward
                self._cumulative_rewards[ag] += reward
                self.terminations[ag] = game_over
                self.truncations[ag] = truncated
                self.infos[ag] = {
                    "action_mask": self._action_masks.get(
                        ag, np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=bool)
                    ),
                    "reward_breakdown": breakdown,
                    "illegal_action": (ag == agent and is_illegal),
                }
        else:
            # Calculate reward for current agent
            reward, breakdown = self.reward_calc.compute(
                game=self.game,
                agent_color=color,
                move_result=move_result,
                cfg=self.cfg,
                is_illegal=is_illegal,
                opponent_captures=self._opponent_captures.get(agent, 0),
                terminated=False,
            )

            self.rewards[agent] = reward
            self._cumulative_rewards[agent] += reward
            self.infos[agent] = {
                "action_mask": self._action_masks.get(
                    agent, np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=bool)
                ),
                "reward_breakdown": breakdown,
                "illegal_action": is_illegal,
            }

            # Reset opponent captures for current agent
            self._opponent_captures[agent] = 0

        # Advance to next agent or handle extra turn
        if not (game_over or truncated):
            if move_result.extra_turn:
                # Same agent gets another turn, roll dice again
                self._roll_dice_for_current()
            else:
                # Move to next agent
                self.game.next_turn()
                self.agent_selection = self._agent_selector.next()
                self._roll_dice_for_current()
        else:
            # Game is over
            self.agents = []

    def observe(self, agent: str):
        """Return observation for specified agent."""
        if agent not in self._obs_builders:
            return None

        dice_val = self._pending_dice.get(agent, 0)
        obs = self._obs_builders[agent].build(dice_val)
        return obs

    def action_mask(self, agent: str):
        """Return action mask for specified agent."""
        return self._action_masks.get(
            agent, np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=bool)
        )

    def _no_move_result(self, player: Player, dice: int) -> MoveResult:
        """Create a MoveResult for when no valid moves are available."""
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
        """Render the environment (optional)."""
        if self.game:
            print(f"Turn {self.turn_count}, Current agent: {self.agent_selection}")
