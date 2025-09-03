import numpy as np
import torch
from stable_baselines3 import PPO

from ludo.constants import BoardConstants, Colors, GameConstants

from .envs.ludo_env import EnvConfig, LudoGymEnv


class PPOStrategy:
    """Wrapper to use PPO model as a Ludo strategy."""

    def __init__(
        self, model_path: str, model_name: str, env_config: EnvConfig | None = None
    ):
        self.model_name = model_name
        self.model = PPO.load(model_path)
        self.env_cfg = env_config or EnvConfig()
        # Backward compatibility: older EnvConfig (from classic env) may lack attributes
        if not hasattr(self.env_cfg, "randomize_training_color"):
            setattr(self.env_cfg, "randomize_training_color", False)
        # Ensure agent_color exists (classic EnvConfig might use agent_color or training_color)
        if not hasattr(self.env_cfg, "agent_color") and hasattr(
            self.env_cfg, "training_color"
        ):
            self.env_cfg.agent_color = self.env_cfg.training_color  # type: ignore
        self.dummy_env = LudoGymEnv(self.env_cfg)
        dummy_obs, _ = self.dummy_env.reset(seed=self.env_cfg.seed)
        self.obs_dim = dummy_obs.shape[0]
        self.description = f"PPO Model: {model_name} (obs_dim={self.obs_dim})"
        # Enforce consistent training agent color (critical for observation semantics)
        if self.env_cfg.agent_color != Colors.RED:
            raise ValueError(
                f"PPOStrategy agent_color mismatch: expected 'red' (training seat), got '{self.env_cfg.agent_color}'"
            )

    @staticmethod
    def _normalize_position_static(pos: int) -> float:
        if pos == GameConstants.HOME_POSITION:
            return -1.0
        if pos >= BoardConstants.HOME_COLUMN_START:
            depth = (
                pos - BoardConstants.HOME_COLUMN_START
            ) / GameConstants.HOME_COLUMN_DEPTH_SCALE
            return (
                GameConstants.POSITION_NORMALIZATION_FACTOR
                + depth * GameConstants.POSITION_NORMALIZATION_FACTOR
            )
        return (
            pos / (GameConstants.MAIN_BOARD_SIZE - 1)
        ) * GameConstants.POSITION_NORMALIZATION_FACTOR

    def _build_action_mask(self, valid_moves: list[dict]) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.float32)
        for mv in valid_moves:
            tid = mv.get("token_id")
            if isinstance(tid, int) and 0 <= tid < GameConstants.TOKENS_PER_PLAYER:
                mask[tid] = 1.0
        return mask

    def _ordered_players(self, players: list[dict]) -> list[dict]:
        indexed = {p.get("color"): p for p in players}
        return [indexed[c] for c in Colors.ALL_COLORS if c in indexed]

    def decide(self, game_context: dict) -> int:
        obs = self._build_observation_from_context(game_context)
        if obs.shape[0] != self.obs_dim:
            if obs.shape[0] < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
            else:
                obs = obs[: self.obs_dim]
        valid_moves = game_context.get("valid_moves", [])
        if not valid_moves:
            return 0
        mask = self._build_action_mask(valid_moves)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            try:
                probs = dist.distribution.probs.squeeze(0).cpu().numpy()
            except Exception:
                probs = dist.distribution.logits.softmax(-1).squeeze(0).cpu().numpy()
        masked = probs * mask
        if masked.sum() <= 0:
            return int(np.argmax(mask))
        return int(np.argmax(masked))

    def _build_observation_from_context(self, context: dict) -> np.ndarray:
        if "players" in context:
            game_info = context.get("game_info", {})
            players = self._ordered_players(context.get("players", []))
            current_player_color = game_info.get(
                "current_player", self.env_cfg.agent_color
            )
            current_player = next(
                (p for p in players if p.get("color") == current_player_color), None
            )
            if current_player is None:
                return np.zeros(self.obs_dim, dtype=np.float32)
            vec: list[float] = []
            for tok in current_player.get("tokens", []):
                vec.append(self._normalize_position_static(tok.get("position", -1)))
            while len(vec) < 4:
                vec.append(-1.0)
            for color in Colors.ALL_COLORS:
                if color == current_player_color:
                    continue
                opp = next((p for p in players if p.get("color") == color), None)
                if opp:
                    token_list = opp.get("tokens", [])
                    for tok in token_list:
                        vec.append(
                            self._normalize_position_static(tok.get("position", -1))
                        )
                    while len(token_list) < 4:
                        vec.append(-1.0)
                else:
                    vec.extend([-1.0] * 4)
            for color in Colors.ALL_COLORS:
                pl = next((p for p in players if p.get("color") == color), None)
                finished = pl.get("finished_tokens", 0) if pl else 0
                vec.append(finished / GameConstants.TOKENS_PER_PLAYER)
            can_finish = 0.0
            for tok in current_player.get("tokens", []):
                pos = tok.get("position", -1)
                if 0 <= pos < BoardConstants.HOME_COLUMN_START:
                    remaining = GameConstants.FINISH_POSITION - pos
                    if remaining <= GameConstants.DICE_MAX:
                        can_finish = 1.0
                        break
            vec.append(can_finish)
            dice_value = context.get("dice_value") or game_info.get("dice_value") or 0
            vec.append(
                (dice_value - GameConstants.DICE_NORMALIZATION_MEAN)
                / GameConstants.DICE_NORMALIZATION_MEAN
            )
            agent_finished = (
                current_player.get("finished_tokens", 0)
                / GameConstants.TOKENS_PER_PLAYER
            )
            opp_progress = []
            for color in Colors.ALL_COLORS:
                if color == current_player_color:
                    continue
                pl = next((p for p in players if p.get("color") == color), None)
                if pl:
                    opp_progress.append(
                        pl.get("finished_tokens", 0) / GameConstants.TOKENS_PER_PLAYER
                    )
            opp_mean = (
                sum(opp_progress) / max(1, len(opp_progress)) if opp_progress else 0.0
            )
            vec.append(agent_finished)
            vec.append(opp_mean)
            if self.env_cfg.obs_cfg.include_turn_index:
                turn_count = game_info.get("turn_count", 0)
                vec.append(min(1.0, turn_count / self.env_cfg.max_turns))
            if self.env_cfg.obs_cfg.include_blocking_count:
                vec.append(0.0)
            return np.asarray(vec, dtype=np.float32)
        player_state = context.get("player_state", {})
        opponents = context.get("opponents", [])
        vec: list[float] = []
        for tok in player_state.get("tokens", []):
            vec.append(self._normalize_position_static(tok.get("position", -1)))
        while len(vec) < 4:
            vec.append(-1.0)
        added_opp_tokens = 0
        for opp in opponents:
            token_list = opp.get("tokens", [])
            for tok in token_list:
                vec.append(self._normalize_position_static(tok.get("position", -1)))
            while len(token_list) < 4:
                vec.append(-1.0)
            added_opp_tokens += 4
            if added_opp_tokens >= 12:
                break
        while added_opp_tokens < 12:
            vec.extend([-1.0] * 4)
            added_opp_tokens += 4
        vec.append(
            player_state.get("finished_tokens", 0) / GameConstants.TOKENS_PER_PLAYER
        )
        for opp in opponents[:3]:
            vec.append(opp.get("tokens_finished", 0) / GameConstants.TOKENS_PER_PLAYER)
        while len(vec) < 4 + 12 + 4:
            vec.append(0.0)
        can_finish = 0.0
        for tok in player_state.get("tokens", []):
            pos = tok.get("position", -1)
            if 0 <= pos < BoardConstants.HOME_COLUMN_START:
                remaining = GameConstants.FINISH_POSITION - pos
                if remaining <= GameConstants.DICE_MAX:
                    can_finish = 1.0
                    break
        vec.append(can_finish)
        dice_value = context.get("dice_value", 0)
        vec.append(
            (dice_value - GameConstants.DICE_NORMALIZATION_MEAN)
            / GameConstants.DICE_NORMALIZATION_MEAN
        )
        agent_finished = (
            player_state.get("finished_tokens", 0) / GameConstants.TOKENS_PER_PLAYER
        )
        opp_progress = [
            opp.get("tokens_finished", 0) / GameConstants.TOKENS_PER_PLAYER
            for opp in opponents[:3]
        ]
        opp_mean = (
            sum(opp_progress) / max(1, len(opp_progress)) if opp_progress else 0.0
        )
        vec.append(agent_finished)
        vec.append(opp_mean)
        if self.env_cfg.obs_cfg.include_turn_index:
            turn_count = context.get("turn_count", 0)
            vec.append(min(1.0, turn_count / self.env_cfg.max_turns))
        if self.env_cfg.obs_cfg.include_blocking_count:
            vec.append(0.0)
        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape[0] != self.obs_dim:
            if arr.shape[0] < self.obs_dim:
                arr = np.pad(arr, (0, self.obs_dim - arr.shape[0]))
            else:
                arr = arr[: self.obs_dim]
        return arr
