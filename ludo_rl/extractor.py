import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .ludo_king.config import config, net_config


class BaseTokenSeqExtractor(BaseFeaturesExtractor):
    """Shared embedding + input prep for token-sequence extractors."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        assert "positions" in observation_space.spaces, (
            "Token-sequence observation required"
        )
        pos_shape = observation_space["positions"].shape  # (T, 16)
        self.T = pos_shape[0]
        self.N = pos_shape[1]
        self.dice_roll_dim = 6
        self.embed_dim = net_config.token_embed_dim
        # Embeddings shared by both variants
        self.pos_emb = nn.Embedding(config.PATH_LENGTH, self.embed_dim)
        self.color_emb = nn.Embedding(4, self.embed_dim)
        self.piece_idx_emb = nn.Embedding(4, self.embed_dim)
        self.time_emb = nn.Embedding(self.T, self.embed_dim)
        self.frame_dice_emb = nn.Embedding(self.dice_roll_dim + 1, self.embed_dim)
        self.player_emb = nn.Embedding(4, self.embed_dim)
        self.curr_dice_emb = nn.Embedding(self.dice_roll_dim + 1, self.embed_dim)

    def _prepare_inputs(self, observations: dict):
        positions = observations["positions"].long()
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        dice_hist = observations["dice_history"].long()
        if dice_hist.dim() == 1:
            dice_hist = dice_hist.unsqueeze(0)
        player_hist = observations["player_history"].long()
        if player_hist.dim() == 1:
            player_hist = player_hist.unsqueeze(0)
        token_mask = observations["token_mask"].to(dtype=torch.bool)
        if token_mask.dim() == 2:
            token_mask = token_mask.unsqueeze(0)
        token_colors = observations["token_colors"].long()
        if token_colors.dim() == 1:
            token_colors = token_colors.unsqueeze(0)
        current_dice = observations["current_dice"].long()
        if current_dice.dim() == 1:
            current_dice = current_dice
        B, T, N = positions.shape
        device = positions.device
        return (
            positions,
            dice_hist,
            player_hist,
            token_mask,
            token_colors,
            current_dice,
            B,
            T,
            N,
            device,
        )

    def _sum_token_embeddings(
        self,
        positions: torch.Tensor,
        dice_hist: torch.Tensor,
        player_hist: torch.Tensor,
        token_mask: torch.Tensor,
        token_colors: torch.Tensor,
        B: int,
        T: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        pos_e = self.pos_emb(positions)  # (B,T,N,d)
        colors = token_colors.unsqueeze(1).expand(B, T, N)
        color_e = self.color_emb(colors)
        piece_idx = (torch.arange(N, device=device) % 4).view(1, 1, N).expand(B, T, N)
        piece_e = self.piece_idx_emb(piece_idx)
        time_idx = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)
        time_e = self.time_emb(time_idx)
        frame_dice = (
            dice_hist.clamp(0, self.dice_roll_dim).view(B, T, 1).expand(B, T, N)
        )
        frame_dice_e = self.frame_dice_emb(frame_dice)
        player_idx = player_hist.view(B, T, 1).expand(B, T, N)
        player_e = self.player_emb(player_idx)
        tok = pos_e + color_e + piece_e + time_e + frame_dice_e + player_e
        # Apply mask later in subclasses (after optional projections)
        return tok

    def _embed_current_dice(self, current_dice: torch.Tensor) -> torch.Tensor:
        return self.curr_dice_emb(current_dice.clamp(0, self.dice_roll_dim).squeeze(1))


class LudoCnnExtractor(BaseTokenSeqExtractor):
    """Token-sequence features with LSTM for temporality + MLP head."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # Token projection and LSTM for temporality
        self.token_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )
        bidirectional = True
        self.lstm = nn.LSTM(
            self.embed_dim,
            self.embed_dim,
            num_layers=2,
            dropout=0.1,
            bidirectional=bidirectional,
            batch_first=True,
        )
        # Features: pool over tokens (token_feat_dim), then concat current dice (embed_dim)
        self.token_feat_dim = self.embed_dim * (2 if bidirectional else 1)
        self.total_feature_dim = self.token_feat_dim + self.embed_dim
        self.feature_norm = nn.LayerNorm(self.total_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(self.total_feature_dim, features_dim),
            nn.GELU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        (
            positions,
            dice_hist,
            player_hist,
            token_mask,
            token_colors,
            current_dice,
            B,
            T,
            N,
            device,
        ) = self._prepare_inputs(observations)

        tok = self._sum_token_embeddings(
            positions, dice_hist, player_hist, token_mask, token_colors, B, T, N, device
        )
        tok = self.token_proj(tok)
        m = token_mask.to(dtype=tok.dtype).unsqueeze(-1)
        tok = tok * m

        # Apply LSTM over time for each token
        # tok: (B, T, N, d) -> permute to (B, N, T, d) for easier reshaping
        tok_permuted = tok.permute(0, 2, 1, 3)  # (B, N, T, d)
        tok_reshaped = tok_permuted.contiguous().view(
            B * N, T, self.embed_dim
        )  # (B*N, T, d)
        lstm_out, _ = self.lstm(tok_reshaped)  # (B*N, T, token_feat_dim)
        # Take the last time step
        last_hidden = lstm_out[:, -1, :]  # (B*N, token_feat_dim)
        # Reshape back to (B, N, d)
        pooled_per_token = last_hidden.view(B, N, self.token_feat_dim)

        # Pool over tokens, masking invalid ones
        token_valid = token_mask.any(dim=1)  # (B, N) - valid if any frame has data
        mask_float = token_valid.float().unsqueeze(-1)  # (B, N, 1)
        valid_counts = mask_float.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = (pooled_per_token * mask_float).sum(dim=1) / valid_counts  # (B, d)

        curr_e = self._embed_current_dice(current_dice)

        combined = torch.cat([pooled, curr_e], dim=1)
        combined = self.feature_norm(combined)
        return self.head(combined)


class LudoTransformerExtractor(BaseTokenSeqExtractor):
    """Transformer over token sequence: (TxN tokens) with dice conditioning."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=net_config.trans_nhead,
            dim_feedforward=self.embed_dim * int(net_config.trans_ff_mult),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(net_config.trans_num_layers)
        )
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, features_dim),
            nn.GELU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        (
            positions,
            dice_hist,
            player_hist,
            token_mask,
            token_colors,
            current_dice,
            B,
            T,
            N,
            device,
        ) = self._prepare_inputs(observations)

        tok = self._sum_token_embeddings(
            positions, dice_hist, player_hist, token_mask, token_colors, B, T, N, device
        )
        seq = tok.view(B, T * N, self.embed_dim)
        mask = token_mask.view(B, T * N)

        cls = self.cls_token.expand(B, 1, -1)
        dice_tok = self._embed_current_dice(current_dice).unsqueeze(1)
        sequence = torch.cat([cls, dice_tok, seq], dim=1)

        pad = torch.zeros(B, 2, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([pad, ~mask], dim=1)

        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
        cls_feature = self.output_norm(encoded[:, 0])
        return self.head(cls_feature)
