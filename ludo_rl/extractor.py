import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .ludo.config import config, net_config


def _extract_piece_positions(
    my_channel: torch.Tensor, board_length: int
) -> torch.Tensor:
    """Recover individual piece positions from the agent's occupancy channel."""
    with torch.autograd.profiler.record_function("extract_piece_positions"):
        counts = my_channel.round().clamp(min=0).to(dtype=torch.long)
        indices = torch.arange(board_length, device=my_channel.device, dtype=torch.long)
        expanded_indices = indices.unsqueeze(0).expand_as(counts)

        flat_positions = torch.repeat_interleave(
            expanded_indices.reshape(-1), counts.reshape(-1), dim=0
        )
        return flat_positions.view(my_channel.shape[0], -1)


class LudoCnnExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the Ludo environment.
    It processes the board with a 1D CNN and concatenates the dice roll.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        board_shape = observation_space["board"].shape  # (10, 58)
        self.board_length = board_shape[1]
        self.dice_roll_dim = 6
        self.position_channels = 4  # my pieces + three opponents
        self.context_channels = board_shape[0] - self.position_channels

        # Position-focused CNN stream (agent + opponents)
        self.position_conv = nn.Sequential(
            nn.Conv1d(
                self.position_channels,
                net_config.conv_configs[0],
                kernel_size=net_config.kernel_sizes[0],
                stride=1,
                padding=net_config.paddings[0],
            ),
            nn.ReLU(),
            nn.Conv1d(
                net_config.conv_configs[0],
                net_config.conv_configs[1],
                kernel_size=net_config.kernel_sizes[1],
                stride=1,
                padding=net_config.paddings[1],
            ),
            nn.ReLU(),
        )
        self.position_pool = nn.AdaptiveAvgPool1d(
            output_size=net_config.pooled_output_size
        )
        self.position_norm = nn.LayerNorm(
            net_config.conv_configs[1] * net_config.pooled_output_size
        )

        # Contextual CNN stream (safe zones, heatmaps, knockouts, etc.)
        self.context_conv = nn.Sequential(
            nn.Conv1d(
                self.context_channels,
                net_config.conv_configs[0],
                kernel_size=net_config.kernel_sizes[0],
                stride=1,
                padding=net_config.paddings[0],
            ),
            nn.ReLU(),
            nn.Conv1d(
                net_config.conv_configs[0],
                net_config.conv_configs[1],
                kernel_size=net_config.kernel_sizes[1],
                stride=1,
                padding=net_config.paddings[1],
            ),
            nn.ReLU(),
        )
        self.context_pool = nn.AdaptiveAvgPool1d(
            output_size=net_config.pooled_output_size
        )
        self.context_norm = nn.LayerNorm(
            net_config.conv_configs[1] * net_config.pooled_output_size
        )

        # Per-piece embeddings and projection
        self.position_embed = nn.Embedding(config.PATH_LENGTH, net_config.embed_dim)
        self.piece_index_embed = nn.Embedding(
            config.PIECES_PER_PLAYER, net_config.embed_dim
        )
        self.piece_mlp = nn.Sequential(
            nn.Linear(
                net_config.embed_dim * 2 + net_config.conv_configs[1],
                net_config.embed_dim,
            ),
            nn.ReLU(),
        )

        # Dice embedding to mirror transformer behaviour
        self.dice_embed = nn.Embedding(self.dice_roll_dim, net_config.embed_dim)

        self.register_buffer(
            "piece_indices",
            torch.arange(config.PIECES_PER_PLAYER, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

        # Infer flattened dimensions using a sample observation
        with torch.no_grad():
            dummy_board = torch.as_tensor(
                observation_space["board"].sample()[None]
            ).float()
            pos_flat = self.position_pool(
                self.position_conv(dummy_board[:, : self.position_channels, :])
            ).flatten(1)
            ctx_flat = self.context_pool(
                self.context_conv(dummy_board[:, self.position_channels :, :])
            ).flatten(1)

        self.piece_feature_dim = config.PIECES_PER_PLAYER * net_config.embed_dim
        self.total_feature_dim = (
            pos_flat.shape[1]
            + ctx_flat.shape[1]
            + self.piece_feature_dim
            + net_config.embed_dim
        )

        self.feature_norm = nn.LayerNorm(self.total_feature_dim)
        self.head = nn.Sequential(
            nn.Linear(self.total_feature_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        board = observations["board"].float()

        position_board = board[:, : self.position_channels, :]
        context_board = board[:, self.position_channels :, :]

        position_conv = self.position_conv(position_board)
        position_flat = self.position_pool(position_conv).flatten(1)
        position_flat = self.position_norm(position_flat)

        context_conv = self.context_conv(context_board)
        context_flat = self.context_pool(context_conv).flatten(1)
        context_flat = self.context_norm(context_flat)

        my_channel = board[:, 0, :]
        piece_positions = _extract_piece_positions(my_channel, self.board_length)
        piece_indices = self.piece_indices.expand(board.shape[0], -1)

        position_embed = self.position_embed(piece_positions)
        index_embed = self.piece_index_embed(piece_indices)
        conv_gather = self._gather_piece_context(position_conv, piece_positions)

        piece_features = torch.cat([position_embed, index_embed, conv_gather], dim=-1)
        piece_features = self.piece_mlp(piece_features)
        piece_flat = piece_features.reshape(board.shape[0], -1)

        dice_roll = (
            observations["dice_roll"].long().clamp(0, self.dice_roll_dim - 1).squeeze(1)
        )
        dice_emb = self.dice_embed(dice_roll)

        combined = torch.cat([position_flat, context_flat, piece_flat, dice_emb], dim=1)
        combined = self.feature_norm(combined)
        return self.head(combined)

    def _gather_piece_context(
        self, conv_map: torch.Tensor, piece_positions: torch.Tensor
    ) -> torch.Tensor:
        """Extract local conv features for each agent piece."""

        _, channels, seq_len = conv_map.shape
        clamped_positions = piece_positions.clamp(0, seq_len - 1)
        gather_index = clamped_positions.unsqueeze(1).expand(-1, channels, -1)
        gathered = torch.gather(conv_map, dim=2, index=gather_index)
        return gathered.transpose(1, 2)


class LudoTransformerExtractor(BaseFeaturesExtractor):
    """Transformer-based extractor using semantic tokens for board squares and pieces."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        nhead: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        board_channels, board_length = observation_space["board"].shape

        self.board_length = board_length
        self.dice_roll_dim = 6
        self.embed_dim = net_config.embed_dim

        # Square token encoding (per square: combine all channels)
        reduced_dim = max(self.embed_dim // 2, 1)
        self.square_norm = nn.LayerNorm(board_channels)
        self.square_proj = nn.Sequential(
            nn.Linear(board_channels, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, self.embed_dim),
        )
        self.square_pos_embed = nn.Embedding(board_length, self.embed_dim)

        # Learnable tokens and embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dice_embed = nn.Embedding(self.dice_roll_dim, self.embed_dim)
        self.piece_position_embed = nn.Embedding(config.PATH_LENGTH, self.embed_dim)
        self.piece_index_embed = nn.Embedding(config.PIECES_PER_PLAYER, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dim_feedforward=self.embed_dim * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.sequence_dropout = nn.Dropout(p=0.1)
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

        self.register_buffer(
            "square_position_indices",
            torch.arange(board_length, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "transformer_piece_indices",
            torch.arange(config.PIECES_PER_PLAYER, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(self, observations: dict) -> torch.Tensor:
        board = observations["board"].float()
        batch_size = board.shape[0]

        # Square tokens: (batch, 58, embed_dim)
        square_features = board.permute(0, 2, 1)  # (batch, squares, channels)
        square_features = self.square_norm(square_features)
        square_tokens = self.square_proj(square_features)
        position_indices = self.square_position_indices.expand(batch_size, -1)
        square_tokens = square_tokens + self.square_pos_embed(position_indices)

        # Dice token: (batch, 1, embed_dim)
        dice_roll = observations["dice_roll"].long().clamp(0, self.dice_roll_dim - 1)
        dice_token = self.dice_embed(dice_roll.squeeze(1)).unsqueeze(1)

        # Piece tokens derived from my_pieces channel
        my_channel = board[:, 0, :]  # (batch, 58)
        piece_positions = _extract_piece_positions(my_channel, self.board_length)
        piece_tokens = self.piece_position_embed(piece_positions)
        piece_indices = self.transformer_piece_indices.expand(batch_size, -1)
        piece_tokens = piece_tokens + self.piece_index_embed(piece_indices)

        # Assemble sequence: [CLS] + dice + squares + pieces
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat(
            [cls_tokens, dice_token, square_tokens, piece_tokens], dim=1
        )
        sequence = self.sequence_dropout(sequence)

        encoded = self.encoder(sequence)
        cls_feature = self.output_norm(encoded[:, 0])
        return self.head(cls_feature)
