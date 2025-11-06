import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .ludo.config import config, net_config


class LudoCnnExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the Ludo environment.
    It processes the board with a 1D CNN and concatenates the dice roll.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # Extract dimensions from the observation space
        self.board_shape = observation_space["board"].shape  # (10, 58)
        self.dice_roll_dim = 6

        n_input_channels = self.board_shape[0]  # Should be 10

        # --- CNN Stream for Board ---
        # Input: (batch_size, 10, 58)
        self.cnn = nn.Sequential(
            nn.Conv1d(
                n_input_channels,
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
            nn.AdaptiveAvgPool1d(output_size=net_config.pooled_output_size),
            nn.Flatten(),
        )

        # --- Compute the flattened size of the CNN output ---
        # To do this, we pass a dummy tensor through the CNN
        with torch.no_grad():
            dummy_board = torch.as_tensor(
                observation_space["board"].sample()[None]
            ).float()
            n_flatten = self.cnn(dummy_board).shape[1]

        # Normalize CNN features to stabilize scale before fusion
        self.board_norm = nn.LayerNorm(n_flatten)

        # --- Linear Layer to combine CNN output and dice roll ---
        # The dice roll will be one-hot encoded to a size of 6
        # Optional pre-linear normalization and a light MLP head
        self.pre_linear_norm = nn.LayerNorm(n_flatten + self.dice_roll_dim)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.dice_roll_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # Process board with CNN
        board_features = self.cnn(observations["board"].float())
        board_features = self.board_norm(board_features)

        # One-hot encode the dice roll
        # The dice roll is 0-5 (since we did dice-1)
        dice_roll = observations["dice_roll"].long()
        # Defensive clamp in case upstream normalization corrupts the categorical input
        dice_roll = torch.clamp(dice_roll, 0, self.dice_roll_dim - 1)
        one_hot_dice = nn.functional.one_hot(
            dice_roll.squeeze(1), num_classes=self.dice_roll_dim
        ).float()

        # Concatenate CNN features and one-hot dice roll
        combined_features = torch.cat([board_features, one_hot_dice], dim=1)
        combined_features = self.pre_linear_norm(combined_features)

        # Pass through final linear layer
        return self.linear(combined_features)


class LudoTransformerExtractor(BaseFeaturesExtractor):
    """Transformer-based extractor using semantic tokens for board squares and pieces."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        board_channels, board_length = observation_space["board"].shape

        self.board_length = board_length
        self.dice_roll_dim = 6
        self.embed_dim = net_config.embed_dim

        # Square token encoding (per square: combine all channels)
        self.square_norm = nn.LayerNorm(board_channels)
        self.square_proj = nn.Linear(board_channels, self.embed_dim)
        self.square_pos_embed = nn.Embedding(board_length, self.embed_dim)

        # Learnable tokens and embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dice_embed = nn.Embedding(self.dice_roll_dim, self.embed_dim)
        self.piece_position_embed = nn.Embedding(config.PATH_LENGTH, self.embed_dim)
        self.piece_index_embed = nn.Embedding(config.PIECES_PER_PLAYER, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.sequence_dropout = nn.Dropout(p=0.1)
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        board = observations["board"].float()
        batch_size = board.shape[0]

        # Square tokens: (batch, 58, embed_dim)
        square_features = board.permute(0, 2, 1)  # (batch, squares, channels)
        square_features = self.square_norm(square_features)
        square_tokens = self.square_proj(square_features)
        position_indices = (
            torch.arange(self.board_length, device=board.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        square_tokens = square_tokens + self.square_pos_embed(position_indices)

        # Dice token: (batch, 1, embed_dim)
        dice_roll = observations["dice_roll"].long().clamp(0, self.dice_roll_dim - 1)
        dice_token = self.dice_embed(dice_roll.squeeze(1)).unsqueeze(1)

        # Piece tokens derived from my_pieces channel
        my_channel = board[:, 0, :]  # (batch, 58)
        piece_positions = self._extract_piece_positions(my_channel)
        piece_tokens = self.piece_position_embed(piece_positions)
        piece_indices = (
            torch.arange(config.PIECES_PER_PLAYER, device=board.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
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

    def _extract_piece_positions(self, my_channel: torch.Tensor) -> torch.Tensor:
        """Recover individual piece positions from the agent's occupancy channel."""

        counts = my_channel.round().clamp(min=0).to(dtype=torch.long)
        indices = torch.arange(self.board_length, device=my_channel.device)

        return torch.stack(
            [
                torch.repeat_interleave(indices, count_row, dim=0)
                for count_row in counts
            ],
            dim=0,
        )
