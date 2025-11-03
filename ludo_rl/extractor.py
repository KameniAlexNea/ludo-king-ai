import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.config import net_config


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

        # --- Linear Layer to combine CNN output and dice roll ---
        # The dice roll will be one-hot encoded to a size of 6
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.dice_roll_dim, features_dim), nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # Process board with CNN
        board_features = self.cnn(observations["board"].float())

        # One-hot encode the dice roll
        # The dice roll is 0-5 (since we did dice-1)
        dice_roll = observations["dice_roll"].long()
        one_hot_dice = nn.functional.one_hot(
            dice_roll.squeeze(1), num_classes=self.dice_roll_dim
        ).float()

        # Concatenate CNN features and one-hot dice roll
        combined_features = torch.cat([board_features, one_hot_dice], dim=1)

        # Pass through final linear layer
        return self.linear(combined_features)
