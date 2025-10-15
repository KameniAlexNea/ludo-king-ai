import gymnasium as gym
import torch
import torch.nn as nn
from ludo_engine.models import ALL_COLORS, GameConstants
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiDiscreteFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict observation spaces with discrete components.

    Handles agent_color (one-hot), agent_progress (discrete bins), agent_vulnerable (binary),
    opponents_positions (discrete ranks), opponents_active (binary), and dice (discrete).
    """

    def __init__(self, observation_space: gym.Space, embed_dim: int = 8):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                "MultiDiscreteFeatureExtractor requires a Dict observation_space"
            )

        # Define embedding dimensions for each component
        SIZE = 1 + GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        MAX_OPPONENTS = GameConstants.MAX_PLAYERS - 1
        self.component_configs = {
            "agent_color": {"n": 2, "size": len(ALL_COLORS)},  # 4 colors, each 0/1
            "agent_progress": {
                "n": SIZE,
                "size": GameConstants.TOKENS_PER_PLAYER,
            },  # 4 tokens, each 0-10
            "agent_vulnerable": {
                "n": 2,
                "size": GameConstants.TOKENS_PER_PLAYER,
            },  # 4 tokens, each 0/1
            "opponents_positions": {
                "n": SIZE,
                "size": GameConstants.TOKENS_PER_PLAYER * MAX_OPPONENTS,
            },  # 12 positions (3 opp × 4 tokens), each 0-64
            "opponents_active": {
                "n": 2,
                "size": MAX_OPPONENTS,
            },  # 3 opponents, each 0/1
            "dice": {"n": 7, "size": 1},  # 1 dice value, 0-6
        }

        # Calculate total feature dimension
        total_features = sum(
            config["size"] * embed_dim for config in self.component_configs.values()
        )
        super().__init__(observation_space, features_dim=total_features)

        self.embed_dim = embed_dim

        # Create embedding layers for each unique cardinality
        unique_ns = sorted(
            set(config["n"] for config in self.component_configs.values())
        )
        self.embed_by_n = nn.ModuleDict(
            {
                str(n): nn.Embedding(num_embeddings=n, embedding_dim=embed_dim)
                for n in unique_ns
            }
        )

        # Multi-head attention for processing embeddings
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, observations: dict) -> torch.Tensor:
        """Process Dict observation into concatenated embeddings with attention."""
        all_embeddings = []

        # Process each component and collect individual embeddings
        for key, config in self.component_configs.items():
            obs_tensor = observations[key]  # Shape: (batch, n * size)
            n, size = config["n"], config["size"]
            batch = obs_tensor.size(0)

            # Reshape to (batch, size, n) and argmax to get discrete indices
            obs_reshaped = obs_tensor.view(batch, size, n)
            discrete_values = torch.argmax(obs_reshaped, dim=-1)  # (batch, size)

            # Embed all discrete values at once
            embs: torch.Tensor = self.embed_by_n[str(n)](discrete_values)  # (batch, size, embed_dim)
            all_embeddings.extend(embs.unbind(dim=1))  # List of (batch, embed_dim) tensors

        # Stack all embeddings: (batch, num_embeddings, embed_dim)
        embs_tensor = torch.stack(all_embeddings, dim=1)

        # Apply multi-head attention
        attended, _ = self.attention(embs_tensor, embs_tensor, embs_tensor)

        # Flatten to match expected features_dim
        return attended.view(attended.size(0), -1)
