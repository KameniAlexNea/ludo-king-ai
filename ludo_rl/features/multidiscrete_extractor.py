import gymnasium as gym
import torch
import torch.nn as nn
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
        self.component_configs = {
            "agent_color": {"n": 2, "size": 4},  # 4 colors, each 0/1
            "agent_progress": {"n": 11, "size": 4},  # 4 tokens, each 0-10
            "agent_vulnerable": {"n": 2, "size": 4},  # 4 tokens, each 0/1
            "opponents_positions": {
                "n": 65,
                "size": 12,
            },  # 12 positions (3 opp × 4 tokens), each 0-64
            "opponents_active": {"n": 2, "size": 3},  # 3 opponents, each 0/1
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

    def forward(self, observations: dict) -> torch.Tensor:
        """Process Dict observation into concatenated embeddings."""
        features = []

        # Process each component
        for key, config in self.component_configs.items():
            obs_tensor = observations[key]  # Shape: (batch, size)
            n, size = config["n"], config["size"]

            # Embed each position in this component
            embeddings = []
            for i in range(size):
                emb = self.embed_by_n[str(n)](obs_tensor[:, i].long())
                embeddings.append(emb)

            # Concatenate embeddings for this component
            component_features = torch.cat(
                embeddings, dim=1
            )  # (batch, size * embed_dim)
            features.append(component_features)

        # Concatenate all component features
        return torch.cat(features, dim=1)  # (batch, total_features)
