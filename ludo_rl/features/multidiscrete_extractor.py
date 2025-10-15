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

    def __init__(self, observation_space: gym.Space, embed_dim: int = 64):
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

        # Calculate total feature dimension: agent features concatenated + aggregated opponent features
        agent_keys = ["agent_color", "agent_progress", "agent_vulnerable", "dice"]
        agent_features = (
            sum(
                config["size"]
                for key, config in self.component_configs.items()
                if key in agent_keys
            )
            * embed_dim
        )
        opponent_features = 1 * embed_dim  # Aggregated opponent vector
        total_features = agent_features + opponent_features
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
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads=embed_dim // 4, batch_first=True
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """Process Dict observation into concatenated agent features + aggregated opponent features with attention."""
        agent_embeddings = []
        opponent_embeddings = []
        agent_keys = {"agent_color", "agent_progress", "agent_vulnerable", "dice"}

        # Process each component and collect embeddings
        for key, config in self.component_configs.items():
            obs_tensor = observations[key]  # Shape: (batch, n * size)
            n, size = config["n"], config["size"]
            batch = obs_tensor.size(0)

            # Reshape to (batch, size, n) and argmax to get discrete indices
            obs_reshaped = obs_tensor.view(batch, size, n)
            discrete_values = torch.argmax(obs_reshaped, dim=-1)  # (batch, size)

            # Embed all discrete values at once
            embs = self.embed_by_n[str(n)](discrete_values)  # (batch, size, embed_dim)

            if key in agent_keys:
                # For agent, unbind and extend to list for later concatenation
                agent_embeddings.extend(embs.unbind(dim=1))
            else:
                # For opponents, unbind and extend
                opponent_embeddings.extend(embs.unbind(dim=1))

        # Concatenate agent embeddings: (batch, agent_features)
        agent_features = torch.cat(agent_embeddings, dim=1)

        # Stack opponent embeddings: (batch, num_opponent_embeddings, embed_dim)
        opp_tensor = torch.stack(opponent_embeddings, dim=1)

        # Apply attention to opponent embeddings
        attended_opp, _ = self.attention(opp_tensor, opp_tensor, opp_tensor)

        # Aggregate opponent features (mean pooling for permutation invariance)
        opp_aggregated = attended_opp.mean(dim=1)  # (batch, embed_dim)

        # Concatenate agent and aggregated opponent features
        return torch.cat([agent_features, opp_aggregated], dim=1)
