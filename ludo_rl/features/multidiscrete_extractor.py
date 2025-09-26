import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiDiscreteFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor that embeds each MultiDiscrete dimension and concatenates embeddings.

    Observation space must be gym.spaces.MultiDiscrete.
    """

    def __init__(self, observation_space: gym.Space, embed_dim: int = 8):
        if not isinstance(observation_space, gym.spaces.MultiDiscrete):
            raise ValueError(
                "MultiDiscreteFeatureExtractor requires a MultiDiscrete observation_space"
            )

        # Compute total embedding size
        self.nvec = observation_space.nvec.tolist()
        self.embed_dim = embed_dim
        total_embed = embed_dim * len(self.nvec)

        super().__init__(observation_space, features_dim=total_embed)

        # Create embedding layers per discrete dim
        self.embeds = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=int(n), embedding_dim=embed_dim)
                for n in self.nvec
            ]
        )

        # Attention layer to let the model learn to weight/interact between discrete dims
        # Use a single-head MultiheadAttention (batch_first=True so inputs are (batch, seq, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch, n_dims)
        embeds = []
        # Ensure long dtype for embedding lookup
        obs_long = observations.long()
        for i, emb in enumerate(self.embeds):
            embeds.append(emb(obs_long[:, i]))
        # stack into (batch, seq_len, embed_dim)
        embeds_tensor = torch.stack(embeds, dim=1)

        # apply self-attention (queries=keys=values)
        # attn_out shape: (batch, seq_len, embed_dim)
        attn_out, attn_weights = self.attn(embeds_tensor, embeds_tensor, embeds_tensor)

        # flatten attended outputs to produce final feature vector
        batch_size = attn_out.shape[0]
        return attn_out.reshape(batch_size, -1)
