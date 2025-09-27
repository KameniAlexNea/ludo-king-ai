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

        # Create shared embedding layers for each unique cardinality (n)
        # This means all dims with the same number of categories reuse the same
        # nn.Embedding instance (e.g. all token position dims, all vulnerable
        # flags, etc.). This avoids duplicating parameters for repeated
        # identical vocabularies like token positions.
        unique_ns = sorted(set(int(n) for n in self.nvec))
        self.embed_by_n = nn.ModuleDict(
            {
                str(n): nn.Embedding(num_embeddings=n, embedding_dim=embed_dim)
                for n in unique_ns
            }
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
        # For each discrete dim, look up the shared embedding for that dim's
        # cardinality (nvec[i]). This reuses parameters for identical dims.
        for i, n in enumerate(self.nvec):
            emb = self.embed_by_n[str(int(n))]
            embeds.append(emb(obs_long[:, i]))
        # stack into (batch, seq_len, embed_dim)
        embeds_tensor = torch.stack(embeds, dim=1)

        # apply self-attention (queries=keys=values)
        # attn_out shape: (batch, seq_len, embed_dim)
        attn_out, attn_weights = self.attn(embeds_tensor, embeds_tensor, embeds_tensor)

        # flatten attended outputs to produce final feature vector
        batch_size = attn_out.shape[0]
        return attn_out.reshape(batch_size, -1)
