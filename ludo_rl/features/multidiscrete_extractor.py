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
        """Convert observations to per-dimension indices, embed, attend, and flatten.

        Accepts either:
        - indices: shape (batch, len(nvec)) with integer category ids
        - one-hot blocks: shape (batch, sum(nvec)), concatenation of one-hot vectors per dim
        """
        B, D = observations.shape[0], observations.shape[1]
        n_dims = len(self.nvec)
        sum_n = int(sum(int(n) for n in self.nvec))

        # Collapse one-hot blocks to indices if needed
        if D == n_dims:
            idx = observations.long()
        elif D == sum_n:
            # Split into blocks according to nvec and take argmax per block
            indices = []
            start = 0
            # ensure float for argmax stability
            obs_f = observations.to(dtype=torch.float32)
            for n in self.nvec:
                end = start + int(n)
                block = obs_f[:, start:end]
                # argmax along category axis -> index in [0, n-1]
                indices.append(block.argmax(dim=1))
                start = end
            idx = torch.stack(indices, dim=1).long()
        else:
            raise ValueError(
                f"Unexpected observation shape {tuple(observations.shape)}: expected (batch, {n_dims}) or (batch, {sum_n})"
            )

        # Embed each discrete dim using shared embeddings by cardinality
        embeds = []
        for i, n in enumerate(self.nvec):
            emb = self.embed_by_n[str(int(n))]
            embeds.append(emb(idx[:, i]))

        # stack into (batch, seq_len, embed_dim)
        embeds_tensor = torch.stack(embeds, dim=1)

        # apply self-attention (queries=keys=values)
        attn_out, _ = self.attn(embeds_tensor, embeds_tensor, embeds_tensor)

        # flatten attended outputs to produce final feature vector
        return attn_out.reshape(B, -1)
