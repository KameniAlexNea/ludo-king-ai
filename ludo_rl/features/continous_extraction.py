from typing import Dict

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ContinuousFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observation spaces with continuous components.

    This extractor manually processes agent and opponent features to ensure
    opponent processing is permutation-invariant using attention and pooling.
    """

    def __init__(self, observation_space: gym.Space, embed_dim: int = 64):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("Extractor requires a Dict observation_space")

        # Determine the dimensions for the agent and opponent sections based on
        # the structure defined in your ContinuousObservationBuilder

        # --- Define Feature Grouping and Sizes (Assuming 4 tokens, 3 opponents) ---
        self.agent_keys = {
            "agent_tokens",  # 4 floats
            "agent_progress",  # 4 floats
            "agent_vulnerable",  # 4 floats
            "dice",  # 1 float (if not one-hot) or 6 floats (if one-hot)
        }
        self.opponent_keys = {
            "opponents_positions",  # 12 floats (3 opp * 4 tokens)
            "opponents_active",  # 3 floats
        }

        # Calculate the size of the opponent sequence (12 positions + 3 flags)
        # This is used for grouping the features to represent each opponent token
        self.opp_sequence_len = 12 + 3

        # Calculate the size of the single opponent feature vector: 4 pos + 1 active flag = 5 features per token set
        # Since we are aggregating ALL opponent data together, we just need to ensure the features are combined correctly.

        # Calculate the raw dimension for a single opponent "entity" (e.g., token)
        # Assuming we treat the 12 pos + 3 active flags as 15 separate items in a sequence
        self.opp_item_dim = (
            1  # Each position/flag is its own scalar item in the sequence
        )

        # We need a small linear layer to project the scalar items into a richer feature space
        self.projection_dim = 16
        self.input_projection = nn.Linear(self.opp_item_dim, self.projection_dim)

        # Multi-head attention for opponent features
        self.attention = nn.MultiheadAttention(
            self.projection_dim,
            num_heads=max(1, self.projection_dim // 4),
            batch_first=True,
        )

        # Output dimension of the aggregated opponent feature
        self.opp_aggregated_dim = self.projection_dim

        # Calculate the dimension of the concatenated agent features
        agent_raw_dim = 0
        for key in self.agent_keys:
            # We assume the output from your ContinuousObserver is a 1D tensor for each key
            agent_raw_dim += observation_space[key].shape[0]

        # Final feature dimension is Agent (raw concatenated) + Opponent (aggregated)
        total_features_dim = agent_raw_dim + self.opp_aggregated_dim
        super().__init__(observation_space, features_dim=total_features_dim)

        # Final layer to concatenate the agent features
        self.final_concat_layer = nn.Linear(
            agent_raw_dim, embed_dim - self.opp_aggregated_dim
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process Dict observation, applying permutation-invariant processing."""

        # --- 1. Process Agent Features (Simple Concatenation) ---
        agent_features_list = []
        for key in self.agent_keys:
            # Flatten the continuous tensor (it should already be 1D, but we ensure it)
            agent_features_list.append(observations[key].flatten(start_dim=1))

        agent_raw_features = torch.cat(agent_features_list, dim=1)

        # Apply a linear layer to give the agent features some initial depth
        agent_processed = self.activation(self.final_concat_layer(agent_raw_features))

        # --- 2. Process Opponent Features (Permutation Invariant) ---
        opp_items_list = []
        for key in self.opponent_keys:
            # Gather all opponent data into a sequence of scalar items
            opp_items_list.append(observations[key].flatten(start_dim=1))

        # opp_sequence_raw shape: (batch_size, 15) -> 12 pos + 3 active flags
        opp_sequence_raw = torch.cat(opp_items_list, dim=1)

        # Reshape to sequence of items: (batch_size, 15, 1)
        opp_sequence = opp_sequence_raw.unsqueeze(-1)

        # Project each scalar item into the shared feature space: (batch_size, 15, projection_dim)
        opp_projected = self.input_projection(opp_sequence)

        # Apply attention to the sequence of opponent items
        # attended_opp shape: (batch_size, 15, projection_dim)
        attended_opp, _ = self.attention(opp_projected, opp_projected, opp_projected)

        # Aggregate opponent features (mean pooling for permutation invariance)
        # opp_aggregated shape: (batch_size, projection_dim)
        opp_aggregated = attended_opp.mean(dim=1)

        # --- 3. Final Concatenation ---
        # Final output: [Agent Features (Fixed Order), Opponent Features (Aggregated)]
        return torch.cat([agent_processed, opp_aggregated], dim=1)
