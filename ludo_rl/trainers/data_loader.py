"""
Handles loading of the Ludo game data from Hugging Face Hub.
"""

from typing import Dict, List

import datasets


class DataLoader:
    """Loads game data from a Hugging Face dataset."""

    def __init__(self, repo_id: str = "alexneakameni/ludo-king-rl"):
        self.repo_id = repo_id

    def load_from_hf(self) -> List[Dict]:
        """
        Loads the training data from the specified Hugging Face dataset repository.

        Returns:
            List[Dict]: A list of game decision records.
        """
        ds = datasets.load_dataset(self.repo_id, split="train")
        print("Dataset Loaded")
        return ds.to_list()
