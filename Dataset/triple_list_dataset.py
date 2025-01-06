import torch
from torch.utils.data import Dataset
import json

class TripleListDataset(Dataset):
    def __init__(self, data_path):
        """
        Initialize the dataset for TripleListEncoder.
        Args:
            data_path (str): Path to preprocessed data (e.g., JSON file).
        """
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an item for TripleListEncoder.
        Args:
            idx (int): Index of the data sample.
        Returns:
            dict: Dictionary with triple_input_ids and attention mask.
        """
        entry = self.data[idx]
        triple_input_ids = torch.tensor(entry["triple_input_ids"], dtype=torch.long)
        triple_attention_mask = torch.tensor(entry["triple_attention_mask"], dtype=torch.long)
        return {
            "triple_input_ids": triple_input_ids,
            "triple_attention_mask": triple_attention_mask
        }
