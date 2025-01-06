import torch
from torch.utils.data import Dataset
import json

class TextDataset(Dataset):
    def __init__(self, data_path):
        """
        Initialize the dataset for Text Unicoder.
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
        Retrieve an item for the Text Unicoder.
        Args:
            idx (int): Index of the data sample.
        Returns:
            dict: Dictionary with tokenized text input and attention mask.
        """
        entry = self.data[idx]
        text_input_ids = torch.tensor(entry["text_input_ids"], dtype=torch.long)
        text_attention_mask = torch.tensor(entry["text_attention_mask"], dtype=torch.long)
        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask
        }
