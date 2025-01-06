import torch
import json
from torch.utils.data import Dataset

class GATDataset(Dataset):
    def __init__(self, data_path):
        """
        Initialize the dataset.
        Args:
            data_path (str): Path to preprocessed data (e.g., JSON file).
        """
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))  # Parse each line as a JSON object

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an item for GATEncoder.
        Args:
            idx (int): Index of the data sample.
        Returns:
            dict: Dictionary with node features and edge list.
        """
        entry = self.data[idx]

        node_features = torch.tensor(entry["node_features"], dtype=torch.float32)
        edge_list = torch.tensor(entry["edge_list"], dtype=torch.long).t().contiguous()
        
        return {
            "node_features": node_features,
            "edge_list": edge_list
        }

