import torch
from torch.nn import Module
from torch_geometric.nn import GATConv


class GATEncoder(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        """
        Initialize a GAT-based encoder.
        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Size of hidden layers in GAT.
            out_channels (int): Size of output node embeddings.
            heads (int): Number of attention heads in GATConv.
        """
        super(GATEncoder, self).__init__()
        # First GAT layer
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # Second GAT layer
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, node_features, edge_index):
        """
        Forward pass for GAT.
        Args:
            node_features (torch.Tensor): Input node features [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge list [2, num_edges].
        Returns:
            torch.Tensor: Updated node features [num_nodes, out_channels].
        """
        # Apply first GAT layer
        x = self.gat1(node_features, edge_index)
        x = torch.relu(x)  # Non-linear activation
        # Apply second GAT layer
        x = self.gat2(x, edge_index)
        return x
