"""Graph Convolutional Network (GCN) model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network for node classification.
    
    A simple 2-layer GCN model.
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.5):
        """Initialize the GCN model.
        
        Args:
            num_features (int): Number of input features
            num_classes (int): Number of output classes
            hidden_dim (int): Hidden layer dimension
            dropout (float): Dropout rate
        """
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge index [2, num_edges]
            
        Returns:
            logits: Class logits [num_nodes, num_classes]
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        return x
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings from the first layer.
        
        Args:
            x: Node features
            edge_index: Edge index
            
        Returns:
            embeddings: Node embeddings from first layer
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        return x
