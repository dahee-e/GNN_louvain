import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv , GATConv

# class GraphTransformer(torch.nn.Module):
#     def __init__(self, num_node_features, hidden_channels, dropout, num_heads=4):
#         super(GraphTransformer, self).__init__()
#         self.conv1 = TransformerConv(num_node_features, hidden_channels, heads=num_heads)
#         self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
#         self.fc = nn.Linear(hidden_channels * num_heads, num_node_features)  # Final layer to project back to original dimension
#         self.dropout = dropout
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.fc(x)  # Project back to the original feature dimension (64)
#         return x


class GraphTransformer(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, dropout, num_heads=4):
        super(GraphTransformer, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.fc = nn.Linear(hidden_channels * num_heads, num_node_features)  # Final layer to project back to original dimension
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc(x)  # Project back to the original feature dimension (64)
        return x
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_node_features)  # Output matches input dimensions
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
