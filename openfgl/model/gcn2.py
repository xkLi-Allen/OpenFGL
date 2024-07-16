import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv


class GCN2(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, alpha=0.1, num_layers=2, dropout=0.5):
        super(GCN2, self).__init__()
        self.alpha = alpha
        self.linear1 = nn.Linear(input_dim, hid_dim)        
        self.layers = nn.ModuleList([GCN2Conv(hid_dim, alpha=alpha) for _ in range(num_layers)])
        self.linear2 = nn.Linear(hid_dim, output_dim)
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x0 = x
        for conv in self.layers:
            x = conv(x, x0, edge_index)
        logits = self.linear2(x)
        return x, logits



