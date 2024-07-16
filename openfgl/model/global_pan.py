import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PANConv
from torch_geometric.nn.pool import global_add_pool, PANPooling
import torch

class GlobalPAN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(GlobalPAN, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        
        if num_layers > 1:
            self.convs.append(PANConv(input_dim, hid_dim, filter_size=0)) 
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers - 2):
                self.convs.append(PANConv(hid_dim, hid_dim, filter_size=0))
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))
            self.convs.append(PANConv(hid_dim, hid_dim, filter_size=0))
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
        else:
            self.convs.append(PANConv(input_dim, hid_dim, filter_size=0))
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
            
            
            
            
        self.pan_pooling = PANPooling(hid_dim, ratio=0.5)
        
        self.lin1 = nn.Linear(hid_dim, hid_dim)
        self.batch_norm1 = nn.BatchNorm1d(hid_dim)
        self.lin2 = nn.Linear(hid_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x, M = conv(x, edge_index)
            x = F.relu(batch_norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x, connect_out_edge_index, connect_out_edge_attr, connect_out_batch, perm, score = self.pan_pooling(x, M, batch)
        embedding = global_add_pool(x, connect_out_batch)
        
        if data.y.shape[0] != 1:
            x = self.batch_norm1(self.lin1(embedding))
        else:
            x = embedding
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lin2(x)
        return embedding, logits
