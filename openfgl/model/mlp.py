import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(nn.Linear(input_dim, hid_dim)) 
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hid_dim, hid_dim))
            self.layers.append(nn.Linear(hid_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, data):
        x = data.x
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.layers[-1](x)
        
        return x, logits
