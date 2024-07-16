import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias

class FedGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(FedGCN, self).__init__()
        self.layers = nn.ModuleList()
        if nlayer > 1:
            self.layers.append(GraphConvolution(nfeat, nhid))
            for _ in range(nlayer - 2):
                self.layers.append(GraphConvolution(nhid, nhid))
            self.layers.append(GraphConvolution(nhid, nclass))
        else:
            self.layers.append(GraphConvolution(nfeat, nclass))

        self.dropout = dropout

    def forward(self, data):
        x, adj = data.x, data.adj
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.layers[-1](x, adj)

        return logits, logits

