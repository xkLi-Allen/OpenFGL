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

        return x, logits

    def aug(self, data):

        # 使用GNN提取特征
        with torch.no_grad():
            node_features,_ = self.forward(data)

        # 任意两节点间的特征组合（这里简化处理，实际可能需要复杂的操作）
        # 假设使用外积来模拟节点间潜在的连接特征
        logits = torch.matmul(node_features, node_features.t())

        # 应用Gumbel-Softmax采样
        adj_sampled = self.gumbel_softmax(logits, tau=0.5)
        return adj_sampled, logits

    def gumbel_softmax(self, logits, tau):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(MLP, self).__init__()
        # 定义第一个线性层
        self.fc1 = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        # 定义第二个线性层
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 输入通过第一个线性层
        x = self.fc1(x)
        # 应用ReLU激活函数
        x = F.relu(x)
        # 应用dropout
        x = self.dropout(x)
        # 输出通过第二个线性层
        x = self.fc2(x)
        return x