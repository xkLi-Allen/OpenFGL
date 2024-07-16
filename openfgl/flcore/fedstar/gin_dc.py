import torch
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
import torch.nn.functional as F

class serverGIN_dc(torch.nn.Module):
    def __init__(self, n_se, num_layers, hid_dim):
        super(serverGIN_dc, self).__init__()

        self.embedding_s = torch.nn.Linear(n_se, hid_dim)
        self.Whp = torch.nn.Linear(hid_dim + hid_dim, hid_dim)

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(hid_dim + hid_dim, hid_dim), torch.nn.ReLU(), torch.nn.Linear(hid_dim, hid_dim))
        self.graph_convs.append(GINConv(self.nn1))
        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(hid_dim, hid_dim))

        for l in range(num_layers - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(hid_dim + hid_dim, hid_dim), torch.nn.ReLU(), torch.nn.Linear(hid_dim, hid_dim))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(hid_dim, hid_dim))


class DecoupledGIN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, n_se, num_layers=2, dropout=0.5):
        super(DecoupledGIN, self).__init__()
        self.n_se = n_se
        self.num_layers = num_layers
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(input_dim, hid_dim))

        self.embedding_s = torch.nn.Linear(n_se, hid_dim)

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(hid_dim + hid_dim, hid_dim), torch.nn.ReLU(), torch.nn.Linear(hid_dim, hid_dim))
        self.graph_convs.append(GINConv(self.nn1))
        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(hid_dim, hid_dim))

        for l in range(num_layers - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(hid_dim + hid_dim, hid_dim), torch.nn.ReLU(), torch.nn.Linear(hid_dim, hid_dim))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(hid_dim, hid_dim))

        self.Whp = torch.nn.Linear(hid_dim + hid_dim, hid_dim)
        self.post = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(hid_dim, output_dim))

    def forward(self, data):
        x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc
        x = self.pre(x)
        s = self.embedding_s(s)
        for i in range(len(self.graph_convs)):
            x = torch.cat((x, s), -1)
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.graph_convs_s_gcn[i](s, edge_index)
            s = torch.tanh(s)
        x = self.Whp(torch.cat((x, s), -1))
        x = global_add_pool(x, batch)
        x = self.post(x)
        y = F.dropout(x, self.dropout, training=self.training)
        y = self.readout(y)
        y = F.log_softmax(y, dim=1)
        return x,y

