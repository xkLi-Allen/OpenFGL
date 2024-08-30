import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from openfgl.model.graphsage import GraphSAGE


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hid_dim = hid_dim

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hid_dim))
        for _ in range(self.num_layers - 1):
            self.layers.append(SAGEConv(hid_dim, hid_dim))
        self.linear = nn.Linear(hid_dim, output_dim)

    def forward(self, x, edge_index=None, adjs=None):
        if self.num_layers == 1:
            edge_index, _, size = adjs
            edge_index = edge_index.to(x.device)
            x_target = x[: size[1]]
            x = self.layers[0]((x, x_target), edge_index)
            x = torch.tanh(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[: size[1]]
                edge_index = edge_index.to(x.device)
                x = self.layers[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                else:
                    x = torch.tanh(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        x = F.softmax(F.dropout(x, p=self.dropout, training=self.training), dim=1)
        return x

    def get_encoder(self, x_all, subgraph_loader):
        self.eval()
        xs = []
        for _, n_id, adjs in subgraph_loader:
            x = x_all[n_id]
            if self.num_layers == 1:
                edge_index, _, size = adjs
                x_target = x[: size[1]]
                edge_index = edge_index.to(x.device)
                x = self.layers[0]((x, x_target), edge_index)
                x = torch.tanh(x)
            else:
                for i, (edge_index, _, size) in enumerate(adjs):
                    x_target = x[: size[1]]
                    edge_index = edge_index.to(x.device)
                    x = self.layers[i]((x, x_target), edge_index)
                    if i != self.num_layers - 1:
                        x = F.relu(x)
                    else:
                        x = torch.tanh(x)
            xs.append(x)
        xs = torch.cat(xs, dim=0).reshape((-1, self.hid_dim))
        return xs


class Classifier_F(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(Classifier_F, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.feat_shape, self.emb_shape = input_dim

        self.add_convs = nn.ModuleList()
        convs = nn.ModuleList()
        convs.append(SAGEConv(self.feat_shape, hid_dim))
        convs.append(nn.modules.Linear(self.emb_shape, hid_dim))
        self.add_convs.append(convs)
        for _ in range(self.num_layers - 2):
            convs = nn.ModuleList()
            convs.append(SAGEConv(hid_dim, hid_dim))
            convs.append(nn.modules.Linear(self.emb_shape, hid_dim))
            self.add_convs.append(convs)
        convs = torch.nn.ModuleList()
        convs.append(SAGEConv(hid_dim, output_dim))
        convs.append(nn.modules.Linear(self.emb_shape, output_dim))
        self.add_convs.append(convs)

    def forward_full(self, data):
        if "mend_emb" not in data:
            x_feat, edge_index = data.x, data.edge_index
            x_emb = torch.zeros((len(x_feat), self.emb_shape)).to(x_feat.device)
        else:
            x_feat, x_emb, edge_index = data.x, data.mend_emb, data.edge_index
            x_emb = x_emb.mean(dim=1)
        for i, layer in enumerate(self.add_convs):
            x_feat = layer[0]((x_feat, x_feat), edge_index)
            x_emb = layer[1](x_emb)
            if (i + 1) == len(self.add_convs):
                x_feat = x_feat + x_emb
                break
            x_feat = F.relu(F.dropout(x_feat+x_emb, p=self.dropout, training=self.training))
        return x_feat

    def forward(self, x, adjs=None):
        if isinstance(x, tuple) and len(x) == 2:
            x_feat, x_emb = x
            x_emb = x_emb.mean(dim=1)
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x_feat[: size[1]]
                x_emb = x_emb[: size[1]]
                edge_index = edge_index.to(x_feat.device)
                x_feat = self.add_convs[i][0]((x_feat, x_target), edge_index)
                x_emb = self.add_convs[i][1](x_emb)
                x_feat = x_feat + x_emb
                if i != self.num_layers - 1:
                    x_feat = F.relu(x_feat)
                    x_feat = F.dropout(x_feat, p=self.dropout, training=self.training)
            return x_feat
        else:
            return self.forward_full(x)


class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, inputs):
        rand = torch.normal(0, 1, size=inputs.shape)
        return inputs + rand.to(inputs.device)


class EmbGenerator(nn.Module):
    def __init__(self, latent_dim, dropout, num_preds, feat_shape):
        super(EmbGenerator, self).__init__()
        self.num_preds = num_preds
        self.feat_shape = feat_shape
        self.dropout = dropout
        self.sample = Sampling()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 2048)
        self.fc_flat = nn.Linear(2048, self.num_preds * self.feat_shape)

    def forward(self, x):
        x = self.sample(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x


class NumPredictor(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(NumPredictor, self).__init__()
        self.reg_1 = nn.Linear(self.latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.reg_1(x))
        return x


# Mend the graph via NeighGen
class MendGraph(nn.Module):
    def __init__(self, num_preds):
        super(MendGraph, self).__init__()
        self.num_preds = num_preds
        for param in self.parameters():
            param.requires_grad = False

    def mend_graph(self, pred_degree, gen_embs):
        device = gen_embs.device
        if pred_degree.device.type != "cpu":
            pred_degree = pred_degree.cpu()
        num_nodes = len(pred_degree)
        mend_emb = gen_embs.view(num_nodes, self.num_preds, -1)
        pred_degree = torch._cast_Int(torch.round(pred_degree)).detach()
        mend_num = torch.zeros(mend_emb.shape).to(device)
        for i in range(num_nodes):
            mend_num[i][: min(self.num_preds, max(0, pred_degree[i]))] = 1
        mend_emb = mend_emb * mend_num
        return mend_emb

    def forward(self, pred_missing, gen_feats):
        mend_emb = self.mend_graph(pred_missing, gen_feats)
        return mend_emb


class LocalDGen(nn.Module):
    def __init__(self, input_dim, emb_shape, output_dim, hid_dim, gen_dim, dropout=0.5, num_preds=5):
        super(LocalDGen, self).__init__()
        self.encoder_model = GraphSAGE(
            input_dim=input_dim, hid_dim=hid_dim,
            num_layers=2, output_dim=gen_dim,
            dropout=dropout)
        self.reg_model = NumPredictor(latent_dim=gen_dim)
        self.gen = EmbGenerator(
            latent_dim=gen_dim, dropout=dropout,
            num_preds=num_preds, feat_shape=emb_shape)
        self.mend_graph = MendGraph(num_preds)
        self.classifier = Classifier_F(
            input_dim=(input_dim, emb_shape),
            hid_dim=hid_dim, output_dim=output_dim,
            num_layers=2, dropout=dropout)

    def forward(self, data):
        _, x = self.encoder_model(data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats = self.mend_graph(degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=data.x, mend_emb=mend_feats, edge_index=data.edge_index))
        return degree, gen_feat, nc_pred[: data.num_nodes]


class FedDEP(nn.Module):
    def __init__(self, local_graph: LocalDGen):
        super(FedDEP, self).__init__()
        self.encoder_model = local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph = local_graph.mend_graph
        self.classifier = local_graph.classifier
        # self.encoder_model.requires_grad_(False)
        # self.reg_model.requires_grad_(False)
        # self.mend_graph.requires_grad_(False)
        # self.classifier.requires_grad_(False)

    def forward(self, data):
        _, x = self.encoder_model(data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats = self.mend_graph(degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=data.x, mend_emb=mend_feats, edge_index=data.edge_index))
        return degree, gen_feat, nc_pred[: data.num_nodes]
