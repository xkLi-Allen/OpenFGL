import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from torch import Tensor
from scipy.sparse import csr_matrix
from openfgl.flcore.adafgl.label_propagation_models import NonParaLP
from openfgl.flcore.adafgl.op import LaplacianGraphOp, ConcatMessageOp



class AdaFGLModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, hidden_dim, output_dim, train_mask, val_mask, test_mask, alpha=0.5, r=0.5):
        super(AdaFGLModel, self).__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        
        self.NonPLP = NonParaLP(prop_steps=10, num_class=self.output_dim, alpha=alpha, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, r=r)
        self.pre_graph_op = LaplacianGraphOp(prop_steps=self.prop_steps, r=r)
        self.post_graph_op = LaplacianGraphOp(prop_steps=100, r=r)
        self.post_msg_op = None


    def homo_init(self):
        self.homo_model = HomoPropagateModel(num_layers=2,
        feat_dim=self.feat_dim,
        hidden_dim=self.hidden_dim,
        output_dim=self.output_dim,
        dropout=0.5,
        prop_steps=self.prop_steps,
        bn=False,
        ln=False)

        self.total_trainable_params = round(sum(p.numel() for p in self.homo_model.parameters() if p.requires_grad)/1000000, 3)

    def hete_init(self):
        self.hete_model = HetePropagateModel(num_layers=3,
        feat_dim=self.feat_dim,
        hidden_dim=self.hidden_dim,
        output_dim=self.output_dim,
        dropout=0.5,
        prop_steps=self.prop_steps,
        bn=False,
        ln=False)

        self.total_trainable_params = round(sum(p.numel() for p in self.hete_model.parameters() if p.requires_grad)/1000000, 3)


    def non_para_lp(self, subgraph, soft_label, x, device):
        self.soft_label = soft_label
        self.ori_feature = x
        self.NonPLP.preprocess(self.soft_label, subgraph, device)
        self.NonPLP.propagate(adj=subgraph.adj)
        self.reliability_acc = self.NonPLP.eval()
        self.homo_init()
        self.hete_init()


    def preprocess(self, adj):
        self.pre_msg_op = ConcatMessageOp(start=0, end=self.prop_steps+1)
        self.universal_re = getre_scale(self.soft_label)
        self.universal_re_smooth = torch.where(self.universal_re>0.999, 1, 0)
        self.universal_re = torch.where(self.universal_re>0.999, 1, 0)
        edge_u = torch.where(self.universal_re_smooth != 0)[0].cpu().numpy()
        edge_v = torch.where(self.universal_re_smooth != 0)[1].cpu().numpy()
        self.universal_re_smooth = np.vstack((edge_u,edge_v))
        universal_re_smooth_adj = sp.coo_matrix((torch.ones([len(self.universal_re_smooth[0])]), (self.universal_re_smooth[0], self.universal_re_smooth[1])), shape=(self.soft_label.shape[0], self.soft_label.shape[0]))
        self.adj = self.alpha * adj + (1-self.alpha) * universal_re_smooth_adj
        self.adj = self.adj.tocoo()
        row, col, edge_weight = self.adj.row, self.adj.col, self.adj.data
        if isinstance(row, Tensor) or isinstance(col, Tensor):
            self.adj = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),
                                            shape=(self.soft_label.shape[0], self.soft_label.shape[0]))
        else:
            self.adj = csr_matrix((edge_weight, (row, col)), shape=(self.soft_label.shape[0], self.soft_label.shape[0]))

        self.processed_feat_list = self.pre_graph_op.propagate(self.adj, self.ori_feature)
        self.smoothed_feature = self.pre_msg_op.aggregate(self.processed_feat_list)
        self.processed_feature = self.soft_label

    def homo_forward(self, device):
        local_smooth_logits, global_logits= self.homo_model(
            smoothed_feature=self.smoothed_feature,
            global_logits=self.processed_feature,
            device=device
        )
        return local_smooth_logits, global_logits

    def hete_forward(self, device):
        local_ori_logits, local_smooth_logits, local_message_propagation  = self.hete_model(
            ori_feature=self.ori_feature,
            smoothed_feature=self.smoothed_feature,
            processed_feature=self.processed_feature,
            universal_re=self.universal_re,
            device=device)

        return local_ori_logits, local_smooth_logits, local_message_propagation 

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)
        return output


class HetePropagateLayer(nn.Module):
    def __init__(self, feat_dim, output_dim, prop_steps, hidden_dim, num_layers, dropout=0.5, beta=0, bn=False, ln=False):
        super(HetePropagateLayer, self).__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.prop_steps = prop_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.bn = bn
        self.ln = ln
        self.beta = beta

        self.lr_hete_trans = nn.ModuleList()
        self.lr_hete_trans.append(nn.Linear((self.prop_steps+1) * self.feat_dim, self.hidden_dim))

        for _ in range(num_layers - 2):
            self.lr_hete_trans.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lr_hete_trans.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.norms = nn.ModuleList()
        if self.bn:
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        if self.ln:
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.softmax = nn.Softmax(dim=1)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lr_hete_tran in self.lr_hete_trans:
            nn.init.xavier_uniform_(lr_hete_tran.weight, gain=gain)
            nn.init.zeros_(lr_hete_tran.bias)


    def forward(self, feature, device, learnable_re=None):

        for i in range(self.num_layers - 1):
            feature = self.lr_hete_trans[i](feature)
            if self.bn is True or self.ln is True:
                feature = self.norms[i](feature)
            feature = self.prelu(feature)
            feature = self.dropout(feature)
        feature_emb = self.lr_hete_trans[-1](feature)


        feature_emb_re = getre_scale(feature_emb)
        learnable_re = self.beta * learnable_re + (1-self.beta) * feature_emb_re
        learnable_re_mean = torch.mean(learnable_re)
        learnable_re_max = torch.max(learnable_re)
        
        learnable_re_pos_min = 0
        learnable_re_pos_difference = learnable_re_max - learnable_re_mean - learnable_re_pos_min 

        learnable_re_neg_min = -learnable_re_mean
        learnable_re_neg_difference = 0 - learnable_re_neg_min 

        learnable_re = learnable_re - learnable_re_mean
        learnable_re = torch.where(learnable_re>0, (learnable_re-learnable_re_pos_min) / learnable_re_pos_difference, -((learnable_re-learnable_re_neg_min) / learnable_re_neg_difference))
        
        learnable_re = add_diag(learnable_re, device)

        pos_signal = self.prelu(learnable_re)
        neg_signal = self.prelu(-learnable_re)

        prop_pos = self.softmax(torch.mm(pos_signal, feature_emb))
        prop_neg = self.softmax(torch.mm(neg_signal, feature_emb))

        local_message_propagation =  ((prop_pos - prop_neg) + feature_emb) / 2

        return local_message_propagation

class HetePropagateModel(nn.Module):
    def __init__(self, num_layers, feat_dim, hidden_dim, output_dim, prop_steps, dropout=0.5, bn=False, ln=False):
        super(HetePropagateModel, self).__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.prop_steps = prop_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bn = bn
        self.ln = ln
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.softmax = nn.Softmax(dim=1)

        
        self.lr_smooth_trans = nn.ModuleList()
        self.lr_smooth_trans.append(nn.Linear((self.prop_steps+1) * self.feat_dim,  self.hidden_dim))
        for _ in range(num_layers - 2):
            self.lr_smooth_trans.append(nn.Linear( self.hidden_dim,  self.hidden_dim))
        self.lr_smooth_trans.append(nn.Linear( self.hidden_dim, self.output_dim))

        self.lr_local_trans = nn.ModuleList()
        self.lr_local_trans.append(nn.Linear(self.feat_dim, self.hidden_dim))
        for _ in range(num_layers - 2):
            self.lr_local_trans.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lr_local_trans.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.hete_propagation = HetePropagateLayer(self.feat_dim, self.output_dim, self.prop_steps, self.hidden_dim, self.num_layers)

        self.norms = nn.ModuleList()
        if self.bn:
            if self.num_layers != 1:
                for _ in range(num_layers-1):
                    self.norms.append(nn.BatchNorm1d(self.hidden_dim))
        if self.ln:
            if self.num_layers != 1:
                for _ in range(num_layers-1):
                    self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lr_local_tran in self.lr_local_trans:
            nn.init.xavier_uniform_(lr_local_tran.weight, gain=gain)
            nn.init.zeros_(lr_local_tran.bias)

        for lr_smooth_tran in self.lr_smooth_trans:
            nn.init.xavier_uniform_(lr_smooth_tran.weight, gain=gain)
            nn.init.zeros_(lr_smooth_tran.bias)


    def forward(self, ori_feature, smoothed_feature, processed_feature, universal_re, device):
        ori_feature = ori_feature.to(device)
        smoothed_feature = smoothed_feature.to(device)
        processed_feature = processed_feature.to(device)

        input_prop_feature = smoothed_feature
        learnable_re = universal_re.to(device)

        for i in range(self.num_layers - 1):
            smoothed_feature = self.lr_smooth_trans[i](smoothed_feature)
            if self.bn is True or self.ln is True:
                smoothed_feature = self.norms[i](smoothed_feature)
            smoothed_feature = self.prelu(smoothed_feature)
            smoothed_feature = self.dropout(smoothed_feature)
        local_smooth_emb = self.lr_smooth_trans[-1](smoothed_feature)

        for i in range(self.num_layers - 1):
            ori_feature = self.lr_local_trans[i](ori_feature)
            if self.bn is True or self.ln is True:
                ori_feature = self.norms[i](ori_feature)
            ori_feature = self.prelu(ori_feature)
            ori_feature = self.dropout(ori_feature)
        local_ori_emb = self.lr_local_trans[-1](ori_feature)

        local_message_propagation = self.hete_propagation(input_prop_feature, device, learnable_re)
    

        return local_ori_emb, local_smooth_emb, local_message_propagation 




class HomoPropagateModel(nn.Module):
    def __init__(self, num_layers, feat_dim, hidden_dim, output_dim, prop_steps, dropout=0.5, bn=False, ln=False):
        super(HomoPropagateModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.lr_smooth_trans = nn.ModuleList()
        self.lr_smooth_trans.append(nn.Linear((prop_steps+1) * feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lr_smooth_trans.append(nn.Linear(hidden_dim, hidden_dim))
        self.lr_smooth_trans.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        self.ln = ln
        self.norms = nn.ModuleList()
        if bn:
            if self.num_layers != 1:
                for _ in range(num_layers-1):
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
        if ln:
            if self.num_layers != 1:
                for _ in range(num_layers-1):
                    self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lr_smooth_tran in self.lr_smooth_trans:
            nn.init.xavier_uniform_(lr_smooth_tran.weight, gain=gain)
            nn.init.zeros_(lr_smooth_tran.bias)

    def forward(self, smoothed_feature, global_logits, device):
        smoothed_feature = smoothed_feature.to(device)
        global_logits = global_logits.to(device)

        for i in range(self.num_layers - 1):
            smoothed_feature = self.lr_smooth_trans[i](smoothed_feature)
            if self.bn is True or self.ln is True:
                smoothed_feature = self.norms[i](smoothed_feature)
            smoothed_feature = self.prelu(smoothed_feature)
            smoothed_feature = self.dropout(smoothed_feature)

        local_smooth_logits = self.lr_smooth_trans[-1](smoothed_feature)



        return local_smooth_logits, global_logits
    
    

def getre_scale(emb):
    emb_softmax = nn.Softmax(dim=1)(emb)
    re = torch.mm(emb_softmax, emb_softmax.transpose(0,1))
    re_self = torch.unsqueeze(torch.diag(re),1)
    scaling = torch.mm(re_self, torch.transpose(re_self, 0, 1))
    re = re / torch.max(torch.sqrt(scaling),1e-9*torch.ones_like(scaling))
    re = re - torch.diag(torch.diag(re))
    return re

def add_diag(re_matrix, device):
    re_diag = torch.diag(re_matrix)
    re_diag_matrix = torch.diag_embed(re_diag)
    re = re_matrix - re_diag_matrix
    re = re_matrix + torch.eye(re.shape[0]).to(device)
    return re