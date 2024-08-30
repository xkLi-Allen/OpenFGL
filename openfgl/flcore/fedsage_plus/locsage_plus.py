import torch
import torch.nn as nn
import torch.nn.functional as F
from openfgl.model.graphsage import GraphSAGE
from torch_geometric.data import Data


    


class dGen(nn.Module):
    def __init__(self, latent_dim):
        super(dGen,self).__init__()
        self.reg = nn.Linear(latent_dim, 1)

    def forward(self,x):
        x = F.relu(self.reg(x))
        return x



class fGen(nn.Module):
    def __init__(self,latent_dim, max_pred, feat_shape, dropout):
        super(fGen, self).__init__()
        self.max_pred = max_pred
        self.feat_shape = feat_shape

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256,2048)
        self.fc_flat = nn.Linear(2048, self.max_pred * self.feat_shape)

        self.dropout = dropout

    def forward(self, x):
        # add random gaussian noise
        x = x + torch.normal(0, 1, size=x.shape).to(x.device)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        
        x = x.view(-1, self.max_pred, self.feat_shape)
        return x


class NeighGen(nn.Module):
    
    
    def __init__(self, input_dim, hid_dim, latent_dim, max_pred, dropout):
        super(NeighGen, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.max_pred = max_pred
        self.encoder = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, output_dim=latent_dim, num_layers=2, dropout=dropout)

        self.dGen = dGen(latent_dim)
        self.fGen = fGen(latent_dim=latent_dim, max_pred=max_pred, feat_shape=input_dim, dropout=dropout)
        
        

    def mend(self, impaired_data, pred_degree_float, pred_neig_feat):
        num_impaired_nodes = impaired_data.x.shape[0]    
        ptr = num_impaired_nodes
        remain_feat = []
        remain_edges = []

        pred_degree = torch._cast_Int(pred_degree_float).detach()
        
        for impaired_node_i in range(num_impaired_nodes):
            for gen_neighbor_j in range(min(self.max_pred, pred_degree[impaired_node_i])):
                remain_feat.append(pred_neig_feat[impaired_node_i, gen_neighbor_j])
                remain_edges.append(torch.tensor([impaired_node_i, ptr]).view(2, 1).to(pred_degree.device))
                ptr += 1
                
        
        
        if pred_degree.sum() > 0:
            mend_x = torch.vstack((impaired_data.x, torch.vstack(remain_feat)))
            mend_edge_index = torch.hstack((impaired_data.edge_index, torch.hstack(remain_edges)))
            mend_y = torch.hstack((impaired_data.y, torch.zeros(ptr-num_impaired_nodes).long().to(pred_degree.device)))
        else:
            mend_x = torch.clone(impaired_data.x)
            mend_edge_index = torch.clone(impaired_data.edge_index)
            mend_y = torch.clone(impaired_data.y)
        
        mend_data = Data(x=mend_x, edge_index=mend_edge_index, y=mend_y)
        return mend_data
        
        
        
    def forward(self, data):
        _, node_encoding = self.encoder(data)
        pred_degree = self.dGen(node_encoding).squeeze() # [N]
        pred_neig_feat = self.fGen(node_encoding) # [N, max_pred, feat_dim]

        mend_graph = self.mend(data, pred_degree, pred_neig_feat)
        
        return pred_degree, pred_neig_feat, mend_graph



class LocSAGEPlus(nn.Module):
    
    def __init__(self, input_dim, hid_dim, latent_dim, output_dim, max_pred, dropout):
        super(LocSAGEPlus, self).__init__()
        
        self.neighGen = NeighGen(input_dim, hid_dim, latent_dim, max_pred, dropout)
        self.classifier = GraphSAGE(input_dim, hid_dim, output_dim, num_layers=2, dropout=dropout)
        
        self.output_pred_degree = None
        self.output_pred_neig_feat = None
        self.output_mend_graph = None 
        self.phase = 0
        
    def forward(self, data):
        if self.phase == 0:
            pred_degree, pred_neig_feat, mend_graph = self.neighGen.forward(data)
            mend_embedding, mend_logits = self.classifier.forward(mend_graph)
            
            self.output_pred_degree = pred_degree
            self.output_pred_neig_feat = pred_neig_feat
            self.output_mend_graph = mend_graph
            return mend_embedding, mend_logits # 原始的节点的顺序都没变切在最前面
        else:
            fill_embedding, fill_logits = self.classifier(data)
            return fill_embedding, fill_logits