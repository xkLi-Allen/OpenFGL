import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
import os
from openfgl.flcore.fedtad._utils import cal_topo_emb
from openfgl.flcore.fedtad.fedtad_config import config

class FedTADClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedTADClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.fedtad_initialization()
        
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "ckr": self.ckr,
                "model": self.task.model
            }
    
    def fedtad_initialization(self):        
        ckr_root = os.path.join(self.task.data_dir, "ckr")
        ckr_filename = os.path.join(ckr_root, f"client_{self.client_id}.pt")
        
        if os.path.exists(ckr_filename):
            ckr = torch.load(ckr_filename).to(self.device)
        else:
            ckr = torch.zeros(self.task.num_global_classes).to(self.device)
            data = self.task.data  
            graph_emb = cal_topo_emb(edge_index=data.edge_index, num_nodes=self.task.num_samples, max_walk_length=config["max_walk_length"]).to(self.device)    
            ft_emb = torch.cat((data.x, graph_emb), dim=1).to(self.device)
            for train_i in self.task.train_mask.nonzero().squeeze():
                neighbor = data.edge_index[1,:][data.edge_index[0, :] == train_i] 
                node_all = 0
                for neighbor_j in neighbor:
                    node_kr = torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
                    node_all += node_kr
                node_all += 1
                node_all /= (neighbor.shape[0] + 1)
                
                label = data.y[train_i]
                ckr[label] += node_all
            
            ckr = ckr / ckr.sum(0)
            if config["save_ckr"]:
                os.makedirs(ckr_root, exist_ok=True)
                torch.save(ckr, ckr_filename)
        
        self.ckr = ckr
        