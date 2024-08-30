import torch
import torch.nn.functional as F
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedgta._utils import info_entropy_rev, compute_moment
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import degree
from openfgl.flcore.fedgta.fedgta_config import config





class FedGTAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedGTAClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.LP = LabelPropagation(num_layers=config["prop_steps"], alpha=config["lp_alpha"])
        self.num_neig = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long) + degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        self.train_label_onehot = F.one_hot(self.task.data.y[self.task.train_mask].view(-1), self.task.num_global_classes).to(torch.float).to(self.device)  
        

        
    def execute(self):
        if f"personalized_{self.client_id}" in self.message_pool["server"]:
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"][f"personalized_{self.client_id}"] ):
                    local_param.data.copy_(global_param)
        else:            
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"] ):
                    local_param.data.copy_(global_param)

        self.task.train()
        self.fedgta_postprocess()
        
        
        
    def fedgta_postprocess(self):
        logits = self.task.evaluate(mute=True)["logits"]
        soft_label = F.softmax(logits.detach(), dim=1)
        output = self.LP.forward(y=soft_label, edge_index=self.task.data.edge_index, mask=self.task.train_mask)
        output_raw = F.softmax(output, dim=1)
        output_dis = F.softmax(output / config["temperature"], dim=1)
        
        
        
        
        output_raw[self.task.train_mask] = self.train_label_onehot
        output_dis[self.task.train_mask] = self.train_label_onehot
        lp_moment_v = compute_moment(x=output_raw, num_moments=config["num_moments"], dim="v", moment_type=config["moment_type"])
        self.lp_moment_v = lp_moment_v.view(-1)
        self.agg_w = info_entropy_rev(output_dis, self.num_neig)
        
        
        
    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "lp_moment_v" : self.lp_moment_v,
                "agg_w" : self.agg_w
            }
        

