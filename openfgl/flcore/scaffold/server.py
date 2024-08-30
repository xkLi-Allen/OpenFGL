import torch
from openfgl.flcore.base import BaseServer
import copy

class ScaffoldServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(ScaffoldServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.global_control  =  [torch.zeros_like(p.data, requires_grad=False) for p in self.task.model.parameters()]
   
    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param.data)
                    else:
                        global_param.data += weight * local_param.data
        
        self.update_global_control()
        
    def send_message(self):
        self.message_pool["server"] = {
            "global_control": self.global_control,
            "weight": list(self.task.model.parameters())
        }
    
    def update_global_control(self):
        with torch.no_grad():
            for it_, client_id in enumerate(self.message_pool["sampled_clients"]):
                for it, local_param in enumerate(self.message_pool[f"client_{client_id}"]["local_control"]):
                    if it_ == 0:
                        self.global_control[it].data.copy_(1 / len(self.message_pool["sampled_clients"]) * local_param)
                    else:
                        self.global_control[it] += 1 / len(self.message_pool["sampled_clients"]) * local_param
