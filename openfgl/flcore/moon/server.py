import torch
import copy
from openfgl.flcore.base import BaseServer


class MoonServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(MoonServer, self).__init__(args, global_data, data_dir, message_pool, device)

   
    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
        
        
    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }