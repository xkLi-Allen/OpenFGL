import torch
from openfgl.flcore.base import BaseServer

class FedProtoServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedProtoServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.global_prototype = {}
   
    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for class_i in range(self.task.num_global_classes):
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                    
                    if it == 0:
                        self.global_prototype[class_i] = weight * self.message_pool[f"client_{client_id}"]["local_prototype"][class_i]
                    else:
                        self.global_prototype[class_i] += weight * self.message_pool[f"client_{client_id}"]["local_prototype"][class_i]
            
        
    def send_message(self):
        self.message_pool["server"] = {
            "global_prototype": self.global_prototype
        }