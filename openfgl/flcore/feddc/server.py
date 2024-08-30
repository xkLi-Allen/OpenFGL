import torch
from openfgl.flcore.base import BaseServer


class FedDCServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedDCServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.avg_update = [torch.zeros_like(p) for p in self.task.model.parameters()]
   
    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                
                for (local_param, global_param, drift_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters(), self.message_pool[f"client_{client_id}"]["local_drift"]):
                    if it == 0:
                        global_param.data.copy_(weight * (local_param + drift_param))
                    else:
                        global_param.data += weight * (local_param + drift_param)

            for it_, client_id in enumerate(self.message_pool["sampled_clients"]):
                for it, update_param in enumerate(self.message_pool[f"client_{client_id}"]["last_update"]):
                    if it_ == 0:
                        self.avg_update[it] = torch.zeros_like(update_param).to(self.device)
                    self.avg_update[it] += update_param
                    if it_ == len(self.message_pool["sampled_clients"]) - 1:
                        self.avg_update[it] /= len(self.message_pool["sampled_clients"])
        
    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "avg_update": self.avg_update
        }