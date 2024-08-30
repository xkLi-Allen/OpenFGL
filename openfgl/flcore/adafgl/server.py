import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.adafgl.adafgl_config import config
from openfgl.flcore.adafgl._utils import adj_initialize
from scipy import sparse as sp 



        
        
        
class AdaFGLServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(AdaFGLServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self.phase = 0

    def execute(self):
        
        # switch phase
        if self.message_pool["round"] == config["num_rounds_vanilla"]:
            self.phase = 1
        
        
        # execute
        if self.phase == 0:
            with torch.no_grad():
                num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                    
                    for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                        if it == 0:
                            global_param.data.copy_(weight * local_param)
                        else:
                            global_param.data += weight * local_param
        else:
            pass # do nothing
        
    def send_message(self):
        if self.phase == 0:
            self.message_pool["server"] = {
                "weight": list(self.task.model.parameters())
            }
        else:
            self.message_pool["server"] = {}
        
