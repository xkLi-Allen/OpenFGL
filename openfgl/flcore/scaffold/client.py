import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from torch.optim import Optimizer, Adam
import torch
import copy



class ScaffoldClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(ScaffoldClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
            
        self.local_control  =  [torch.zeros_like(p.data, requires_grad=False) for p in self.task.model.parameters()]
        
        
    def execute(self):       
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.step_preprocess = self.step_preprocess
        self.task.train()    
        
        self.update_local_control()
        
    def step_preprocess(self):
        for p, local_control, global_control in zip(self.task.model.parameters(), self.local_control, self.message_pool["server"]["global_control"]):
            if p.grad is None:
                continue
            p.grad.data += global_control - local_control
        
    def update_local_control(self):
        with torch.no_grad():
            for it, (local_state, global_state, global_control) in enumerate(zip(self.task.model.parameters(), self.message_pool["server"]["weight"], self.message_pool["server"]["global_control"])):
                self.local_control[it].data = self.local_control[it].data - global_control.data + (global_state.data - local_state.data) / (self.args.num_epochs * self.args.lr)
        

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "local_control": self.local_control
            }
        

            
        