import torch
from openfgl.flcore.base import BaseClient
from openfgl.flcore.feddc.feddc_config import config



class FedDCClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedDCClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
            
        self.local_drift = [torch.zeros_like(p, requires_grad=False) for p in self.task.model.parameters()]
        self.last_update = [torch.zeros_like(p, requires_grad=False) for p in self.task.model.parameters()]
        
        
    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, label, mask):
            task_loss = self.task.default_loss_fn(logits[mask], label[mask])
            
            if self.message_pool["round"] != 0:
                loss_drift = 0
                loss_grad = 0        
                
                for local_state, global_state, drift_param, update_param, avg_param in zip(self.task.model.parameters(), self.message_pool["server"]["weight"], self.local_drift, self.last_update, self.message_pool["server"]["avg_update"]):
                    loss_drift += torch.sum(torch.pow(drift_param + local_state - global_state, 2))
                    loss_grad += torch.sum(local_state * update_param - avg_param)
            
                return task_loss + (config["feddc_alpha"] / 2) * loss_drift + (1/(self.args.lr * self.args.num_epochs)) * loss_grad  
            else:
                return task_loss
        
        return custom_loss_fn    
    
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):   
                local_param.data.copy_(global_param)


        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()
        
        with torch.no_grad():
            for it, (update_param, local_state, global_state) in enumerate(zip(self.last_update, self.task.model.parameters(), self.message_pool["server"]["weight"])):
                self.last_update[it] = local_state.detach() - global_state.detach()
                self.local_drift[it] += update_param
        

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "last_update": self.last_update,
                "local_drift": self.local_drift
            }
        

            
        