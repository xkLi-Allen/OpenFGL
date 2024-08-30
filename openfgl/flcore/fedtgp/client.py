import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedtgp.fedtgp_config import config

class FedTGPClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedTGPClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.local_prototype = {}
    
    
    def execute(self):
        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()
        self.update_local_prototype()


    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, label, mask):
            if self.message_pool["round"] == 0 or self.task.num_samples != label.shape[0]: # first round or eval on global
                return self.task.default_loss_fn(logits[mask], label[mask]) 
            else:
                loss_fedtgp = 0
                for class_i in range(self.task.num_global_classes):
                    selected_idx = self.task.train_mask & (label == class_i)
                    if selected_idx.sum() == 0:
                        continue
                    input = embedding[selected_idx]
                    target = self.message_pool["server"]["global_prototype"][class_i].expand_as(input)
                    loss_fedtgp += nn.MSELoss()(input, target)
                return self.task.default_loss_fn(logits[mask], label[mask]) + config["fedtgp_lambda"] * loss_fedtgp
        return custom_loss_fn    
    
    
    def update_local_prototype(self):
        with torch.no_grad():
            embedding = self.task.evaluate(mute=True)["embedding"]
            for class_i in range(self.task.num_global_classes):
                selected_idx = self.task.train_mask & (self.task.data.y.to(self.device) == class_i)
                if selected_idx.sum() == 0:
                    self.local_prototype[class_i] = torch.zeros(self.args.hid_dim).to(self.device)
                else:
                    input = embedding[selected_idx]
                    self.local_prototype[class_i] = torch.mean(input, dim=0)
  
            
    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "local_prototype": self.local_prototype
            }
