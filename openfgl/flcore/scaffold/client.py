import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from torch.optim import Optimizer, Adam
import torch
import copy



class ScaffoldClient(BaseClient):
    """
    ScaffoldClient implements the client-side logic for the SCAFFOLD algorithm in Federated Learning.
    SCAFFOLD aims to reduce the variance caused by client drift by introducing control variates (local and global control variables)
    that adjust the client updates during training.

    Attributes:
        local_control (list[torch.Tensor]): A list of tensors representing the local control variates for each parameter in the model.
    """
    
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the ScaffoldClient with the provided arguments, client ID, data, and device.
        
        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the client and the server.
            device (torch.device): Device to run the computations on (CPU or GPU).
        """
        super(ScaffoldClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
            
        self.local_control  =  [torch.zeros_like(p.data, requires_grad=False) for p in self.task.model.parameters()]
        
        
    def execute(self):      
        """
        Executes the local training process for the client. It involves updating the local model with the global model
        parameters and applying the control variates to adjust the gradients before training.
        """ 
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.step_preprocess = self.step_preprocess
        self.task.train()    
        
        self.update_local_control()
        
        
    def step_preprocess(self):
        """
        Adjusts the gradients using the local and global control variates before the optimizer step is taken.
        This helps in controlling the variance caused by client drift during the local updates.
        """
        for p, local_control, global_control in zip(self.task.model.parameters(), self.local_control, self.message_pool["server"]["global_control"]):
            if p.grad is None:
                continue
            p.grad.data += global_control - local_control
        
    def update_local_control(self):
        """
        Updates the local control variates based on the difference between the global and local model parameters
        after training. This adjustment is crucial for reducing the variance in the federated learning process.
        """
        with torch.no_grad():
            for it, (local_state, global_state, global_control) in enumerate(zip(self.task.model.parameters(), self.message_pool["server"]["weight"], self.message_pool["server"]["global_control"])):
                self.local_control[it].data = self.local_control[it].data - global_control.data + (global_state.data - local_state.data) / (self.args.num_epochs * self.args.lr)
        

    def send_message(self):
        """
        Sends the updated model parameters and local control variates to the server after local training is completed.
        This information is used by the server to update the global model and control variates for the next round.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "local_control": self.local_control
            }
        

            
        