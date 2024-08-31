import torch
from openfgl.flcore.base import BaseServer
import copy

class ScaffoldServer(BaseServer):
    """
    ScaffoldServer implements the server-side logic for the SCAFFOLD algorithm in Federated Learning.
    SCAFFOLD aims to reduce the variance caused by client drift by introducing control variates (local and global control variables)
    that adjust the client updates during training.

    Attributes:
        global_control (list[torch.Tensor]): A list of tensors representing the global control variates for each parameter in the model.
    """
    
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the ScaffoldServer with the provided arguments, global data, and device.
        
        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): The global dataset used for training (if applicable).
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the server and the clients.
            device (torch.device): Device to run the computations on (CPU or GPU).
        """
        super(ScaffoldServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.global_control  =  [torch.zeros_like(p.data, requires_grad=False) for p in self.task.model.parameters()]
   
   
   
    def execute(self):
        """
        Executes the aggregation of client updates by averaging the local model parameters from the sampled clients.
        It also updates the global control variates based on the local control variates received from the clients.
        """
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
        """
        Sends the updated global model parameters and global control variates to the clients after the aggregation step.
        This information is used by the clients to adjust their local updates in the next round.
        """
        self.message_pool["server"] = {
            "global_control": self.global_control,
            "weight": list(self.task.model.parameters())
        }
    
    def update_global_control(self):
        """
        Updates the global control variates by averaging the local control variates from the sampled clients.
        This step is crucial for mitigating the variance caused by client drift in the federated learning process.
        """
        with torch.no_grad():
            for it_, client_id in enumerate(self.message_pool["sampled_clients"]):
                for it, local_param in enumerate(self.message_pool[f"client_{client_id}"]["local_control"]):
                    if it_ == 0:
                        self.global_control[it].data.copy_(1 / len(self.message_pool["sampled_clients"]) * local_param)
                    else:
                        self.global_control[it] += 1 / len(self.message_pool["sampled_clients"]) * local_param
