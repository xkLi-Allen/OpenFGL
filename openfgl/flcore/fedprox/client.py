import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedprox.fedprox_config import config



class FedProxClient(BaseClient):
    """
    FedProxClient is a client implementation for the Federated Proximal (FedProx) framework, 
    introduced in the paper "Federated Optimization in Heterogeneous Networks." This client 
    handles local training with a custom loss function that includes a proximal term, 
    designed to address the challenges of heterogeneity in federated learning environments.

    Attributes:
        None
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedProxClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedProxClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        
        
    def get_custom_loss_fn(self):
        """
        Returns a custom loss function for the FedProx framework. This loss function combines 
        the standard task-specific loss (e.g., cross-entropy) with a proximal term that penalizes 
        the deviation of local model parameters from the global model parameters.

        Returns:
            custom_loss_fn (function): A custom loss function that includes the proximal term.
        """
        def custom_loss_fn(embedding, logits, label, mask):
            loss_fedprox = 0
            for local_param, global_param in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                loss_fedprox += config["fedprox_mu"] / 2 * (local_param - global_param).norm(2)**2
            return self.task.default_loss_fn(logits[mask], label[mask]) + loss_fedprox
        
        return custom_loss_fn    
    
    
    
    def execute(self):
        """
        Executes the local training process. This method first synchronizes the local model 
        with the global model parameters received from the server, and then trains the model 
        locally using the custom loss function that includes the FedProx proximal term.
        """
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):   
                local_param.data.copy_(global_param)


        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()




    def send_message(self):
        """
        Sends a message to the server containing the local model parameters and the number 
        of samples used for training. This information is used by the server to update the 
        global model parameters.

        The message includes:
            - num_samples: The number of samples used in local training.
            - weight: The updated local model parameters.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }
        

        