import torch
from openfgl.flcore.base import BaseClient
from openfgl.flcore.feddc.feddc_config import config



class FedDCClient(BaseClient):
    """
    FedDCClient is a client implementation for the Federated Learning algorithm with 
    Drift Decoupling and Correction (FedDC). It extends the BaseClient class and manages 
    local training while correcting for local drift to handle non-IID data effectively.

    Attributes:
        local_drift (list): A list of tensors representing the accumulated drift for each model parameter.
        last_update (list): A list of tensors representing the last update applied to each model parameter.
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedDCClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedDCClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
            
        self.local_drift = [torch.zeros_like(p, requires_grad=False) for p in self.task.model.parameters()]
        self.last_update = [torch.zeros_like(p, requires_grad=False) for p in self.task.model.parameters()]
        
        
    def get_custom_loss_fn(self):
        """
        Returns a custom loss function for the FedDC algorithm. This loss function accounts
        for local drift correction in addition to the standard task loss.

        Returns:
            custom_loss_fn (function): A custom loss function.
        """
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
        """
        Executes the local training process. This method first synchronizes the local model
        with the global model parameters received from the server, then trains the model
        using a custom loss function that incorporates drift correction. After training, it
        updates the local drift and last update tensors.
        """
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
        """
        Sends a message to the server containing the model parameters after training, along with
        the last update and local drift tensors, and the number of samples in the client's dataset.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "last_update": self.last_update,
                "local_drift": self.local_drift
            }
        

            
        