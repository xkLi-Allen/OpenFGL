import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient

class FedAvgClient(BaseClient):
    """
    FedAvgClient implements the client-side logic for the Federated Averaging (FedAvg) algorithm,
    introduced in the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    by McMahan et al. (2017). This class extends the BaseClient class and manages local training
    and communication with the server.

    The FedAvg algorithm allows clients to train models locally on their data and send the 
    updated model parameters to the server for aggregation, enabling efficient learning in 
    decentralized environments.

    Attributes:
        None (inherits attributes from BaseClient)
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedAvgClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedAvgClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
            
        
    def execute(self):
        """
        Executes the local training process. This method first synchronizes the local model
        with the global model parameters received from the server, and then trains the model
        on the client's local data.
        """
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.train()

    def send_message(self):
        """
        Sends a message to the server containing the model parameters after training
        and the number of samples in the client's dataset.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }
        
        
        