import torch
from openfgl.flcore.base import BaseServer

class FGSSLServer(BaseServer):
    """
    FGSSLServer implements the server-side functionality for the Federated Graph Semantic and Structural Learning (FGSSL)
    framework. The server is responsible for aggregating model parameters from multiple clients and sending the updated 
    global model parameters back to the clients.

    Attributes:
        task (object): The task object containing the model and data configurations.
        message_pool (dict): A pool for managing messages exchanged between the server and clients.
    """
    
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FGSSLServer with the provided arguments, global data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Data that might be used by the server for global operations.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the server and clients.
            device (torch.device): Device on which computations will be performed (e.g., CPU or GPU).
        """
        super(FGSSLServer, self).__init__(args, global_data, data_dir, message_pool, device)
        

   
    def execute(self):
        """
        Executes the server's main function: aggregating the model parameters from the clients. 
        The aggregation is weighted by the number of samples each client holds. The updated global 
        model parameters are computed and stored.
        """
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
        
    def send_message(self):
        """
        Sends the aggregated global model parameters to the clients. The updated parameters are 
        stored in the message pool, which will be accessed by the clients.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }