import torch
import copy
from openfgl.flcore.base import BaseServer


class MoonServer(BaseServer):
    """
    MoonServer implements the server-side logic for Model-contrastive Federated Learning (MOON).
    The server is responsible for aggregating the model parameters from multiple clients based on their 
    contributions (e.g., number of samples) and then sending the updated global model back to the clients.

    Attributes:
        None (inherits all attributes from BaseServer).
    """
    
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the MoonServer with the provided arguments, global data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): The global dataset used by the server.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the clients and the server.
            device (torch.device): The device (CPU or GPU) to be used for computations.
        """
        super(MoonServer, self).__init__(args, global_data, data_dir, message_pool, device)

   
    def execute(self):
        """
        Aggregates the model parameters received from the clients. The aggregation is done based on the 
        proportion of samples each client has, ensuring that the contributions of each client are weighted 
        appropriately in the global model update.
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
        Sends the updated global model parameters to the clients. This is done by placing the model parameters 
        in the `message_pool` under the "server" key, which clients can then retrieve and use to update their 
        local models.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }