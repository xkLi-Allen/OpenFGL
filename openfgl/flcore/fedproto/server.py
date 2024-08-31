import torch
from openfgl.flcore.base import BaseServer

class FedProtoServer(BaseServer):
    """
    FedProtoServer is a server implementation for the Federated Prototype Learning (FedProto) framework. 
    This server is responsible for aggregating local prototypes from clients to update the global prototypes, 
    which are then used in the federated learning process.

    Attributes:
        global_prototype (dict): A dictionary storing the global prototypes for each class, updated 
                                 based on the local prototypes received from the clients.
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedProtoServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedProtoServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.global_prototype = {}
   
   
   
    def execute(self):
        """
        Executes the server-side operations for aggregating local prototypes from clients. 
        The global prototypes for each class are computed as the weighted average of the 
        local prototypes from the sampled clients.
        """
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for class_i in range(self.task.num_global_classes):
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                    
                    if it == 0:
                        self.global_prototype[class_i] = weight * self.message_pool[f"client_{client_id}"]["local_prototype"][class_i]
                    else:
                        self.global_prototype[class_i] += weight * self.message_pool[f"client_{client_id}"]["local_prototype"][class_i]
            
        
        
    def send_message(self):
        """
        Sends a message to the clients containing the updated global prototypes. These prototypes 
        are used by the clients in their local training processes to ensure alignment with the global model.
        """
        self.message_pool["server"] = {
            "global_prototype": self.global_prototype
        }