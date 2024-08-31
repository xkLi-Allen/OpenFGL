import torch
from openfgl.flcore.base import BaseServer


class FedDCServer(BaseServer):
    """
    FedDCServer is a server implementation for the Federated Learning algorithm with 
    Drift Decoupling and Correction (FedDC). It extends the BaseServer class and manages 
    the aggregation of client model updates, correcting for local drift, and computes the 
    average updates across all clients.

    Attributes:
        avg_update (list): A list of tensors representing the average update across all clients for each model parameter.
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedDCServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedDCServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.avg_update = [torch.zeros_like(p) for p in self.task.model.parameters()]
   
   
    def execute(self):
        """
        Executes the server-side operations. This method aggregates the model updates from 
        the clients by computing a weighted average of the model parameters, considering 
        both the local parameters and the local drift corrections from each client. 
        It also computes the average of the last updates from all clients.
        """
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                
                for (local_param, global_param, drift_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters(), self.message_pool[f"client_{client_id}"]["local_drift"]):
                    if it == 0:
                        global_param.data.copy_(weight * (local_param + drift_param))
                    else:
                        global_param.data += weight * (local_param + drift_param)

            for it_, client_id in enumerate(self.message_pool["sampled_clients"]):
                for it, update_param in enumerate(self.message_pool[f"client_{client_id}"]["last_update"]):
                    if it_ == 0:
                        self.avg_update[it] = torch.zeros_like(update_param).to(self.device)
                    self.avg_update[it] += update_param
                    if it_ == len(self.message_pool["sampled_clients"]) - 1:
                        self.avg_update[it] /= len(self.message_pool["sampled_clients"])
        
    def send_message(self):
        """
        Sends a message to the clients containing the updated global model parameters after 
        aggregation, along with the average of the last updates from all clients.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "avg_update": self.avg_update
        }