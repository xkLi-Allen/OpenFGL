import copy
import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedgta.fedgta_config import config


class FedGTAServer(BaseServer):
    """
    FedGTAServer is a server implementation for the Federated Graph Learning framework 
    with Topology-aware Averaging (FedGTA). This server manages the aggregation of model 
    parameters from multiple clients, applies personalized model updates, and handles 
    communication with the clients.

    Attributes:
        aggregated_models (list): A list of model copies, one for each client, used for 
                                  personalized model aggregation.
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedGTAServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedGTAServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self.aggregated_models = [copy.deepcopy(self.task.model) for _ in range(self.args.num_clients)]
    
    
    
    def switch_personalized_global_model(self, client_id):
        """
        Switches between the personalized and global models for a specific client. If the 
        server has a personalized model for the client, it loads it into the global model; 
        otherwise, it loads the general global model.

        Args:
            client_id (int): The ID of the client for which the model switch is being performed.
        """
        if f"personalized_{client_id}" in self.message_pool["server"]:
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"][f"personalized_{client_id}"] ):
                    local_param.data.copy_(global_param)
        else:            
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"] ):
                    local_param.data.copy_(global_param)
        
        
    def execute(self):
        """
        Executes the server-side operations for aggregating model parameters. The method 
        computes the similarity between clients based on their label propagation moments 
        and aggregates the models from similar clients to create personalized models.
        """
        agg_client_list = {}
        for client_id in self.message_pool["sampled_clients"]:
            agg_client_list[client_id] = []
            sim = torch.tensor([torch.cosine_similarity(self.message_pool[f"client_{client_id}"]["lp_moment_v"], 
                                                        self.message_pool[f"client_{target_id}"]["lp_moment_v"], dim=0) 
                                for target_id in self.message_pool["sampled_clients"]]).to(self.device)
            accept_idx = torch.where(sim > config["accept_alpha"])
            agg_client_list[client_id] = [self.message_pool["sampled_clients"][idx] for idx in accept_idx[0].tolist()]
        
        
        
        for src, clients_list in agg_client_list.items():            
            with torch.no_grad():
                tot_w = [self.message_pool[f"client_{client_id}"]["agg_w"] for client_id in clients_list]
                for it, client_id in enumerate(clients_list):
                    weight = tot_w[it] / sum(tot_w)
                    for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.aggregated_models[src].parameters()):
                        if it == 0:
                            global_param.data.copy_(weight * local_param)
                        else:
                            global_param.data += weight * local_param

        
    def send_message(self):
        """
        Sends a message to the clients. In the initial round, only the global model parameters 
        are sent. In subsequent rounds, the server also sends personalized models to each client 
        based on the results of the aggregation process.
        """
        if self.message_pool["round"] == 0:
            self.message_pool["server"] = {
                "weight": list(self.task.model.parameters())
            }
        else:
            self.message_pool["server"] = {}
            for client_id in self.message_pool["sampled_clients"]:
                self.message_pool["server"][f'personalized_{client_id}'] = list(self.aggregated_models[client_id].parameters())