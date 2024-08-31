import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
import os
from openfgl.flcore.fedtad._utils import cal_topo_emb
from openfgl.flcore.fedtad.fedtad_config import config

class FedTADClient(BaseClient):
    """
    FedTADClient implements the client-side operations for the Federated Learning algorithm
    described in the paper 'FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph 
    Federated Learning'. This class handles the local training, model updates, and knowledge 
    distillation process based on topological data.

    Attributes:
        ckr (torch.Tensor): Class-wise Knowledge Reliability (CKR) vector, which stores the 
                            reliability of the topological knowledge for each class.
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedTADClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedTADClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.fedtad_initialization()
        
        
        
    def execute(self):
        """
        Executes the local training process.

        The method first synchronizes the local model with the global model weights received from 
        the server. Then, it trains the local model using the client's data.
        """
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.train()



    def send_message(self):
        """
        Sends the model parameters and the Class-wise Knowledge Reliability (CKR) to the server.

        The message sent to the server includes the number of samples, the model weights, and 
        the CKR vector for the client's local data.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "ckr": self.ckr,
                "model": self.task.model
            }
    
    
    
    def fedtad_initialization(self):        
        """
        Initializes the Class-wise Knowledge Reliability (CKR) based on the topological data.

        This method calculates the CKR by computing topological embeddings for the graph structure 
        and using cosine similarity between nodes to update the CKR for each class. The CKR is 
        saved to disk if the configuration allows.
        """
        ckr_root = os.path.join(self.task.data_dir, "ckr")
        ckr_filename = os.path.join(ckr_root, f"client_{self.client_id}.pt")
        
        if os.path.exists(ckr_filename):
            ckr = torch.load(ckr_filename).to(self.device)
        else:
            ckr = torch.zeros(self.task.num_global_classes).to(self.device)
            data = self.task.data  
            graph_emb = cal_topo_emb(edge_index=data.edge_index, num_nodes=self.task.num_samples, max_walk_length=config["max_walk_length"]).to(self.device)    
            ft_emb = torch.cat((data.x, graph_emb), dim=1).to(self.device)
            for train_i in self.task.train_mask.nonzero().squeeze():
                neighbor = data.edge_index[1,:][data.edge_index[0, :] == train_i] 
                node_all = 0
                for neighbor_j in neighbor:
                    node_kr = torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
                    node_all += node_kr
                node_all += 1
                node_all /= (neighbor.shape[0] + 1)
                
                label = data.y[train_i]
                ckr[label] += node_all
            
            ckr = ckr / ckr.sum(0)
            if config["save_ckr"]:
                os.makedirs(ckr_root, exist_ok=True)
                torch.save(ckr, ckr_filename)
        
        self.ckr = ckr
        