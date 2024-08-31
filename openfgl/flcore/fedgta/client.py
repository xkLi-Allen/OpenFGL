import torch
import torch.nn.functional as F
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedgta._utils import info_entropy_rev, compute_moment
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import degree
from openfgl.flcore.fedgta.fedgta_config import config





class FedGTAClient(BaseClient):
    """
    FedGTAClient is a client implementation for the Federated Graph Learning framework 
    with Topology-aware Averaging (FedGTA). This client handles local model training, 
    label propagation, and the computation of topology-aware metrics for federated learning.

    Attributes:
        LP (LabelPropagation): A label propagation model for graph-based semi-supervised learning.
        num_neig (torch.Tensor): Tensor representing the degree (number of neighbors) of each node in the graph.
        train_label_onehot (torch.Tensor): One-hot encoded labels for the training nodes.
        lp_moment_v (torch.Tensor): Computed moments from the label propagation results, used for topology-aware averaging.
        agg_w (torch.Tensor): Aggregation weights based on the information entropy of label propagation results.
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedGTAClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedGTAClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.LP = LabelPropagation(num_layers=config["prop_steps"], alpha=config["lp_alpha"])
        self.num_neig = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long) + degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        self.train_label_onehot = F.one_hot(self.task.data.y[self.task.train_mask].view(-1), self.task.num_global_classes).to(torch.float).to(self.device)  
        

        
    def execute(self):
        """
        Executes the local training process. The method first synchronizes the local model with the 
        global model parameters received from the server, then trains the model locally, and finally 
        performs post-processing using label propagation and topology-aware averaging.
        """
        if f"personalized_{self.client_id}" in self.message_pool["server"]:
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"][f"personalized_{self.client_id}"] ):
                    local_param.data.copy_(global_param)
        else:            
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"] ):
                    local_param.data.copy_(global_param)

        self.task.train()
        self.fedgta_postprocess()
        
        
        
    def fedgta_postprocess(self):
        """
        Performs post-processing after local training. This includes label propagation on the graph, 
        followed by the computation of moment vectors and aggregation weights based on the 
        propagated labels and graph topology.
        """
        logits = self.task.evaluate(mute=True)["logits"]
        soft_label = F.softmax(logits.detach(), dim=1)
        output = self.LP.forward(y=soft_label, edge_index=self.task.data.edge_index, mask=self.task.train_mask)
        output_raw = F.softmax(output, dim=1)
        output_dis = F.softmax(output / config["temperature"], dim=1)
        
        
        
        
        output_raw[self.task.train_mask] = self.train_label_onehot
        output_dis[self.task.train_mask] = self.train_label_onehot
        lp_moment_v = compute_moment(x=output_raw, num_moments=config["num_moments"], dim="v", moment_type=config["moment_type"])
        self.lp_moment_v = lp_moment_v.view(-1)
        self.agg_w = info_entropy_rev(output_dis, self.num_neig)
        
        
        
    def send_message(self):
        """
        Sends a message to the server containing the local model parameters, 
        the computed moment vectors, and the aggregation weights after post-processing.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "lp_moment_v" : self.lp_moment_v,
                "agg_w" : self.agg_w
            }
        

