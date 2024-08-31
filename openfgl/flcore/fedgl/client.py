import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedgl.models import FedGCN
from torch_geometric.utils import to_torch_csr_tensor
from openfgl.flcore.fedgl.fedgl_config import config


class FedGLClient(BaseClient):
    """
    FedGLClient is a client implementation for the Federated Graph Learning (FedGL) framework
    with global self-supervision. It extends the BaseClient class and handles the local training 
    of graph neural networks in a federated learning environment, incorporating global self-supervision 
    through pseudo-labels and global graph structures.

    Attributes:
        adj (torch.Tensor): Sparse adjacency matrix in CSR format representing the local graph structure.
        mask (torch.Tensor): Tensor indicating which nodes are included in the global map, used for masking operations.
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedGLClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(FedGCN(nfeat=self.task.num_feats,nhid=self.args.hid_dim,
                                           nclass=self.task.num_global_classes,nlayer=self.args.num_layers,dropout=self.args.dropout))
        self.adj = to_torch_csr_tensor(self.task.data.edge_index)
        self.mask = torch.tensor(list(self.task.data.global_map.values())).to(self.device)


    def get_custom_loss_fn(self):
        """
        Returns a custom loss function for the FedGL framework. This loss function combines 
        the standard cross-entropy loss with an additional self-supervised learning (SSL) loss 
        based on pseudo-labels and a global graph structure.

        Returns:
            custom_loss_fn (function): A custom loss function.
        """
        def custom_loss_fn(embedding, logits, label, mask):
            loss = torch.nn.functional.cross_entropy(logits[mask], label[mask])
            if self.message_pool["round"] != 0 and config['ssl_loss_weight']>0:
                p_g = self.message_pool["server"]["pseudo_labels"][self.client_id]
                p_m = self.message_pool["server"]["pseudo_labels_mask"][self.client_id]
                local_train_mask = self.task.splitted_data['train_mask'].type(torch.int)
                p_m = p_m - local_train_mask
                p_m[p_m < 0] = 0
                if p_m.sum() == 0:
                    index = torch.where(local_train_mask == 0)[0]
                    tmp = torch.randint(0,index.size(0),(1,))
                    p_m[index[tmp]] = 1
                p_m = p_m.type(torch.bool)
                loss_ssl = torch.nn.functional.cross_entropy(logits[p_m], p_g[p_m].type(torch.long))
                loss += config['ssl_loss_weight'] * loss_ssl

            return loss
        return custom_loss_fn

    def execute(self):
        """
        Executes the local training process. The method synchronizes the local model with the global 
        model parameters received from the server, and if applicable, incorporates the global graph 
        structure into the adjacency matrix. It then trains the model using the custom loss function.
        """
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.loss_fn = self.get_custom_loss_fn()
        if self.message_pool["round"] != 0 and config['pseudo_graph_weight']>0:
            self.task.splitted_data["data"].adj = self.adj + self.message_pool["server"]["whole_adj"][self.client_id].type(torch.float)
        else:
            self.task.splitted_data["data"].adj = self.adj
        self.task.train()


    def send_message(self):
        """
        Sends a message to the server containing the model parameters after training, as well as the embeddings 
        and predictions produced by the model. These are used by the server to update the global model and generate 
        pseudo-labels for self-supervised learning.
        """
        self.task.model.eval()
        emb,pred = self.task.model(self.task.data)

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "mask": self.mask,
            "embeddings" : emb,
            "preds": pred
        }

