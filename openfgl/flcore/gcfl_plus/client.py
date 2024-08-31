import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
import numpy as np
from openfgl.flcore.gcfl_plus.models import GIN


class GCFLPlusClient(BaseClient):
    """
    GCFLPlusClient implements the client-side functionality for the Federated Graph Classification framework (GCFL+).
    This client is designed to operate on non-IID graphs and includes personalized training, weight updating, and 
    message passing functionalities. It builds on a Graph Isomorphism Network (GIN) model for graph classification tasks.

    Attributes:
        task (object): The task object containing the model and data configurations.
        W (dict): A dictionary containing the current model parameters.
        dW (dict): A dictionary to store the differences between the current and previous model parameters.
        W_old (dict): A dictionary to store a copy of the model parameters before training.
        gconvNames (list): A list of the names of the graph convolution layers in the model, which are updated during training.
    """
    
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the GCFLPlusClient with the provided arguments, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): The unique identifier for the client.
            data (object): The client's local graph data.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the client and server.
            device (torch.device): Device on which computations will be performed (e.g., CPU or GPU).
        """
        super(GCFLPlusClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.task.load_custom_model(GIN(nfeat=self.task.num_feats,nhid=self.args.hid_dim,nlayer=self.args.num_layers,nclass=self.task.num_global_classes,dropout=self.args.dropout))

        self.W = {key: value for key, value in self.task.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.task.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.task.model.named_parameters()}
        self.gconvNames = None
        # for k,v in self.task.model.named_parameters():
        #     if 'convs' in k:
        #         self.gconvNames.append(k)



    def execute(self):
        """
        Executes the local training process on the client's data. During the first round, the client initializes the 
        graph convolutional layer names. It then updates the model weights based on the server's clustered model 
        weights, trains the model, and calculates the gradients for the graph convolutional layers.
        """
        if self.message_pool["round"] == 0:
            self.gconvNames = self.message_pool["server"]["cluster_weights"][0][0].keys()

        for i, ids in enumerate(self.message_pool["server"]["cluster_indices"]):
            if self.client_id in ids:
                j = ids.index(self.client_id)
                tar = self.message_pool["server"]["cluster_weights"][i][j]
                for k in tar:
                    self.W[k].data = tar[k].data.clone()


        for k in self.gconvNames:
            self.W_old[k].data = self.W[k].data.clone()

        self.task.train()

        for k in self.gconvNames:
            self.dW[k].data = self.W[k].data.clone() - self.W_old[k].data.clone()

        # self.weightsNorm = torch.norm(flatten(self.W)).item()
        #
        # weights_conv = {key: self.W[key] for key in self.gconvNames}
        # self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()
        #
        # dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        # self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        # grads = {key: value.grad for key, value in self.W.items()}
        # self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

        for k in self.gconvNames:
            self.W[k].data = self.W_old[k].data.clone()

    def send_message(self):
        """
        Sends the updated model parameters, gradient norms, and weight differences to the server.
        This information will be used by the server to update the global model and cluster assignments.

        The message contains:
            - num_samples: The number of samples the client trained on.
            - W: The current model parameters.
            - convGradsNorm: The norm of the gradients for the graph convolutional layers.
            - dW: The differences between the current and previous model parameters.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "W": self.W,
            "convGradsNorm": self.convGradsNorm,
            "dW": self.dW
        }



def flatten(w):
    """
    Flattens a dictionary of tensors into a single tensor.

    Args:
        w (dict): A dictionary where the values are tensors.

    Returns:
        torch.Tensor: A flattened tensor containing all elements from the input tensors.
    """
    return torch.cat([v.flatten() for v in w.values()])