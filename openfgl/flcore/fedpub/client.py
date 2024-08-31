import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedpub.fedpub_config import config
from openfgl.flcore.fedpub.maskedgcn import MaskedGCN


class FedPubClient(BaseClient):
    """
    FedPubClient is a client implementation for the Personalized Subgraph Federated Learning 
    (FedPub) framework, introduced in the paper "Personalized Subgraph Federated Learning." 
    This client handles local model training with a customized GCN model that incorporates 
    mask-based personalization and subgraph-specific adjustments.

    Attributes:
        prev_w (dict): Stores the previous global model weights for the current client, 
                       used for regularization in the custom loss function.
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedPubClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedPubClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.task.load_custom_model(MaskedGCN(input_dim=self.task.num_feats, hid_dim=self.args.hid_dim, output_dim=self.task.num_global_classes, l1=config["l1"], laye_mask_one=config["laye_mask_one"], clsf_mask_one=config["clsf_mask_one"]))
        
        
        
        
    def get_custom_loss_fn(self):
        """
        Returns a custom loss function for the FedPub framework. This loss function combines 
        the standard cross-entropy loss with additional L1 regularization on mask parameters 
        and L2 regularization on model parameters, penalizing deviation from the previous 
        global model weights.

        Returns:
            custom_loss_fn (function): A custom loss function that includes mask regularization 
                                       and model weight regularization.
        """
        def custom_loss_fn(embedding, logits, label, mask):
            loss = torch.nn.functional.cross_entropy(logits[mask], label[mask])
            for name, param in self.task.model.state_dict().items():
                if 'mask' in name:
                    loss += torch.norm(param.float(), 1) * config["l1"]
                elif 'conv' in name or 'clsif' in name:
                    if self.message_pool['round'] == 0: continue
                    loss += torch.norm(param.float() - self.prev_w[name], 2) * config["loc_l2"]
            return loss
        return custom_loss_fn



    def execute(self):
        """
        Executes the local training process. The method first loads the appropriate model 
        weights (either personalized or global), sets the custom loss function, and then 
        trains the model locally.
        """
        if f'personalized_{self.client_id}' in self.message_pool["server"]:
            weight = self.message_pool["server"][f'personalized_{self.client_id}']
        else:
            weight = self.message_pool["server"]["weight"]

        self.prev_w = weight
        model_state = self.task.model.state_dict()
        for k, v in weight.items():
            if 'running' in k or 'tracked' in k:
                weight[k] = model_state[k]
                continue
            if 'mask' in k or 'pre' in k or 'pos' in k:
                weight[k] = model_state[k]
                continue

        self.task.model.load_state_dict(weight)
        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()



    @torch.no_grad()
    def get_functional_embedding(self):
        """
        Computes the functional embedding of the model based on a proxy input from the server. 
        This embedding is used to assess the client's contribution to the global model.

        Returns:
            numpy.ndarray: The computed functional embedding as a numpy array.
        """
        self.task.model.eval()
        with torch.no_grad():
            proxy_in = self.message_pool['server']['proxy']
            proxy_in = proxy_in.to(self.device)
            proxy_out,_ = self.task.model(proxy_in)
            proxy_out = proxy_out.mean(dim=0)
            proxy_out = proxy_out.clone().detach().cpu().numpy()
        return proxy_out



    def send_message(self):
        """
        Sends a message to the server containing the local model parameters, the number 
        of samples used for training, and the computed functional embedding. This information 
        is used by the server to update the global model and track the client's contribution.

        The message includes:
            - num_samples: The number of samples used in local training.
            - weight: The updated local model parameters.
            - functional_embedding: The computed functional embedding of the model.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": self.task.model.state_dict(),
            "functional_embedding" : self.get_functional_embedding()
        }





