import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedproto.fedproto_config import config

class FedProtoClient(BaseClient):
    """
    FedProtoClient is a client implementation for the Federated Prototype Learning (FedProto) framework. 
    This client handles the local training of models, computes class-specific prototypes, and interacts 
    with the server to contribute to the global model updates.

    Attributes:
        local_prototype (dict): A dictionary storing the local prototypes for each class after training.
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedProtoClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedProtoClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.local_prototype = {}
    
    
    def execute(self):
        """
        Executes the local training process. This method sets a custom loss function that incorporates 
        the prototype-based regularization term, performs local training, and then updates the local 
        prototypes for each class.
        """
        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()
        self.update_local_prototype()


    def get_custom_loss_fn(self):
        """
        Returns a custom loss function for the FedProto framework. This loss function combines the standard 
        cross-entropy loss with an additional prototype-based regularization term.

        Returns:
            custom_loss_fn (function): A custom loss function.
        """
        def custom_loss_fn(embedding, logits, label, mask):
            if self.message_pool["round"] == 0 or self.task.num_samples != label.shape[0]: # first round or eval on global
                return self.task.default_loss_fn(logits[mask], label[mask]) 
            else:
                loss_fedproto = 0
                for class_i in range(self.task.num_global_classes):
                    selected_idx = self.task.train_mask & (label == class_i)
                    if selected_idx.sum() == 0:
                        continue
                    input = embedding[selected_idx]
                    target = self.message_pool["server"]["global_prototype"][class_i].expand_as(input)
                    loss_fedproto += nn.MSELoss()(input, target)
                return self.task.default_loss_fn(logits[mask], label[mask]) + config["fedproto_lambda"] * loss_fedproto 
        return custom_loss_fn    
    
    
    def update_local_prototype(self):
        """
        Updates the local prototypes for each class after local training. The prototypes are calculated 
        as the mean of the embeddings of the samples belonging to each class.
        """
        with torch.no_grad():
            embedding = self.task.evaluate(mute=True)["embedding"]
            for class_i in range(self.task.num_global_classes):
                selected_idx = self.task.train_mask & (self.task.data.y.to(self.device) == class_i)
                if selected_idx.sum() == 0:
                    self.local_prototype[class_i] = torch.zeros(self.args.hid_dim).to(self.device)
                else:
                    input = embedding[selected_idx]
                    self.local_prototype[class_i] = torch.mean(input, dim=0)
  
            
    def send_message(self):
        """
        Sends a message to the server containing the number of samples used for training and the 
        local prototypes for each class. This information is used by the server to update the 
        global prototypes and the global model.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "local_prototype": self.local_prototype
            }
        
