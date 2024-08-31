import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedsage_plus.fedsage_plus_config import config
from openfgl.flcore.fedsage_plus.locsage_plus import LocSAGEPlus

class FedSagePlusServer(BaseServer):
    """
    FedSagePlusServer is the server-side implementation for the Federated Learning algorithm 
    described in the paper 'Subgraph Federated Learning with Missing Neighbor Generation'.
    It manages the global model, coordinates the phases of training across clients, and 
    aggregates the clients' updates.

    Attributes:
        phase (int): Indicates the current phase of training. It can be 0 (neighbor generation) or 1 (classifier aggregation).
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedSagePlusServer.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global data used for the federated learning process.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between clients and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedSagePlusServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.load_custom_model(LocSAGEPlus(input_dim=self.task.num_feats, 
                                                        hid_dim=self.args.hid_dim, 
                                                        latent_dim=config["latent_dim"], 
                                                        output_dim=self.task.num_global_classes, 
                                                        max_pred=config["max_pred"], 
                                                        dropout=self.args.dropout))

    def execute(self):
        """
        Executes the server-side operations for each round. Depending on the phase, it either 
        does nothing (in phase 0) or aggregates the classifier parameters from all clients 
        (in phase 1).
        """
        # switch phase
        if self.message_pool["round"] == 0:
            self.phase = 0
        elif self.message_pool["round"] == config["gen_rounds"] - 1: # last round for neighGen, server should switch phase 1
            self.phase = 1
            
        # execute
        if self.phase == 0: # do nothing
            pass
        elif self.phase == 1: # classifier aggregation
            with torch.no_grad():
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = 1 / len(self.message_pool["sampled_clients"])
                    for (local_param, global_param_with_name) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.named_parameters()):
                        name = global_param_with_name[0]
                        global_param = global_param_with_name[1]       
                        if "classifier" in name:
                            if it == 0:
                                global_param.data.copy_(weight * local_param)
                            else:
                                global_param.data += weight * local_param
                
            
    def send_message(self):
        """
        Sends the updated global model parameters to the clients after each round of execution.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }