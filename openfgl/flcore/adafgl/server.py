import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.adafgl.adafgl_config import config
from openfgl.flcore.adafgl._utils import adj_initialize
from scipy import sparse as sp 



        
        
        
class AdaFGLServer(BaseServer):
    """
    AdaFGLServer implements the server-side logic for federated learning using the AdaFGL model, as described 
    in the paper "AdaFGL: A New Paradigm for Federated Node Classification with Topology Heterogeneity".
    It extends the BaseServer class by managing the aggregation of model updates from clients, and coordinating 
    the training process across different phases, particularly handling topology heterogeneity.

    Attributes:
        phase (int): Indicates the current phase of the server's operations. It starts at 0 (initial phase) 
                     and switches to 1 (AdaFGL phase) when the vanilla rounds are completed.
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the AdaFGLServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(AdaFGLServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self.phase = 0

    def execute(self):
        """
        Executes the server-side operations. This method handles the switching between
        different phases of the federated learning process, and aggregates model updates
        from clients during the initial phase.
        """
        # switch phase
        if self.message_pool["round"] == config["num_rounds_vanilla"]:
            self.phase = 1
        
        
        # execute
        if self.phase == 0:
            with torch.no_grad():
                num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                    
                    for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                        if it == 0:
                            global_param.data.copy_(weight * local_param)
                        else:
                            global_param.data += weight * local_param
        else:
            pass # do nothing
        
    def send_message(self):
        """
        Sends a message to the clients containing the aggregated model parameters.
        The content of the message depends on the current phase.
        """
        if self.phase == 0:
            self.message_pool["server"] = {
                "weight": list(self.task.model.parameters())
            }
        else:
            self.message_pool["server"] = {}
        
