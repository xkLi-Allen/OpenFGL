import torch
from openfgl.flcore.base import BaseServer

class IsolateServer(BaseServer):
    """
    IsolateServer represents a federated learning server that operates in isolation. Unlike a typical 
    federated server, it does not aggregate client updates or communicate with clients. This class 
    is intended for scenarios where each client trains independently without contributing to a global model.

    Attributes:
        task (object): The task object containing the global model, data, and training configurations.
    """
    
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the IsolateServer with the provided arguments, global data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): The global dataset for server-side operations, if any.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the server and clients.
            device (torch.device): Device on which computations will be performed (e.g., CPU or GPU).
        """
        super(IsolateServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        
   
    def execute(self):
        """
        Executes the server's operations. In this isolated setup, the server does not perform any aggregation 
        or other federated learning tasks. It asserts that all clients are sampled but does not process any updates.
        """
        assert len(self.message_pool["sampled_clients"]) == self.args.num_clients
        pass
        
    def send_message(self):
        """
        An empty send_message method, as this server does not communicate with clients.
        No messages or model updates are sent to the clients.
        """
        pass