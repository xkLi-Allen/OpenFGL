import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient

class IsolateClient(BaseClient):
    """
    IsolateClient represents a federated learning client that operates in isolation, without participating 
    in the typical federated aggregation and communication process. This class is intended for use cases where 
    the client trains a model independently and does not send updates back to the server.

    Attributes:
        task (object): The task object containing the model, data, and training configurations.
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the IsolateClient with the provided arguments, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): Unique identifier for the client.
            data (object): The dataset assigned to this client.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the server and clients.
            device (torch.device): Device on which computations will be performed (e.g., CPU or GPU).
        """
        super(IsolateClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        
        
        
    def execute(self):
        """
        Executes the training process for the client in isolation. The client trains its model using the assigned
        data without communicating with the server.
        """
        self.task.train()



    def send_message(self):
        """
        An empty send_message method, as this client does not participate in communication with the server.
        No model updates or messages are sent back to the server.
        """
        pass
        
