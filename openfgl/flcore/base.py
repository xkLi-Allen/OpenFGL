import torch
import torch.nn as nn
from openfgl.utils.basic_utils import load_task


class BaseClient:
    """
    Base class for a client in a federated learning setup.

    Attributes:
        args (Namespace): Arguments containing model and training configurations.
        client_id (int): ID of the client.
        message_pool (object): Pool for managing messages between client and server.
        device (torch.device): Device to run the computations on.
        task (object): Task-specific data and functions loaded via the `load_task` utility.
        personalized (bool): Flag to indicate if the client is using a personalized algorithm.
    """
    def __init__(self, args, client_id, data, data_dir, message_pool, device, personalized=False):
        """
        Initialize the BaseClient with provided arguments and data.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
            personalized (bool, optional): Flag to indicate if the client is using a personalized algorithm. Defaults to False.
        """
        self.args = args
        self.client_id = client_id
        self.message_pool = message_pool
        self.device = device
        self.task = load_task(args, client_id, data, data_dir, device)
        self.personalized = personalized
    
    def execute(self):
        """
        Client local execution. This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def send_message(self):
        """
        Send a message to the server. This method should be implemented by subclasses.
        """
        raise NotImplementedError



class BaseServer:
    """
    Base class for a server in a federated learning setup.

    Attributes:
        args (Namespace): Arguments containing model and training configurations.
        message_pool (object): Pool for managing messages between client and server.
        device (torch.device): Device to run the computations on.
        task (object): Task-specific data and functions loaded via the `load_task` utility.
        personalized (bool): Flag to indicate if the server is using a personalized algorithm.
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device, personalized=False):
        """
        Initialize the BaseServer with provided arguments and data.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global data accessible to the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
            personalized (bool, optional): Flag to indicate if the server is using a personalized algorithm. Defaults to False.
        """
        self.args = args
        self.message_pool = message_pool
        self.device = device
        self.task = load_task(args, None, global_data, data_dir, device)
        self.personalized = personalized
   
    def execute(self):
        """
        Server global execution. This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def send_message(self):
        """
        Send messages to clients. This method should be implemented by subclasses.
        """
        raise NotImplementedError
    