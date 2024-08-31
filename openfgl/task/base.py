from torch.optim import Adam
    
class BaseTask:
    """
    Base class for defining a task in a federated learning setup.

    Attributes:
        client_id (int): ID of the client.
        data_dir (str): Directory containing the data.
        args (Namespace): Arguments containing model and training configurations.
        device (torch.device): Device to run the computations on.
        data (object): Data specific to the task.
        model (torch.nn.Module): Model to be trained.
        optim (torch.optim.Optimizer): Optimizer for the model.
        override_evaluate (function): Custom evaluation function, if provided.
        step_preprocess (function): Custom preprocessing step, if provided.
    """
    def __init__(self, args, client_id, data, data_dir, device):
        """
        Initialize the BaseTask with provided arguments, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the task.
            data_dir (str): Directory containing the data.
            device (torch.device): Device to run the computations on.
        """

        self.client_id = client_id
        self.data_dir = data_dir
        self.args = args
        self.device = device
        
        if data is not None:
            self.data = data
            if hasattr(self.data, "_data_list"):
                self.data._data_list = None
            self.data = self.data.to(device)
            self.load_train_val_test_split()
            self.model = self.default_model.to(device)
            self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.override_evaluate = None
        self.step_preprocess = None
    
    def train(self):
        """
        Train the model on the provided data. This method should be implemented by subclasses.
        """
        raise NotImplementedError
        
    def evaluate(self):
        """
        Evaluate the model on the provided data. This method should be implemented by subclasses.
        """
        raise NotImplementedError
    
    
    @property
    def num_samples(self):
        """
        Get the number of samples in the dataset. This method should be implemented by subclasses.

        Returns:
            int: Number of samples.
        """
        raise NotImplementedError
    
    @property
    def default_model(self):
        """
        Get the default model for the task. This method should be implemented by subclasses.

        Returns:
            torch.nn.Module: Default model.
        """
        raise NotImplementedError
    
    @property
    def default_optim(self):
        """
        Get the default optimizer for the task. This method should be implemented by subclasses.

        Returns:
            torch.optim.Optimizer: Default optimizer.
        """
        raise NotImplementedError
    
    @property
    def default_loss_fn(self):
        """
        Get the default loss function for the task. This method should be implemented by subclasses.

        Returns:
            function: Default loss function.
        """
        raise NotImplementedError
    
    @property
    def train_val_test_path(self):
        """
        Get the path to the train/validation/test split file. This method should be implemented by subclasses.

        Returns:
            str: Path to the split file.
        """
        raise NotImplementedError
    
    @property
    def default_train_val_test_split(self):
        """
        Get the default train/validation/test split. This method should be implemented by subclasses.

        Returns:
            dict: Default train/validation/test split.
        """
        raise NotImplementedError    

    def load_train_val_test_split(self):
        """
        Load the train/validation/test split from a file. This method should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def load_custom_model(self, custom_model):
        """
        Load a custom model for the task and reinitialize the optimizer.

        Args:
            custom_model (torch.nn.Module): Custom model to be used.
        """
        self.model = custom_model.to(self.device)
        self.optim = self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

            
            
