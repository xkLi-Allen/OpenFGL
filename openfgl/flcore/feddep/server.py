import torch
import torch.nn.functional as F
from openfgl.flcore.base import BaseServer

from openfgl.flcore.feddep.localdep import Classifier_F


class FedDEPEServer(BaseServer):
    """
    FedDEPEServer is a server implementation for the Federated Learning algorithm with Deep 
    Efficient Private Neighbor Generation for Subgraph Federated Learning (FedDEP). This 
    server manages the aggregation of model parameters from multiple clients and oversees 
    the global model updates in a federated learning environment.

    Attributes:
        None (inherits attributes from BaseServer)
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedDEPEServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedDEPEServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.load_custom_model(Classifier_F(
            input_dim=(self.task.num_feats, self.args.hid_dim),
            hid_dim=self.args.hid_dim, output_dim=self.task.num_global_classes,
            num_layers=self.args.num_layers, dropout=self.args.dropout))
        self.task.loss_fn = F.cross_entropy
        self.task.override_evaluate = self.get_override_evaluate()

    def execute(self):
        """
        Executes the server-side operations. If it's not the initial round, this method 
        aggregates the model parameters received from sampled clients by computing their 
        weighted average to update the global model.
        """
        if self.message_pool["round"] == 0:
            pass
        else:
            with torch.no_grad():
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = 1 / len(self.message_pool["sampled_clients"])
                    for local_param, global_param in zip(
                        self.message_pool[f"client_{client_id}"]["weight"],
                        self.task.model.parameters()):
                        if it == 0:
                            global_param.data.copy_(weight * local_param)
                        else:
                            global_param.data += weight * local_param

    def send_message(self):
        """
        Sends a message to the clients containing the updated global model parameters 
        after aggregation.
        """
        self.message_pool["server"] = {"weight": list(self.task.model.parameters())}



    def get_override_evaluate(self):
        """
        Overrides the default evaluation method. This method evaluates the global model 
        on the training, validation, and test datasets using the specified evaluation 
        metrics.

        Returns:
            function: A custom evaluation function.
        """
        from openfgl.utils.metrics import compute_supervised_metrics

        def override_evaluate(splitted_data=None, mute=False):
            """
            Evaluates the model on the provided dataset splits (or the default splits) and 
            computes relevant metrics. Outputs evaluation information unless muted.

            Args:
                splitted_data (dict, optional): The dataset splits to evaluate on. Defaults to None.
                mute (bool, optional): If True, suppresses the print output. Defaults to False.

            Returns:
                dict: Evaluation output containing losses and metrics for training, validation, and test datasets.
            """
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            else:
                names = ["train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data
            splitted_data["data"] = splitted_data["data"].to(self.device)
            eval_output = {}
            self.task.model.eval()
            with torch.no_grad():
                logits = self.task.model.forward(splitted_data["data"])
                loss_train = self.task.loss_fn(logits[splitted_data["train_mask"]], splitted_data["data"].y[splitted_data["train_mask"]])
                loss_val = self.task.loss_fn(logits[splitted_data["val_mask"]], splitted_data["data"].y[splitted_data["val_mask"]])
                loss_test = self.task.loss_fn(logits[splitted_data["test_mask"]], splitted_data["data"].y[splitted_data["test_mask"]])

            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test

            metric_train = compute_supervised_metrics(
                metrics=self.args.metrics,
                logits=logits[splitted_data["train_mask"]],
                labels=splitted_data["data"].y[splitted_data["train_mask"]],
                suffix="train"
            )
            metric_val = compute_supervised_metrics(
                metrics=self.args.metrics,
                logits=logits[splitted_data["val_mask"]],
                labels=splitted_data["data"].y[splitted_data["val_mask"]],
                suffix="val"
            )
            metric_test = compute_supervised_metrics(
                metrics=self.args.metrics,
                logits=logits[splitted_data["test_mask"]],
                labels=splitted_data["data"].y[splitted_data["test_mask"]],
                suffix="test"
            )
            eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}

            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = "[server]"
            if not mute:
                print(prefix + info)
            return eval_output

        return override_evaluate
