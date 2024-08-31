import torch
import torch.nn as nn
from openfgl.task.base import BaseTask
from openfgl.utils.basic_utils import extract_floats, idx_to_mask_tensor, mask_tensor_to_idx
from os import path as osp
from openfgl.utils.metrics import compute_supervised_metrics
import os
import torch
from openfgl.utils.task_utils import load_node_edge_level_default_model
import pickle
import numpy as np
from sklearn.cluster import KMeans
    
def compute_edge_logits(node_embedding, edge_index):
    """
    Compute edge logits based on node embeddings and edge index.

    Args:
        node_embedding (torch.Tensor): Node embeddings.
        edge_index (torch.Tensor): Edge indices.

    Returns:
        torch.Tensor: Edge logits.
    """
    source_node_embedding = node_embedding[edge_index[0]]
    target_node_embedding = node_embedding[edge_index[1]]
    edge_logits = (source_node_embedding * target_node_embedding).sum(dim=1)
    return edge_logits



class NodeClustTask(BaseTask):
    """
    Task class for node clustering in a federated learning setup.

    Attributes:
        client_id (int): ID of the client.
        data_dir (str): Directory containing the data.
        args (Namespace): Arguments containing model and training configurations.
        device (torch.device): Device to run the computations on.
        data (object): Data specific to the task.
        model (torch.nn.Module): Model to be trained.
        optim (torch.optim.Optimizer): Optimizer for the model.
        splitted_data (dict): Dictionary containing split data and DataLoaders.
    """

    def __init__(self, args, client_id, data, data_dir, device):
        """
        Initialize the NodeClustTask with provided arguments, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the task.
            data_dir (str): Directory containing the data.
            device (torch.device): Device to run the computations on.
        """
        super(NodeClustTask, self).__init__(args, client_id, data, data_dir, device)

        merged_edge_index_list = []
        
        for source in range(self.num_samples):
            for target in range(self.num_samples):
                merged_edge_index_list.append((source, target))
                
        merged_edge_index = torch.tensor(merged_edge_index_list).T.long().to(self.device)
        merged_edge_label = torch.zeros((merged_edge_index.shape[1],)).float().to(self.device)
            
        for edge_id in range(self.data.edge_index.shape[1]):
            source = self.data.edge_index[0, edge_id].item()
            target = self.data.edge_index[1, edge_id].item()
            idx = source * self.num_samples + target
            merged_edge_label[idx] = 1
        for source in range(self.num_samples):
            idx = source * self.num_samples + source
            merged_edge_label[idx] = 1
        
        merged_edge_label = merged_edge_label.to(self.device)
        
        
        self.splitted_data = {
            "data": self.data,
            "merged_edge_index": merged_edge_index,
            "merged_edge_label": merged_edge_label
        }
        

    def train(self, splitted_data=None):
        """
        Train the model on the provided or processed data.

        Args:
            splitted_data (dict, optional): Dictionary containing split data and DataLoaders. Defaults to None.
        """
        if splitted_data is None:
            splitted_data = self.splitted_data
        else:
            names = ["data", "merged_edge_index", "merged_edge_label"]
            for name in names:
                assert name in splitted_data
        
        self.model.train()
        for _ in range(self.args.num_epochs):
            self.optim.zero_grad()
            node_embedding, node_logits = self.model.forward(splitted_data["data"])
            edge_logits = compute_edge_logits(node_embedding, splitted_data["merged_edge_index"])
            
            loss = self.loss_fn(None, edge_logits, splitted_data["merged_edge_label"], mask=None)
            loss.backward()
            
            if self.step_preprocess is not None:
                self.step_preprocess()
            
            self.optim.step()
    

    
    def evaluate(self, splitted_data=None, mute=False):
        """
        Evaluate the model on the provided or processed data.

        Args:
            splitted_data (dict, optional): Dictionary containing split data and DataLoaders. Defaults to None.
            mute (bool, optional): If True, suppress the print statements. Defaults to False.

        Returns:
            dict: Dictionary containing evaluation metrics and results.
        """
        if self.override_evaluate is None:
            if splitted_data is None:
                splitted_data = self.splitted_data
            else:
                names = ["data", "merged_edge_index", "merged_edge_label"]
                for name in names:
                    assert name in splitted_data
                
            eval_output = {}
            self.model.eval()
            with torch.no_grad():
                node_embedding, node_logits = self.model.forward(splitted_data["data"])
                edge_logits = compute_edge_logits(node_embedding, splitted_data["merged_edge_index"])
                loss = self.loss_fn(None, edge_logits, splitted_data["merged_edge_label"], mask=None)

                kmeans = KMeans(n_clusters=self.args.num_clusters, random_state=0)
                node_embeddings_np = node_embedding.detach().cpu().numpy()
                cluster_label_tensor = torch.tensor(kmeans.fit_predict(node_embeddings_np)).to(self.device)
        
            eval_output["embedding"] = None
            eval_output["logits"] = cluster_label_tensor
            eval_output["loss"] = loss
            
            
            metric = compute_supervised_metrics(metrics=self.args.metrics, logits=cluster_label_tensor, labels=splitted_data["data"].y, suffix="")
            eval_output = {**eval_output, **metric}
            
            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
            if not mute:
                print(prefix+info)
            return eval_output

        else:
            return self.override_evaluate(splitted_data, mute)
    
    def loss_fn(self, embedding, logits, label, mask):
        """
        Calculate the loss for the model.

        Args:
            embedding (torch.Tensor): Embeddings from the model.
            logits (torch.Tensor): Logits from the model.
            label (torch.Tensor): Ground truth labels.
            mask (torch.Tensor): Mask to filter the logits and labels.

        Returns:
            torch.Tensor: Calculated loss.
        """
        return self.default_loss_fn(logits, label)
    
    @property
    def train_val_test_path(self):
        """
        Get the path to the train/validation/test split file.

        Returns:
            str: Path to the split file.
        """
        return osp.join(self.data_dir, f"node_clust")
    
    @property
    def default_model(self):        
        """
        Get the default model for node and edge level tasks.

        Returns:
            torch.nn.Module: Default model.
        """    
        return load_node_edge_level_default_model(self.args, input_dim=self.num_feats, output_dim=self.num_global_classes, client_id=self.client_id)
    
    @property
    def default_optim(self):
        """
        Get the default optimizer for the task.

        Returns:
            torch.optim.Optimizer: Default optimizer.
        """
        if self.args.optim == "adam":
            from torch.optim import Adam
            return Adam
    
    @property
    def num_samples(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.data.x.shape[0]
    
    @property
    def num_feats(self):
        """
        Get the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return self.data.x.shape[1]
    
    @property
    def num_global_classes(self):
        """
        Get the number of global classes in the dataset.

        Returns:
            int: Number of global classes.
        """
        return self.data.num_global_classes
        
    @property
    def default_loss_fn(self):
        """
        Get the default loss function for the task.

        Returns:
            function: Default loss function.
        """
        return nn.BCEWithLogitsLoss(weight=None)
    
    @property
    def default_train_val_test_split(self):
        """
        Get the default train/validation/test split. Not used in this task.

        Returns:
            None
        """
        return None

    @property
    def train_val_test_path(self):
        """
        Get the path to the train/validation/test split file. Not used in this task.

        Returns:
            None
        """
        pass
    

    def load_train_val_test_split(self):
        """
        Load the train/validation/test split from a file. Not used in this task.
        """
        pass
            