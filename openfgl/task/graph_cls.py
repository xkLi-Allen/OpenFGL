import torch
import torch.nn as nn
from openfgl.task.base import BaseTask
from openfgl.utils.basic_utils import extract_floats, idx_to_mask_tensor, mask_tensor_to_idx
from os import path as osp
from openfgl.utils.metrics import compute_supervised_metrics
import os
import torch
from openfgl.utils.task_utils import load_graph_cls_default_model
import pickle
from torch_geometric.loader import DataLoader
import numpy as np
from openfgl.data.processing import processing



class GraphClsTask(BaseTask):
    """
    Task class for graph classification in a federated learning setup.

    Attributes:
        client_id (int): ID of the client.
        data_dir (str): Directory containing the data.
        args (Namespace): Arguments containing model and training configurations.
        device (torch.device): Device to run the computations on.
        data (object): Data specific to the task.
        model (torch.nn.Module): Model to be trained.
        optim (torch.optim.Optimizer): Optimizer for the model.
        train_mask (torch.Tensor): Mask for the training set.
        val_mask (torch.Tensor): Mask for the validation set.
        test_mask (torch.Tensor): Mask for the test set.
        train_dataloader (DataLoader): DataLoader for the training set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        test_dataloader (DataLoader): DataLoader for the test set.
        splitted_data (dict): Dictionary containing split data and DataLoaders.
        processed_data (object): Processed data for training.
    """
    
    def __init__(self, args, client_id, data, data_dir, device):
        """
        Initialize the GraphClsTask with provided arguments, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the task.
            data_dir (str): Directory containing the data.
            device (torch.device): Device to run the computations on.
        """
        super(GraphClsTask, self).__init__(args, client_id, data, data_dir, device)
        
        
        
    def train(self, splitted_data=None):
        """
        Train the model on the provided or processed data.

        Args:
            splitted_data (dict, optional): Dictionary containing split data and DataLoaders. Defaults to None.
        """
        if splitted_data is None:
            splitted_data = self.processed_data # use processed_data to train
        else:
            names = ["data", "train_dataloader", "val_dataloader", "test_dataloader", "train_mask", "val_mask", "test_mask"]
            for name in names:
                assert name in splitted_data
                
        self.model.train()
        for _ in range(self.args.num_epochs):
            for batch in splitted_data["train_dataloader"]:
                self.optim.zero_grad()
                embedding, logits = self.model.forward(batch)
                loss_train = self.loss_fn(embedding, logits, batch.y, torch.ones_like(batch.y).bool())
                loss_train.backward()
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
        if splitted_data is None:
            splitted_data = self.splitted_data # use splitted_data to evaluate
        else:
            names = ["data", "train_dataloader", "val_dataloader", "test_dataloader", "train_mask", "val_mask", "test_mask"]
            for name in names:
                assert name in splitted_data
                
        eval_output = {}
        self.model.eval()
        
        num_samples = len(splitted_data["data"])
        num_global_classes = splitted_data["data"].num_global_classes
        
        embedding_all = torch.zeros((num_samples, self.args.hid_dim)).to(self.device)
        logits_all = torch.zeros((num_samples, num_global_classes)).to(self.device)
        label_all = torch.zeros((num_samples)).to(self.device).long()
        
        train_idx = splitted_data["train_mask"].nonzero().squeeze().tolist()
        if isinstance(train_idx, int):
            train_idx = [train_idx]
        val_idx = splitted_data["val_mask"].nonzero().squeeze().tolist()
        if isinstance(val_idx, int):
            val_idx = [val_idx]
        test_idx = splitted_data["test_mask"].nonzero().squeeze().tolist()
        if isinstance(test_idx, int):
            test_idx = [test_idx]
        
        
        train_cnt = 0
        val_cnt = 0
        test_cnt = 0
        
        with torch.no_grad():
            for batch in splitted_data["train_dataloader"]:
                embedding, logits = self.model.forward(batch)
                embedding_all[train_idx[train_cnt:train_cnt+batch.num_graphs]] = embedding
                logits_all[train_idx[train_cnt:train_cnt+batch.num_graphs]] = logits
                label_all[train_idx[train_cnt:train_cnt+batch.num_graphs]] = batch.y
                train_cnt += batch.num_graphs
            for batch in splitted_data["val_dataloader"]:
                embedding, logits = self.model.forward(batch)
                embedding_all[val_idx[val_cnt:val_cnt+batch.num_graphs]] = embedding
                logits_all[val_idx[val_cnt:val_cnt+batch.num_graphs]] = logits
                label_all[val_idx[val_cnt:val_cnt+batch.num_graphs]] = batch.y
                val_cnt += batch.num_graphs
            for batch in splitted_data["test_dataloader"]:
                embedding, logits = self.model.forward(batch)
                embedding_all[test_idx[test_cnt:test_cnt+batch.num_graphs]] = embedding
                logits_all[test_idx[test_cnt:test_cnt+batch.num_graphs]] = logits
                label_all[test_idx[test_cnt:test_cnt+batch.num_graphs]] = batch.y
                test_cnt += batch.num_graphs

            loss_train = self.loss_fn(embedding_all, logits_all, label_all, splitted_data["train_mask"])
            loss_val = self.loss_fn(embedding_all, logits_all, label_all, splitted_data["val_mask"])
            loss_test = self.loss_fn(embedding_all, logits_all, label_all, splitted_data["test_mask"])

        eval_output["embedding"] = embedding_all
        eval_output["logits"] = logits_all
        eval_output["loss_train"] = loss_train 
        eval_output["loss_val"]   = loss_val
        eval_output["loss_test"]  = loss_test
        
        
        metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits_all[splitted_data["train_mask"]], labels=label_all[splitted_data["train_mask"]], suffix="train")
        metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits_all[splitted_data["val_mask"]], labels=label_all[splitted_data["val_mask"]], suffix="val")
        metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits_all[splitted_data["test_mask"]], labels=label_all[splitted_data["test_mask"]], suffix="test")
        eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
        
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
        return self.default_loss_fn(logits[mask], label[mask])
        
    @property
    def default_model(self):   
        """
        Get the default model for graph classification.

        Returns:
            torch.nn.Module: Default model.
        """         
        return load_graph_cls_default_model(self.args, input_dim=self.num_feats, output_dim=self.num_global_classes, client_id=self.client_id)
    
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
        return len(self.data)
    
    @property
    def num_feats(self):
        """
        Get the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return self.data[0].x.shape[1]
    
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
        return nn.CrossEntropyLoss()
    
    @property
    def default_train_val_test_split(self):
        """
        Get the default train/validation/test split.

        Returns:
            tuple: Default train/validation/test split ratios.
        """
        return 0.8, 0.1, 0.1
        
  
    @property
    def train_val_test_path(self):
        """
        Get the path to the train/validation/test split file.

        Returns:
            str: Path to the split file.
        """
        if self.args.train_val_test == "default_split":
            return osp.join(self.data_dir, f"graph_cls", "default_split")
        else:
            split_dir = f"split_{self.args.train_val_test}" 
            return osp.join(self.data_dir, f"graph_cls", split_dir)
    

    def load_train_val_test_split(self):
        """
        Load the train/validation/test split from a file.
        """
        if self.client_id is None and len(self.args.dataset) == 1: # server
            glb_train = []
            glb_val = []
            glb_test = []
            
            for client_id in range(self.args.num_clients):
                glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{client_id}.pkl")
                glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{client_id}.pkl")
                glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{client_id}.pkl")
                
                with open(glb_train_path, 'rb') as file:
                    glb_train_data = pickle.load(file)
                    glb_train += glb_train_data
                    
                with open(glb_val_path, 'rb') as file:
                    glb_val_data = pickle.load(file)
                    glb_val += glb_val_data
                    
                with open(glb_test_path, 'rb') as file:
                    glb_test_data = pickle.load(file)
                    glb_test += glb_test_data
                
            train_mask = idx_to_mask_tensor(glb_train, self.num_samples).bool()
            val_mask = idx_to_mask_tensor(glb_val, self.num_samples).bool()
            test_mask = idx_to_mask_tensor(glb_test, self.num_samples).bool()
            
        else: # client
            train_path = osp.join(self.train_val_test_path, f"train_{self.client_id}.pt")
            val_path = osp.join(self.train_val_test_path, f"val_{self.client_id}.pt")
            test_path = osp.join(self.train_val_test_path, f"test_{self.client_id}.pt")
            glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{self.client_id}.pkl")
            glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{self.client_id}.pkl")
            glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{self.client_id}.pkl")
            
            if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path)\
                and osp.exists(glb_train_path) and osp.exists(glb_val_path) and osp.exists(glb_test_path): 
                train_mask = torch.load(train_path)
                val_mask = torch.load(val_path)
                test_mask = torch.load(test_path)
            else:
                train_mask, val_mask, test_mask = self.local_graph_train_val_test_split(self.data, self.args.train_val_test)
                
                if not osp.exists(self.train_val_test_path):
                    os.makedirs(self.train_val_test_path)
                    
                torch.save(train_mask, train_path)
                torch.save(val_mask, val_path)
                torch.save(test_mask, test_path)
                
                if len(self.args.dataset) == 1:
                    # map to global
                    glb_train_id = []
                    glb_val_id = []
                    glb_test_id = []
                    for id_train in train_mask.nonzero():
                        glb_train_id.append(self.data.global_map[id_train.item()])
                    for id_val in val_mask.nonzero():
                        glb_val_id.append(self.data.global_map[id_val.item()])
                    for id_test in test_mask.nonzero():
                        glb_test_id.append(self.data.global_map[id_test.item()])
                    with open(glb_train_path, 'wb') as file:
                        pickle.dump(glb_train_id, file)
                    with open(glb_val_path, 'wb') as file:
                        pickle.dump(glb_val_id, file)
                    with open(glb_test_path, 'wb') as file:
                        pickle.dump(glb_test_id, file)
            
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.train_dataloader = DataLoader([basedata for basedata in self.data[self.train_mask]], batch_size=self.args.batch_size, shuffle=False)
        self.val_dataloader = DataLoader([basedata for basedata in self.data[self.val_mask]], batch_size=self.args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader([basedata for basedata in self.data[self.test_mask]], batch_size=self.args.batch_size, shuffle=False)
        
        self.splitted_data = {
            "data": self.data,
            "train_dataloader": self.train_dataloader,
            "val_dataloader": self.val_dataloader,
            "test_dataloader": self.test_dataloader,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask
        }
        
        
        self.processed_data = processing(args=self.args, splitted_data=self.splitted_data, processed_dir=self.data_dir, client_id=self.client_id)
        

    def local_graph_train_val_test_split(self, local_graphs, split, shuffle=True):
        """
        Split the local graphs into train, validation, and test sets.

        Args:
            local_graphs (object): Local graphs to be split.
            split (str or tuple): Split ratios or default split identifier.
            shuffle (bool, optional): If True, shuffle the graphs before splitting. Defaults to True.

        Returns:
            tuple: Masks for the train, validation, and test sets.
        """
        num_graphs = self.num_samples
        
        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split
        else:
            train_, val_, test_ = extract_floats(split)
        
        train_mask = idx_to_mask_tensor([], num_graphs)
        val_mask = idx_to_mask_tensor([], num_graphs)
        test_mask = idx_to_mask_tensor([], num_graphs)
        for class_i in range(local_graphs.num_global_classes):
            class_i_graph_mask = local_graphs.y == class_i
            num_class_i_graphs = class_i_graph_mask.sum()
            class_i_graph_list = mask_tensor_to_idx(class_i_graph_mask)
            if shuffle:
                np.random.shuffle(class_i_graph_list)
            train_mask += idx_to_mask_tensor(class_i_graph_list[:int(train_ * num_class_i_graphs)], num_graphs)
            val_mask += idx_to_mask_tensor(class_i_graph_list[int(train_ * num_class_i_graphs) : int((train_+val_) * num_class_i_graphs)], num_graphs)
            test_mask += idx_to_mask_tensor(class_i_graph_list[int((train_+val_) * num_class_i_graphs): min(num_class_i_graphs, int((train_+val_+test_) * num_class_i_graphs))], num_graphs)
        
        
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
        return train_mask, val_mask, test_mask