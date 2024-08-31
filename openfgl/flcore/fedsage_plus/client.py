import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedsage_plus.fedsage_plus_config import config
import numpy as np
from openfgl.flcore.fedsage_plus.locsage_plus import LocSAGEPlus
from openfgl.flcore.fedsage_plus._utils import greedy_loss
from openfgl.data.simulation import get_subgraph_pyg_data
import torch.nn.functional as F
from torch_geometric.data import Data
from openfgl.utils.metrics import compute_supervised_metrics


def accuracy_missing(output, labels):
    """Computes the accuracy for the missing neighbor prediction."""
    preds = torch._cast_Int(output)
    correct=0.0
    for pred,label in zip(preds,labels):
        if int(pred)==int(label):
            correct+=1.0
    return correct / len(labels)


def accuracy(pred,true):
    """Computes the classification accuracy."""
    correct = (pred.max(1)[1] == true).sum()
    tot = true.shape[0]
    acc = float(correct / tot)
    return acc



class FedSagePlusClient(BaseClient):
    """
    FedSagePlusClient is the client-side implementation for the Federated Learning algorithm 
    described in the paper 'Subgraph Federated Learning with Missing Neighbor Generation'.
    This class handles local training, missing neighbor generation, and subgraph reconstruction 
    within a federated learning framework.

    Attributes:
        splitted_impaired_data (dict): The subgraph data with impaired/missing neighbors.
        num_missing (torch.Tensor): Tensor representing the number of missing neighbors for each node.
        missing_feat (torch.Tensor): Tensor containing the missing features of the graph.
        original_neighbors (dict): A dictionary mapping node indices to their original neighbors.
        impaired_neighbors (dict): A dictionary mapping node indices to their neighbors in the impaired subgraph.
    """
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedSagePlusClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedSagePlusClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(LocSAGEPlus(input_dim=self.task.num_feats, 
                                                hid_dim=self.args.hid_dim, 
                                                latent_dim=config["latent_dim"], 
                                                output_dim=self.task.num_global_classes, 
                                                max_pred=config["max_pred"], 
                                                dropout=self.args.dropout))
        self.splitted_impaired_data, self.num_missing, self.missing_feat, self.original_neighbors, self.impaired_neighbors = self.get_impaired_subgraph()
        self.send_message() # initial message for first-round neighGen training 
        
        
        
    def get_custom_loss_fn(self):
        """
        Returns a custom loss function for the training process, which changes depending 
        on the current phase of the training (neighbor generation or classification).

        Returns:
            function: A custom loss function for training.
        """
        if self.phase == 0:
            def custom_loss_fn(embedding, logits, label, mask):    
                pred_degree = self.task.model.output_pred_degree
                pred_neig_feat = self.task.model.output_pred_neig_feat


                num_impaired_nodes = self.splitted_impaired_data["data"].x.shape[0]
                impaired_logits = logits[: num_impaired_nodes]


                loss_train_missing = F.smooth_l1_loss(pred_degree[mask], self.num_missing[mask])
                loss_train_feat = greedy_loss(pred_neig_feat[mask], 
                                              self.missing_feat[mask], 
                                              pred_degree[mask], 
                                              self.num_missing[mask], 
                                              max_pred=config["max_pred"])
                loss_train_label= F.cross_entropy(impaired_logits[mask], label[mask])

                loss_other = 0

                for client_id in self.message_pool["sampled_clients"]:
                    if client_id != self.client_id:
                        others_central_ids = np.random.choice(self.message_pool[f"client_{client_id}"]["num_samples"], int(self.task.train_mask.sum()))
                        global_target_feat = []
                        for node_id in others_central_ids:
                            other_neighbors = self.message_pool[f"client_{client_id}"]["original_neighbors"][node_id]
                            while len(other_neighbors) == 0:
                                node_id = np.random.choice(self.message_pool[f"client_{client_id}"]["num_samples"], 1)[0]
                                other_neighbors = self.message_pool[f"client_{client_id}"]["original_neighbors"][node_id]
                            others_neig_ids = np.random.choice(list(other_neighbors), config["max_pred"])
                            for neig_id in others_neig_ids:
                                global_target_feat.append(self.message_pool[f"client_{client_id}"]["feat"][neig_id])
                        global_target_feat = torch.stack(global_target_feat, 0).view(-1, config["max_pred"], self.task.num_feats)
                        loss_train_feat_other = greedy_loss(pred_neig_feat[mask],
                                                            global_target_feat,
                                                            pred_degree[mask],
                                                            self.num_missing[mask],
                                                            max_pred=config["max_pred"])

                        loss_other += loss_train_feat_other    

                loss = (config["num_missing_trade_off"] * loss_train_missing + \
                       config["missing_feat_trade_off"] * loss_train_feat + \
                       config["cls_trade_off"] * loss_train_label + \
                       config["missing_feat_trade_off"] * loss_other) / len(self.message_pool["sampled_clients"])
                       
                acc_degree = accuracy_missing(pred_degree[mask], self.num_missing[mask])
                acc_cls = accuracy(impaired_logits[mask], label[mask])

                print(f"[client {self.client_id} neighGen phase]\tacc_degree: {acc_degree:.4f}\tacc_cls: {acc_cls:.4f}\tloss_train: {loss:.4f}\tloss_degree: {loss_train_missing:.4f}\tloss_feat: {loss_train_feat:.4f}\tloss_cls: {loss_train_label:.4f}\tloss_other: {loss_other:.4f}")

                return loss
        else:
            def custom_loss_fn(embedding, logits, label, mask):    
                return F.cross_entropy(logits[mask], label[mask])
        return custom_loss_fn



    def execute(self):
        """
        Executes the training process. This method handles the switching between different 
        phases of training, initializes the missing neighbor generation, and performs training 
        based on the current phase.
        """
        # switch phase
        if self.message_pool["round"] < config["gen_rounds"]:
            self.phase = 0
            self.task.override_evaluate = self.get_phase_0_override_evaluate()
        elif self.message_pool["round"] == config["gen_rounds"]:
            self.phase = 1
            self.splitted_filled_data = self.get_filled_subgraph()
            self.task.model.phase = 1
            def get_evaluate_splitted_data():
                return self.splitted_filled_data
            self.task.evaluate_splitted_data = get_evaluate_splitted_data()
            self.task.override_evaluate = self.get_phase_1_override_evaluate()
            
        # execute
        if not hasattr(self, "phase"): # miss the generator training phase due to partial participation
            self.phase = 1
            self.splitted_filled_data = self.get_filled_subgraph()
            self.task.model.phase = 1
            def get_evaluate_splitted_data():
                return self.splitted_filled_data
            self.task.evaluate_splitted_data = get_evaluate_splitted_data()
            self.task.override_evaluate = self.get_phase_1_override_evaluate()
            
            
        if self.phase == 0:
            self.task.loss_fn = self.get_custom_loss_fn()
            self.task.train(self.splitted_impaired_data)
        else:
            with torch.no_grad():
                for (local_param_with_name, global_param) in zip(self.task.model.named_parameters(), self.message_pool["server"]["weight"]):
                    name = local_param_with_name[0]
                    local_param = local_param_with_name[1]
                    if "classifier" in name:
                        local_param.data.copy_(global_param)
                        
            self.task.loss_fn = self.get_custom_loss_fn()
            self.task.train(self.splitted_filled_data)

            
            
            
            


    def send_message(self):
        """
        Sends a message to the server containing the current model parameters and, 
        if in the neighbor generation phase, additional information needed for 
        cross-client missing neighbor prediction.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
            }

        if "round" not in self.message_pool or (hasattr(self, "phase") and self.phase == 0):
            self.message_pool[f"client_{self.client_id}"]["feat"] = self.task.data.x  # for 'loss_other'
            self.message_pool[f"client_{self.client_id}"]["original_neighbors"] = self.original_neighbors  # for 'loss_other'


    def get_impaired_subgraph(self):
        """
        Creates an impaired subgraph by randomly hiding a portion of the graph structure.

        Returns:
            splitted_impaired_data (dict): The impaired subgraph data and corresponding masks.
            num_missing (torch.Tensor): Tensor containing the number of missing neighbors for each node.
            missing_feat (torch.Tensor): Tensor containing the features of missing neighbors.
            original_neighbors (dict): Dictionary of original neighbors for each node in the graph.
            impaired_neighbors (dict): Dictionary of neighbors in the impaired subgraph.
        """
        hide_len = int(config["hidden_portion"] * (self.task.val_mask).sum())
        could_hide_ids = self.task.val_mask.nonzero().squeeze().tolist()
        hide_ids = np.random.choice(could_hide_ids, hide_len, replace=False)
        all_ids = list(range(self.task.num_samples))
        remained_ids = list(set(all_ids) - set(hide_ids))

        impaired_subgraph = get_subgraph_pyg_data(global_dataset=self.task.data, node_list=remained_ids)

        impaired_subgraph = impaired_subgraph.to(self.device)
        num_missing_list = []
        missing_feat_list = []


        original_neighbors = {node_id: set() for node_id in range(self.task.data.x.shape[0])}
        for edge_id in range(self.task.data.edge_index.shape[1]):
            source = self.task.data.edge_index[0, edge_id].item()
            target = self.task.data.edge_index[1, edge_id].item()
            if source != target:
                original_neighbors[source].add(target)
                original_neighbors[target].add(source)

        impaired_neighbors = {node_id: set() for node_id in range(impaired_subgraph.x.shape[0])}
        for edge_id in range(impaired_subgraph.edge_index.shape[1]):
            source = impaired_subgraph.edge_index[0, edge_id].item()
            target = impaired_subgraph.edge_index[1, edge_id].item()
            if source != target:
                impaired_neighbors[source].add(target)
                impaired_neighbors[target].add(source)


        for impaired_id in range(impaired_subgraph.x.shape[0]):
            original_id = impaired_subgraph.global_map[impaired_id]
            num_original_neighbor = len(original_neighbors[original_id])
            num_impaired_neighbor = len(impaired_neighbors[impaired_id])
            impaired_neighbor_in_original = set()
            for impaired_neighbor in impaired_neighbors[impaired_id]:
                impaired_neighbor_in_original.add(impaired_subgraph.global_map[impaired_neighbor])

            num_missing_neighbors = num_original_neighbor - num_impaired_neighbor
            num_missing_list.append(num_missing_neighbors)
            missing_neighbors = original_neighbors[original_id] - impaired_neighbor_in_original



            if num_missing_neighbors == 0:
                current_missing_feat = torch.zeros((config["max_pred"], self.task.num_feats)).to(self.device)
            else:
                if num_missing_neighbors <= config["max_pred"]:
                    zeros = torch.zeros((max(0, config["max_pred"] - num_missing_neighbors), self.task.num_feats)).to(self.device)
                    current_missing_feat = torch.vstack((self.task.data.x[list(missing_neighbors)], zeros)).view(config["max_pred"], self.task.num_feats)
                else:
                    current_missing_feat = self.task.data.x[list(missing_neighbors)[:config["max_pred"]]].view(config["max_pred"], self.task.num_feats)

            missing_feat_list.append(current_missing_feat)

        num_missing = torch.tensor(num_missing_list).squeeze().float().to(self.device)
        missing_feat = torch.stack(missing_feat_list, 0)

        impaired_train_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)
        impaired_val_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)
        impaired_test_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)

        for impaired_id in range(impaired_subgraph.x.shape[0]):
            original_id = impaired_subgraph.global_map[impaired_id]

            if self.task.train_mask[original_id]:
                impaired_train_mask[impaired_id] = 1

            if self.task.val_mask[original_id]:
                impaired_val_mask[impaired_id] = 1

            if self.task.test_mask[original_id]:
                impaired_test_mask[impaired_id] = 1

        splitted_impaired_data = {
            "data": impaired_subgraph,
            "train_mask": impaired_train_mask,
            "val_mask": impaired_val_mask,
            "test_mask": impaired_test_mask
        }

        return splitted_impaired_data, num_missing, missing_feat, original_neighbors, impaired_neighbors
    
    
    
    
    def get_filled_subgraph(self):
        """
        Fills the impaired subgraph with generated neighbors to create a filled subgraph.

        Returns:
            dict: The filled subgraph data and corresponding masks.
        """
        with torch.no_grad():
            embedding, logits = self.task.model.forward(self.splitted_impaired_data["data"])
            pred_degree_float = self.task.model.output_pred_degree.detach()
            pred_neig_feat = self.task.model.output_pred_neig_feat.detach()
            num_impaired_nodes = self.splitted_impaired_data["data"].x.shape[0]
            global_map = self.splitted_impaired_data["data"].global_map
            num_original_nodes = self.task.data.x.shape[0]    
            ptr = num_original_nodes
            remain_feat = []
            remain_edges = []

            pred_degree = torch._cast_Int(pred_degree_float)
            

            for impaired_node_i in range(num_impaired_nodes):
                original_node_i = global_map[impaired_node_i]
                
                for gen_neighbor_j in range(min(config["max_pred"], pred_degree[impaired_node_i])):
                    remain_feat.append(pred_neig_feat[impaired_node_i, gen_neighbor_j])
                    remain_edges.append(torch.tensor([original_node_i, ptr]).view(2, 1).to(self.device))
                    ptr += 1
                    
            
            
            if pred_degree.sum() > 0:
                filled_x = torch.vstack((self.task.data.x, torch.vstack(remain_feat)))
                filled_edge_index = torch.hstack((self.task.data.edge_index, torch.hstack(remain_edges)))
                filled_y = torch.hstack((self.task.data.y, torch.zeros(ptr-num_original_nodes).long().to(self.device)))
                filled_train_mask = torch.hstack((self.task.train_mask, torch.zeros(ptr-num_original_nodes).bool().to(self.device)))
                filled_val_mask = torch.hstack((self.task.val_mask, torch.zeros(ptr-num_original_nodes).bool().to(self.device)))
                filled_test_mask = torch.hstack((self.task.test_mask, torch.zeros(ptr-num_original_nodes).bool().to(self.device)))
            else:
                filled_x = torch.clone(self.task.data.x)
                filled_edge_index = torch.clone(self.task.data.edge_index)
                filled_y = torch.clone(self.task.data.y)
                filled_train_mask = torch.clone(self.task.train_mask)
                filled_val_mask = torch.clone(self.task.val_mask)
                filled_test_mask = torch.clone(self.task.test_mask)
                
            filled_data = Data(x=filled_x, edge_index=filled_edge_index, y=filled_y)
                
            splitted_filled_data = {
                "data": filled_data,
                "train_mask": filled_train_mask,
                "val_mask": filled_val_mask,
                "test_mask": filled_test_mask
            }

            return splitted_filled_data
        
        
        
    def get_phase_0_override_evaluate(self):
        """
        Overrides the default evaluation method for the neighbor generation phase.

        Returns:
            function: The custom evaluation function for phase 0.
        """
        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            else:
                names = ["data", "train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data
                    
            self.task.model.phase = 1 # temporary modification for evaluation
            with torch.no_grad():
                embedding, logits = self.task.model.forward(splitted_data["data"])
                
                loss_train = F.cross_entropy(logits[splitted_data["train_mask"]], splitted_data["data"].y[splitted_data["train_mask"]])
                loss_val = F.cross_entropy(logits[splitted_data["val_mask"]], splitted_data["data"].y[splitted_data["val_mask"]])
                loss_test = F.cross_entropy(logits[splitted_data["test_mask"]], splitted_data["data"].y[splitted_data["test_mask"]])

            eval_output = {}
            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]], suffix="train")
            metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]], suffix="val")
            metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]], suffix="test")
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
            self.task.model.phase = 0 # reset
            return eval_output
        return override_evaluate
    
    
    
    def get_phase_1_override_evaluate(self):
        """
        Overrides the default evaluation method for the classification phase.

        Returns:
            function: The custom evaluation function for phase 1.
        """
        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.splitted_filled_data
            else:
                names = ["data", "train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data
            
            
            eval_output = {}
            self.task.model.eval()
            with torch.no_grad():
                embedding, logits = self.task.model.forward(splitted_data["data"])
                loss_train = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["train_mask"])
                loss_val = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["val_mask"])
                loss_test = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["test_mask"])

            
            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]], suffix="train")
            metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]], suffix="val")
            metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]], suffix="test")
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
        
        return override_evaluate