import torch
import torch.nn as nn
from openfgl.task.base import BaseTask
from openfgl.utils.basic_utils import extract_floats, idx_to_mask_tensor, mask_tensor_to_idx
from os import path as osp
from openfgl.utils.metrics import compute_supervised_metrics
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from openfgl.utils.task_utils import load_node_edge_level_default_model
import pickle
import numpy as np

def compute_edge_logits(node_embedding, edge_index):
    source_node_embedding = node_embedding[edge_index[0]]
    target_node_embedding = node_embedding[edge_index[1]]
    edge_logits = (source_node_embedding * target_node_embedding).sum(dim=1)
    return edge_logits

    
class LinkPredTask(BaseTask):
    def __init__(self, args, client_id, data, data_dir, device):
        super(LinkPredTask, self).__init__(args, client_id, data, data_dir, device)
        

    def train(self, splitted_data=None):
        if splitted_data is None:
            splitted_data = self.splitted_data
        else:
            names = ["forward_data"] + [f"merged_edge_{i}" for i in ["index", "label", "train_mask", "val_mask", "test_mask"]]
            for name in names:
                assert name in splitted_data
        
        self.model.train()
        for _ in range(self.args.num_epochs):
            self.optim.zero_grad()
            node_embedding, node_logits = self.model.forward(splitted_data["forward_data"])
            edge_logits = compute_edge_logits(node_embedding, splitted_data["merged_edge_index"])
            loss_train = self.loss_fn(node_embedding, edge_logits, splitted_data["merged_edge_label"], splitted_data["merged_edge_train_mask"])
            loss_train.backward()
            
            if self.step_preprocess is not None:
                self.step_preprocess()
            
            self.optim.step()
    

    
    def evaluate(self, splitted_data=None, mute=False):
        if self.override_evaluate is None:
            if splitted_data is None:
                splitted_data = self.splitted_data
            else:
                names = ["forward_data"] + [f"merged_edge_{i}" for i in ["index", "label", "train_mask", "val_mask", "test_mask"]]
                for name in names:
                    assert name in splitted_data
            
            
            eval_output = {}
            self.model.eval()
            with torch.no_grad():
                node_embedding, node_logits = self.model.forward(splitted_data["forward_data"])
                edge_logits = compute_edge_logits(node_embedding, splitted_data["merged_edge_index"])
                loss_train = self.loss_fn(None, edge_logits, splitted_data["merged_edge_label"], splitted_data["merged_edge_train_mask"])
                loss_val = self.loss_fn(None, edge_logits, splitted_data["merged_edge_label"], splitted_data["merged_edge_val_mask"])
                loss_test = self.loss_fn(None, edge_logits, splitted_data["merged_edge_label"], splitted_data["merged_edge_test_mask"])

            
            eval_output["embedding"] = None
            eval_output["logits"] = edge_logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=edge_logits[splitted_data["merged_edge_train_mask"]], labels=splitted_data["merged_edge_label"][splitted_data["merged_edge_train_mask"]], suffix="train")
            metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=edge_logits[splitted_data["merged_edge_val_mask"]], labels=splitted_data["merged_edge_label"][splitted_data["merged_edge_val_mask"]], suffix="val")
            metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=edge_logits[splitted_data["merged_edge_test_mask"]], labels=splitted_data["merged_edge_label"][splitted_data["merged_edge_test_mask"]], suffix="test")
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

        else:
            return self.override_evaluate(splitted_data, mute)
    
    def loss_fn(self, embedding, logits, label, mask):
        return self.default_loss_fn(logits[mask], label[mask])
        
    @property
    def default_model(self):            
        return load_node_edge_level_default_model(self.args, input_dim=self.num_feats, output_dim=self.num_global_classes, client_id=self.client_id)
    
    @property
    def default_optim(self):
        if self.args.optim == "adam":
            from torch.optim import Adam
            return Adam
    
    
    @property
    def num_samples(self):
        return self.splitted_data["merged_edge_label"].shape[0]
    
    @property
    def num_feats(self):
        return self.data.x.shape[1]
    
    @property
    def num_global_classes(self):
        return self.data.num_global_classes
        
    @property
    def default_loss_fn(self):
        return nn.BCEWithLogitsLoss()
    
    @property
    def default_train_val_test_split(self):
        if self.client_id is None:
            return None
        
        if len(self.args.dataset) > 1:
            name = self.args.dataset[self.client_id]
        else:
            name = self.args.dataset[0]
            
        return 0.8, 0.1, 0.1
        
        
    @property
    def train_val_test_path(self):
        return osp.join(self.data_dir, f"link_pred")
    

    def load_train_val_test_split(self):
        if self.client_id is None and len(self.args.dataset) == 1: # server
            glb_merged_edge_index_list = []
            glb_label_list = []
            glb_train_mask_list = []
            glb_val_mask_list = []
            glb_test_mask_list = []
            
            for client_id in range(self.args.num_clients):
                glb_merged_edge_index_path = osp.join(self.train_val_test_path, f"glb_merged_edge_index_{client_id}.pkl")
                merged_edge_label_path = osp.join(self.train_val_test_path, f"merged_edge_label_{client_id}.pt")
                merged_edge_train_path = osp.join(self.train_val_test_path, f"merged_edge_train_{client_id}.pt")
                merged_edge_val_path = osp.join(self.train_val_test_path, f"merged_edge_val_{client_id}.pt")
                merged_edge_test_path = osp.join(self.train_val_test_path, f"merged_edge_test_{client_id}.pt")
                
                
                glb_merged_edge_index = torch.load(glb_merged_edge_index_path)
                merged_edge_label = torch.load(merged_edge_label_path)
                merged_edge_train_mask = torch.load(merged_edge_train_path)
                merged_edge_val_mask = torch.load(merged_edge_val_path)
                merged_edge_test_mask = torch.load(merged_edge_test_path)
                
                glb_merged_edge_index_list.append(glb_merged_edge_index)
                glb_label_list.append(merged_edge_label)
                glb_train_mask_list.append(merged_edge_train_mask)
                glb_val_mask_list.append(merged_edge_val_mask)
                glb_test_mask_list.append(merged_edge_test_mask)
                
            # => hstack
            merged_edge_index = torch.hstack(glb_merged_edge_index_list).long()
            merged_edge_label = torch.hstack(glb_label_list).long()
            merged_edge_train_mask = torch.hstack(glb_train_mask_list).bool()
            merged_edge_val_mask = torch.hstack(glb_val_mask_list).bool()
            merged_edge_test_mask = torch.hstack(glb_test_mask_list).bool()
            
            # obtain global forward data
            remove_edge_set = set()
            remove_merged_ids = merged_edge_val_mask | merged_edge_test_mask | (merged_edge_train_mask & (merged_edge_label == 0))
            remove_merged_edge_index = merged_edge_index[:, remove_merged_ids]
            for edge_id in range(remove_merged_edge_index.shape[1]):
                source = remove_merged_edge_index[0, edge_id].item()
                target = remove_merged_edge_index[1, edge_id].item()
                remove_edge_set.add((source, target))
            
            forward_edge_set = set()
            for edge_id in range(self.data.edge_index.shape[1]):
                source = self.data.edge_index[0, edge_id].item()
                target = self.data.edge_index[1, edge_id].item()
                if source != target \
                        and (source, target) not in remove_edge_set \
                        and (target, source) not in remove_edge_set:
                    forward_edge_set.add((source, target))
            forward_edge_index = torch.tensor(list(forward_edge_set)).T.long()
            forward_edge_index = to_undirected(forward_edge_index)
            forward_data = Data(self.data.x, forward_edge_index, y=self.data.y)
                
            
        else: # client
            forward_data_path = osp.join(self.train_val_test_path, f"forward_data_{self.client_id}.pt")
            merged_edge_index_path = osp.join(self.train_val_test_path, f"merged_edge_index_{self.client_id}.pt")
            merged_edge_label_path = osp.join(self.train_val_test_path, f"merged_edge_label_{self.client_id}.pt")
            merged_edge_train_path = osp.join(self.train_val_test_path, f"merged_edge_train_{self.client_id}.pt")
            merged_edge_val_path = osp.join(self.train_val_test_path, f"merged_edge_val_{self.client_id}.pt")
            merged_edge_test_path = osp.join(self.train_val_test_path, f"merged_edge_test_{self.client_id}.pt")
            glb_merged_edge_index_path = osp.join(self.train_val_test_path, f"glb_merged_edge_index_{self.client_id}.pkl")
            
            if osp.exists(forward_data_path) and osp.exists(merged_edge_index_path) and osp.exists(merged_edge_label_path) \
                and osp.exists(merged_edge_train_path) and osp.exists(merged_edge_val_path) and osp.exists(merged_edge_test_path) \
                and osp.exists(glb_merged_edge_index_path): 
                
                forward_data = torch.load(forward_data_path)
                merged_edge_index = torch.load(merged_edge_index_path)
                merged_edge_label = torch.load(merged_edge_label_path)
                merged_edge_train_mask = torch.load(merged_edge_train_path)
                merged_edge_val_mask = torch.load(merged_edge_val_path)
                merged_edge_test_mask = torch.load(merged_edge_test_path)
            else:
                forward_data, merged_edge_index, merged_edge_label, \
                merged_edge_train_mask, merged_edge_val_mask, merged_edge_test_mask = self.local_subgraph_train_val_test_split(self.data, self.args.train_val_test)
                
                if not osp.exists(self.train_val_test_path):
                    os.makedirs(self.train_val_test_path)
                    
                torch.save(forward_data, forward_data_path)
                torch.save(merged_edge_index, merged_edge_index_path)
                torch.save(merged_edge_label, merged_edge_label_path)
                torch.save(merged_edge_train_mask, merged_edge_train_path)
                torch.save(merged_edge_val_mask, merged_edge_val_path)
                torch.save(merged_edge_test_mask, merged_edge_test_path)
                
                if len(self.args.dataset) == 1:
                    # map to global
                    glb_merged_edge_index = torch.zeros_like(merged_edge_index)
                    for edge_id in range(glb_merged_edge_index.shape[1]):
                        glb_merged_edge_index[0, edge_id] = self.data.global_map[merged_edge_index[0, edge_id].item()]
                        glb_merged_edge_index[1, edge_id] = self.data.global_map[merged_edge_index[1, edge_id].item()]
                        
                torch.save(glb_merged_edge_index, glb_merged_edge_index_path)
                
   
            
            
        self.forward_data = forward_data.to(self.device)
        self.merged_edge_index = merged_edge_index.to(self.device)
        self.merged_edge_label = merged_edge_label.to(self.device)
        self.merged_edge_train_mask = merged_edge_train_mask.to(self.device)
        self.merged_edge_val_mask = merged_edge_val_mask.to(self.device)
        self.merged_edge_test_mask = merged_edge_test_mask.to(self.device)
        
        self.splitted_data = {
            "forward_data": self.forward_data,
            "merged_edge_index": self.merged_edge_index,
            "merged_edge_label": self.merged_edge_label,
            "merged_edge_train_mask": self.merged_edge_train_mask,
            "merged_edge_val_mask": self.merged_edge_val_mask,
            "merged_edge_test_mask": self.merged_edge_test_mask
        }
        
        
        
        
        
            
        
        
        
        
        
            
            
    def local_subgraph_train_val_test_split(self, local_subgraph, split, shuffle=True):
        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split
        else:
            train_, val_, test_ = extract_floats(split)
        
        num_nodes = local_subgraph.x.shape[0]
        
        # to directed
        pos_set = set()
        for edge_id in range(local_subgraph.edge_index.shape[1]):
            source = local_subgraph.edge_index[0, edge_id].item()
            target = local_subgraph.edge_index[1, edge_id].item()
            
            if (source, target) not in pos_set and (target, source) not in pos_set:
                pos_set.add((source, target))

        # count all pos edges        
        num_pos_all = len(pos_set)
        
        num_pos_train = int(train_ * num_pos_all)
        num_pos_val = int(val_ * num_pos_all)
        num_pos_test = min(int(test_ * num_pos_all), num_pos_all-num_pos_train-num_pos_val)
        num_pos_others = num_pos_all - num_pos_train - num_pos_val - num_pos_test
            
        pos_edge_ids = list(range(num_pos_all))
        
        if shuffle:
            np.random.shuffle(pos_edge_ids)
        

        # sample negative train
        neg_train_set = set()
        while len(neg_train_set) < num_pos_train:
            source = np.random.randint(0, num_nodes)
            target = np.random.randint(0, num_nodes)
            if source == target:
                continue
            if (source, target) in pos_set or (target, source) in pos_set:
                continue
            if (source, target) in neg_train_set or (target, source) in neg_train_set:
                continue
            neg_train_set.add((source, target))



        # sample negative val
        neg_val_set = set()
        while len(neg_val_set) < num_pos_val:
            source = np.random.randint(0, num_nodes)
            target = np.random.randint(0, num_nodes)
            if source == target:
                continue
            if (source, target) in pos_set or (target, source) in pos_set:
                continue
            if (source, target) in neg_train_set or (target, source) in neg_train_set:
                continue
            if (source, target) in neg_val_set or (target, source) in neg_val_set:
                continue
            neg_val_set.add((source, target))
            
            
            
        # sample negative test
        neg_test_set = set()
        while len(neg_test_set) < num_pos_test:
            source = np.random.randint(0, num_nodes)
            target = np.random.randint(0, num_nodes)
            if source == target:
                continue
            if (source, target) in pos_set or (target, source) in pos_set:
                continue
            if (source, target) in neg_train_set or (target, source) in neg_train_set:
                continue
            if (source, target) in neg_val_set or (target, source) in neg_val_set:
                continue
            if (source, target) in neg_test_set or (target, source) in neg_test_set:
                continue
            neg_test_set.add((source, target))

        
        # create graph data for GNN forward during training & evaluation
        pos_train_edge_ids = pos_edge_ids[:num_pos_train]
        pos_val_edge_ids = pos_edge_ids[num_pos_train: num_pos_train+num_pos_val]
        pos_test_edge_ids = pos_edge_ids[num_pos_train+num_pos_val: num_pos_train+num_pos_val+num_pos_test]
        pos_other_edge_ids = pos_edge_ids[num_pos_train+num_pos_val+num_pos_test: ]
        
        neg_train_edge_ids = list(range(num_pos_all, num_pos_all+num_pos_train))
        neg_val_edge_ids = list(range(num_pos_all+num_pos_train, num_pos_all+num_pos_train+num_pos_val))
        neg_test_edge_ids = list(range(num_pos_all+num_pos_train+num_pos_val, num_pos_all+num_pos_train+num_pos_val+num_pos_test))
        
        num_neg_all = len(neg_train_set) + len(neg_val_set) + len(neg_test_set)
        
        forward_edge_ids = pos_train_edge_ids + pos_other_edge_ids
        
        
        
        forward_edge_index = to_undirected(local_subgraph.edge_index[:, forward_edge_ids])
        forward_data = Data(x=local_subgraph.x, edge_index=forward_edge_index, y=local_subgraph.y)
        
        # merge pos edge_index & neg edge_index & others
        pos_edge_index_directed = torch.tensor(list(pos_set)).T.long() # pos directed
        neg_edge_index_directed = torch.tensor(list(neg_train_set)+list(neg_val_set)+list(neg_test_set)).T.long() # neg directed
        
        num_all_edges = num_pos_all + num_neg_all
        merged_edge_index = torch.hstack((pos_edge_index_directed, neg_edge_index_directed)).long()
        merged_edge_label = torch.hstack((torch.ones(num_pos_all), torch.zeros(num_neg_all))).float()
        merged_edge_train_mask = idx_to_mask_tensor(idx_list=pos_train_edge_ids+neg_train_edge_ids, length=num_all_edges).bool()
        merged_edge_val_mask = idx_to_mask_tensor(idx_list=pos_val_edge_ids+neg_val_edge_ids, length=num_all_edges).bool()
        merged_edge_test_mask = idx_to_mask_tensor(idx_list=pos_test_edge_ids+neg_test_edge_ids, length=num_all_edges).bool()
        
        return forward_data, merged_edge_index, merged_edge_label, merged_edge_train_mask, merged_edge_val_mask, merged_edge_test_mask
        
        
        
        
        