import os
from os import path as osp
from openfgl.data.global_dataset_loader import load_global_dataset
from torch_geometric.data import Dataset
from torch_geometric.utils import remove_self_loops, to_undirected
import copy
import torch
import json

class FGLDataset(Dataset):
    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        self.check_args(args)
        self.args = args
        super(FGLDataset, self).__init__(args.root, transform, pre_transform, pre_filter)
        self.load_data()

    
    @property
    def global_root(self) -> str:
        return osp.join(self.root, "global")
    
    @property
    def distrib_root(self) -> str:
        return osp.join(self.root, "distrib")
    
    
    @property
    def raw_dir(self) -> str:
        return self.root

    def check_args(self, args):
        if args.scenario == "graph_fl":
            from openfgl.config import supported_graph_fl_datasets, supported_graph_fl_simulations, supported_graph_fl_task
            for dataset in args.dataset:
                assert dataset in supported_graph_fl_datasets, f"Invalid graph-fl dataset '{dataset}'."
            assert args.simulation_mode in supported_graph_fl_simulations, f"Invalid graph_fl simulation mode '{args.simulation_mode}'."
            assert args.task in supported_graph_fl_task, f"Invalid graph-fl task '{args.task}'."
            
            
        elif args.scenario == "subgraph_fl":
            from openfgl.config import supported_subgraph_fl_datasets, supported_subgraph_fl_simulations, supported_subgraph_fl_task
            for dataset in args.dataset:
                assert dataset in supported_subgraph_fl_datasets, f"Invalid subgraph_fl dataset '{dataset}'."
            assert args.simulation_mode in supported_subgraph_fl_simulations, f"Invalid subgraph_fl simulation mode '{args.simulation_mode}'."
            assert args.task in supported_subgraph_fl_task, f"Invalid graph_fl task '{args.task}'."
        
        if args.simulation_mode == "graph_fl_cross_domain":
            assert len(args.dataset) == args.num_clients , f"For graph-fl cross domain simulation, the number of clients must be equal to the number of used datasets (args.num_clients={args.num_clients}; used_datasets: {args.dataset})."
        elif args.simulation_mode == "graph_fl_label_skew":
            assert len(args.dataset) == 1, f"For graph-fl label skew simulation, only single dataset is supported."
        elif args.simulation_mode == "subgraph_fl_label_skew":
            assert len(args.dataset) == 1, f"For subgraph-fl label skew simulation, only single dataset is supported."
        elif args.simulation_mode == "subgraph_fl_louvain_plus":
            assert len(args.dataset) == 1, f"For subgraph-fl louvain clustering simulation, only single dataset is supported."
        elif args.simulation_mode == "subgraph_fl_metis_plus":
            assert len(args.dataset) == 1, f"For subgraph-fl metis clustering simulation, only single dataset is supported."
            
        
    
    @property
    def processed_dir(self) -> str:
        if self.args.simulation_mode in ["subgraph_fl_label_skew", "graph_fl_label_skew"]:
            simulation_name = f"{self.args.simulation_mode}_{self.args.skew_alpha:.2f}"
        elif self.args.simulation_mode in ["subgraph_fl_louvain_plus", "subgraph_fl_louvain"]:
            simulation_name = f"{self.args.simulation_mode}_{self.args.louvain_resolution}"
        elif self.args.simulation_mode in ["subgraph_fl_metis_plus"]:
            simulation_name = f"{self.args.simulation_mode}_{self.args.metis_num_coms}"
        else:
            simulation_name = self.args.simulation_mode
            
        fmt_dataset_list = copy.deepcopy(self.args.dataset)
        fmt_dataset_list = sorted(fmt_dataset_list)
           
        
        return osp.join(self.distrib_root,
                        "_".join([simulation_name, "_".join(fmt_dataset_list), f"client_{self.args.num_clients}"]))
        
                            
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        files_names = ["data_{}.pt".format(i) for i in range(self.args.num_clients)]
        return files_names


    def get_client_data(self, client_id):
        data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(client_id)))
        if hasattr(data, "x"):
            data.x = data.x.to(torch.float32)
        if hasattr(data, "y"):
            data.y = data.y.squeeze() # could be int64 (for classification) / float32 (for regression)
        if hasattr(data, "edge_attr"):
            data.edge_index, data.edge_attr = remove_self_loops(*to_undirected(data.edge_index, data.edge_attr))
        else:
            data.edge_index = remove_self_loops(to_undirected(data.edge_index))[0]
        data.edge_index = data.edge_index.to(torch.int64)
        # reset cache
        data._data_list = None
        return data

    def save_client_data(self, data, client_id):
        torch.save(data, osp.join(self.processed_dir, "data_{}.pt".format(client_id)))

    def process(self):
        if len(self.args.dataset) == 1:
            global_dataset = load_global_dataset(self.global_root, scenario=self.args.scenario, dataset=self.args.dataset[0])
        else:
            global_dataset = [load_global_dataset(self.global_root, scenario=self.args.scenario, dataset=dataset_i) for dataset_i in self.args.dataset]

        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if self.args.simulation_mode == "graph_fl_label_skew":
            from openfgl.data.simulation import graph_fl_label_skew
            self.local_data = graph_fl_label_skew(self.args, global_dataset)
        elif self.args.simulation_mode == "graph_fl_cross_domain":
            from openfgl.data.simulation import graph_fl_cross_domain
            self.local_data = graph_fl_cross_domain(self.args, global_dataset)
        elif self.args.simulation_mode == "graph_fl_topology_skew":
            from openfgl.data.simulation import graph_fl_topology_skew
            self.local_data = graph_fl_topology_skew(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_label_skew":
            from openfgl.data.simulation import subgraph_fl_label_skew
            self.local_data = subgraph_fl_label_skew(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_louvain_plus":
            from openfgl.data.simulation import subgraph_fl_louvain_plus
            self.local_data = subgraph_fl_louvain_plus(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_metis_plus":
            from openfgl.data.simulation import subgraph_fl_metis_plus
            self.local_data = subgraph_fl_metis_plus(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_louvain":
            from openfgl.data.simulation import subgraph_fl_louvain
            self.local_data = subgraph_fl_louvain(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_metis":
            from openfgl.data.simulation import subgraph_fl_metis
            self.local_data = subgraph_fl_metis(self.args, global_dataset)
        elif self.args.simulation_mode == "graph_fl_feature_skew":
            from openfgl.data.simulation import graph_fl_feature_skew
            self.local_data = graph_fl_feature_skew(self.args, global_dataset)

        
        
        for client_id in range(self.args.num_clients):
            self.save_client_data(self.local_data[client_id], client_id)
            
        self.save_dataset_description()
        
    def save_dataset_description(self):
        file_path = os.path.join(self.processed_dir, "description.txt")
        args_str = json.dumps(vars(self.args), indent=4)
        with open(file_path, 'w') as file:
            file.write(args_str)
            print(f"Saved dataset arguments to {file_path}.")


    def load_data(self):
        self.local_data = [self.get_client_data(client_id) for client_id in range(self.args.num_clients)]
        
        
        if len(self.args.dataset) == 1:
            global_dataset = load_global_dataset(self.global_root, scenario=self.args.scenario, dataset=self.args.dataset[0])
            if self.args.scenario == "graph_fl":
                self.global_data = global_dataset
            else:
                self.global_data = global_dataset.data
                if hasattr(self.global_data, "x"):
                    self.global_data.x = self.global_data.x.to(torch.float32)
                if hasattr(self.global_data, "y"):
                    self.global_data.y = self.global_data.y.squeeze() # could be int64 (for classification) / float32 (for regression)
                if hasattr(self.global_data, "edge_attr"):
                    self.global_data.edge_index, self.global_data.edge_attr = remove_self_loops(*to_undirected(self.global_data.edge_index, self.global_data.edge_attr))
                else:
                    self.global_data.edge_index = remove_self_loops(to_undirected(self.global_data.edge_index))[0]
                # reset cache
                self.global_data._data_list = None
                
                
            self.global_data.num_global_classes = global_dataset.num_classes
        else:
            self.global_data = None
        
