import os.path as osp
import os
import torch
import numpy as np
import copy
from torch_geometric.loader import DataLoader
import random
import argparse
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_random_edge



def processing(args: argparse.ArgumentParser, splitted_data, processed_dir, client_id):
    """Process the given splitted data based on the specified processing type in args.

    This function applies various data processing techniques such as feature sparsity,
    feature noise, topology sparsity, topology noise, label sparsity, and label noise
    to the given dataset based on the `args.processing` parameter.

    Args:
        args (argparse.ArgumentParser): Command-line arguments.
        splitted_data (dict): The splitted data (created by specific task) to be processed.
        processed_dir (str): Directory where processed data should be stored.
        client_id (int or None): Identifier for the client or None for global processing.

    Returns:
        dict: The processed splitted data based on the specified processing type.
    """
    if args.processing == "raw" or client_id is None: # no processing or global data
        processed_data = splitted_data
    elif args.processing == "random_feature_sparsity":
        processed_data = random_feature_sparsity(args, splitted_data, processed_dir=processed_dir, client_id=client_id, mask_prob=args.processing_percentage)
    elif args.processing == "random_feature_noise":
        from openfgl.data.processing import random_feature_noise
        processed_data = random_feature_noise(args, splitted_data, processed_dir=processed_dir, client_id=client_id, noise_std=args.processing_percentage)
    elif args.processing == "random_topology_sparsity":
        from openfgl.data.processing import random_topology_sparsity
        processed_data = random_topology_sparsity(args, splitted_data, processed_dir=processed_dir, client_id=client_id, mask_prob=args.processing_percentage)
    elif args.processing == "random_topology_noise":
        from openfgl.data.processing import random_topology_noise
        processed_data = random_topology_noise(args, splitted_data, processed_dir=processed_dir, client_id=client_id, noise_prob=args.processing_percentage)
    elif args.processing == "random_label_sparsity":
        from openfgl.data.processing import random_label_sparsity
        processed_data = random_label_sparsity(args, splitted_data, processed_dir=processed_dir, client_id=client_id, mask_prob=args.processing_percentage)
    elif args.processing == "random_label_noise":
        from openfgl.data.processing import random_label_noise
        processed_data = random_label_noise(args, splitted_data, processed_dir=processed_dir, client_id=client_id, noise_prob=args.processing_percentage)
    
    return processed_data
        
            
            
            
            

def random_feature_sparsity(args: argparse.ArgumentParser, splitted_data: dict, processed_dir: str, client_id: int, mask_prob: float=0.1):
    """Apply random feature sparsity to the splitted data.

    This function creates a mask to introduce sparsity in the features of the splitted data
    by randomly setting a percentage of feature values to zero.

    Args:
        args (argparse.ArgumentParser): Command-line arguments.
        splitted_data (dict): The input data dictionary to be processed.
        processed_dir (str): Directory where processed data and masks should be stored.
        client_id (int): Identifier for the client.
        mask_prob (float, optional): Probability of masking each feature. Defaults to 0.1.

    Returns:
        dict: The data dictionary with features sparsified according to the mask.
    """
    mask_filename = osp.join(processed_dir, "sparsity", f"random_feature_sparsity_{mask_prob:.2f}_client_{client_id}.pt")
    if osp.exists(mask_filename):
        mask = torch.load(mask_filename)
    else:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        mask = torch.ones_like(splitted_data["data"].x, dtype=torch.float32)
        for i in range(splitted_data["data"].x.shape[0]):
            nonzero = (splitted_data["data"].x[i, :]).nonzero().squeeze().tolist()
            if type(nonzero) is not list:
                nonzero = [nonzero]
            num_to_mask = int(mask_prob * len(nonzero))
            random.shuffle(nonzero)
            mask[i, nonzero[:num_to_mask]] = 0
        torch.save(mask, mask_filename)        
        
    masked_splitted_data = copy.deepcopy(splitted_data)
    
    
    if args.task == "node_cls":
        masked_splitted_data["data"].x *= mask
    elif args.task == "graph_cls":
        mask = mask.to(masked_splitted_data["data"].data.x.device)
        masked_splitted_data["data"].data.x *= mask
        masked_splitted_data["train_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][masked_splitted_data["train_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
        masked_splitted_data["val_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][masked_splitted_data["val_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
        masked_splitted_data["test_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][masked_splitted_data["test_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
    
    return masked_splitted_data




def random_feature_noise(args: argparse.ArgumentParser, splitted_data: dict, processed_dir: str, client_id: int, noise_std: float=0.1):
    """Apply random feature noise to the given data.

    This function adds Gaussian noise to the features of the data to introduce variability
    based on the specified standard deviation of the noise.

    Args:
        args (argparse.ArgumentParser): Command-line arguments.
        splitted_data (dict): The input data dictionary to be processed.
        processed_dir (str): Directory where processed data and noise masks should be stored.
        client_id (int): Identifier for the client.
        noise_std (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.1.

    Returns:
        dict: The data dictionary with features noised according to the generated noise.
    """
    noise_filename = osp.join(processed_dir, "noise", f"random_feature_noise_{noise_std:.2f}_client_{client_id}.pt")
    if os.path.exists(noise_filename):
        noise = torch.load(noise_filename)
    else:
        os.makedirs(os.path.dirname(noise_filename), exist_ok=True)
        noise = torch.randn_like(splitted_data["data"].x) * noise_std
        torch.save(noise, noise_filename)
        
    noised_splitted_data = copy.deepcopy(splitted_data)
        
        
    if args.task == "node_cls":
        noised_splitted_data["data"].x += noise
    elif args.task == "graph_cls":
        noise = noise.to(noised_splitted_data["data"].data.x.device)
        noised_splitted_data["data"].data.x += noise
        noised_splitted_data["train_dataloader"] = DataLoader([ basedata for basedata in noised_splitted_data["data"][noised_splitted_data["train_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
        noised_splitted_data["val_dataloader"] = DataLoader([ basedata for basedata in noised_splitted_data["data"][noised_splitted_data["val_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
        noised_splitted_data["test_dataloader"] = DataLoader([ basedata for basedata in noised_splitted_data["data"][noised_splitted_data["test_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
        
    return noised_splitted_data
 


def random_label_sparsity(args: argparse.ArgumentParser, splitted_data: dict, processed_dir: str, client_id: int, mask_prob=0.1):
    """Apply random label sparsity to the given data.

    This function masks a percentage of training labels to introduce sparsity in the labels
    based on the specified mask probability.

    Args:
        args (argparse.ArgumentParser): Command-line arguments.
        splitted_data (dict): The input data dictionary to be processed.
        processed_dir (str): Directory where processed data and masks should be stored.
        client_id (int): Identifier for the client.
        mask_prob (float, optional): Probability of masking each label. Defaults to 0.1.

    Returns:
        dict: The data dictionary with labels sparsified according to the mask.
    """
    mask_filename = osp.join(processed_dir, "sparsity", f"random_label_sparsity_{mask_prob:.2f}_client_{client_id}.pt")
    if osp.exists(mask_filename):
        masked_train_mask = torch.load(mask_filename)
    else:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        masked_train_mask = torch.clone(splitted_data["train_mask"])
        indices = masked_train_mask.nonzero().squeeze().tolist()
        if type(indices) is not list:
            indices = [indices]
        num_samples = int(round(len(indices) * mask_prob))
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        masked_train_mask[selected_indices] = 0
        torch.save(masked_train_mask, mask_filename)
            
    masked_splitted_data = copy.deepcopy(splitted_data)
                
    if args.task == "node_cls":
        masked_splitted_data["train_mask"] = masked_train_mask
    elif args.task == "graph_cls":
        masked_splitted_data["train_mask"] = masked_train_mask
        masked_splitted_data["train_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][masked_splitted_data["train_mask"]]], 
                                                              batch_size=args.batch_size, shuffle=False)
         
    return masked_splitted_data




def random_label_noise(args: argparse.ArgumentParser, splitted_data: dict, processed_dir: str, client_id: int, noise_prob: float=0.1):
    """Apply random label noise to the given data.

    This function changes a percentage of labels to random classes to introduce noise
    based on the specified noise probability.

    Args:
        args (argparse.ArgumentParser): Command-line arguments.
        splitted_data (dict): The input data dictionary to be processed.
        processed_dir (str): Directory where processed data and noise masks should be stored.
        client_id (int): Identifier for the client.
        noise_prob (float, optional): Probability of changing each label. Defaults to 0.1.

    Returns:
        dict: The data dictionary with labels noised according to the generated noise.
    """
    noise_filename = osp.join(processed_dir, "noise", f"random_label_noise_{noise_prob:.2f}_client_{client_id}.pt")
    if osp.exists(noise_filename):
        noised_label = torch.load(noise_filename)
    else:
        os.makedirs(os.path.dirname(noise_filename), exist_ok=True)
        noised_label = torch.clone(splitted_data["data"].y)
        all_labels = [class_i for class_i in range(splitted_data["data"].num_global_classes)]

        train_mask = splitted_data["train_mask"]
        indices = train_mask.nonzero().squeeze().tolist()
        if type(indices) is not list:
            indices = [indices]
        num_samples = int(round(len(indices) * noise_prob))
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        for idx in selected_indices:
            real_label = splitted_data["data"].y[idx].item()
            new_label = real_label
            while(new_label == real_label):
                new_label = np.random.choice(all_labels)                
            noised_label[idx] = new_label
        
        torch.save(noised_label, noise_filename)
        
    noised_splitted_data = copy.deepcopy(splitted_data)    
    if args.task == "node_cls":
        noised_splitted_data["data"].y = noised_label
    elif args.task == "graph_cls":
        noised_splitted_data["data"].data.y = noised_label
    
    return noised_splitted_data


 


def random_topology_sparsity(args: argparse.ArgumentParser, splitted_data: dict, processed_dir: str, client_id: int, mask_prob: float=0.1):
    """Apply random topology sparsity to the given data.

    This function removes a percentage of edges to introduce sparsity in the topology
    of the data based on the specified mask probability.

    Args:
        args (argparse.ArgumentParser): Command-line arguments.
        splitted_data (dict): The input data dictionary to be processed.
        processed_dir (str): Directory where processed data and masks should be stored.
        client_id (int): Identifier for the client.
        mask_prob (float, optional): Probability of masking each edge. Defaults to 0.1.

    Returns:
        dict: The data dictionary with topology sparsified according to the mask.
    """
    masked_edge_index_filename = osp.join(processed_dir, "sparsity", f"random_topology_sparsity_{mask_prob:.2f}_client_{client_id}.pt")

    
    if args.task == "node_cls":
        if osp.exists(masked_edge_index_filename):
            masked_edge_index = torch.load(masked_edge_index_filename)
        else:
            os.makedirs(os.path.dirname(masked_edge_index_filename), exist_ok=True)
            edge_index = splitted_data["data"].edge_index
            directed_edge_index_ids = (edge_index[0,:]>edge_index[1,:]).nonzero().squeeze().tolist()
            if type(directed_edge_index_ids) is not list:
                directed_edge_index_ids = [directed_edge_index_ids]
                
            num_remained = int(round(len(directed_edge_index_ids) * (1-mask_prob)))
            remained_ids = np.random.choice(directed_edge_index_ids, num_remained, replace=False)            
            masked_edge_index = to_undirected(edge_index=edge_index[:, remained_ids])
            torch.save(masked_edge_index, masked_edge_index_filename)  

        masked_splitted_data = copy.deepcopy(splitted_data)
        masked_splitted_data["data"].edge_index = masked_edge_index
        
    elif args.task == "graph_cls":
        if osp.exists(masked_edge_index_filename):
            masked_edge_index = torch.load(masked_edge_index_filename)

        else:
            os.makedirs(os.path.dirname(masked_edge_index_filename), exist_ok=True)
            masked_edge_index = []
            for basedata in splitted_data["data"]:
                edge_index = basedata.edge_index
                directed_edge_index_ids = (edge_index[0,:]>edge_index[1,:]).nonzero().squeeze().tolist()
                if type(directed_edge_index_ids) is not list:
                    directed_edge_index_ids = [directed_edge_index_ids]
                    
                num_remained = int(round(len(directed_edge_index_ids) * (1-mask_prob)))
                remained_ids = np.random.choice(directed_edge_index_ids, num_remained, replace=False)            
                masked_edge_index_i = to_undirected(edge_index=edge_index[:, remained_ids])
                masked_edge_index.append(masked_edge_index_i)
            torch.save(masked_edge_index, masked_edge_index_filename)  
            
        masked_splitted_data = copy.deepcopy(splitted_data)
        
        # modify each basedata 
        for basedata, masked_edge_index_i in zip(masked_splitted_data["data"], masked_edge_index):
            basedata.edge_index = masked_edge_index_i
            
        # modify dataloader
        masked_splitted_data["train_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][masked_splitted_data["train_mask"]]], 
                                                            batch_size=args.batch_size, shuffle=False)
        masked_splitted_data["val_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][masked_splitted_data["val_mask"]]], 
                                                            batch_size=args.batch_size, shuffle=False)
        masked_splitted_data["test_dataloader"] = DataLoader([ basedata for basedata in masked_splitted_data["data"][masked_splitted_data["test_mask"]]], 
                                                            batch_size=args.batch_size, shuffle=False)

    return masked_splitted_data

    
def random_topology_noise(args, splitted_data: dict, processed_dir: str, client_id: int, noise_prob: float=0.1):
    """Apply random topology noise to the given data.

    This function introduces noise in the topology of the data by adding and removing edges
    based on the specified noise probability.

    Args:
        args (argparse.ArgumentParser): Command-line arguments.
        splitted_data (dict): The input data dictionary to be processed.
        processed_dir (str): Directory where processed data and noise masks should be stored.
        client_id (int): Identifier for the client.
        noise_prob (float, optional): Probability of modifying each edge. Defaults to 0.1.

    Returns:
        dict: The data dictionary with topology noised according to the generated noise.
    """
    noised_edge_index_filename = osp.join(processed_dir, "noise", f"random_topology_noise_{noise_prob:.2f}_client_{client_id}.pt")
    
    if args.task == "node_cls":
        if osp.exists(noised_edge_index_filename):
            noised_edge_index = torch.load(noised_edge_index_filename)
        else:
            os.makedirs(os.path.dirname(noised_edge_index_filename), exist_ok=True)
            edge_index = splitted_data["data"].edge_index
            
            # add negative edges
            retained_edge_index, added_edge_index = add_random_edge(edge_index=edge_index, p=noise_prob, force_undirected=True)
            
            # remove positive edges
            directed_edge_index_ids = (edge_index[0,:]>edge_index[1,:]).nonzero().squeeze().tolist()
            if type(directed_edge_index_ids) is not list:
                directed_edge_index_ids = [directed_edge_index_ids]
                
            num_remained = int(round(len(directed_edge_index_ids) * (1-noise_prob)))
            remained_ids = np.random.choice(directed_edge_index_ids, num_remained, replace=False)            
            remained_edge_index = to_undirected(edge_index=edge_index[:, remained_ids])
            
            # noised
            noised_edge_index = torch.hstack((remained_edge_index, added_edge_index))
            torch.save(noised_edge_index, noised_edge_index_filename)  

        noised_splitted_data = copy.deepcopy(splitted_data)
        noised_splitted_data["data"].edge_index = noised_edge_index
    elif args.task == "graph_cls":
        if osp.exists(noised_edge_index_filename):
            noised_edge_index = torch.load(noised_edge_index_filename)

        else:
            os.makedirs(os.path.dirname(noised_edge_index_filename), exist_ok=True)
            noised_edge_index = []
            for basedata in splitted_data["data"]:
                edge_index = basedata.edge_index
                # add negative edges
                retained_edge_index, added_edge_index = add_random_edge(edge_index=edge_index, p=noise_prob, force_undirected=True)
                
                # remove positive edges
                directed_edge_index_ids = (edge_index[0,:]>edge_index[1,:]).nonzero().squeeze().tolist()
                if type(directed_edge_index_ids) is not list:
                    directed_edge_index_ids = [directed_edge_index_ids]
                    
                num_remained = int(round(len(directed_edge_index_ids) * (1-noise_prob)))
                remained_ids = np.random.choice(directed_edge_index_ids, num_remained, replace=False)            
                remained_edge_index = to_undirected(edge_index=edge_index[:, remained_ids])
                
                # noised
                noised_edge_index_i = torch.hstack((remained_edge_index, added_edge_index))
                
                noised_edge_index.append(noised_edge_index_i)
            torch.save(noised_edge_index, noised_edge_index_filename)  
            
        noised_splitted_data = copy.deepcopy(splitted_data)
        
        # modify each basedata 
        for basedata, noised_edge_index_i in zip(noised_splitted_data["data"], noised_edge_index):
            basedata.edge_index = noised_edge_index_i
            
        # modify dataloader
        
        noised_splitted_data["train_dataloader"] = DataLoader([ basedata for basedata in noised_splitted_data["data"][noised_splitted_data["train_mask"]]], 
                                                            batch_size=args.batch_size, shuffle=False)
        noised_splitted_data["val_dataloader"] = DataLoader([ basedata for basedata in noised_splitted_data["data"][noised_splitted_data["val_mask"]]], 
                                                            batch_size=args.batch_size, shuffle=False)
        noised_splitted_data["test_dataloader"] = DataLoader([ basedata for basedata in noised_splitted_data["data"][noised_splitted_data["test_mask"]]], 
                                                            batch_size=args.batch_size, shuffle=False)


    return noised_splitted_data


