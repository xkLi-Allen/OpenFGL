import torch
from openfgl.flcore.base import BaseServer
from dtaidistance import dtw
import networkx as nx
from openfgl.flcore.gcfl_plus.gcfl_plus_config import config
import numpy as np
from openfgl.flcore.gcfl_plus.models import CrossDomainGIN


class GCFLPlusServer(BaseServer):
    """
    GCFLPlusServer implements the server-side functionality for the Federated Graph Classification framework (GCFL+).
    This server manages client updates, performs clustering of clients based on their gradient sequences, and 
    updates the global model. The server also handles the distribution of model updates to clients based on 
    their cluster assignments.

    Attributes:
        task (object): The task object containing the model and data configurations.
        W (dict): A dictionary containing the current global model parameters.
        cluster_indices (list): A list of lists, where each inner list contains the client indices that belong to a cluster.
        seqs_grads (dict): A dictionary that stores the sequence of gradient norms for each client.
        EPS_1 (float): A threshold for the mean update norm to determine if clustering should occur.
        EPS_2 (float): A threshold for the maximum update norm to determine if clustering should occur.
        seq_length (int): The length of the gradient sequence to be considered for DTW-based clustering.
        standardize (bool): A flag indicating whether to standardize the gradient sequences before computing DTW distances.
        num_clients (int): The number of clients in the federated learning setup.
        cluster_weights (list): A list of lists, where each inner list contains the model weights for each client in a cluster.
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the GCFLPlusServer with the provided arguments, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): The global dataset available to the server (if any).
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the server and clients.
            device (torch.device): Device on which computations will be performed (e.g., CPU or GPU).
        """
        super(GCFLPlusServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self.task.load_custom_model(
            CrossDomainGIN(nhid=self.args.hid_dim, nlayer=self.args.num_layers))
        self.W = {key: value for key, value in self.task.model.named_parameters()}

        self.cluster_indices = [[i for i in range(args.num_clients)]]
        self.seqs_grads = {i:[] for i in range(args.num_clients)}
        self.EPS_1 = config['eps1']
        self.EPS_2 = config['eps2']
        self.seq_length = config['seq_length']
        self.standardize = config['standardize']
        self.num_clients = args.num_clients
        self.cluster_weights = [[self.W for i in range(self.num_clients)]]



    def execute(self):
        """
        Executes the server-side update procedure. The server collects gradient norms from clients, 
        computes pairwise DTW distances, and performs clustering of clients based on these distances. 
        If clustering conditions are met, the clients are split into new clusters, and their weights are updated accordingly.
        """
        for i in self.message_pool["sampled_clients"]:
            self.seqs_grads[i].append(self.message_pool[f"client_{i}"]["convGradsNorm"])

        cluster_indices_new = []
        for idc in self.cluster_indices:
            max_norm = self.compute_max_update_norm(idc)
            mean_norm = self.compute_mean_update_norm(idc)
            if mean_norm < self.EPS_1 and max_norm > self.EPS_2 and len(idc) > 2 and self.message_pool["round"] > 20 \
                    and all(len(value) >= self.seq_length for value in self.seqs_grads.values()):

                tmp = [self.seqs_grads[id_][-self.seq_length:] for id_ in idc]
                dtw_distances = self.compute_pairwise_distances(tmp, self.standardize)
                c1, c2 = self.min_cut(np.max(dtw_distances) - dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                self.seqs_grads = {i: [] for i in range(self.num_clients)}
            else:
                cluster_indices_new += [idc]

        self.cluster_indices = cluster_indices_new
        self.cluster_weights = self.get_cluster_weights()



    def send_message(self):
        """
        Sends the updated cluster indices and corresponding weights back to the clients.
        This information is used by the clients to update their local models based on their cluster assignment.
        """

        self.message_pool["server"] = {
            "cluster_indices": self.cluster_indices,
            "cluster_weights": self.cluster_weights
        }


    def get_cluster_weights(self):
        """
        Aggregates the model weights for each cluster of clients based on their gradient updates.

        Returns:
            list: A list of model weights for each cluster.
        """
        weights = []
        for cluster in self.cluster_indices:
            targs = []
            sours = []
            total_size = 0
            for client_id in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = self.message_pool[f"client_{client_id}"]["W"][k].data.clone()
                    dW[k] = self.message_pool[f"client_{client_id}"]["dW"][k].data.clone()
                targs.append(W)
                sours.append((dW, self.message_pool[f"client_{client_id}"]["num_samples"]))
                total_size += self.message_pool[f"client_{client_id}"]["num_samples"]

            for target in targs:
                for name in target:
                    tmp = torch.div(
                        torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sours]),
                                  dim=0), total_size).clone()
                    target[name].data += tmp

            weights.append(targs)
        return weights



    def compute_max_update_norm(self, cluster):
        """
        Computes the maximum norm of gradient updates for a given cluster of clients.

        Args:
            cluster (list): A list of client IDs in the cluster.

        Returns:
            float: The maximum gradient update norm within the cluster.
        """
        max_dW = -np.inf
        for client_id in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = self.message_pool[f"client_{client_id}"]["dW"][k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW

    def compute_mean_update_norm(self, cluster):
        """
        Computes the mean norm of gradient updates for a given cluster of clients.

        Args:
            cluster (list): A list of client IDs in the cluster.

        Returns:
            float: The mean gradient update norm within the cluster.
        """
        cluster_dWs = []
        for client_id in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = self.message_pool[f"client_{client_id}"]["dW"][k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()



    def compute_pairwise_distances(self, seqs, standardize=False):
        """
        Computes the pairwise DTW (Dynamic Time Warping) distances between sequences of gradient norms.

        Args:
            seqs (list): A list of sequences of gradient norms.
            standardize (bool): Whether to standardize the sequences before computing DTW distances.

        Returns:
            np.array: A matrix of pairwise DTW distances.
        """
        """ computes DTW distances """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances



    def min_cut(self, similarity, idc):
        """
        Performs a min-cut on a similarity graph to split clients into two clusters.

        Args:
            similarity (np.array): A matrix of similarities between clients.
            idc (list): A list of client IDs corresponding to the similarity matrix.

        Returns:
            tuple: Two lists of client IDs representing the two new clusters.
        """
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        # c1 = np.array([idc[x] for x in partition[0]])
        # c2 = np.array([idc[x] for x in partition[1]])
        c1 = [idc[x] for x in partition[0]]
        c2 = [idc[x] for x in partition[1]]
        return c1, c2



def flatten(w):
    """
    Flattens a dictionary of tensors into a single tensor.

    Args:
        w (dict): A dictionary where the values are tensors.

    Returns:
        torch.Tensor: A flattened tensor containing all elements from the input tensors.
    """
    return torch.cat([v.flatten() for v in w.values()])
