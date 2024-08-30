import torch
from openfgl.flcore.base import BaseServer
from dtaidistance import dtw
import networkx as nx
from openfgl.flcore.gcfl_plus.gcfl_plus_config import config
import numpy as np
from openfgl.flcore.gcfl_plus.models import CrossDomainGIN


class GCFLPlusServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
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

        self.message_pool["server"] = {
            "cluster_indices": self.cluster_indices,
            "cluster_weights": self.cluster_weights
        }


    def get_cluster_weights(self):
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
        cluster_dWs = []
        for client_id in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = self.message_pool[f"client_{client_id}"]["dW"][k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def compute_pairwise_distances(self, seqs, standardize=False):
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
    return torch.cat([v.flatten() for v in w.values()])
