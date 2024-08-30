import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedgl.models import FedGCN
from openfgl.flcore.fedgl.fedgl_config import config
from scipy.spatial.distance import cdist
import scipy as sp
import numpy as np
from torch_geometric.utils import to_torch_csr_tensor

class FedGLServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGLServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.load_custom_model(FedGCN(nfeat=self.task.num_feats, nhid=self.args.hid_dim,
                                           nclass=self.task.num_global_classes, nlayer=self.args.num_layers,
                                           dropout=self.args.dropout))
        self.task.splitted_data["data"].adj = to_torch_csr_tensor(self.task.data.edge_index)
        self.pseudo_labels = []
        self.pseudo_labels_mask = []
        self.whole_adj = []

    def send_message(self):
        if self.message_pool["round"] == 0 :
            self.message_pool["server"] = {
                "weight": list(self.task.model.parameters())
            }
        else:

            self.message_pool["server"] = {
                "weight": list(self.task.model.parameters()),
                "pseudo_labels": self.pseudo_labels,
                "pseudo_labels_mask": self.pseudo_labels_mask,
                "whole_adj": self.whole_adj
            }


    def execute(self):
        sample_weights = []
        client_masks = []
        client_embeddings = []
        client_preds = []

        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in
                                   self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                client_masks.append(self.message_pool[f"client_{client_id}"]["mask"])
                client_embeddings.append(self.message_pool[f"client_{client_id}"]["embeddings"])
                client_preds.append(self.message_pool[f"client_{client_id}"]["preds"])

                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                sample_weights.append(weight)
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                       self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param


        if self.message_pool["round"]%config["pseudo_labels_update_epoch"] == 0:

            # ------- compute weight --------
            if config["ssl_loss_weight"]>0 or config["pseudo_graph_weight"] > 0:
            # ---- weight of each client per node
                if config["pred_weight"] == 'mean':
                    random_clients_data_rate = [1 for i in range(len(self.message_pool["sampled_clients"]))]
                elif config["pred_weight"] == 'sampling_rate':
                    random_clients_data_rate = sample_weights
                else:
                    raise ("error pred weight")

                random_clients_weights_per_node = torch.zeros(self.task.data.x.shape[0])
                for rate, mask in zip(random_clients_data_rate, client_masks):
                    mask_all = torch.zeros(self.task.data.x.shape[0])
                    mask_all[mask] = 1.
                    random_clients_weights_per_node += rate * mask_all
                random_clients_weights_per_node[random_clients_weights_per_node == 0] = 1.


            # -------- reconstruct the adj matrix in server ------

                if config["pseudo_graph_weight"] > 0:
                    # ----- obtain global embedding
                    global_emb = np.zeros((self.task.data.x.shape[0], self.task.num_global_classes))
                    for rate, embed, mask in zip(random_clients_data_rate, client_embeddings, client_masks):
                        client_emb = np.zeros((self.task.data.x.shape[0], self.task.num_global_classes))
                        client_emb[mask.detach().cpu().numpy()] = embed.detach().cpu().numpy()
                        global_emb += rate * client_emb
                    # row normalization
                    global_emb = global_emb / random_clients_weights_per_node[:, None].numpy()

                    server_adj = construct_server_adj(global_emb, type='dot', s=config['k'], mode=0, sigma=2)

                    np.fill_diagonal(server_adj, 1)
                    whole_adj = config['pseudo_graph_weight'] * normalize_server_adj(server_adj).to(self.device)

                    for i,mask in enumerate(client_masks):
                        self.whole_adj.append(whole_adj[mask,:][:,mask].to_sparse_csr())

                    # print(server_adj_final)

                    # --------- pseudo labels -------
                if config['ssl_loss_weight'] > 0:
                    # ----- obtain global prediction
                    global_pred = torch.zeros((self.task.data.x.shape[0], self.task.num_global_classes))
                    for rate, pred, mask in zip(random_clients_data_rate, client_preds, client_masks):
                        client_pred = torch.zeros((self.task.data.x.shape[0], self.task.num_global_classes))
                        client_pred[mask] = torch.nn.functional.softmax(pred.detach(),dim=1).cpu()
                        # set pred value to 0 if it less than probability_threshold
                        # print(pred.max(axis=1))
                        client_pred[client_pred < config['probability_threshold']] = 0
                        # weight sum of client prediction
                        global_pred += rate * client_pred
                        # global_pred += client_pred
                    global_pred = global_pred / random_clients_weights_per_node[:, None]

                    # ----- label sharpening
                    # print('before sharpening:\n', global_pred.argmax(axis=1)[:20])
                    # global_pred = global_pred / global_pred.sum(axis=1)[:, None]
                    # global_pred = np.square(global_pred) / np.square(global_pred).sum(axis=1)[:, None]
                    # print('after sharpening:\n', global_pred)

                    # ----- self-supervised learning ----
                    # select pseudo labels
                    pseudo_labels_col_index = torch.argmax(global_pred, dim=1)  # 维度是 行数，找到每一行中最大值的列下标

                    global_pred_rowsum = global_pred.sum(dim=1)
                    pseudo_labels_row_index = torch.where(global_pred_rowsum > 0)[0]
                    # print('pseudo labels num: ', len(pseudo_labels_row_index), 'class: ', set(pseudo_labels_col_index))

                    # update pseudo labels mask
                    p_mask = torch.zeros(self.task.data.x.shape[0]).to(self.device)
                    p_mask[pseudo_labels_row_index] = 1
                    for i,mask in enumerate(client_masks):
                        self.pseudo_labels_mask.append(p_mask[mask])

                    # update global pseudo labels
                    p_global = torch.zeros(self.task.data.x.shape[0]).to(self.device)
                    p_global[pseudo_labels_row_index] = pseudo_labels_col_index[pseudo_labels_row_index].type(torch.float).to(self.device)
                    for i,mask in enumerate(client_masks):
                        self.pseudo_labels.append(p_global[mask])




def construct_server_adj(data, type, s, mode=0, sigma=2):
    if type == 'dot':
        Z_full = data.dot(data.T)
        Z_full[Z_full < 0] = 0
    elif type == 'kernel':
        Z_full = gaussian_kernel(data, mode, sigma)
    else:
        print('type is error!')
    Z = np.zeros(Z_full.shape)
    for i in range(Z.shape[0]):
        index_s = np.argsort(-Z_full[i, :])[0:s]  # 选择s个最近的锚点，就是高斯核函数值最大的s个点
        Z[i, index_s] = Z_full[i, index_s]
    Z /= Z.sum(axis=1)[:, None]

    # 若Z有某一列全为0，则随机赋一个很小的值
    for i in range(Z.shape[1]):
        if (Z[:, i] == np.zeros(Z.shape[0])).all():
            row = np.random.choice(Z.shape[0], 1)
            Z[row[0]][i] = 0.1
    return Z

#高斯核函数3-根据邻居算segma
def gaussian_kernel(X, Y, mode=1, segma=2, K=5):
    sqdist = cdist(X, Y, metric='sqeuclidean')
    if mode == 0:
        #直接给定segma
        segma_ij = 2 * segma **2
    elif mode == 1:
        # 前k个邻居的距离平均值作为segma
        sqdist_sort_row = np.sort(sqdist, axis=1)
        sqdist_sort_col = np.sort(sqdist, axis=0)
        segma_i = 1/K * np.sqrt(sqdist_sort_row[:, 0:K - 1].sum(axis=1)).reshape(sqdist.shape[0], 1)
        segma_j = 1/K * np.sqrt(sqdist_sort_col[0:K - 1, :].sum(axis=0)).reshape(1, sqdist.shape[1])
        segma_ij = segma_i.dot(segma_j)
    else:
        #第k个邻居的距离作为segma
        sqdist_sort_row = np.sort(sqdist, axis=1)
        sqdist_sort_col = np.sort(sqdist, axis=0)
        segma_i = np.sqrt(sqdist_sort_row[:,K-1]).reshape(sqdist.shape[0], 1)
        segma_j = np.sqrt(sqdist_sort_col[K-1, :]).reshape(1, sqdist.shape[1])
        segma_ij = segma_i.dot(segma_j)
    #print(sqdist, segma_ij)
    return np.exp(-sqdist / segma_ij)

def normalize_server_adj(adj):
    """Symmetrically normalize server adjacency matrix."""
    adj = torch.tensor(adj)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return adj.mm(d_mat_inv_sqrt).T.mm(d_mat_inv_sqrt)