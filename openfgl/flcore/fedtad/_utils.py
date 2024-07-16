import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse
from torch_geometric.data import Data



def student_loss(s_logit, t_logit, return_t_logits=False, method="kl"):
    """Kl/ L1 Loss for student"""
    print_logits = False
    if method == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif method == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(method)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


class DiversityLoss(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

    
def construct_graph(node_logits, adj_logits, k=5):
    adjacency_matrix = torch.zeros_like(adj_logits)
    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
    for i in range(node_logits.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = add_self_loops(edge_index)[0]
    data = Data(x=node_logits, edge_index=edge_index)
    return data   



def random_walk_with_matrix(T, walk_length, start):
    current_node = start
    walk = [current_node]
    for _ in range(walk_length - 1):
        probabilities = F.softmax(T[current_node, :], dim=0)
        probabilities /= torch.sum(probabilities)
        next_node = torch.multinomial(probabilities, 1).item()
        walk.append(next_node)
        current_node = next_node
    return walk




def cal_topo_emb(edge_index, num_nodes, max_walk_length):
    A = to_dense_adj(add_self_loops(edge_index)[0], max_num_nodes=num_nodes).squeeze()
    D = torch.diag(torch.sum(A, dim=1))
    T = A * torch.pinverse(D)
    result_each_length = []
    
    for i in range(1, max_walk_length+1):    
        result_per_node = []
        for start in range(num_nodes):
            result_walk = random_walk_with_matrix(T, i, start)
            result_per_node.append(torch.tensor(result_walk).view(1,-1))
        result_each_length.append(torch.vstack(result_per_node))
    topo_emb = torch.hstack(result_each_length)
    return topo_emb    