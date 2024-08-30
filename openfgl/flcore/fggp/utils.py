import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings
from pynndescent import NNDescent

pynndescent_available = True


ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine'):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean_old(M, u):
    _, nf = np.unique(u, return_counts=True)
    idx = np.argsort(u)
    M = M[idx, :]
    M = np.vstack((np.zeros((1, M.shape[1])), M))

    np.cumsum(M, axis=0, out=M)
    cnf = np.cumsum(nf)
    nf1 = np.insert(cnf, 0, 0)
    nf1 = nf1[:-1]

    M = M[cnf, :] - M[nf1, :]
    M = M / nf[:, None]
    return M


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    # Cast input data to float32
    data = data.astype(np.float32)

    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        adj, orig_dist = clust_rank(mat, initial_rank, distance)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c


import torch
import torch.nn.functional as F


def proto_align_loss(proto1, proto2, num_classes, temperature=0.5):
    """
    :param proto1: 来自第一个视图的原型，形状为 [num_classes, feature_dim]
    :param proto2: 来自第二个视图的原型，形状为 [num_classes, feature_dim]
    :param num_classes: 类别总数
    :param temperature: 温度参数，用于调整损失函数的敏感性
    """
    # 计算所有正样本对 (proto1[i], proto2[i]) 之间的相似度
    positive_sim = F.cosine_similarity(proto1, proto2, dim=1) / temperature

    # 计算所有可能的负样本对
    negative_sim = torch.mm(proto1, proto2.t()) / temperature
    # 确保正样本对的相似度在负样本矩阵中为无效，防止自比较
    mask = torch.eye(num_classes).bool().to(proto1.device)
    negative_sim.masked_fill_(mask, float('-inf'))

    # 通过softmax来计算每一行的对数概率
    negative_sim_exp = torch.exp(negative_sim)
    positive_sim_exp = torch.exp(positive_sim)
    sum_negatives = negative_sim_exp.sum(dim=1)

    # Info-NCE loss 计算
    loss = -torch.log(positive_sim_exp / (positive_sim_exp + sum_negatives))

    return loss.mean()

def get_proto_norm_weighted(num_classes, embedding, class_label, weight,unique_labels):
    m1= F.one_hot(class_label, num_classes=num_classes)
    m2 = (m1 * weight[:, None]).t()
    m = m2 / (m2.sum(dim=1, keepdim=True)+ 1e-6)
    m = m[unique_labels]
    return torch.mm(m, embedding)


import torch

def get_norm_and_orig(data):
    # 假设data_loader已经有了原始的邻接矩阵存储在data_loader.adj中
    edge_index = data.edge_index
    num_nodes = edge_index.max().item() + 1

    # 构造邻接矩阵
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1

    # 计算度矩阵D并进行归一化处理
    degree = adj.sum(dim=1, keepdim=True)
    deg_inv_sqrt = degree.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # 计算标准化后的邻接矩阵 \hat{A} = D^{-1/2} * A * D^{-1/2}
    norm_adj = deg_inv_sqrt * adj * deg_inv_sqrt.t()

    # 更新data_loader中的属性
    if data.is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    data.adj_orig = adj.to(device)  # 原始邻接矩阵
    data.norm_adj = norm_adj.to(device)  # 标准化邻接矩阵

    return data


def proto_align_loss(proto1, proto2,  temperature=0.5):
    """
    :param proto1: 来自第一个视图的原型，形状为 [num_classes, feature_dim]
    :param proto2: 来自第二个视图的原型，形状为 [num_classes, feature_dim]
    :param num_classes: 类别总数
    :param temperature: 温度参数，用于调整损失函数的敏感性
    """

    num_classes = proto1.shape[0]
    # 计算所有正样本对 (proto1[i], proto2[i]) 之间的相似度
    positive_sim = F.cosine_similarity(proto1, proto2, dim=1) / temperature

    # 计算所有可能的负样本对
    negative_sim = torch.mm(proto1, proto2.t()) / temperature
    # 确保正样本对的相似度在负样本矩阵中为无效，防止自比较
    mask = torch.eye(num_classes).bool().to(proto1.device)
    negative_sim.masked_fill_(mask, float('-inf'))

    # 通过softmax来计算每一行的对数概率
    negative_sim_exp = torch.exp(negative_sim)
    positive_sim_exp = torch.exp(positive_sim)
    sum_negatives = negative_sim_exp.sum(dim=1)

    # Info-NCE loss 计算
    loss = -torch.log(positive_sim_exp / (positive_sim_exp + sum_negatives))

    return loss.mean()

