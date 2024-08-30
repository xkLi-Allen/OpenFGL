import torch
import numpy as np
import scipy.sparse as sp
import numpy.ctypeslib as ctl
import os.path as osp

from ctypes import c_int

def adj_initialize(data):
    data.adj = sp.coo_matrix((torch.ones([len(data.edge_index[0])]), (data.edge_index[0].cpu(), data.edge_index[1].cpu())), shape=(data.x.shape[0], data.x.shape[0]))
    data.row, data.col, data.edge_weight = data.adj.row, data.adj.col, data.adj.data
    if isinstance(data.adj.row, torch.Tensor) or isinstance(data.adj.col, torch.Tensor):
        data.adj = sp.csr_matrix((data.edge_weight.cpu().numpy(), (data.row.numpy(), data.col.numpy())),
                                        shape=(data.num_nodes, data.num_nodes))
    else:
        data.adj = sp.csr_matrix((data.edge_weight, (data.row, data.col)), shape=(data.num_nodes, data.num_nodes))
    
    return data

def adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]

    ctl_lib = ctl.load_library("/home/ai2/work/OPENFGL/flcore/adafgl/csrc/libmatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )

    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

def cuda_csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    
    ctl_lib = ctl.load_library("./flcore/adafgl/csrc/libcudamatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                         c_int]
    ctl_lib.FloatCSRMulDense.restypes = c_int

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    data_nnz = len(data)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDense(answer, data_nnz, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)