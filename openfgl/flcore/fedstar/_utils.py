import torch
from torch_geometric.utils import degree, to_scipy_sparse_matrix
from scipy import sparse as sp

def init_structure_encoding(n_rw, n_dg, gs, type_init):
    tmp = []
    if type_init == 'rw':
        for gg in gs:
            # Geometric diffusion features with Random Walk
            g = gg.clone().detach().cpu()
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv=sp.diags(D)
            RW=A*Dinv
            M=RW

            SE_rw=[torch.from_numpy(M.diagonal()).float()]
            M_power=M
            for _ in range(n_rw-1):
                M_power=M_power*M
                SE_rw.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw=torch.stack(SE_rw,dim=-1)

            gg['stc_enc'] = SE_rw.to(gg.x.device)
            tmp.append(gg)

    elif type_init == 'dg':
        for gg in gs:
            # PE_degree
            g = gg.clone().detach().cpu()
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, n_dg)
            SE_dg = torch.zeros([g.num_nodes, n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1

            gg['stc_enc'] = SE_dg.to(gg.x.device)

    elif type_init == 'rw_dg':
        for gg in gs:
            # SE_rw
            g = gg.clone().detach().cpu()
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv=sp.diags(D)
            RW=A*Dinv
            M=RW

            SE=[torch.from_numpy(M.diagonal()).float()]
            M_power=M
            for _ in range(n_rw-1):
                M_power=M_power*M
                SE.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw=torch.stack(SE,dim=-1)

            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, n_dg)
            SE_dg = torch.zeros([g.num_nodes, n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1

            gg['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1).to(gg.x.device)
            tmp.append(gg)



    return tmp