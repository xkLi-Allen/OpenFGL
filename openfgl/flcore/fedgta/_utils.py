import torch
import math

def info_entropy_rev(vec, num_neig, eps=1e-8):
    return (num_neig.sum()) * vec.shape[1] * math.exp(-1) + torch.sum(torch.multiply(num_neig, torch.sum(torch.multiply(vec, torch.log(vec+eps)), dim=1)))


def raw_moment(x:torch.Tensor, moment, dim=0):
    tmp = torch.pow(x, moment)
    return torch.mean(tmp, dim=dim)

def central_moment(x:torch.Tensor, moment, dim=0):
    tmp = torch.mean(x, dim=dim)
    if dim == 0:
        tmp = x - tmp.view(1, -1)
    else:
        tmp = x - tmp.view(-1,1)
    tmp = torch.pow(tmp, moment)
    return  torch.mean(tmp, dim=dim)



def compute_moment(x, num_moments=5, dim="h", moment_type="raw"):
    if moment_type == "raw":
        if dim not in ["h", "v"]:
            raise ValueError
        else:
            if dim == "h":
                dim = 1
            else:
                dim = 0
        moment_type = raw_moment
        moment_list = []
        for p in range(num_moments):
            moment_list.append(moment_type(x, moment=p + 1, dim=dim).view(1, -1))
        moment_tensor = torch.cat(moment_list)
        return moment_tensor
    elif moment_type == "central":
        if dim not in ["h", "v"]:
            raise ValueError
        else:
            if dim == "h":
                dim = 1
            else:
                dim = 0
        moment_type = central_moment
        moment_list = []
        for p in range(num_moments):
            moment_list.append(moment_type(x, moment=p + 1, dim=dim).view(1, -1))
        moment_tensor = torch.cat(moment_list)
        return moment_tensor
    elif moment_type == "hybrid":
        o_ = compute_moment(x, num_moments, dim, moment_type="raw")
        m_ = compute_moment(x, num_moments, dim, moment_type="central")
        return torch.cat((o_, m_))
    
    

