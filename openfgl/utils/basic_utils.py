import torch
import random
import numpy as np
import sys
from collections.abc import Iterable
from fvcore.nn import FlopCountAnalysis, parameter_count

def check_args(args):
    pass


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    

    
    
def load_client(args, client_id, data, data_dir, message_pool, device):
    if args.fl_algorithm == "isolate":
        from flcore.isolate.client import IsolateClient
        return IsolateClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedavg":
        from flcore.fedavg.client import FedAvgClient
        return FedAvgClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedprox":
        from flcore.fedprox.client import FedProxClient
        return FedProxClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "scaffold":
        from flcore.scaffold.client import ScaffoldClient
        return ScaffoldClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "moon":
        from flcore.moon.client import MoonClient
        return MoonClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "feddc":
        from flcore.feddc.client import FedDCClient
        return FedDCClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedproto":
        from flcore.fedproto.client import FedProtoClient
        return FedProtoClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedtgp":
        from flcore.fedtgp.client import FedTGPClient
        return FedTGPClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedpub":
        from flcore.fedpub.client import FedPubClient
        return FedPubClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedstar":
        from flcore.fedstar.client import FedStarClient
        return FedStarClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedgta":
        from flcore.fedgta.client import FedGTAClient
        return FedGTAClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedtad":
        from flcore.fedtad.client import FedTADClient
        return FedTADClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedsage_plus":
        from flcore.fedsage_plus.client import FedSagePlusClient
        return FedSagePlusClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "adafgl":
        from flcore.adafgl.client import AdaFGLClient
        return AdaFGLClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "gcfl_plus":
        from flcore.gcfl_plus.client import GCFLPlusClient
        return GCFLPlusClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "feddep":
        from flcore.feddep.client import FedDEPClient
        return FedDEPClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fggp":
        from flcore.fggp.client import FGGPClient
        return FGGPClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fgssl":
        from flcore.fgssl.client import FGSSLClient
        return FGSSLClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedgl":
        from flcore.fedgl.client import FedGLClient
        return FedGLClient(args, client_id, data, data_dir, message_pool, device)
    
def load_server(args, global_data, data_dir, message_pool, device):
    if args.fl_algorithm == "isolate":
        from flcore.isolate.server import IsolateServer
        return IsolateServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedavg":
        from flcore.fedavg.server import FedAvgServer
        return FedAvgServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedprox":
        from flcore.fedprox.server import FedProxServer
        return FedProxServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "scaffold":
        from flcore.scaffold.server import ScaffoldServer
        return ScaffoldServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "moon":
        from flcore.moon.server import MoonServer
        return MoonServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "feddc":
        from flcore.feddc.server import FedDCServer
        return FedDCServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedproto":
        from flcore.fedproto.server import FedProtoServer
        return FedProtoServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedtgp":
        from flcore.fedtgp.server import FedTGPServer
        return FedTGPServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedpub":
        from flcore.fedpub.server import FedPubServer
        return FedPubServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedstar":
        from flcore.fedstar.server import FedStarServer
        return FedStarServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedgta":
        from flcore.fedgta.server import FedGTAServer
        return FedGTAServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedtad":
        from flcore.fedtad.server import FedTADServer
        return FedTADServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedsage_plus":
        from flcore.fedsage_plus.server import FedSagePlusServer
        return FedSagePlusServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "adafgl":
        from flcore.adafgl.server import AdaFGLServer
        return AdaFGLServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "gcfl_plus":
        from flcore.gcfl_plus.server import GCFLPlusServer
        return GCFLPlusServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "feddep":
        from flcore.feddep.server import FedDEPEServer
        return FedDEPEServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fggp":
        from flcore.fggp.server import FGGPServer
        return FGGPServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fgssl":
        from flcore.fgssl.server import FGSSLServer
        return FGSSLServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedgl":
        from flcore.fedgl.server import FedGLServer
        return FedGLServer(args, global_data, data_dir, message_pool, device)
    
def load_optim(args):
    if args.optim == "adam":
        from torch.optim import Adam
        return Adam
    
    
def load_task(args, client_id, data, data_dir, device):
    if args.task == "node_cls":
        from task.node_cls import NodeClsTask
        return NodeClsTask(args, client_id, data, data_dir, device)
    elif args.task == "graph_cls":
        from task.graph_cls import GraphClsTask
        return GraphClsTask(args, client_id, data, data_dir, device)
    elif args.task == "link_pred":
        from task.link_pred import LinkPredTask
        return LinkPredTask(args, client_id, data, data_dir, device)
    elif args.task == "node_clust":
        from task.node_clust import NodeClustTask
        return NodeClustTask(args, client_id, data, data_dir, device)
    


def extract_floats(s):
    from decimal import Decimal
    parts = s.split('-')
    train = float(parts[0])
    val = float(parts[1])
    test = float(parts[2])
    assert Decimal(parts[0]) + Decimal(parts[1]) + Decimal(parts[2]) == Decimal(1)
    return train, val, test

def idx_to_mask_tensor(idx_list, length):
    mask = torch.zeros(length)
    mask[idx_list] = 1
    return mask



def mask_tensor_to_idx(tensor):
    result = tensor.nonzero().squeeze().tolist()
    if type(result) is not list:
        result = [result]
    return result
    

import sys
import torch

def total_size(o):
    size = 0
    if isinstance(o, torch.Tensor):
        size += o.element_size() * o.numel()
    elif isinstance(o, dict):
        size += sum(total_size(v) for v in o.values())
    elif isinstance(o, Iterable):
        size += sum(total_size(i) for i in o)
    return size



def model_complexity(model:torch.nn.Module):
    params = sum([val for val in parameter_count(model).values()])
    return params
    