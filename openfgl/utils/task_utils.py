from torch_geometric.nn.pool import *




def load_graph_cls_default_model(args, input_dim, output_dim, client_id=None):
    """
    Load the default model for graph classification tasks.

    Args:
        args (Namespace): Arguments containing model configurations.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        client_id (int, optional): ID of the client in federated learning. Defaults to None.

    Returns:
        torch.nn.Module: The initialized model.
    """
    if client_id is None: # server
        model_name = args.model[0]
    else: # client
        if len(args.model) > 1:
            model_id = int(len(args.model) * client_id / args.num_clients)
            model_name = args.model[model_id]
        else:
            model_name = args.model[0]
        
            
    if model_name == "gin":
        from openfgl.model.gin import GIN
        return GIN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "global_edge":
        from openfgl.model.global_edge import GlobalEdge
        return GlobalEdge(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "global_pan":
        from openfgl.model.global_pan import GlobalPAN
        return GlobalPAN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "global_sag":
        from openfgl.model.global_sag import GlobalSAG
        return GlobalSAG(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)



def load_node_edge_level_default_model(args, input_dim, output_dim, client_id=None):
    """
    Load the default model for node and edge level tasks.

    Args:
        args (Namespace): Arguments containing model configurations.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        client_id (int, optional): ID of the client in federated learning. Defaults to None.

    Returns:
        torch.nn.Module: The initialized model.
    """
    if client_id is None: # server
        model_name = args.model[0]
    else: # client
        if len(args.model) > 1:
            model_id = int(len(args.model) * client_id / args.num_clients)
            model_name = args.model[model_id]
        else:
            model_name = args.model[0]
    if model_name == "mlp":
        from openfgl.model.mlp import MLP
        return MLP(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gcn":
        from openfgl.model.gcn import GCN
        return GCN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gat":
        from openfgl.model.gat import GAT
        return GAT(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "graphsage":
        from openfgl.model.graphsage import GraphSAGE
        return GraphSAGE(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "sgc":
        from openfgl.model.sgc import SGC
        return SGC(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gcn2":
        from openfgl.model.gcn2 import GCN2
        return GCN2(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    else:
        raise ValueError

