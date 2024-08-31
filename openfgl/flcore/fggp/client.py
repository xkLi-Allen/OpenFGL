import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
import copy
from torch_geometric.utils import to_torch_csc_tensor
from openfgl.flcore.fggp.models import FedGCN,MLP
from sklearn.neighbors import kneighbors_graph
from openfgl.flcore.fggp.fggp_config import config
from openfgl.flcore.fggp.utils import  get_norm_and_orig,get_proto_norm_weighted,proto_align_loss
import torch.nn.functional as F


class FGGPClient(BaseClient):
    """
    FGGPClient is a client-side implementation for the Federated Graph Learning with Generalizable Prototypes 
    (FGGP) framework. This client handles local training, model updates, and prototype generation in a 
    federated learning setting, focusing on overcoming domain shifts across clients.

    Attributes:
        global_model (nn.Module): A copy of the global model used to compute global embeddings.
        personal_project (nn.Module): A projection layer used for personalizing embeddings.
        data2 (torch_geometric.data.Data): A copy of the data with modified edges for use in the FGGP algorithm.
    """
    
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FGGPClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): The ID of the client.
            data (torch_geometric.data.Data): The graph data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between client and server.
            device (torch.device): The device on which computations will be performed (e.g., CPU or GPU).
        """
        super(FGGPClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(FedGCN(nfeat=self.task.num_feats, nhid=self.args.hid_dim,
                                           nclass=self.task.num_global_classes, nlayer=self.args.num_layers,
                                           dropout=self.args.dropout))
        self.global_model = copy.deepcopy(self.task.model)
        self.task.splitted_data["data"].adj = to_torch_csc_tensor(self.task.data.edge_index)
        self.personal_project = MLP(self.args.hid_dim,self.args.hid_dim,0.5)




    def get_custom_loss_fn(self):
        """
        Returns the custom loss function used during local training. The loss function includes:
        - Cross-entropy loss for classification.
        - Graph augmentation loss for learning on augmented graph structures.
        - Prototype alignment loss to align local and global prototypes.
        """
        def custom_loss_fn(embedding, logits, label, mask):
            loss_ce = torch.nn.functional.cross_entropy(logits[mask], label[mask])

            adj_sampled, adj_logits = self.task.model.aug(self.data2)
            self.data2.adj = adj_sampled
            emb_g,logits_g = self.task.model(self.data2)
            ga_loss = self.data2.norm_w * F.binary_cross_entropy_with_logits(adj_logits, self.data2.adj_orig,
                                                                                pos_weight=self.data2.pos_weight)
            loss_ce2 = F.cross_entropy(logits_g[mask],self.data2.y[mask])
            output_exp = torch.exp(F.log_softmax(logits,dim=1))
            confidences = output_exp.max(1)[0]
            pseudo_labels = output_exp.max(1)[1].type_as(label)
            pseudo_labels[mask] = label[mask]
            confidences[mask] = 1.0
            unique_labels = torch.unique(pseudo_labels)

            proto = get_proto_norm_weighted(self.task.num_global_classes, embedding, pseudo_labels, confidences, unique_labels)
            proto_global = get_proto_norm_weighted(self.task.num_global_classes, emb_g, pseudo_labels, confidences, unique_labels)

            loss_pa = proto_align_loss(proto_global, proto, temperature=0.5)

            loss = loss_ce  + ga_loss + loss_ce2 + loss_pa
            return loss
        return custom_loss_fn




    def execute(self):
        """
        Executes the local training process. This involves:
        - Synchronizing the local and global model parameters with the server.
        - Calculating the k-nearest neighbors graph for global embeddings.
        - Training the model using the custom loss function.
        """
        with torch.no_grad():
            for (local_param, g_p,global_param) in zip(self.task.model.parameters(), self.global_model.parameters(),self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)
                g_p.data.copy_(global_param)


        self.task.loss_fn = self.get_custom_loss_fn()

        self.global_model.eval()
        globel_emb, _ = self.global_model(self.task.data)
        adj = kneighbors_graph(globel_emb.detach().cpu(), config['neibor_num'], metric='cosine')
        del globel_emb, _
        adj.setdiag(1)
        coo = adj.tocoo()
        self.task.data.global_edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long).to(self.device)
        del coo
        del adj
        combined_edge_index = torch.cat([self.task.data.edge_index, self.task.data.global_edge_index], dim=1)
        # combined_edge_index = torch.cat([train_loader.edge_index, train_loader.edge_index], dim=1)
        edge_set = set(zip(combined_edge_index[0].cpu().tolist(), combined_edge_index[1].cpu().tolist()))
        union_edge_index = torch.tensor([[i[0] for i in edge_set], [i[1] for i in edge_set]], dtype=torch.long)

        self.data2 = self.task.splitted_data["data"].clone()

        self.data2.edge_index = union_edge_index
        self.data2 = get_norm_and_orig(self.data2)
        adj_orig = self.data2.adj_orig
        norm_w = adj_orig.shape[0] ** 2 / float((adj_orig.shape[0] ** 2 - adj_orig.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_orig.shape[0] ** 2 - adj_orig.sum()) / adj_orig.sum()]).to(
            self.device)
        self.data2.norm_w = norm_w
        self.data2.pos_weight = pos_weight

        self.task.train()

    def send_message(self):
        """
        Sends the client's local model parameters and the computed prototypes to the server.
        """
        self.task.model.eval()
        emb,logits = self.task.model(self.task.splitted_data["data"])
        #feat = self.personal_project(emb)

        confidences = logits.max(1)[0]
        pseudo_labels = logits.max(1)[1].type_as(self.task.splitted_data["data"].y)
        pseudo_labels[self.task.splitted_data['train_mask']] = self.task.splitted_data["data"].y[self.task.splitted_data['train_mask']]
        confidences[self.task.splitted_data['train_mask']] = 1.0
        unique_labels = torch.unique(pseudo_labels)
        proto = get_proto_norm_weighted(config['N_CLASS'], emb, pseudo_labels, confidences, unique_labels)
        tensor_dict = {i: proto[i].data for i in range(proto.shape[0])}


        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "protos" : tensor_dict
        }

