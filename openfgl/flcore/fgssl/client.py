import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
import copy
from openfgl.flcore.fgssl.models import *
import openfgl.flcore.fgssl.augment as A
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
import openfgl.flcore.fgssl.losses as L



class FGSSLClient(BaseClient):
    """
    FGSSLClient implements the client-side functionality for the Federated Graph Semantic and Structural Learning (FGSSL)
    framework. This client performs graph neural network (GNN) training while considering both semantic and structural
    aspects of the graph. The client leverages global model knowledge and augments the data with various contrastive
    learning techniques.

    Attributes:
        global_model (nn.Module): A copy of the global model received from the server.
        cos (nn.Module): Cosine similarity function used for contrastive learning.
        contrast_model (nn.Module): Model for performing single-branch contrastive learning.
        withcontrast_model (nn.Module): Model for performing within-embedding contrastive learning.
        augWeak (A.Compose): Weak data augmentation pipeline.
        augStrongF (A.Compose): Strong data augmentation pipeline.
        augNone (A.Identity): Identity augmentation (no augmentation).
        ccKD (nn.Module): Correlation-based knowledge distillation module.
    """
    
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FGSSLClient with the provided arguments, client ID, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): Unique identifier for the client.
            data (object): Graph data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between the client and server.
            device (torch.device): Device on which computations will be performed (e.g., CPU or GPU).
        """
        super(FGSSLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.global_model = copy.deepcopy(self.task.model).to(self.device)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.contrast_model = SingleBranchContrast(loss=L.InfoNCE(tau=0.1), mode='L2L').to(device)
        self.withcontrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)
        self.augWeak = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
        self.augStrongF = A.Compose([A.EdgeRemoving(pe=0.8), A.FeatureMasking(pf=0.5)])
        self.augNone = A.Identity()
        self.ccKD = Correlation()
            
            
            
    def get_custom_loss_fn(self):
        """
        Returns the custom loss function for training the model. The loss function includes cross-entropy loss 
        for classification, as well as contrastive and distillation losses when the model is in training mode.

        Returns:
            function: A custom loss function that computes the loss based on model outputs and target labels.
        """

        def custom_loss_fn(embedding, logits, label, mask):
            loss1 = torch.nn.functional.cross_entropy(logits[mask], label[mask])
            if self.task.model.training:

                self.global_model.eval()
                batch = self.task.splitted_data["data"]
                embedding_g, logits_g = self.global_model.forward(batch)

                batch1 = copy.deepcopy(batch)
                batch2 = copy.deepcopy(batch)

                g1, edge_index1, edge_weight1 = self.augWeak(batch.x, batch.edge_index)
                g2, edge_index2, edge_weight2 = self.augStrongF(batch.x, batch.edge_index)

                batch1.x = g1
                batch1.edge_index = edge_index1

                batch2.x = g2
                batch2.edge_index = edge_index2

                with torch.no_grad():
                    pred_aug_global, globalOne = self.global_model(batch1)

                _, now2 = self.task.model(batch1)

                pred_aug_local, now = self.task.model(batch2)

                adj_orig = to_dense_adj(batch.edge_index, max_num_nodes=batch.x.shape[0]).squeeze(0).to(self.device)

                struct_kd = com_distillation_loss(pred_aug_global, pred_aug_local, adj_orig, adj_orig, 3)
                simi_kd_loss = simi_kd(pred_aug_global, pred_aug_local, batch.edge_index, 4)

                # rkd_Loss = rkd_loss(pred_aug_local , pred_aug_global)
                # "tag"
                cc_loss = self.ccKD(pred_aug_local, pred_aug_global)
                loss3 = simi_kd_2(adj_orig, pred_aug_local, pred_aug_global)
                loss_ff = edge_distribution_high(batch.edge_index, pred_aug_local, pred_aug_global)
                globalOne = globalOne[mask]
                now = now[mask]
                now2 = now2[mask]
                extra_pos_mask = torch.eq(label[mask], label[mask].unsqueeze(dim=1)).to(self.device)
                extra_pos_mask.fill_diagonal_(True)

                extra_neg_mask = torch.ne(label[mask], label[mask].unsqueeze(dim=1)).to(self.device)
                extra_neg_mask.fill_diagonal_(False)

                loss3 = self.contrast_model(globalOne, now, extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
                loss3 = self.contrast_model(now2, now, extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)

                loss = loss1 + loss_ff * 0.1
            else:
                loss = loss1
            return loss
        return custom_loss_fn



    def execute(self):
        """
        Executes the local training process. The global model weights are first updated with the received 
        global parameters. The model is then trained using the custom loss function defined in `get_custom_loss_fn`.
        """
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)
            for (local_param, global_param) in zip(self.global_model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()
        
        

    def send_message(self):
        """
        Sends the locally trained model parameters to the server.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }
        
def com_distillation_loss(t_logits, s_logits, adj_orig, adj_sampled, temp):

    s_dist = F.log_softmax(s_logits / temp, dim=-1)
    t_dist = F.softmax(t_logits / temp, dim=-1)
    kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())


    adj = torch.triu(adj_orig).detach()
    edge_list = (adj + adj.T).nonzero().t()

    s_dist_neigh = F.log_softmax(s_logits[edge_list[0]] / temp, dim=-1)
    t_dist_neigh = F.softmax(t_logits[edge_list[1]] / temp, dim=-1)

    kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss

def simi_kd_2(adj_orig, feats, out):
    tau = 0.1

    adj = torch.triu(adj_orig)
    edge_idx = (adj + adj.T).nonzero().t()
    feats = F.softmax(feats / tau, dim=-1)
    out = F.softmax(out / tau, dim=-1)

    src = edge_idx[0]
    dst = edge_idx[1]

    _1 = torch.cosine_similarity(feats[src], feats[dst], dim=-1)
    _2 = torch.cosine_similarity(out[src], out[dst], dim=-1)

    loss = F.kl_div(_1, _2)
    return loss

def edge_distribution_high(edge_idx, feats, out):

    tau =0.1
    src = edge_idx[0]
    dst = edge_idx[1]
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    feats_abs = torch.abs(feats[src] - feats[dst])
    e_softmax = F.log_softmax(feats_abs / tau, dim=-1)

    out_1 = torch.abs(out[src] - out[dst])
    e_softmax_2 = F.log_softmax(out_1 / tau, dim=-1)

    loss_s = criterion_t(e_softmax, e_softmax_2)
    return loss_s

def simi_kd(global_nodes, local_nodes, edge_index, temp):
    adj_orig = to_dense_adj(edge_index).squeeze(0)
    adj_orig.fill_diagonal_(True)
    s_dist = F.log_softmax(local_nodes / temp, dim=-1)
    t_dist = F.softmax(global_nodes / temp, dim=-1)
    # kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())
    local_simi = torch.cosine_similarity(local_nodes.unsqueeze(1), local_nodes.unsqueeze(0), dim=-1)
    global_simi = torch.cosine_similarity(global_nodes.unsqueeze(1), global_nodes.unsqueeze(0), dim=-1)

    local_simi = torch.where(adj_orig > 0, local_simi, torch.zeros_like(local_simi))
    global_simi = torch.where(adj_orig > 0, global_simi, torch.zeros_like(global_simi))

    s_dist_neigh = F.log_softmax(local_simi / temp, dim=-1)
    t_dist_neigh = F.softmax(global_simi / temp, dim=-1)

    kd_loss = temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss