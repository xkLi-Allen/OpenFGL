import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.loader import NeighborSampler

from openfgl.flcore.feddep.localdep import Encoder
from openfgl.flcore.feddep.dec_cluster.clustering import train_clustering
from openfgl.flcore.feddep.feddep_config import config


def LocalRecLoss(pred_embs, true_embs, pred_missing, true_missing, num_preds):
    use_cuda, device = (pred_embs.device.type != "cpu"), pred_embs.device
    if use_cuda:
        true_missing = true_missing.cpu()
        pred_missing = pred_missing.cpu()
    pred_len = len(pred_embs)
    pred_embs = pred_embs.view(pred_len, num_preds, -1)
    loss = torch.zeros(pred_embs.shape[:2])
    if use_cuda:
        loss = loss.to(device)

    pred_missing_np = (
        np.round(pred_missing.detach().numpy()).reshape(-1).astype(np.int32))
    true_missing_np = true_missing.detach().numpy().reshape(-1).astype(np.int32)
    true_missing_np = np.clip(true_missing_np, 0, num_preds)
    pred_missing_np = np.clip(pred_missing_np, 0, num_preds)

    for i in range(pred_len):
        if true_missing_np[i] > 0:
            if isinstance(true_embs[i][true_missing_np[i] - 1], np.ndarray):
                true_emb_i = torch.tensor(true_embs[i]).to(device)
            else:
                true_emb_i = true_embs[i].to(device)
            for pred_j in range(min(num_preds, pred_missing_np[i])):
                true_embs_tensor = true_emb_i[true_missing_np[i] - 1]
                loss[i][pred_j] = F.mse_loss(
                    pred_embs[i][pred_j].unsqueeze(0).float(),
                    true_embs_tensor.unsqueeze(0).float())

                for true_k in range(min(num_preds, true_missing_np[i] - 1)):
                    true_embs_tensor = true_emb_i[true_k]
                    loss_ijk = F.mse_loss(
                        pred_embs[i][pred_j].unsqueeze(0).float(),
                        true_embs_tensor.unsqueeze(0).float())
                    if torch.sum(loss_ijk.data) < torch.sum(loss[i][pred_j].data):
                        loss[i][pred_j] = loss_ijk
        else:
            continue
    return loss.mean(1).mean(0).float()


def FedRecLoss(pred_embs, true_embs, pred_missing, num_preds):
    use_cuda, device = (pred_embs.device.type != "cpu"), pred_embs.device
    if use_cuda:
        pred_missing = pred_missing.cpu()

    pred_len = len(pred_embs)
    pred_embs = pred_embs.view(pred_len, num_preds, -1)
    loss = torch.zeros(pred_embs.shape[:2])
    if use_cuda:
        loss = loss.to(device)
    pred_missing_np = pred_missing.detach().numpy().reshape(-1).astype(np.int32)
    pred_missing_np = np.clip(pred_missing_np, 0, num_preds)
    if isinstance(true_embs[0], np.ndarray):
        true_embs = torch.tensor(true_embs).to(device)
    else:
        true_embs = true_embs.to(device)
    for i in range(pred_len):
        for pred_j in range(min(num_preds, pred_missing_np[i])):
            loss[i][pred_j] += F.mse_loss(
                pred_embs[i][pred_j].unsqueeze(0).float(),
                true_embs[i][0].unsqueeze(0).float(),
            )
            for true_k in true_embs[i][1:]:
                loss_ijk = F.mse_loss(
                    pred_embs[i][pred_j].unsqueeze(0).float(),
                    true_k.unsqueeze(0).float(),
                )
                if torch.sum(loss_ijk.data) < torch.sum(loss[i][pred_j].data):
                    loss[i][pred_j] = loss_ijk
    return loss.mean(1).mean(0).float()


def get_prototypes(emb, K, batch_size, device, ae_pretrained_epochs, ae_finetune_epochs, dec_epochs):
    emb_shape = emb.shape[1]
    proto_idx = train_clustering(
        node_embs=emb, num_prototypes=K,
        batch_size=batch_size, device=device, ae_pretrained_epochs=ae_pretrained_epochs,
        ae_finetune_epochs=ae_finetune_epochs, dec_epochs=dec_epochs).reshape(-1)

    prototypes = np.zeros(shape=(K, emb_shape))
    proto_idx = np.asarray(proto_idx, dtype=np.int32).reshape(-1)
    if emb.device != "cpu":
        emb = emb.cpu()
    emb = emb.numpy()
    for cluster in range(K):
        row_ix = np.where(proto_idx == cluster)
        prototypes[cluster] = emb[row_ix].mean(axis=0)
    return prototypes, proto_idx


def get_emb(data, hid_dim, output_dim, num_layers, device):
    subgraph_sampler = NeighborSampler(
        data.edge_index,
        num_nodes=data.num_nodes,
        node_idx=torch.tensor([i for i in range(data.num_nodes)]),
        sizes=[5] * num_layers,
        batch_size=4096,
        shuffle=False)
    train_idx = torch.where(data.train_mask == True)[0]
    dataloader = {
        "data": data,
        "train": NeighborSampler(
            data.edge_index,
            num_nodes=data.num_nodes,
            node_idx=train_idx,
            sizes=[5] * num_layers,
            batch_size=config["encoder_batch_size"],
            shuffle=True
        ),
        "val": subgraph_sampler,
        "test": subgraph_sampler
    }

    encoder = Encoder(
        input_dim=data.x.shape[1],
        hid_dim=hid_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.5).to(device)

    encoder.train()
    optim = torch.optim.Adam(encoder.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(config["encoder_epochs"]):
        total_loss, total_correct = 0, 0
        for batch_size, n_id, adjs in dataloader["train"]:
            x, y = data.x[n_id].to(device), data.y[n_id[:batch_size]].to(device)
            out = encoder.forward(x=x, adjs=adjs)
            loss = F.cross_entropy(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_correct += int(out.argmax(dim=-1).cpu().eq(y.cpu()).sum())

        loss = total_loss / len(dataloader["train"])
        approx_acc = total_correct / int(data.train_mask.sum())
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}")

    encoder.eval()
    all_emb = encoder.get_encoder(dataloader["data"].x.to(device), dataloader["test"]).detach()
    return all_emb


class HideGraph(BaseTransform):
    def __init__(self, encoder_hid_dim, encoder_output_dim, encoder_num_layers, hidden_portion, num_preds, num_protos, device):
        self.encoder_hid_dim = encoder_hid_dim
        self.encoder_output_dim = encoder_output_dim
        self.encoder_num_layers = encoder_num_layers
        self.hidden_portion = hidden_portion
        self.num_preds = num_preds
        self.num_protos = num_protos
        self.device = device
        
        
    def __call__(self, data):
        # get prototypes
        emb = get_emb(data=data, hid_dim=self.encoder_hid_dim, output_dim=self.encoder_output_dim, num_layers=self.encoder_num_layers, device=self.device)
        self.prototypes, self.proto_idx = get_prototypes(
            emb=emb,
            K=self.num_protos,
            batch_size=config["cluster_batch_size"],
            device=self.device,
            ae_pretrained_epochs=config["ae_pretrained_epochs"],
            ae_finetune_epochs=config["ae_finetune_epochs"],
            dec_epochs=config["dec_epochs"])
        self.emb = np.zeros((len(self.proto_idx), len(self.prototypes[0])))
        for i in range(len(self.emb)):
            self.emb[i] = self.prototypes[self.proto_idx[i]]

        self.x_missing = torch.zeros((len(data.x), self.num_preds, len(self.emb[0])))

        val_ids = torch.where(data.val_mask == True)[0].to("cpu")
        hide_ids = np.random.choice(
            val_ids, int(len(val_ids) * self.hidden_portion), replace=False)
        remaining_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        remaining_mask[hide_ids] = False
        remaining_nodes = torch.where(remaining_mask == True)[0].numpy()

        data.ids_missing = [[] for _ in range(data.num_nodes)]

        data.emb = torch.FloatTensor(self.emb)
        data.global_map = torch.tensor(
            [i for i in range(data.num_nodes)], dtype=torch.long)

        G = to_networkx(
            data,
            node_attrs=[
                "x", "y", "train_mask",
                "val_mask", "test_mask",
                "global_map", "ids_missing", "emb"
            ],
            to_undirected=True)
        for missing_node in hide_ids:
            neighbors = G.neighbors(missing_node)
            for i in neighbors:
                G.nodes[i]["ids_missing"].append(missing_node)
        for i in G.nodes:
            ids_missing = G.nodes[i]["ids_missing"]
            del G.nodes[i]["ids_missing"]
            G.nodes[i]["num_missing"] = torch.tensor([len(ids_missing)], dtype=torch.float32)
            if len(ids_missing) > 0:
                if len(ids_missing) <= self.num_preds:
                    G.nodes[i]["x_missing"] = torch.tensor(
                        np.vstack((self.emb[ids_missing],
                                np.zeros((self.num_preds-len(ids_missing),
                                    self.emb.shape[1]))))
                    )
                else:
                    G.nodes[i]["x_missing"] = torch.tensor(self.emb[ids_missing[:self.num_preds]])
            else:
                G.nodes[i]["x_missing"] = torch.zeros((self.num_preds, self.emb.shape[1]))
            self.x_missing[i] = G.nodes[i]["x_missing"]
        impaired_graph = from_networkx(nx.subgraph(G, remaining_nodes))
        return impaired_graph, self.emb, self.x_missing


@torch.no_grad()
def GraphMender(model, impaired_data, original_data, num_preds):
    pred_missing, pred_feats, _ = model(impaired_data)
    # Mend the original data
    original_data = original_data.detach().cpu()
    new_edge_index = original_data.edge_index.T
    pred_missing = pred_missing.detach().cpu().numpy()

    pred_feats = pred_feats.detach().cpu().reshape((len(pred_missing), num_preds, -1))

    emb_len = pred_feats.shape[-1]
    start_id = original_data.num_nodes
    mend_emb = torch.zeros(size=(start_id, num_preds, emb_len))
    for node in range(len(pred_missing)):
        num_fill_nodes = np.around(pred_missing[node]).astype(np.int32).item()
        if num_fill_nodes > 0:
            org_id = impaired_data.global_map[node]
            mend_emb[org_id][:num_fill_nodes] += pred_feats[node][:num_fill_nodes]

    filled_data = {
        "data": Data(
                x=original_data.x,
                edge_index=new_edge_index.T,
                y=original_data.y,
                # train_idx=torch.where(original_data.train_mask == True)[0],
                # valid_idx=torch.where(original_data.val_mask == True)[0],
                # test_idx=torch.where(original_data.test_mask == True)[0],
                mend_emb=mend_emb),
        "train_mask": original_data.train_mask,
        "val_mask": original_data.val_mask,
        "test_mask": original_data.test_mask
    }
    return filled_data
