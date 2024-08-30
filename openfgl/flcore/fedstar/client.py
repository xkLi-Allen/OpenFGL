import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedstar._utils import init_structure_encoding
from openfgl.flcore.fedstar.gin_dc import DecoupledGIN
from torch_geometric.loader import DataLoader
from openfgl.flcore.fedstar.fedstar_config import config


class FedStarClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):

        super(FedStarClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.task.load_custom_model(DecoupledGIN(input_dim=self.task.num_feats, hid_dim=self.args.hid_dim, output_dim=self.task.num_global_classes, n_se=config["n_rw"] + config["n_dg"], num_layers=self.args.num_layers, dropout=self.args.dropout).to(self.device))
        self.task.data = init_structure_encoding(config["n_rw"], config["n_dg"], self.task.data, config["type_init"])


        tmp = torch.nonzero(self.task.train_mask, as_tuple=True)[0]
        self.task.splitted_data['train_dataloader'] = DataLoader([self.task.data[i] for i in tmp], batch_size=self.args.batch_size, shuffle=False)
        tmp = torch.nonzero(self.task.val_mask, as_tuple=True)[0]
        self.task.splitted_data['val_dataloader'] = DataLoader([self.task.data[i] for i in tmp], batch_size=self.args.batch_size, shuffle=False)
        tmp = torch.nonzero(self.task.test_mask, as_tuple=True)[0]
        self.task.splitted_data['test_dataloader'] = DataLoader([self.task.data[i] for i in tmp], batch_size=self.args.batch_size, shuffle=False)
        
    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, label, mask):
            loss = torch.nn.functional.nll_loss(logits[mask], label[mask])
            return loss
        return custom_loss_fn


    
    def execute(self):
        with torch.no_grad():
            g_w = self.message_pool["server"]["weight"]
            for k,v in self.task.model.state_dict().items():
                if '_s' in k:
                    v.data = g_w[k].data.clone()
        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": self.task.model.state_dict()
        }

