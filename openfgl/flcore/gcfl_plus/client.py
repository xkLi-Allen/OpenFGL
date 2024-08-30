import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
import numpy as np
from openfgl.flcore.gcfl_plus.models import GIN


class GCFLPlusClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(GCFLPlusClient, self).__init__(args, client_id, data, data_dir, message_pool, device, personalized=True)
        self.task.load_custom_model(GIN(nfeat=self.task.num_feats,nhid=self.args.hid_dim,nlayer=self.args.num_layers,nclass=self.task.num_global_classes,dropout=self.args.dropout))

        self.W = {key: value for key, value in self.task.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.task.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.task.model.named_parameters()}
        self.gconvNames = None
        # for k,v in self.task.model.named_parameters():
        #     if 'convs' in k:
        #         self.gconvNames.append(k)

    def execute(self):
        if self.message_pool["round"] == 0:
            self.gconvNames = self.message_pool["server"]["cluster_weights"][0][0].keys()

        for i, ids in enumerate(self.message_pool["server"]["cluster_indices"]):
            if self.client_id in ids:
                j = ids.index(self.client_id)
                tar = self.message_pool["server"]["cluster_weights"][i][j]
                for k in tar:
                    self.W[k].data = tar[k].data.clone()


        for k in self.gconvNames:
            self.W_old[k].data = self.W[k].data.clone()

        self.task.train()

        for k in self.gconvNames:
            self.dW[k].data = self.W[k].data.clone() - self.W_old[k].data.clone()

        # self.weightsNorm = torch.norm(flatten(self.W)).item()
        #
        # weights_conv = {key: self.W[key] for key in self.gconvNames}
        # self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()
        #
        # dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        # self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        # grads = {key: value.grad for key, value in self.W.items()}
        # self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

        for k in self.gconvNames:
            self.W[k].data = self.W_old[k].data.clone()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "W": self.W,
            "convGradsNorm": self.convGradsNorm,
            "dW": self.dW
        }

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])