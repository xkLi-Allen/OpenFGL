import torch
from openfgl.flcore.base import BaseServer
import numpy as np
from openfgl.flcore.fggp.utils import FINCH
from openfgl.flcore.fggp.fggp_config import config
from openfgl.flcore.fggp.models import FedGCN,MLP


class FGGPServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FGGPServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.load_custom_model(FedGCN(nfeat=self.task.num_feats, nhid=self.args.hid_dim,
                                           nclass=self.task.num_global_classes, nlayer=self.args.num_layers,
                                           dropout=self.args.dropout))

    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in
                                   self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                if config["params_weight"] == "samples_num":
                    weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                else:
                    weight = 1/len(self.message_pool["sampled_clients"])

                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                       self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
        self.global_protos = self.proto_aggregation()

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }

    def proto_aggregation(self):
        agg_protos_label = dict()
        for idx in self.message_pool["sampled_clients"]:
            local_protos = self.message_pool[f"client_{idx}"]["protos"]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].data]

        for num, each_class_proto in agg_protos_label.items():
            if len(each_class_proto) == 1:
                proto = each_class_proto[0].to(self.device)
            else:
                proto = torch.cat(each_class_proto, dim=0).to(self.device)
            y_hat = torch.ones(proto.shape[0]).to(self.device) * num
        return agg_protos_label