import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedstar._utils import init_structure_encoding
from openfgl.flcore.fedstar.gin_dc import serverGIN_dc,DecoupledGIN
from openfgl.flcore.fedstar.fedstar_config import config
from torch_geometric.loader import DataLoader


class FedStarServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedStarServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self.task.load_custom_model(serverGIN_dc(n_se=config["n_rw"] + config["n_dg"],
                                                 num_layers=self.args.num_layers, hid_dim=self.args.hid_dim).to(self.device))
        if global_data is not None:
            self.task.data = init_structure_encoding(config["n_rw"], config["n_dg"], self.task.data,
                                                     config["type_init"])

            tmp = torch.nonzero(self.task.train_mask, as_tuple=True)[0]
            self.task.splitted_data['train_dataloader'] = DataLoader([self.task.data[i] for i in tmp],
                                                                     batch_size=self.args.batch_size, shuffle=False)
            tmp = torch.nonzero(self.task.val_mask, as_tuple=True)[0]
            self.task.splitted_data['val_dataloader'] = DataLoader([self.task.data[i] for i in tmp],
                                                                   batch_size=self.args.batch_size, shuffle=False)
            tmp = torch.nonzero(self.task.test_mask, as_tuple=True)[0]
            self.task.splitted_data['test_dataloader'] = DataLoader([self.task.data[i] for i in tmp],
                                                                    batch_size=self.args.batch_size, shuffle=False)





    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in
                                   self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                local_weight = self.message_pool[f"client_{client_id}"]["weight"]

                for k,v in self.task.model.state_dict().items():
                    if '_s' in k:
                        if it == 0:
                            v.data.copy_(weight * local_weight[k])
                        else:
                            v.data += weight * local_weight[k]



    def send_message(self):
        self.message_pool["server"] = {
            "weight": self.task.model.state_dict()
        }