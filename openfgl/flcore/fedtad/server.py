import torch
from openfgl.flcore.base import BaseServer
from openfgl.flcore.fedtad.fedtad_config import config
from torch.optim import Adam
from openfgl.flcore.fedtad.generator import FedTAD_ConGenerator
import torch.nn.functional as F
from openfgl.flcore.fedtad._utils import construct_graph, DiversityLoss
import torch.nn as nn




class FedTADServer(BaseServer):
    """
    FedTADServer implements the server-side operations for the Federated Learning algorithm
    described in the paper 'FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph 
    Federated Learning'. This class handles global model aggregation, the training of a generator 
    for knowledge distillation, and the coordination of knowledge sharing between clients.

    Attributes:
        generator (FedTAD_ConGenerator): A generator model used for creating pseudo graphs to 
                                         facilitate knowledge distillation.
        generator_optimizer (torch.optim.Optimizer): Optimizer for the generator model.
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedTADServer.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible to the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedTADServer, self).__init__(args, global_data, data_dir, message_pool, device)

        self.task.optim = Adam(self.task.model.parameters(), lr=config["lr_d"], weight_decay=self.args.weight_decay)
        self.generator = FedTAD_ConGenerator(noise_dim=config["noise_dim"], feat_dim=args.hid_dim if config["distill_mode"] == 'rep_distill' else self.task.num_feats, out_dim=self.task.num_global_classes, dropout=config["gen_dropout"]).to(device)
        self.generator_optimizer = Adam(self.generator.parameters(), lr=config["lr_g"], weight_decay=self.args.weight_decay)
        
        
        
    def execute(self):
        """
        Executes the main operations of the server during a federated learning round.

        This includes aggregating the model parameters from the clients, training a generator 
        to create pseudo graphs for knowledge distillation, and updating the global model based 
        on the generated data and the knowledge shared by the clients.
        """
        # global aggregation
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
        
        # initialize
        c_cnt = [0] * self.task.num_global_classes
        for class_i in range(self.task.num_global_classes):
            c_cnt[class_i] = int(config["num_gen"] * 1 / self.task.num_global_classes)
        c_cnt[-1] += config["num_gen"] - sum(c_cnt)
        c = torch.zeros(config["num_gen"]).to(self.device).long()
        ptr = 0
        for class_i in range(self.task.num_global_classes):
            for _ in range(c_cnt[class_i]):
                c[ptr] = class_i
                ptr += 1
                
                
        each_class_mask = {}
        for class_i in range(self.task.num_global_classes):
            each_class_mask[class_i] = c == class_i
            each_class_mask[class_i] = each_class_mask[class_i].to(self.device)


        
        for client_id in self.message_pool["sampled_clients"]:
            self.message_pool[f"client_{client_id}"]["model"].eval()
        
        
        for _ in range(config["glb_epochs"]):
            
            ############ sampling noise ##############
            z = torch.randn((config["num_gen"], 32)).to(self.device)
            
            
            ############ train generator ##############
            self.generator.train()
            self.task.model.eval()
            
            for it_g in range(config["it_g"]):
                loss_sem = 0
                loss_diverg = 0
                loss_div = 0
                
                
                self.generator_optimizer.zero_grad()
                for client_id in self.message_pool["sampled_clients"]:    
                    ######  generator forward  ########
                    node_logits = self.generator.forward(z=z, c=c) 
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(node_logits, adj_logits, k=config["topk"])
                    
                    ##### local & global model -> forward #########
                    
                    local_embedding, local_logits = self.message_pool[f"client_{client_id}"]["model"].forward(pseudo_graph)
                    global_embedding, global_logits = self.task.model.forward(pseudo_graph)
                    
                    if config["distill_mode"] == 'rep_distill':
                        local_pred = local_embedding
                        global_pred = global_embedding
                    else:
                        local_pred = local_logits
                        global_pred = global_logits
                        
                        
                    ##########  semantic loss  #############
                    for class_i in range(self.task.num_global_classes):
                        loss_sem += self.message_pool[f"client_{client_id}"]["ckr"][class_i] * nn.CrossEntropyLoss()(local_pred[each_class_mask[class_i]], c[each_class_mask[class_i]])
                        

                    ############  diversity loss  ##############
                    loss_div += DiversityLoss(metric='l1').to(self.device)(z.view(z.shape[0],-1), node_logits) 
                
                
                    ############  divergence loss  ############   
                    for class_i in range(self.task.num_global_classes):
                        loss_diverg += - self.message_pool[f"client_{client_id}"]["ckr"][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_mask[class_i]] - local_pred[each_class_mask[class_i]].detach()), dim=1))


                ############ generator loss #############
                loss_G = config["lam1"] * loss_sem + loss_diverg + config["lam2"] * loss_div
                
                loss_G.backward()
                self.generator_optimizer.step()
                    
                    
                    
                
            ########### train global model ###########

            
            self.generator.eval()
            self.task.model.train()
            
            
            ######  generator forward  ########
            node_logits = self.generator.forward(z=z, c=c)                    
            node_norm = F.normalize(node_logits, p=2, dim=1)
            adj_logits = torch.mm(node_norm, node_norm.t())
            pseudo_graph = construct_graph(node_logits.detach(), adj_logits.detach(), k=config["topk"])
            
            
            
            for it_d in range(config["it_d"]):
                self.task.optim.zero_grad()
                loss_D = 0
                
                for client_id in self.message_pool["sampled_clients"]:    
                    #######  local & global model -> forward  #######
                    local_embedding, local_logits = self.message_pool[f"client_{client_id}"]["model"].forward(pseudo_graph)
                    global_embedding, global_logits = self.task.model.forward(pseudo_graph)
                    
                                        
                    if config["distill_mode"] == 'rep_distill':
                        local_pred = local_embedding
                        global_pred = global_embedding
                    else:
                        local_pred = local_logits
                        global_pred = global_logits
                        
                    ############  divergence loss  ############   
                    for class_i in range(self.task.num_global_classes):
                        loss_D += self.message_pool[f"client_{client_id}"]["ckr"][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_mask[class_i]] - local_pred[each_class_mask[class_i]]), dim=1))

                loss_D.backward()
                self.task.optim.step()

        
        
    def send_message(self):
        """
        Sends the updated global model weights to the clients.
        
        The message sent to the clients includes the updated model parameters after aggregation 
        and knowledge distillation.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }