import random
import torch
import torch.nn.functional as F
from openfgl.flcore.adafgl.op import LaplacianGraphOp
import scipy.sparse as sp
import torch


    
def idx_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


class NonParaLP():
    def __init__(self, prop_steps, num_class, alpha, train_mask, val_mask, test_mask, r=0.5):
        self.prop_steps = prop_steps
        self.r = r
        self.num_class = num_class
        self.alpha = alpha
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.graph_op = LaplacianGraphOp(prop_steps=self.prop_steps, r=self.r)

    def preprocess(self, nodes_embedding, subgraph, device):
        self.subgraph = subgraph
        self.y = subgraph.y
        self.label = F.one_hot(self.y.view(-1), self.num_class).to(torch.float).to(device)
        num_nodes = len(self.train_mask)

        train_idx_list = torch.where(self.train_mask == True)[0].cpu().numpy().tolist()
        num_train = int(len(train_idx_list) / 2)

        random.shuffle(train_idx_list)
        self.lp_train_idx = idx_to_mask(train_idx_list[: num_train], num_nodes)
        self.lp_eval_idx = idx_to_mask(train_idx_list[num_train: ], num_nodes)

        unlabel_idx = self.lp_eval_idx | self.val_mask.cpu() | self.test_mask.cpu()
        unlabel_init = torch.full([self.label[unlabel_idx].shape[0], self.label[unlabel_idx].shape[1]], 1 / self.num_class).to(device)
        self.label[self.lp_eval_idx + self.val_mask.cpu() + self.test_mask.cpu()] = unlabel_init
        
    def propagate(self, adj):
        self.output = self.graph_op.init_lp_propagate(adj, self.label, init_label=self.lp_train_idx, alpha=self.alpha)
        self.output = self.output[-1]

    def eval(self, i=None):
        pred = self.output.max(1)[1].type_as(self.subgraph.y)
        correct = pred[self.lp_eval_idx].eq(self.subgraph.y[self.lp_eval_idx]).double()
        correct = correct.sum()
        reliability_acc = (correct / self.subgraph.y[self.lp_eval_idx].shape[0]).item()
        return reliability_acc