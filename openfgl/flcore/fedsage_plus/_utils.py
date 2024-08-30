import torch
import numpy as np
import torch.nn.functional as F

def greedy_loss(pred_feats, true_feats, pred_missing, true_missing, max_pred):
    num_samples = pred_feats.shape[0] 
    loss = torch.zeros(pred_feats.shape).to(pred_feats.device)
    true_missing = torch.clip(true_missing,0, max_pred).long()
    pred_missing = torch.clip(pred_missing, 0, max_pred).long()
    
    
    for i in range(num_samples):
        for pred_j in range(min(max_pred, int(pred_missing[i]))):
            if true_missing[i]>0:
                true_feats_tensor = true_feats[i][true_missing[i]-1]
                loss[i][pred_j] += F.mse_loss(pred_feats[i][pred_j].unsqueeze(0),
                                                  true_feats_tensor.unsqueeze(0)).squeeze(0)

                for true_k in range(min(max_pred, true_missing[i])):
                    true_feats_tensor = true_feats[i][true_k]
                    loss_ijk = F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                        true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                    if torch.sum(loss_ijk) < torch.sum(loss[i][pred_j].data):
                        loss[i][pred_j]=loss_ijk
            else:
                continue
    return loss.unsqueeze(0).mean()
