from cytoolz.itertoolz import sliding_window
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from typing import List, Optional


class Classifier(nn.Module):
    def __init__(self, dimensions: List[int]):
        super(Classifier, self).__init__()
        units = []
        for from_dimension, to_dimension in sliding_window(2, dimensions):
            units.append(nn.Linear(from_dimension, to_dimension))
            units.append(nn.ReLU())
        self.classifier = nn.Sequential(*units[:-1])
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, batch):
        return self.softmax(self.classifier(batch))


def pretrain_accuracy(output: torch.Tensor, batch: torch.Tensor) -> float:
    numerator = float((output == batch).view(-1).long().sum().data.cpu())
    denominator = float(output.view(-1).size()[0])
    return numerator / denominator


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy
