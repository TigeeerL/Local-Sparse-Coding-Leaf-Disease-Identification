import torch
import torch.nn
import torch.nn.functional as F
import math

from scipy.optimize import linear_sum_assignment


class LocalDictionaryLoss(torch.nn.Module):
    def __init__(self, penalty):
        super(LocalDictionaryLoss, self).__init__()
        self.penalty = penalty

    def forward(self, A, y, x):
        return self.forward_detailed(A, y, x)[2]

    def forward_detailed(self, A, y, x):
        weight = (y.unsqueeze(1) - A.unsqueeze(0)).pow(2).sum(dim=2)
        a = 0.5 * (y - x @ A).pow(2).sum(dim=1).mean() # average across the number of data points
        b = (weight * x).sum(dim=1).mean()
        return a, b, a + b * self.penalty


def clustering_accuracy(targets, predictions, k=None, return_matching=False):
    assert len(targets) == len(predictions)
    n = len(targets)
    if k is None:
        k = int(targets.max() + 1)
    cost = torch.zeros(k, k)
    for i in range(n):
        cost[targets[i].item(), predictions[i].item()] -= 1
    matching = linear_sum_assignment(cost)
    breakdown = -cost[matching]
    total = breakdown.sum().item() / n
    for i in range(k):
        breakdown[i] /= -cost[i].sum()
    if return_matching:
        return total, breakdown, matching
    else:
        return total, breakdown

def split_in_half(data, labels, k):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in range(k):
        n = sum(labels == i)
        p = torch.randperm(n)
        train_idx = p[:n//2]
        test_idx = p[n//2:]
        train_data.append(data[labels == i][train_idx])
        train_labels.append(labels[labels == i][train_idx])
        test_data.append(data[labels == i][test_idx])
        test_labels.append(labels[labels == i][test_idx])
    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)
    test_data = torch.cat(test_data)
    test_labels = torch.cat(test_labels)

    return train_data, train_labels, test_data, test_labels

def split_data(data, labels, k, train_amount = 0.5):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    train_output_idx = []
    test_output_idx = []
    if train_amount > 1:
        train_amount = train_amount/labels.shape[0]
    for i in range(k):
        n = sum(labels == i)
        p = torch.randperm(n)
        if train_amount > 1:
            train_idx = p[:train_amount//k]
            test_idx = p[train_amount//k:]
        else:
            train_idx = p[:math.floor(n*train_amount)]
            test_idx = p[math.floor(n*train_amount):]
        ###
        interm_idx = labels == i
        interm_idx = interm_idx.nonzero().squeeze()
        train_idx = interm_idx[train_idx]
        train_data.append(data[train_idx])
        train_labels.append(labels[train_idx])
        train_output_idx.append(train_idx)
        # train_data.append(data[labels == i][train_idx])
        # train_labels.append(labels[labels == i][train_idx])
        test_idx = interm_idx[test_idx]
        test_data.append(data[test_idx])
        test_labels.append(labels[test_idx])
        test_output_idx.append(test_idx)
        # test_data.append(data[labels == i][test_idx])
        # test_labels.append(labels[labels == i][test_idx])
    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)
    test_data = torch.cat(test_data)
    test_labels = torch.cat(test_labels)
    train_output_idx = torch.cat(train_output_idx)
    test_output_idx = torch.cat(test_output_idx)
    ###

    return train_data, train_labels, test_data, test_labels

def generate_text(comp_result, compare_poss):
    text = "\""
    for i in range(len(compare_poss)):
        if compare_poss[i] == "reg":
            text += compare_poss[i] + "?"
            if comp_result[i] == 0:
                text += "(ex1)" + " "
            elif comp_result[i] == 1:
                text += "(abs)" + " "
            elif comp_result[i] == 2:
                text += "(ex2)" + " "
        else:
            text += compare_poss[i] + "?"
            if comp_result[i] == False:
                text += "(no)" + " "
            elif comp_result[i] == True:
                text += "(yes)" + " "
        if i == len(compare_poss)-1:
            text = text[:-1]
            # print("123")
            text += "\""
    # if comp_result[0] == False:
    #     text += "no " + compare_poss[0] + " "
    # elif comp_result[0] == True:
    #     text += "with " + compare_poss[0] + " "
    # if comp_result[1] == False:
    #     text += "no " + compare_poss[1] + " "
    # elif comp_result[1] == True:
    #     text += "with " + compare_poss[1] + " "
    # if comp_result[2] == False:
    #     text += "not " + compare_poss[2] + " "
    # elif comp_result[2] == True:
    #     text += "with " + compare_poss[2] + " "
    # if comp_result[3] == False:
    #     text += "not " + compare_poss[3] + " "
    # elif comp_result[3] == True:
    #     text += "with " + compare_poss[3] + " "
    return text

