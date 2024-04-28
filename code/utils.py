import torch
import torch.nn as nn
import numpy as np


def entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def single_entropy(fc2_s):
    fc2_s = nn.Softmax(-1)(fc2_s)
    entropy = torch.sum(- fc2_s * torch.log(fc2_s + 1e-10), dim=1)
    entropy_norm = np.log(fc2_s.size(1))
    entropy = entropy / entropy_norm
    return entropy


def margin(out):
    out = nn.Softmax(-1)(out)
    top2 = torch.topk(out, 2).values
    return 1 - (top2[:, 0] - top2[:, 1])


def get_target_weight(entropy, consistency, threshold):
    sorce = (entropy + consistency) / 2
    weight = [0.0 for i in range(len(sorce))]
    for i in range(len(sorce)):
        if sorce[i] < (threshold / 2):
            weight[i] = 1.0
    return torch.tensor(weight)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()


def nega_normalize_weight(x):
    x = 1 - x
    return x.detach()
