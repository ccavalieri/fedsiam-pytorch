#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedReptile(w_global, w_clients, epsilon=1.0):
    """
    Federated Reptile aggregation.
    """
    w_avg = FedAvg(w_clients)
    
    w_new = copy.deepcopy(w_global)
    for k in w_new.keys():
        w_new[k] = w_global[k] + epsilon * (w_avg[k] - w_global[k])
    
    return w_new
