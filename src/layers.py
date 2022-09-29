import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import subgraph


class GNNLayer(nn.Module):
    
    def __init__(self, in_features, out_features, params) -> None:
        super(GNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.keys = params['keys']

        self.d = params['d']
        self.t = params['t']
        self.scalar = params['scalar']
        self.combination = params['combination']

        self.one_hot_length = subgraph.one_hot_length(self.t)

        if self.scalar:
            self.eps = nn.ParameterDict()

        self.transform = nn.ModuleDict()
        for key in self.keys:
            k = str(key)
            if self.combination == "multi":
                self.transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.Sigmoid())
            elif self.combination == "sum":
                self.transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.ReLU())
    
            if self.scalar:
                self.eps[k] = torch.nn.Parameter(torch.Tensor([0]))

        self.linear = nn.Linear(self.in_features, self.out_features)

        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def forward(self, h, pair_info):

        # transform roots
        h3 = self.linear(h)

        pairs, degrees, scatter = pair_info

        for key in pairs:
            if len(scatter[key]) == 0:
                continue

            k = str(key)
            
            if self.combination == "multi":  # s(h_x @ W) * s(h_y @ W)
                h_temp = 1
                for i in range(self.t):
                    h_t = torch.hstack((h[pairs[key][i]], degrees[key][i]))
                    h_temp = h_temp * self.transform[k](h_t)
            elif self.combination == "sum":  # s(h_x @ W + h_y @ W)
                h_temp = 0
                for i in range(self.t):
                    h_t = torch.hstack((h[pairs[key][i]], degrees[key][i]))
                    h_temp = h_temp + h_t
                h_temp = self.transform[k](h_temp)

            h_sum = torch.zeros((h.shape[0], self.out_features)).to(self.dummy_param.device)
            scatter_add(src=h_temp, out=h_sum, index=scatter[key], dim=0)

            if self.scalar:
                h_sum = (1 + self.eps[k]) * h_sum

            h3 = h3 + h_sum

        return h3