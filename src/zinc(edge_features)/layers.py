import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import numpy as np
import subgraph

class GNNLayer(nn.Module):
    
    def __init__(self, in_features, out_features, params) -> None:
        super(GNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.keys = params['keys']
        
        self.edge_infeat = params['edge_attr']
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
                self.transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length + self.edge_infeat, self.out_features), torch.nn.Sigmoid())
            elif self.combination == "sum":
                self.transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length + self.edge_infeat, self.out_features), torch.nn.ReLU())
    
            if self.scalar:
                self.eps[k] = torch.nn.Parameter(torch.Tensor([0]))

        self.linear = nn.Linear(self.in_features, self.out_features)

        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def forward(self, h, edge_attr, edge_index, edge_feat_mat, pair_info):

        # transform roots
        h3 = self.linear(h)

        pairs, degrees, scatter = pair_info

        for key in pairs:
            if len(scatter[key]) == 0:
                continue

            k = str(key)
            
            edge_keys_1_2 = []
            edge_keys_3_2 = []
            edge_keys_3_1 = []
            subgraph_edge_attributes = []
            
            if(pairs[key].shape[0]==2):
                row, col = pairs[key]
                for i in np.arange(len(row)):
                    row_index = pairs[key][0][i].item()
                    col_index = pairs[key][1][i].item()
                    
                    index = int(edge_feat_mat[row_index, col_index].item())
                    subgraph_edge_attributes.append(edge_attr[index])
            else:
                pos1, pos2, pos3 = pairs[key]
                for i in np.arange(len(pos1)):
                    pos1_index = pairs[key][0][i].item()
                    pos2_index = pairs[key][1][i].item()
                    pos3_index = pairs[key][2][i].item()
                    
                    index_1_2 = int(edge_feat_mat[pos1_index, pos2_index].item())
                    index_3_2 = int(edge_feat_mat[pos2_index, pos3_index].item())
                    index_3_1 = int(edge_feat_mat[pos1_index, pos3_index].item())
                    
                    if(index_1_2 != -1):
                        edge_keys_1_2.append(edge_attr[index_1_2])
                    else:
                        edge_keys_1_2.append(torch.zeros(edge_attr.shape[1]))
                        
                    if(index_3_2 != -1):
                        edge_keys_3_2.append(edge_attr[index_3_2])
                    else:
                        edge_keys_3_2.append(torch.zeros(edge_attr.shape[1]))
                        
                    if(index_3_1 != -1):
                        edge_keys_3_1.append(edge_attr[index_3_1])
                    else:
                        edge_keys_3_1.append(torch.zeros(edge_attr.shape[1]))
                    
                    subgraph_edge_attributes.append(edge_keys_1_2[i]+edge_keys_3_2[i]+edge_keys_3_1[i])
            
            edge_attributes = torch.stack(subgraph_edge_attributes)
            
            if self.combination == "multi":  # s(h_x @ W) * s(h_y @ W)
                h_temp = 1
                for i in range(self.t):
                    h_t = torch.hstack((h[pairs[key][i]], degrees[key][i], edge_attributes))
                    h_temp = h_temp * self.transform[k](h_t)
            elif self.combination == "sum":  # s(h_x @ W + h_y @ W)
                h_temp = 0
                for i in range(self.t):
                    h_t = torch.hstack((h[pairs[key][i]], degrees[key][i], edge_attributes))
                    h_temp = h_temp + h_t
                h_temp = self.transform[k](h_temp)

            h_sum = torch.zeros((h.shape[0], self.out_features)).to(self.dummy_param.device)
            scatter_add(src=h_temp, out=h_sum, index=scatter[key], dim=0)

            if self.scalar:
                h_sum = (1 + self.eps[k]) * h_sum

            h3 = h3 + h_sum

        return h3