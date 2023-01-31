import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from layers import *
from torch.nn import Linear, Sequential, ReLU, Sigmoid, Dropout, BatchNorm1d


class GNN_ogb(nn.Module):  # for molhiv
    def __init__(self, params):
        super(GNN_ogb, self).__init__()

        self.d = params['d']
        self.t = params['t']
        self.scalar = params['scalar']
        
        self.nfeat = params['nfeat']
        self.nhid = params['nhid']
        self.nlayers = params['nlayers']
        self.nclass = params['nclass']

        self.readout = params['readout']
        self.dropout = params['dropout']
        self.jk = params['jk']

        self.atom_encoder = AtomEncoder(self.nhid)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(self.nlayers):
            self.convs.append(GNNLayer(self.nhid, self.nhid, params))
            self.batch_norms.append(BatchNorm1d(self.nhid))

        # pooler
        if self.readout == "sum":
            self.pool = global_add_pool
        elif self.readout == "mean":
            self.pool = global_mean_pool
        else:
            raise NotImplementedError

        # classifier
        if self.jk:
            self.linears_prediction = torch.nn.ModuleList()
            for layer in range(self.nlayers+1):
                self.linears_prediction.append(nn.Linear(self.nhid, self.nclass))
        else:
            self.graph_pred_linear = torch.nn.Linear(self.nhid, self.nclass)

    def forward(self, h, edge_index, pair_info, batch):

        h_list = [self.atom_encoder(h)]
        for layer in range(self.nlayers):
            h = self.convs[layer](h_list[layer], pair_info)
            h = self.batch_norms[layer](h)
            if layer == self.nlayers - 1:  #remove relu for the last layer
                h = F.dropout(h, self.dropout, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training = self.training)
            
            h_list.append(h)

        if self.jk:  # perform pooling over all nodes in each graph in every layer
            ret = 0
            for layer in range(self.nlayers + 1):
                ret += self.linears_prediction[layer](self.pool(h_list[layer], batch))
        else:
            ret = self.graph_pred_linear(self.pool(h_list[-1], batch))
        
        return ret

class GNN_bench(nn.Module):  # for zinc, csl and tu datasets
    
    def __init__(self, params):
        super().__init__()

        self.d = params['d']
        self.t = params['t']
        self.scalar = params['scalar']
        
        self.nfeat = params['nfeat']
        self.edge_feat = params['edge_attr']
        self.nhid = params['nhid']
        self.nlayers = params['nlayers']
        self.nclass = params['nclass']

        self.readout = params['readout']
        self.dropout = params['dropout']
        self.jk = params['jk']

        self.tu = 'tu' in params

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.embedding_h = nn.Linear(self.nfeat, self.nhid)
        
        for layer in range(self.nlayers):
            self.convs.append(GNNLayer(self.nhid, self.nhid, params))
            self.batch_norms.append(BatchNorm1d(self.nhid))
            
        # pooler
        if self.readout == "sum":
            self.pool = global_add_pool
        elif self.readout == "mean":
            self.pool = global_mean_pool
        else:
            raise NotImplementedError

        if self.tu or self.nclass > 1:
            if self.jk:
                self.linears_prediction = torch.nn.ModuleList()
                for layer in range(self.nlayers+1):
                    self.linears_prediction.append(nn.Linear(self.nhid, self.nclass))
            else:
                self.linears_prediction = nn.Linear(self.nhid, self.nclass)
        else: # mlp readout for zinc
            hidden_multiplier = params['multiplier']
            if self.jk:
                self.linears_prediction = torch.nn.ModuleList()
                for layer in range(self.nlayers+1):
                    self.linears_prediction.append(nn.Linear(self.nhid, hidden_multiplier * self.nhid))
            else:
                self.linears_prediction = nn.Linear(self.nhid, hidden_multiplier * self.nhid)
            self.fc2 = nn.Linear(hidden_multiplier * self.nhid, self.nclass)
        

    def forward(self, h, edge_attr, edge_index, edge_feat_mat, pair_info, batch):
        h = self.embedding_h(h)
        
        h_list = [h]
        
        for layer in range(len(self.convs)):
            h = self.convs[layer](h, edge_attr, edge_index, edge_feat_mat, pair_info)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            h = h + h_list[layer]  # residual 
            h = F.dropout(h, self.dropout, training=self.training)
            h_list.append(h)

        if self.jk:  # waste of parameter budget for zinc
            h = 0
            for layer in range(self.nlayers + 1):
                h += self.linears_prediction[layer](self.pool(h_list[layer], batch))
        else:
            h = self.linears_prediction(self.pool(h_list[-1], batch))

        if self.tu or self.nclass > 1:
            return h
        else:  #mlp readout for zinc
            h = F.relu(h)
            h = self.fc2(h)
            return h

class GNN_synthetic(nn.Module):  # for all other synthetic datasets
    
    def __init__(self, params):
        super().__init__()

        self.d = params['d']
        self.t = params['t']
        self.scalar = params['scalar']
        
        self.nfeat = params['nfeat']
        self.nhid = params['nhid']
        self.nlayers = params['nlayers']
        self.nclass = params['nclass']

        if 'counting' in params:
            self.counting = params['counting']
        else:
            self.counting = False

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.embedding_h = nn.Linear(self.nfeat, self.nhid)
        for layer in range(self.nlayers):
            self.convs.append(GNNLayer(self.nhid, self.nhid, params))
            self.batch_norms.append(BatchNorm1d(self.nhid))
            
        # classifier
        if self.counting:  # mlp output        
            self.fc1 = torch.nn.Linear(self.nhid, self.nhid)
            self.fc2 = torch.nn.Linear(self.nhid, self.nclass)
        else:  # linear output
            self.fc1 = torch.nn.Linear(self.nhid, self.nclass)

    def forward(self, h, edge_index, pair_info, batch):
        h = self.embedding_h(h)
        
        for layer in range(len(self.convs)):
            h = self.convs[layer](h, pair_info)
            h = self.batch_norms[layer](h)
            h = F.relu(h)

        h = global_add_pool(h, batch)

        if self.counting:  # mlp output
            h = F.relu(self.fc1(h))
            h = self.fc2(h)
        else:  # linear output
            h = self.fc1(h)

        return h