import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()
    avg_loss = 0
    y_preds, y_trues = [], []
    for iter, data in enumerate(train_loader):
        pairs, degrees, scatter = data.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = data.x.to(device)
        # x = torch.ones((data.x.shape[0], 1)).to(device)
        labels = data.y.to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)

        optimizer.zero_grad()
        output = model(x, edge_index, (pairs, degrees, scatter), batch_idx)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item()
        y_preds.append(torch.argmax(output, dim=-1))
        y_trues.append(labels)

    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    avg_loss /= (iter + 1)
    return avg_loss, (y_preds == y_trues).float().mean()

# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()
    y_preds, y_trues = [], []
    for iter, data in enumerate(loader):
        pairs, degrees, scatter = data.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = data.x.to(device)
        # x = torch.ones((data.x.shape[0], 1)).to(device)
        labels = data.y.to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)

        output = model(x, edge_index, (pairs, degrees, scatter), batch_idx)

        y_preds.append(torch.argmax(output, dim=-1))
        y_trues.append(labels)

    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    return (y_preds == y_trues).float().mean()
