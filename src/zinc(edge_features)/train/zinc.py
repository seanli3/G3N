import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    return MAE

def train(model, optimizer, loader, epoch, device):
    model.train()
    epoch_loss = 0
    nb_data = 0
    gpu_mem = 0
    for iter, batch in enumerate(loader):
        pairs, degrees, scatter = batch.pair_info[0]
        edge_dictionary = {}
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = batch.x
        edge_attr = batch.edge_attr
            
        x = F.one_hot(x, num_classes=28).float().squeeze(1).to(device)
        edge_attr = F.one_hot(edge_attr, num_classes=4).float().squeeze(1).to(device)
        
        row, col =batch.edge_index
        edge_feat_mat = torch.zeros((len(row), len(col))) - 1

        for i in np.arange(len(row)):
            edge_feat_mat[row[i].item(), col[i].item()] = i
            
        batch_idx = batch.batch.to(device)

        optimizer.zero_grad()

        # zinc pos enc not used
        batch_scores = model(x, edge_attr, batch.edge_index, edge_feat_mat, (pairs, degrees, scatter), batch_idx).squeeze(1)
        batch_targets = batch.y.to(device)
        
        loss = MAE(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    
    return epoch_loss, optimizer

def eval(model, loader, epoch, device):
    model.eval()
    epoch_test_loss = 0
    nb_data = 0
    with torch.no_grad():
        for iter, batch in enumerate(loader):

            pairs, degrees, scatter = batch.pair_info[0]
            for key in pairs:
                degrees[key] = degrees[key].to(device)
                scatter[key] = scatter[key].to(device)
            
            x = batch.x
            edge_attr = batch.edge_attr
            
            x = F.one_hot(x, num_classes=28).float().squeeze(1).to(device)
            edge_attr = F.one_hot(edge_attr, num_classes=4).float().squeeze(1).to(device)
            
            row, col =batch.edge_index
            edge_feat_mat = torch.zeros((len(row), len(col)))

            for i in np.arange(len(row)):
                edge_feat_mat[row[i].item(), col[i].item()] = i
            
            batch_idx = batch.batch.to(device)

            # zinc pos enc not used
            batch_scores = model(x, edge_attr, batch.edge_index, edge_feat_mat, (pairs, degrees, scatter), batch_idx).squeeze(1)
            batch_targets = batch.y.to(device)

            loss = MAE(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        
    return epoch_test_loss