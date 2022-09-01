import torch
import torch.nn.functional as F

def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def train(model, optimizer, loader, device, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    # for iter, batch in enumerate(tqdm(loader, desc="Iteration")):
    for iter, batch in enumerate(loader):
        pairs, degrees, scatter = batch.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)

        x = torch.zeros((batch.num_nodes, 1)).to(device)
        batch_idx = batch.batch.to(device)

        optimizer.zero_grad()

        batch_scores = model(x, None,(pairs, degrees, scatter), batch_idx)
        batch_labels = batch.y.to(device)
        
        loss = F.cross_entropy(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def eval(model, loader, device, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        # for iter, batch in enumerate(tqdm(loader, desc="Iteration")):
        for iter, batch in enumerate(loader):
            pairs, degrees, scatter = batch.pair_info[0]
            for key in pairs:
                degrees[key] = degrees[key].to(device)
                scatter[key] = scatter[key].to(device)
                
            x = torch.zeros((batch.num_nodes, 1)).to(device)
            batch_idx = batch.batch.to(device)

            batch_scores = model(x, None,(pairs, degrees, scatter), batch_idx)
            batch_labels = batch.y.to(device)

            loss = F.cross_entropy(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc