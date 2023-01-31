import torch

def train(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:

        pairs, degrees, scatter = data.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)
        y = data.y.to(device)

        optimizer.zero_grad()
        out = model(x, edge_index, (pairs, degrees, scatter), batch_idx).squeeze()
        loss = torch.nn.CrossEntropyLoss()(out, y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

def test(loader, model, evaluator, device):
    with torch.no_grad():
        model.train() # eliminate the effect of BN
        y_preds, y_trues = [], []
        for data in loader:

            pairs, degrees, scatter = data.pair_info[0]
            for key in pairs:
                degrees[key] = degrees[key].to(device)
                scatter[key] = scatter[key].to(device)
            
            x = data.x.float().to(device)
            edge_index = data.edge_index.to(device)
            batch_idx = data.batch.to(device)

            y_preds.append(torch.argmax(model(x, edge_index, (pairs, degrees, scatter), batch_idx), dim=-1))
            y_trues.append(data.y)
        y_preds = torch.cat(y_preds, -1)
        y_trues = torch.cat(y_trues, -1)
        return (y_preds == y_trues).float().mean()