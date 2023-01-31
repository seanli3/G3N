import torch
import torch.nn.functional as F

def train(epoch, model, optimizer, train_loader, device):
    model.train()
    
    L=0
    correct=0
    for data in train_loader:
        pairs, degrees, scatter = data.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)

        optimizer.zero_grad()
        y_grd= (data.y).type(torch.float) 
        pre= model(x, edge_index, (pairs, degrees, scatter), batch_idx)
        pred=torch.sigmoid(pre)              
        lss=F.binary_cross_entropy(pred, y_grd.unsqueeze(-1),reduction='sum')
        
        lss.backward()
        optimizer.step()
        
        correct += torch.round(pred[:,0]).eq(y_grd).sum().item()

        L+=lss.item()
    return correct/800,L/800

def test(model, val_loader, test_loader, device):
    model.eval()
    correct = 0
    L=0
    for data in test_loader:
        pairs, degrees, scatter = data.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)

        pre= model(x, edge_index, (pairs, degrees, scatter), batch_idx)
        pred=torch.sigmoid(pre)
        y_grd= (data.y).type(torch.float) 
        correct += torch.round(pred[:,0]).eq(y_grd).sum().item()

        
        lss=F.binary_cross_entropy(pred, y_grd.unsqueeze(-1),reduction='sum')
        L+=lss.item()
    L=L/200
    s1= correct / 200
    correct = 0
    Lv=0
    for data in val_loader:
        pairs, degrees, scatter = data.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)

        pre= model(x, edge_index, (pairs, degrees, scatter), batch_idx)
        pred=torch.sigmoid(pre)
        y_grd= (data.y).type(torch.float) 
        correct += torch.round(pred[:,0]).eq(y_grd).sum().item()

        lss=F.binary_cross_entropy(pred, y_grd.unsqueeze(-1),reduction='sum')
        Lv+=lss.item()
    s2= correct / 200    
    Lv=Lv/200

    return s1,L, s2, Lv