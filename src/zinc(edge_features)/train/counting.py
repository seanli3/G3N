import torch
from sklearn.metrics import r2_score


def train(epoch, model, train_loader, optimizer, device, ntask, trid):
    model.train()
    
    L=0
    correct=0
    for batch in train_loader:
        pairs, degrees, scatter = batch.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.to(device)
        batch_idx = batch.batch.to(device)
        y = batch.y.to(device)

        optimizer.zero_grad()
        
        pre=model(x, edge_index, (pairs, degrees, scatter), batch_idx)
        
        lss= torch.square(pre- y[:,ntask:ntask+1]).sum() 
        
        lss.backward()
        optimizer.step()  
        L+=lss.item()

    return L/len(trid)

def test(model, val_loader, test_loader,device, ntask, vlid, tsid):
    model.eval()
    yhat=[]
    ygrd=[]
    L=0
    for batch in test_loader:
        pairs, degrees, scatter = batch.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.to(device)
        batch_idx = batch.batch.to(device)
        y = batch.y.to(device)
        
        pre=model(x, edge_index, (pairs, degrees, scatter), batch_idx)

        yhat.append(pre.cpu().detach())
        ygrd.append(y[:,ntask:ntask+1].cpu().detach())
        lss= torch.square(pre- y[:,ntask:ntask+1]).sum()         
        L+=lss.item()
    yhat=torch.cat(yhat)
    ygrd=torch.cat(ygrd)
    testr2=r2_score(ygrd.numpy(),yhat.numpy())

    Lv=0
    for batch in val_loader:
        pairs, degrees, scatter = batch.pair_info[0]
        for key in pairs:
            degrees[key] = degrees[key].to(device)
            scatter[key] = scatter[key].to(device)
        
        x = batch.x.float().to(device)
        edge_index = batch.edge_index.to(device)
        batch_idx = batch.batch.to(device)
        y = batch.y.to(device)
        
        pre=model(x, edge_index, (pairs, degrees, scatter), batch_idx)

        lss= torch.square(pre- y[:,ntask:ntask+1]).sum() 
        Lv+=lss.item()    
    return L/len(tsid), Lv/len(vlid),testr2