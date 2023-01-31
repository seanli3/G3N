import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, device, loader, optimizer, evaluator):
    model.train()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pairs, degrees, scatter = batch.pair_info[0]
            for key in pairs:
                degrees[key] = degrees[key].to(device)
                scatter[key] = scatter[key].to(device)
            
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_idx = batch.batch.to(device)

            pred = model(x, edge_index, (pairs, degrees, scatter), batch_idx)

            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            loss = F.binary_cross_entropy_with_logits(pred.to(torch.float32)[is_labeled], batch.y.to(device).to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pairs, degrees, scatter = batch.pair_info[0]
                for key in pairs:
                    degrees[key] = degrees[key].to(device)
                    scatter[key] = scatter[key].to(device)
                
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                batch_idx = batch.batch.to(device)

                pred = model(x, edge_index,(pairs, degrees, scatter), batch_idx)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)