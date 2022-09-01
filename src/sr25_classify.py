import torch
import argparse
import time

from torch_geometric.loader import DataLoader
from tqdm import tqdm
import subgraph
import os
import util
import numpy as np
import models as models

from train.sr25_classify import train, test

def main():
    parser = argparse.ArgumentParser(description='SR-25')
    parser.add_argument('--d', type=int, default=1,
                        help='distance of neighbourhood (default: 1)')
    parser.add_argument('--t', type=int, default=2,
                        help='size of t-subsets (default: 2)')
    parser.add_argument('--scalar', type=bool, default=True,
                        help='learn scalars')
    parser.add_argument('--no-connected', dest='connected', action='store_false',
                        help='also consider disconnected t-subsets')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=40,
                        help='dimensionality of hidden units in GNNs')
    parser.add_argument('--combination', type=str, default="multi", choices=["sum", "multi"],
                        help='pair combination operation (default: multi)')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    dataset = util.SRDataset(root="dataset/sr25/")
    dataset.data.y = torch.arange(len(dataset.data.y)).long() # each graph is a unique class
    train_loader = []
    for batch in tqdm(DataLoader(dataset, batch_size=100, shuffle=True)):
        train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    train_loader = DataLoader(train_loader, batch_size=1, shuffle=True)
    val_loader = []  # same as train
    for batch in tqdm(DataLoader(dataset, batch_size=100, shuffle=False)):
        val_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    val_loader = DataLoader(val_loader, batch_size=1, shuffle=True)

    params = {
        'nfeat':dataset.data.x.shape[1],
        'nhid':args.emb_dim, 
        'nclass':15,  # 15 graphs
        'nlayers':args.num_layer,
        'd':args.d,
        't':args.t, 
        'scalar':args.scalar,  
        'combination':args.combination,
        'keys':subgraph.get_keys_from_loaders([train_loader, val_loader]),
    }

    model = models.GNN_synthetic(params).to(device)
    if iter==0:
        print('emb_dim:', args.emb_dim)
        print('number of parameters:', util.get_n_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=20,
                                                            verbose=True)

    # train loop here
    best_val_perf = test_perf = float('-inf')
    for epoch in range(1, 301):
        start = time.time()
        model.train()
        train_loss = train(train_loader, model, optimizer, device=device)

        model.eval()
        val_perf = test(val_loader, model, evaluator=None, device=device)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = test(val_loader, model, evaluator=None, device=device) 
        time_per_epoch = time.time() - start 

        scheduler.step(train_loss)

        # logger here
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, Seconds: {time_per_epoch:.4f}')

if __name__ == "__main__":
    main()