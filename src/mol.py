import os
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import models as models
from tqdm import tqdm

import argparse
import time
import numpy as np
import pathlib

import subgraph
from train.mol import train, eval

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def main():
    parser = argparse.ArgumentParser(description='ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--d', type=int, default=1,
                        help='distance of neighbourhood (default: 1)')
    parser.add_argument('--t', type=int, default=2,
                        help='size of t-subsets (default: 2)')
    parser.add_argument('--scalar', type=bool, default=True,
                        help='learn scalars')
    parser.add_argument('--no-connected', dest='connected', action='store_false',
                        help='also consider disconnected t-subsets')

    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--readout', type=str, default="mean", choices=["sum", "mean"],
                        help='readout (default: mean)')
    parser.add_argument('--combination', type=str, default="multi", choices=["sum", "multi"],
                        help='pair combination operation (default: multi)')
    parser.add_argument('--mlp', type=bool, default=False,
                        help="mlp (default: False)")
    parser.add_argument('--jk', type=bool, default=False,
                        help="jk (default: False)")

    parser.add_argument('--virtual', type=bool, default=False,
                        help='virtual node (default: False)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv", choices=["ogbg-molhiv", "ogbg-molpcba"],
                            help='dataset name (default: ogbg-molhiv)')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    dataset = PygGraphPropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)
    
    # offline preprocessing and caching
    pathlib.Path('preprocessed').mkdir(parents=True, exist_ok=True) 
    save_file = f'preprocessed/{args.dataset}_{args.d}_{args.t}_{args.connected}.data'
    try:
        print('Loading pair infomation...')
        time_t = time.time()
        train_loader, valid_loader, test_loader = torch.load(save_file)
        print('Pair infomation loaded! Time:', time.time() - time_t)
    except:
        print('Computing pair infomation...')
        time_t = time.time()
        train_loader = []
        valid_loader = []
        test_loader = []
        for batch in tqdm(DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)):
            train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=True)
        for batch in tqdm(DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)):
            valid_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        valid_loader = DataLoader(valid_loader, batch_size=1, shuffle=False)
        for batch in tqdm(DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)):
            test_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        test_loader = DataLoader(test_loader, batch_size=1, shuffle=False)
        print('Pair infomation computed! Time:', time.time() - time_t)
        print('Saving pair infomation...')
        time_t = time.time()
        torch.save((train_loader, valid_loader, test_loader), save_file)
        print('Pair infomation saved! Time:', time.time() - time_t)

    params = {
        'nfeat': dataset.data.x.shape[1] + args.emb_dim,
        'nhid':args.emb_dim, 
        'nclass':dataset.num_tasks,
        'nlayers':args.num_layer,
        'dropout':args.drop_ratio,
        'readout':args.readout,
        'd':args.d,
        't':args.t, 
        'scalar':args.scalar,  
        'mlp':args.mlp, 
        'jk':args.jk, 
        'virtual':args.virtual,
        'combination':args.combination,
        'keys':subgraph.get_keys_from_loaders([train_loader, valid_loader, test_loader]),
    }

    model = models.GNN_ogb(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        time_t = time.time()
        train_perf = train(model, device, train_loader, optimizer, evaluator)

        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)
        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        best_val_epoch = np.argmax(np.array(valid_curve))
        print(f'epoch: {epoch} time: {time.time()-time_t:.4f} train: {train_perf[dataset.eval_metric]:.4f} vali: {valid_perf[dataset.eval_metric]:.4f} test: {test_perf[dataset.eval_metric]:.4f} | best_val: {valid_curve[best_val_epoch]:.4f} best_test: {test_curve[best_val_epoch]:.4f}')

    best_val_epoch = np.argmax(np.array(valid_curve))

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

if __name__ == "__main__":
    main()