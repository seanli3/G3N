import torch
import argparse

from torch_geometric.loader import DataLoader
from tqdm import tqdm
import subgraph
import os
import util

from torch_geometric.datasets import GNNBenchmarkDataset
from train.iso import run

def main():
    parser = argparse.ArgumentParser(description='iso tests')
    parser.add_argument('--dataset', type=str, required=True, choices=['exp', 'csl', 'graph8c', 'sr25'])
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

    dataset_name = args.dataset
    if dataset_name == "exp":
        transform = util.SpectralDesign(nmax=0, recfield=1, dv=2, nfreq=5, adddegree=True)
        dataset = util.PlanarSATPairsDataset(root="dataset/EXP/", pre_transform=transform)
        train_loader = []
        for batch in tqdm(DataLoader(dataset, batch_size=100, shuffle=False)):
            train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=False)
    elif dataset_name == "csl":
        dataset = GNNBenchmarkDataset(root='dataset/CSL', name='CSL')
        added = [False for _ in range(15)]
        trainset = []
        for data in dataset:
            y = data.y[0]
            if not added[y]:
                added[y]=True
                trainset.append(data)
        train_loader = []
        for batch in tqdm(DataLoader(trainset, batch_size=100, shuffle=False)):
            train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=False)
    elif dataset_name == "graph8c":
        transform = util.SpectralDesign(nmax=0, recfield=1, dv=2, nfreq=5, adddegree=True)
        dataset = util.Grapg8cDataset(root="dataset/graph8c/", pre_transform=transform)
        train_loader = []
        for batch in tqdm(DataLoader(dataset, batch_size=100, shuffle=False)):
            train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=False)
    elif dataset_name == "sr25":
        transform = util.SpectralDesign(nmax=0, recfield=1, dv=2, nfreq=5, adddegree=True)
        dataset = util.SRDataset(root="dataset/sr25/", pre_transform=transform)
        train_loader = []
        for batch in tqdm(DataLoader(dataset, batch_size=100, shuffle=False)):
            train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=False)

    run(args, device, train_loader, tol=0.001)

if __name__ == "__main__":
    main()