import os.path as osp
import numpy as np
from torch_geometric.datasets import TUDataset
from util import get_dataset
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from train.tu import train, test
import subgraph
import argparse
import os
import models as models
import time
from tqdm import tqdm
import util 

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IMDB-BINARY', choices=["IMDB-BINARY","IMDB-MULTI","NCI1","PROTEINS","PTC_MR","MUTAG"])
    parser.add_argument('--d', type=int, default=2,
                        help='distance of neighbourhood (default: 1)')
    parser.add_argument('--t', type=int, default=2,
                        help='size of t-subsets (default: 2)')
    parser.add_argument('--scalar', type=bool, default=True,
                        help='learn scalars')
    parser.add_argument('--no-connected', dest='connected', action='store_false',
                        help='also consider disconnected t-subsets')

    parser.add_argument('--mlp', type=bool, default=False,
                        help="mlp (default: False)")
    parser.add_argument('--jk', type=bool, default=True,
                        help="jk")
    parser.add_argument('--drop_ratio', type=float, default=0.1,
                        help='dropout ratio')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs')
    parser.add_argument('--combination', type=str, default="multi", choices=["sum", "multi"],
                        help='pair combination operation')
    parser.add_argument('--readout', type=str, default="sum", choices=["sum", "mean"],
                        help='readout')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial lr')
    parser.add_argument('--step', type=int, default=50,
                        help='lr decrease steps')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--print_params', type=bool, default=False,
                        help='print number of parameters')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    #set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        print('cuda available with GPU:',torch.cuda.get_device_name(0))

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Download dataset.
    ds_name = args.dataset
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dir_path, 'data', ds_name)
    dataset = TUDataset(path, name=ds_name)


    # One-hot degree if node labels are not available.
    # The following if clause is taken from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py.
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

        for data in dataset:
            nfeat = dataset.transform(data).x.shape[1]
            break
    else:
        nfeat = dataset.data.x.shape[1]
    nclass = max(dataset.data.y) + 1

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)

    val_curves=[]
    test_curves=[]
    for train_idxs, val_idxs in skf.split(dataset, dataset.data.y):
        # Split data... following GIN and CIN, only look at validation accuracy
        train_dataset = dataset[train_idxs]
        val_dataset = dataset[val_idxs]

        # Prepare batching.
        # print('Computing pair infomation...', end=" ")
        time_t = time.time()
        train_loader = []
        valid_loader = []

        for batch in DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True):
            if dataset.data.x is None:
                batch = dataset.transform(batch)
            train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=True)
        for batch in DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True):
            if dataset.data.x is None:
                batch = dataset.transform(batch)
            valid_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
        valid_loader = DataLoader(valid_loader, batch_size=1, shuffle=False)
        # print('Pair infomation computed! Time:', time.time() - time_t)

        params = {
                'nfeat':nfeat,
                'nhid':args.emb_dim, 
                'nclass':nclass,
                'nlayers':args.num_layer,
                'dropout':args.drop_ratio,
                'readout':args.readout,
                'd':args.d,
                't':args.t, 
                'scalar':args.scalar,  
                'mlp':args.mlp, 
                'jk':args.jk, 
                'combination':args.combination,
                'keys':subgraph.get_keys_from_loaders([train_loader, valid_loader]),
                'tu':True,
            }

        # Setup model.
        model = models.GNN_bench(params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)

        if args.print_params:
            print('number of parameters:', util.get_n_params(model))

        val_accs = []
        with tqdm(range(args.epochs)) as tq:
            for epoch in tq:
                time_t = time.time()
                train_loss, train_acc = train(train_loader, model, optimizer, device)
                val_acc = test(valid_loader, model, device)
                scheduler.step()

                val_accs.append(val_acc.cpu().detach().item())

                tq.set_description(f'train_loss={train_loss:.2f}, train_acc={train_acc:.2f}, val_acc={val_acc:.2f}, time={time.time()-time_t:.2f}')  
        val_curves.append(val_accs)

        # aggregate val results (from https://github.com/twitter-research/cwn/blob/main/exp/run_tu_exp.py)
        val_curves_combined = np.asarray(val_curves)
        avg_val_curve = val_curves_combined.mean(axis=0)
        best_index = np.argmax(avg_val_curve)
        mean_perf = avg_val_curve[best_index] * 100
        std_perf = val_curves_combined.std(axis=0)[best_index] * 100

    # print('all folds completed! result: ')
    print(mean_perf, std_perf)


if __name__ == "__main__":
    main()