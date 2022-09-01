import torch
import argparse
import time

from torch_geometric.loader import DataLoader
from tqdm import tqdm
import subgraph
import models as models
import os
import util

from train.counting import train, test
import scipy.io as sio


def main():
    parser = argparse.ArgumentParser(description='counting')
    parser.add_argument('--ntask', type=int, default=0,
                        help='0: triangle, 1: tailed_triangle; 2: star; 3: 4-cycle')
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

    # get dataset
    dataset = util.GraphCountDataset(root="dataset/subgraphcount/")
    dataset.data.y=(dataset.data.y/dataset.data.y.std(0))  # normalize outputs
    dataset.data.x[:,1]=dataset.data.x[:,1]/dataset.data.x[:,1].max()  # normalize degree of the node
    a=sio.loadmat('dataset/subgraphcount/raw/randomgraph.mat')
    trid=a['train_idx'][0]
    vlid=a['val_idx'][0]
    tsid=a['test_idx'][0]
    
    # offline preprocessing
    print('Computing pair infomation...')
    time_t = time.time()
    train_loader = []
    for batch in tqdm(DataLoader(dataset[[i for i in trid]], batch_size=10, shuffle=True)):
        train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    train_loader = DataLoader(train_loader, batch_size=1, shuffle=True)

    valid_loader = []
    for batch in tqdm(DataLoader(dataset[[i for i in vlid]], batch_size=100, shuffle=False)):
        valid_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    valid_loader = DataLoader(valid_loader, batch_size=1, shuffle=False)

    test_loader = []
    for batch in tqdm(DataLoader(dataset[[i for i in tsid]], batch_size=100, shuffle=False)):
        test_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    test_loader = DataLoader(test_loader, batch_size=1, shuffle=False)
    print('Pair infomation computed! Time:', time.time() - time_t)
        
    # init model
    params = {
        'nfeat':dataset.num_features,
        'nhid':args.emb_dim, 
        'nclass':1,
        'nlayers':args.num_layer,
        'd':args.d,
        't':args.t, 
        'scalar':args.scalar,  
        'combination':args.combination,
        'keys':subgraph.get_keys_from_loaders([train_loader,valid_loader,test_loader]),
        'counting':True,
    }

    model = models.GNN_synthetic(params).to(device)

    n_params = util.get_n_params(model)
    print('emb_dim:', args.emb_dim)
    print('number of parameters:', util.get_n_params(model))
    if n_params > 30000:
        print('Warning: parameter budget exceeded')

    # select task, 0: triangle, 1: tailed_triangle 2: star  3: 4-cycle  4:custom
    ntask=args.ntask

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    bval=1000
    btest=0
    btestr2=0
    for epoch in range(1, 201):
        t = time.time()
        trloss=train(epoch, model, train_loader, optimizer, device, ntask, trid)
        test_loss,val_loss,testr2 = test(model, valid_loader, test_loader,device, ntask, vlid, tsid)
        if bval>val_loss:
            bval=val_loss
            btest=test_loss
            btestr2=testr2
        time_taken = time.time() - t
        
        print('Epoch: {:02d}, trloss: {:.6f},  Valloss: {:.6f}, Testloss: {:.6f}, best test loss: {:.6f}, bestr2:{:.6f}, time:{:.6f}'.format(epoch,trloss,val_loss,test_loss,btest,btestr2,time_taken))
        
        if bval<1e-4:
            break


if __name__ == "__main__":
    main()
