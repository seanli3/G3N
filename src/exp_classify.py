import torch
import argparse
import models as models
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import subgraph
import os
import util
from train.exp_classify import train, test

def main():
    parser = argparse.ArgumentParser(description='EXP')
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

    dataset = util.PlanarSATPairsDataset(root="dataset/EXP/")
    
    print('emb_dim:', args.emb_dim)
    train_loader = []
    for batch in tqdm(DataLoader(dataset[400:1200], batch_size=50, shuffle=True)):
        train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    train_loader = DataLoader(train_loader, batch_size=1, shuffle=True)
    val_loader = []
    for batch in tqdm(DataLoader(dataset[0:200], batch_size=50, shuffle=False)):
        val_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    val_loader = DataLoader(val_loader, batch_size=1, shuffle=True)
    test_loader = []
    for batch in tqdm(DataLoader(dataset[200:400], batch_size=50, shuffle=False)):
        test_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    test_loader = DataLoader(test_loader, batch_size=1, shuffle=True)

    params = {
        'nfeat':dataset.num_features,
        'nhid':args.emb_dim, 
        'nclass':1,  # only 2 classes
        'nlayers':args.num_layer,
        'd':args.d,
        't':args.t, 
        'scalar':args.scalar,  
        'combination':args.combination,
        'keys':subgraph.get_keys_from_loaders([train_loader, val_loader, test_loader]),
    }

    model = models.GNN_synthetic(params).to(device)
    n_params = util.get_n_params(model)
    print('number of parameters:', n_params)
    assert(n_params <= 36000)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bval=1000
    btest=0
    for epoch in range(1, 1001):
        tracc,trloss=train(epoch, model, optimizer, train_loader, device)
        test_acc,test_loss,val_acc,val_loss = test(model, val_loader, test_loader, device)
        if bval>val_loss:
            bval=val_loss
            btest=test_acc    
        print('Epoch: {:02d}, trloss: {:.4f}, tracc: {:.4f}, Valloss: {:.4f}, Val acc: {:.4f},Testloss: {:.4f}, Test acc: {:.4f},best test acc: {:.4f}'.format(epoch,trloss,tracc,val_loss,val_acc,test_loss,test_acc,btest))

if __name__ == "__main__":
    main()
