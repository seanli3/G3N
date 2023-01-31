import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import models as models
import os
import pathlib

from tqdm import tqdm
import argparse
import time
import numpy as np
import util
import subgraph
from torch_geometric.datasets import ZINC
from train.zinc import train, eval


# In[2]:


parser = argparse.ArgumentParser(description="ZINC")
parser.add_argument('--d', type=int, default=4,
                    help='distance of neighbourhood (default: 1)')
parser.add_argument('--t', type=int, default=2,
                    help='size of t-subsets (default: 2)')
parser.add_argument('--scalar', type=bool, default=True,
                    help='learn scalars')
parser.add_argument('--no-connected', dest='connected', action='store_false',
                    help='also consider disconnected t-subsets')

parser.add_argument('--drop_ratio', type=float, default=0.0,
                    help='dropout ratio')
parser.add_argument('--num_layer', type=int, default=4,
                    help='number of GNN message passing layers')
parser.add_argument('--emb_dim', type=int, default=80,
                    help='dimensionality of hidden units in GNNs')
parser.add_argument('--readout', type=str, default="sum", choices=["sum", "mean"],
                    help='readout')
parser.add_argument('--combination', type=str, default="multi", choices=["sum", "multi"],
                    help='pair combination operation')
parser.add_argument('--mlp', type=bool, default=False,
                    help="mlp (default: False)")
parser.add_argument('--jk', type=bool, default=False,
                    help="jk")
parser.add_argument('--multiplier', type=int, default=1,
                    help="hidden layer readout multiplier")

parser.add_argument('--edge_features', dest='edge_features', action='store_false',
                    help='exist edge attributes')

parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--step', type=int, default=20,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
args = parser.parse_args('')


# In[4]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  
if torch.cuda.is_available():
    print('cuda available with GPU:',torch.cuda.get_device_name(0))
device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")


# In[5]:


trainset = ZINC(root="dataset/ZINC", split='train', subset=True)  # subset loads 12k instead of 250k
valset = ZINC(root="dataset/ZINC", split='val', subset=True)
testset = ZINC(root="dataset/ZINC", split='test', subset=True)


# In[7]:


# offline preprocessing
pathlib.Path('preprocessed').mkdir(parents=True, exist_ok=True) 
save_file = f'preprocessed/zinc_{args.d}_{args.t}_{args.connected}.data'


# In[7]:


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
    for batch in tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True)):
        train_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    train_loader = DataLoader(train_loader, batch_size=1, shuffle=True)
    for batch in tqdm(DataLoader(valset, batch_size=args.batch_size, shuffle=False)):
        valid_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    valid_loader = DataLoader(valid_loader, batch_size=1, shuffle=False)
    for batch in tqdm(DataLoader(testset, batch_size=args.batch_size, shuffle=False)):
        test_loader.append(subgraph.transform(batch, args.d, args.t, args.connected))
    test_loader = DataLoader(test_loader, batch_size=1, shuffle=False)
    print('Pair infomation computed! Time:', time.time() - time_t)
    print('Saving pair infomation...')
    time_t = time.time()
    torch.save((train_loader, valid_loader, test_loader), save_file)
    print('Pair infomation saved! Time:', time.time() - time_t)

params = {
    'nfeat':28, #num of atom type
    'edge_attr': 4, #num of bond type
    'exist_edge_attr': args.edge_features,
    'nhid':args.emb_dim, 
    'nclass':1,   # 1 out dim since regression problem 
    'nlayers':args.num_layer,
    'dropout':args.drop_ratio,
    'readout':args.readout,
    'd':args.d,
    't':args.t, 
    'scalar':args.scalar,  
    'mlp':args.mlp, 
    'jk':args.jk, 
    'combination':args.combination,
    'multiplier':args.multiplier,
    'keys':subgraph.get_keys_from_loaders([train_loader, valid_loader, test_loader]),
}

model = models.GNN_bench(params).to(device)

n_params = util.get_n_params(model)
print('emb_dim:', args.emb_dim)
print('number of parameters:', util.get_n_params(model))
if n_params > 110000:
    print(f'Warning: 100000 parameter budget exceeded.')

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=0.5,
                                                    patience=args.step,
                                                    verbose=True)

t0 = time.time()
per_epoch_time = []
epoch_train_losses, epoch_val_losses = [], []

# At any point you can hit Ctrl + C to break out of training early.
try:
    with tqdm(range(args.epochs)) as tq:
        for epoch in tq:

            tq.set_description('Epoch %d' % epoch)

            startime_t = time.time()

            epoch_train_loss, optimizer = train(model, optimizer, train_loader, epoch, device)

            epoch_val_loss = eval(model, valid_loader, epoch, device)
            epoch_test_loss = eval(model, test_loader, epoch, device)                

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)

            tq.set_postfix(lr=optimizer.param_groups[0]['lr'],
                          train_loss=epoch_train_loss, val_loss=epoch_val_loss, test_loss=epoch_test_loss)

            per_epoch_time.append(time.time() - startime_t)

            scheduler.step(epoch_val_loss)

            if optimizer.param_groups[0]['lr'] < 1e-5:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early because of KeyboardInterrupt')

test_mae = eval(model, test_loader, epoch, device)
train_mae = eval(model, train_loader, epoch, device)
print("Test MAE: {:.4f}".format(test_mae))
print("Train MAE: {:.4f}".format(train_mae))
print("Convergence Time (Epochs): {:.4f}".format(epoch))
print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))