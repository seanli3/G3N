import torch
import util
import models as models
import subgraph
import numpy as np
from scipy.spatial.distance import pdist

def run(args, device, train_loader, tol):
  for data in train_loader:
    if data.x==None:
      nfeat=1
    else:
      nfeat=data.x.shape[1]

  M=0
  for iter in range(1, 101):
      torch.manual_seed(iter)
      
      params = {
          'nfeat':nfeat,
          'nhid':args.emb_dim, 
          'nclass':10,
          'nlayers':args.num_layer,
          'd':args.d,
          't':args.t, 
          'scalar':args.scalar,
          'combination':args.combination,
          'keys':subgraph.get_keys_from_loaders([train_loader]),
      }

      model = models.GNN_synthetic(params).to(device)
      if iter==0:
          n_params = util.get_n_params(model)
          print('emb_dim:', args.emb_dim)
          print('number of parameters:', util.get_n_params(model))
          assert(n_params <= 33000)

      embeddings=[]
      model.eval()  # no learning
      for batch in train_loader:
          pairs, degrees, scatter = batch.pair_info[0]
          for key in pairs:
              degrees[key] = degrees[key].to(device)
              scatter[key] = scatter[key].to(device)
          
          if batch.x == None:
            x = torch.zeros((batch.num_nodes, 1)).to(device)
          else:
            x = batch.x.float().to(device)
          edge_index = batch.edge_index.to(device)
          batch_idx = batch.batch.to(device)
          pred = model(x, edge_index,(pairs, degrees, scatter), batch_idx)
          embeddings.append(pred)

      E = torch.cat(embeddings).cpu().detach().numpy()
      M = M + 1 * (np.abs(E[0::2] - E[1::2]).sum(1) > 0.001)
      sm = (M == 0).sum()
      print('iter', iter, 'similar:',sm)