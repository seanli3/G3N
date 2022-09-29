import torch
import pickle
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
import networkx as nx
import os.path as osp
import numpy as np
from torch_geometric.datasets import TUDataset

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# TUdataset
# Return classes as a numpy array.
def read_classes(ds_name):
    # Classes
    with open("data/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_targets(ds_name):
    # Classes
    with open("data/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [float(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_multi_targets(ds_name):
    # Classes
    with open("data/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [[float(j) for j in i.split(",")] for i in list(f)]
    f.closed

    return np.array(classes)


# Download dataset, regression problem=False, multi-target regression=False.
def get_dataset(dataset, regression=False, multi_target_regression=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    TUDataset(path, name=dataset)

    if multi_target_regression:
        return read_multi_targets(dataset)
    if not regression:
        return read_classes(dataset)
    else:
        return read_targets(dataset)

# adapted from https://github.com/balcilar/gnn-matlang
class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            tri=np.trace(A3)/6
            tailed=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(np.trace(A3.dot(a))+np.trace(A2)-2*A2.sum())
            cus= a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg=a.sum(0)
            star=0
            for j in range(a.shape[0]):
                star+=comb(int(deg[j]),3)

            expy=torch.tensor([[tri,tailed,star,cyc4,cus]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["EXP.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/EXP.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Grapg8cDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Grapg8cDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graph8c.g6"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SpectralDesign(object):

    def __init__(self, nmax=0, recfield=1, dv=5, nfreq=5, adddegree=False, laplacien=True, addadj=False, vmax=None):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area
        self.recfield = recfield
        # b parameter
        self.dv = dv
        # number of sampled point of spectrum
        self.nfreq = nfreq
        # if degree is added to node feature
        self.adddegree = adddegree
        # use laplacian or adjacency for spectrum
        self.laplacien = laplacien
        # add adjacecny as edge feature
        self.addadj = addadj
        # use given max eigenvalue
        self.vmax = vmax

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax = nmax

    def __call__(self, data):

        n = data.x.shape[0]
        nf = data.x.shape[1]

        data.x = data.x.type(torch.float32)

        nsup = self.nfreq + 1
        if self.addadj:
            nsup += 1

        A = np.zeros((n, n), dtype=np.float32)
        SP = np.zeros((nsup, n, n), dtype=np.float32)
        A[data.edge_index[0], data.edge_index[1]] = 1

        if self.adddegree:
            data.x = torch.cat([data.x, torch.tensor(A.sum(0)).unsqueeze(-1)], 1)

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        if self.recfield == 0:
            M = A
        else:
            M = (A + np.eye(n))
            for i in range(1, self.recfield):
                M = M.dot(M)

        M = (M > 0)

        d = A.sum(axis=0)
        # normalized Laplacian matrix.
        dis = 1 / np.sqrt(d)
        dis[np.isinf(dis)] = 0
        dis[np.isnan(dis)] = 0
        D = np.diag(dis)
        nL = np.eye(D.shape[0]) - (A.dot(D)).T.dot(D)
        V, U = np.linalg.eigh(nL)
        V[V < 0] = 0
        # keep maximum eigenvalue for Chebnet if it is needed
        data.lmax = V.max().astype(np.float32)

        if not self.laplacien:
            V, U = np.linalg.eigh(A)

        # design convolution supports
        vmax = self.vmax
        if vmax is None:
            vmax = V.max()

        freqcenter = np.linspace(V.min(), vmax, self.nfreq)

        # design convolution supports (aka edge features)
        for i in range(0, len(freqcenter)):
            SP[i, :, :] = M * (U.dot(np.diag(np.exp(-(self.dv * (V - freqcenter[i]) ** 2))).dot(U.T)))
            # add identity
        SP[len(freqcenter), :, :] = np.eye(n)
        # add adjacency if it is desired
        if self.addadj:
            SP[len(freqcenter) + 1, :, :] = A

        # set convolution support weigths as an edge feature
        E = np.where(M > 0)
        data.edge_index2 = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
        data.edge_attr2 = torch.Tensor(SP[:, E[0], E[1]].T).type(torch.float32)

        # set tensor for Maron's PPGN
        if self.nmax > 0:
            H = torch.zeros(1, nf + 2, self.nmax, self.nmax)
            H[0, 0, data.edge_index[0], data.edge_index[1]] = 1
            H[0, 1, 0:n, 0:n] = torch.diag(torch.ones(data.x.shape[0]))
            for j in range(0, nf):
                H[0, j + 2, 0:n, 0:n] = torch.diag(data.x[:, j])
            data.X2 = H
            M = torch.zeros(1, 2, self.nmax, self.nmax)
            for i in range(0, n):
                M[0, 0, i, i] = 1
            M[0, 1, 0:n, 0:n] = 1 - M[0, 0, 0:n, 0:n]
            data.M = M

        return data