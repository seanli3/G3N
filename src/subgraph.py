import torch
import networkx as nx
import itertools
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import numpy as np

def one_hot_length(t, get_deg=False):
    max_deg = 5
    iso_type = 2 if t==3 else 0
    if get_deg:
        return max_deg + iso_type, max_deg
    else:
        return max_deg + iso_type

def get_avg_diameter(graph_collections):
    diameters = []
    for graphs in graph_collections:
        for graph in graphs:
            G = to_networkx(graph).to_undirected()
            for g in nx.connected_components(G):  # usually batched graphs
                diameters.append(nx.diameter(G.subgraph(g)))
    avg_diam = np.mean(diameters)
    print('average graph diameter:', avg_diam)
    return avg_diam

def get_keys_from_loaders(loaders):
    keys = set()
    for loader in loaders:
        for batch in loader:
            pairs, _, _ = batch.pair_info[0]
            for key in pairs:
                keys.add(key)
    print('topology types:', keys)
    print('number of topology types:', len(keys))
    return keys

def transform(data, d, t, connected):
    data.pair_info = compute_nhbr_pair_data(to_networkx(data), d, t, connected)
    return data

def knbrs(G_in, start, k):  # get k-hop neighbourhood
    # nbrs = nx.single_source_shortest_path_length(G_in,source=start,cutoff=k).keys()  # slightly slower than below?
    nbrs = set([start])
    for _ in range(1, k+1):
        next_nbrs = set((nbr for n in nbrs for nbr in G_in[n]))
        nbrs = nbrs.union(next_nbrs)
    return nbrs

def induced_degree(u, G, subgraph_nodes):
    ret = len(subgraph_nodes.intersection(set(G[u])))
    return ret

def original_degree(u, G):
    ret = len(set(G[u]))
    return ret

def compute_nhbr_pair_data(G, d, t, require_connected):
    G = G.to_undirected()  # works assuming undirected
    pairs = {}
    degrees = {}
    scatter = {}
    distances = dict(nx.all_pairs_shortest_path_length(G))

    one_hot_len, max_deg = one_hot_length(t, get_deg=True)

    # create pair_neighbourhood
    for node in G.nodes:  # sorted 0,1,.. etc
        subgraph_nodes = knbrs(G, node, d)
        if t==1:
            subgraph_nodes.remove(node)
        for comb in itertools.combinations(subgraph_nodes, t):

            if t == 1:
                is_connected = True
                edges = 0
            elif t==2:
                u, v = comb
                is_connected = (u, v) in G.edges
                edges = int(is_connected)
            elif t==3:
                u, v, w = comb
                edges = int((u, v) in G.edges) + int((u, w) in G.edges) + int((w, v) in G.edges)
                is_connected = edges >= 2
                iso_type = edges % 2
            else:
                raise NotImplementedError

            if require_connected and not is_connected:
                continue

            tuple_list = sorted([(distances[node][u], u) for u in comb], key=lambda x: x[0])

            key = [x[0] for x in tuple_list]
            # key.append(edges)
            key = tuple(key)

            if key not in pairs:
                pairs[key] = [[] for _ in range(t)]
                degrees[key] = [[] for _ in range(t)]
                scatter[key] = []

            for i in range(t):
                u = tuple_list[i][1]
                deg = induced_degree(u, G=G, subgraph_nodes=subgraph_nodes)
                deg = min(max_deg, deg) - 1
                one_hot_deg = [0 for _ in range(one_hot_len)]
                one_hot_deg[deg] = 1
                if t==3:
                    one_hot_deg[max_deg + iso_type] = 1
                pairs[key][i].append(u)
                degrees[key][i].append(one_hot_deg)
            scatter[key].append(node)
    for key in pairs:
        pairs[key] = torch.tensor(pairs[key]).to('cuda')
        degrees[key] = torch.tensor(degrees[key]).to('cuda')
        scatter[key] = torch.tensor(scatter[key]).to('cuda')
    nhbr_info = (pairs, degrees, scatter)
    return nhbr_info