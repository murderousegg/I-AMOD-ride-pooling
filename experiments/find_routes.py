import experiments.build_NYC_subway_net as nyc
import pickle
import re
import matplotlib.pyplot as plt
import networkx as nx
from pyproj import Transformer
import numpy as np
from src.routeFinder import *
import src.tnet as tnet

def s2int(node: str) -> int:
    num = int(re.sub(r'\D', '', node))
    return num 

tNet, tstamp, fcoeffs = nyc.build_NYC_net("data/net/NYC/", only_road=True)
with open("results/normal/45000_3/NYC_supergraph_solved.gpickle", 'rb') as f:
            tNet.G_supergraph = pickle.load(f)

with open("results/normal/45000_3/NYC_roadgraph_solved.gpickle", 'rb') as f:
            original_G = pickle.load(f)

with open("data/gml/NYC_small_roadgraph.gpickle", 'rb') as f:
            tNet.G = pickle.load(f)

with open("data/gml/NYC_small_demands.gpickle", 'rb') as f:
            tNet.g = pickle.load(f)
x_sol = np.load("results/normal/45000_3/x_sol.npy")

tNet.g = tnet.perturbDemandConstant(tNet.g, 3)
car_node_idx = {
            node: i for i, node in enumerate(tNet.G.nodes())
        }
ori_car_node_idx = {
    node:i for i, node in enumerate(original_G.nodes())
}

origins = sorted({o for o, _ in tNet.g.keys()})
edges = list(tNet.G_supergraph.edges())
pt_edges = [i for i, (u,v,d) in enumerate(tNet.G_supergraph.edges(data=True)) if d["type"] == "s"]
tol = 500
mask = (x_sol[pt_edges, :] > tol).any(axis=0)  # boolean per column
cols = np.where(mask)[0]  # indices of origins with any PT-edge flow
print(cols)
origin_idx = 50
o = origins[origin_idx]
xo = {e: float(x_sol[row_idx, origin_idx]) for row_idx, e in enumerate(edges)}
print(o)
print(min(xo.values()), sum(xo.values()))
print(x_sol.shape)

routes = userRouteFinder(tNet.G_supergraph, tNet.g, {o: xo}, eps=1)
print(routes)

def save_routes(routes_by_od, path='routes_by_od.pkl.gz'):
    with gzip.open(path, 'wb') as f:
        pickle.dump(routes_by_od, f, protocol=pickle.HIGHEST_PROTOCOL)


save_routes(routes, path=f'results/routeRecovery/routes_by_od_{origin_idx}.pkl.gz')
