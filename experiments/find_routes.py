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
with open("results/normal/25000_3/NYC_supergraph_solved.gpickle", 'rb') as f:
            tNet.G_supergraph = pickle.load(f)

with open("results/normal/25000_3/NYC_roadgraph_solved.gpickle", 'rb') as f:
            original_G = pickle.load(f)

with open("data/gml/NYC_small_roadgraph.gpickle", 'rb') as f:
            tNet.G = pickle.load(f)

with open("data/gml/NYC_small_demands.gpickle", 'rb') as f:
            tNet.g = pickle.load(f)
x_sol = np.load("results/normal/25000_3/x_sol.npy")

tNet.g = tnet.perturbDemandConstant(tNet.g, 3)
car_node_idx = {
            node: i for i, node in enumerate(tNet.G.nodes())
        }
ori_car_node_idx = {
    node:i for i, node in enumerate(original_G.nodes())
}
x_sol = np.load("results/normal/25000_3/x_sol.npy")


origins = sorted({o for o, _ in tNet.g.keys()})
edges = list(tNet.G_supergraph.edges())
pt_edges = [i for i, (u,v,d) in enumerate(tNet.G_supergraph.edges(data=True)) if d["type"] == "s"]
print(edges[pt_edges[0]])
tol = 10
mask = (x_sol[pt_edges, :] > tol).any(axis=0)  # boolean per column
cols = np.where(mask)[0]  # indices of origins with any PT-edge flow
print(cols)
origin_idx = 40
o = origins[origin_idx]
xo = {e: float(x_sol[row_idx, origin_idx]) for row_idx, e in enumerate(edges)}
print(o)
print(min(xo.values()), sum(xo.values()))
print(x_sol.shape)

# decompose into OD-pairs
xf_by_d = solve_flow_decomposition_D_fast(tNet.G_supergraph, origin=o, xo=xo, g=tNet.g)

# find the actual routes
routes_by_od = {}
for d, x_od in xf_by_d.items():
    demand_od = tNet.g[(o, d)]               # or sum of x_od leaving o
    gw = [(o, d), demand_od]
    routes = routeFinder_OD(tNet.G_supergraph, gw, x_od, eps=1e-3, max_routes=100)
    routes_by_od[(o, d)] = routes

def save_routes(routes_by_od, path='routes_by_od.pkl.gz'):
    with gzip.open(path, 'wb') as f:
        pickle.dump(routes_by_od, f, protocol=pickle.HIGHEST_PROTOCOL)


save_routes(routes_by_od, path=f'routes_by_od_{origin_idx}.pkl.gz')

###### Find RP routes for given OD
# rp_o = s2int('18rp')#some origin
# rp_d = s2int('347rp')#some destination
# D_rp = np.load("full_demand.npy") #OD matrix for rp demand only
# y_sol = np.load("results/normal/25000_3/y_sol.npy")

# np.fill_diagonal(D_rp, 0)
# G_nodes = list(original_G.nodes())
# G_node_idx = {n:i for i,n in enumerate(G_nodes)}

# #define od indices
# G_o_idx = G_node_idx[rp_o]
# G_d_idx = G_node_idx[rp_d]

# #find correct y column (as dict)
# yo = {e: (float(y_sol[row_idx, G_o_idx]) if float(y_sol[row_idx, G_o_idx])>=0 else 0 ) for row_idx, e in enumerate(original_G.edges())}

# # find car demand from D_rp as dict
# car_g = {(rp_o, G_nodes[d]): (D_rp[G_o_idx, d] if D_rp[G_o_idx, d]>=0 else 0) for d in range(len(G_nodes))} #only demand for this O

# # decompose 
# yf_by_d = solve_flow_decomposition_D_fast_tol(
#     original_G, origin=rp_o, xo=yo, g=car_g,
#     link_tol_abs=1e-6, link_tol_rel=1e-6, node_tol_abs=1e-6
# )
# # car_routes_by_od = {}
# car_routes = routeFinder_OD(original_G, [(rp_o, rp_d),car_g[(rp_o, rp_d)]], yf_by_d[rp_d], eps=1e-3, max_routes=100)
# save_routes(car_routes, path='results/car_routes_1.pkl.gz')

# ###### Find RP routes for given OD
# rp_o = s2int('82rp')#some origin
# rp_d = s2int('347rp')#some destination

# #define od indices
# G_o_idx = G_node_idx[rp_o]
# G_d_idx = G_node_idx[rp_d]

# #find correct y column (as dict)
# yo = {e: (float(y_sol[row_idx, G_o_idx]) if float(y_sol[row_idx, G_o_idx])>=0 else 0 ) for row_idx, e in enumerate(original_G.edges())}

# # find car demand from D_rp as dict
# car_g = {(rp_o, G_nodes[d]): (D_rp[G_o_idx, d] if D_rp[G_o_idx, d]>=0 else 0) for d in range(len(G_nodes))} #only demand for this O

# # decompose 
# yf_by_d = solve_flow_decomposition_D_fast_tol(
#     original_G, origin=rp_o, xo=yo, g=car_g,
#     link_tol_abs=1e-6, link_tol_rel=1e-6, node_tol_abs=1e-6
# )
# # car_routes_by_od = {}
# car_routes = routeFinder_OD(original_G, [(rp_o, rp_d),car_g[(rp_o, rp_d)]], yf_by_d[rp_d], eps=1e-3, max_routes=100)
# save_routes(car_routes, path='results/car_routes_2.pkl.gz')

###### debugging
def diag_origin_column(G, o, xo, g, cost_key='t_1', tol=1e-9):
    # 0) tiny negatives?
    neg = {e:v for e,v in xo.items() if v < -tol}
    tiny_neg = {e:v for e,v in xo.items() if -tol <= v < 0}
    print(f"negatives: {len(neg)}, tiny negatives: {len(tiny_neg)} (tol={tol})")

    # 1) divergence of xo
    div = {n:0.0 for n in G.nodes()}
    for (i,j), v in xo.items():
        div[i] -= v     # outflow
        div[j] += v     # inflow

    # 2) target divergence from g
    D = [d for (oo,d),val in g.items() if oo==o and val>tol]
    b = {n:0.0 for n in G.nodes()}
    total_demand = 0.0
    for d in D:
        b[o] -= g[(o,d)]   # origin: net outflow
        b[d] += g[(o,d)]   # destination: net inflow
        total_demand += g[(o,d)]
    print(f"sum_d g[(o,d)] = {total_demand}")

    # 3) compare div(xo) vs b
    bad = {n:(div[n]-b[n]) for n in G.nodes() if abs(div[n]-b[n])>1e-6}
    print(f"nodes with divergence mismatch > 1e-6: {len(bad)}")
    for n,delta in list(bad.items())[:10]:
        print(f"  node {n}: div(xo)={div[n]:.6g}, b={b[n]:.6g}, delta={delta:.6g}")

    # 4) reachability
    unreachable = [d for d in D if not nx.has_path(G, o, d)]
    print(f"unreachable destinations from {o}: {unreachable}")

    # 5) missing cost key?
    missing_cost = [(i,j) for (i,j) in G.edges() if cost_key not in G[i][j]]
    print(f"edges missing '{cost_key}': {len(missing_cost)}")

