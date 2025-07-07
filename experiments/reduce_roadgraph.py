import numpy as np
from scipy import io
import networkx as nx
import experiments.build_NYC_subway_net as nyc
import pickle
from scipy.spatial import cKDTree
from typing import Sequence, Tuple, Dict
import gurobipy as gp
from gurobipy import GRB
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

EARTH_R = 6_371_000.0
# ---------- load base network ---------------------------------------------
tNet, _, _ = nyc.build_NYC_net("data/net/NYC/", only_road=True)
tNet.read_node_coordinates("data/pos/NYC.txt")
tNet_coords = np.array([tNet.G.nodes[i]["pos"] for i in tNet.G.nodes()])
tNet_coords = tNet_coords[:, ::-1]
node_list = list(tNet.G.nodes())                       # index → label
node_idx_map = {n: i for i, n in enumerate(node_list)} # label → index
edge_list = list(tNet.G.edges())

def latlon_to_xy(lat: np.ndarray,
                 lon: np.ndarray,
                 lat0: float | None = None) -> np.ndarray:
    """
    Convert geographic coordinates to a local Euclidean (x, y) system.

    Parameters
    ----------
    lat, lon : 1-D arrays of degrees
    lat0     : reference latitude in degrees.  If None, use mean(lat).

    Returns
    -------
    xy : ndarray shape (N, 2)  - metres
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0 if lat0 is not None else lat.mean())

    x = EARTH_R * (lon_rad - lon_rad.mean()) * np.cos(lat0_rad)
    y = EARTH_R * (lat_rad - lat_rad.mean())
    return np.column_stack((x, y))              # (N, 2)

def prune_greedy(lat: Sequence[float],
                           lon: Sequence[float],
                           k_keep: int,
                           ) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Keep k_keep nodes; minimise total extra distance (greedy heuristic).

    Returns
    -------
    keep_idx      : ndarray of original indices kept (length = k_keep)
    nearest_kept  : dict  orig_index → kept_index   (index into keep_idx)
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    N = len(lat)
    if k_keep >= N:
        raise ValueError("k_keep must be smaller than number of nodes")

    XY = latlon_to_xy(lat, lon)
    keep = np.ones(N, dtype=bool)

    while keep.sum() > k_keep:
        idx_keep = np.flatnonzero(keep)
        tree     = cKDTree(XY[idx_keep])
        dists, _ = tree.query(XY[idx_keep], k=3)        # self, nn1, nn2
        penalty  = dists[:, 2]                          # fallback distance
        drop_idx = idx_keep[np.argmin(penalty)]
        keep[drop_idx] = False

    keep_idx = np.flatnonzero(keep)

    tree_kept = cKDTree(XY[keep_idx])
    d_nearest, nn = tree_kept.query(XY, k=1)
    nearest_kept = {orig: int(nn[i]) for i, orig in enumerate(range(N))}
    # some stats
    max_d   = d_nearest.max()
    mean_d  = d_nearest.mean()
    p95_d   = np.percentile(d_nearest, 95)
    print(f"max  distance to nearest kept node : {max_d/1000:.3f} km")
    print(f"mean distance                      : {mean_d:.1f} m")
    print(f"95-percentile distance              : {p95_d:.1f} m")

    return keep_idx, nearest_kept

#### k-median implementation for optimal pruning
def solve_kmedian(lat: Sequence[float], lon: Sequence[float], k_keep: int) -> Tuple[np.ndarray, Dict[int, int]]:
    XY = latlon_to_xy(lat, lon)
    N = len(XY)
    D = cdist(XY, XY)  # distance matrix (N x N)

    m = gp.Model("k-median")
    max_dist = 800  # in meters
    x = {}
    for i in range(N):
        for j in range(N):
            if D[i, j] <= max_dist:
                x[i, j] = m.addVar(vtype=GRB.BINARY)
    y = m.addVars(N, vtype=GRB.BINARY, name="y")

    # Objective: total distance
    obj = gp.quicksum(D[i, j] * x[i, j] for (i, j) in x)
    m.setObjective(obj, GRB.MINIMIZE)

    # Each node must be assigned to one center (if at least one assignment is allowed)
    for i in range(N):
        assign_vars = [x[i, j] for j in range(N) if (i, j) in x]
        if assign_vars:
            m.addConstr(gp.quicksum(assign_vars) == 1)
        else:
            raise ValueError(f"No feasible assignments for node {i} under max_dist={max_dist}")
    for (i, j) in x:
        m.addConstr(x[i, j] <= y[j])
    # Exactly k centers
    m.addConstr(gp.quicksum(y[j] for j in range(N)) == k_keep)
    m.Params.OutputFlag = 0
    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Optimization failed with status {m.status}")

    # Extract results
    centers = [j for j in range(N) if y[j].X > 0.5]
    assignment = {i: j for (i, j) in x if x[i, j].X > 0.5}
    tree_kept = cKDTree(XY[np.array(centers)])
    d_nearest, _ = tree_kept.query(XY, k=1)
    # some stats
    max_d   = d_nearest.max()
    mean_d  = d_nearest.mean()
    p95_d   = np.percentile(d_nearest, 95)
    
    print(f"max  distance to nearest kept node : {max_d/1000:.3f} km")
    print(f"mean distance                      : {mean_d:.1f} m")
    print(f"95-percentile distance              : {p95_d:.1f} m")

    return np.array(centers), assignment, XY

keep, mapping, XY = solve_kmedian(tNet_coords[:,0], tNet_coords[:,1], k_keep=100)
roadGraph = nx.DiGraph()

position = {}
for node in keep:
    roadGraph.add_node(node_list[node], position = XY[node,:])
    position[node_list[node]]=XY[node,:]

new_g = {}
for (orig, dest), q in tNet.g.items():
    dest_new = node_list[mapping[node_idx_map[int(dest)]]]
    # dest_new = int(dest)
    orig_new = node_list[mapping[node_idx_map[int(orig)]]]
    new_g[(orig_new, dest_new)] = new_g.get((orig_new, dest_new), 0.0) + q
fig, ax = plt.subplots(figsize=(5,10))
#edges = list(G.edges())
#nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=width, edge_cmap=cmap)
nx.draw(roadGraph, position, node_color='k',  width=0.3, edge_cmap=plt.cm.Blues, arrowsize=4, node_size=10, alpha=0.7,
connectionstyle='arc3, rad=0.04')
plt.savefig("results/network_kept_nodes.pdf", format="pdf")


tNet.g = {(f"{k[0]}'", f"{k[1]}''"): v for k, v in new_g.items()}

with open("data/gml/NYC_small_demands.gpickle", "wb") as f:
    pickle.dump(tNet.g, f)
with open("data/gml/NYC_small_roadgraph.gpickle", "wb") as f:
    pickle.dump(roadGraph, f)
plt.show()
