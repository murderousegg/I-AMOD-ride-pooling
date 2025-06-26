import numpy as np
from scipy import io
import networkx as nx
import experiments.build_NYC_subway_net as nyc
import pickle
from scipy.spatial import cKDTree
from typing import Sequence, Tuple, Dict

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
    xy : ndarray shape (N, 2)  – metres
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0 if lat0 is not None else lat.mean())

    x = EARTH_R * (lon_rad - lon_rad.mean()) * np.cos(lat0_rad)
    y = EARTH_R * (lat_rad - lat_rad.mean())
    return np.column_stack((x, y))              # (N, 2)

def prune_greedy_manhattan(lat: Sequence[float],
                           lon: Sequence[float],
                           k_keep: int,
                           *,
                           verbose: bool = True
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
        if verbose and keep.sum() % 500 == 0:
            print(f"... {keep.sum()} left")

    keep_idx = np.flatnonzero(keep)                     # original ordering

    # ------------------------------------------------------------------
    # 3)  map every node to its nearest kept node
    # ------------------------------------------------------------------
    tree_kept = cKDTree(XY[keep_idx])                   # build once
    _, nn = tree_kept.query(XY, k=1)                    # nearest kept
    nearest_kept = {orig: int(nn[i]) for i, orig in enumerate(range(N))}
    # nn[i] gives *position* inside keep_idx, not global index.

    return keep_idx, nearest_kept

keep, mapping = prune_greedy_manhattan(tNet_coords[:,0], tNet_coords[:,1], k_keep=270)
roadGraph = nx.DiGraph()
for node in keep:
    roadGraph.add_node(node)


new_g = {}
for (orig, dest), q in tNet.g.items():
    dest_new = mapping[int(dest)]
    orig_new = mapping[int(orig)]
    new_g[(orig_new, dest_new)] = new_g.get((orig_new, dest_new), 0.0) + q
tNet.g = {(f"{k[0]}'", f"{k[1]}''"): v for k, v in new_g.items()}

with open("data/gml/NYC_small_demands.gpickle", "wb") as f:
    pickle.dump(tNet.g, f)
with open("data/gml/NYC_small_roadgraph.gpickle", "wb") as f:
    pickle.dump(roadGraph, f)
    
