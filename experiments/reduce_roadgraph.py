import numpy as np
from scipy import io
import networkx as nx
import src.tnet as tnet
import experiments.build_NYC_subway_net as nyc
import pickle
from scipy.spatial import KDTree 

# ---------- helpers --------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# ---------- load base network ---------------------------------------------
tNet, _, _ = nyc.build_NYC_net("data/net/NYC/", only_road=True)
tNet.read_node_coordinates("data/pos/NYC.txt")
tNet_coords = np.array([tNet.G.nodes[i]["pos"] for i in tNet.G.nodes()])
tNet_coords = tNet_coords[:, ::-1]
node_list = list(tNet.G.nodes())                       # index → label
node_idx_map = {n: i for i, n in enumerate(node_list)} # label → index

# ---------- MATLAB mini-graph ---------------------------------------------
mat = io.loadmat("small_graph160.mat", squeeze_me=True)
edge_list = np.asarray(mat["edge_list"])     # shape (E, 3)
latLon_mat = np.asarray(mat["NodesLatLon"])[:,:2]  # shape (N_mat, 2)
roadCap = mat["RoadCap"]

# ------------------------------------------------------------------
# 1. build a KD-tree on *road* coords  (lat, lon order) ------------
# ------------------------------------------------------------------
mat_ids_with_edges = np.unique(edge_list[:, :2])         # 1-based IDs
mask = np.isin(np.arange(1, len(latLon_mat)+1), mat_ids_with_edges)

latLon_mat = latLon_mat[mask]          # only 274 nodes instead of 357
road_tree = KDTree(tNet_coords)          # tNet_coords = [[lat, lon], ...]

# map every MATLAB node (1-based) to its nearest road node
# --> returns 0-based indices into tNet_coords / node_list
nearest_idx = road_tree.query(latLon_mat)[1]   # latLon_mat already [lat, lon]
closest_nodes = [node_list[i] for i in nearest_idx]

# ------------------------------------------------------------------
# 2. build the reduced road graph ----------------------------------
# ------------------------------------------------------------------
new_edges, capacities = [], []
for u_id, v_id, t0 in edge_list:
    u_label = closest_nodes[int(u_id) - 1]          # ONE subtraction, not two
    v_label = closest_nodes[int(v_id) - 1]
    new_edges.append((u_label, v_label, t0))
    capacities.append(roadCap[int(u_id) - 1, int(v_id) - 1])
roadGraph = nx.DiGraph()
for node_label in set(closest_nodes):
    roadGraph.add_node(node_label)
for (u, v, t0), cap in zip(new_edges, capacities):
    t0_true = nx.shortest_path_length(tNet.G, source=u, target=v, weight='t_0')
    roadGraph.add_edge(u, v, t_0=t0_true, capacity=cap * 60)
# ------------------------------------------------------------------
# 3. remap each destination in the OD matrix -----------------------
# ------------------------------------------------------------------
mat_tree = KDTree(latLon_mat)            # (lat, lon) – same order as above
new_g = {}
for (orig, dest), q in tNet.g.items():
    dest_idx = node_idx_map[int(dest)]   # dest is plain int (no prime yet)
    dest_coord = tNet_coords[dest_idx]   # [lat, lon]

    orig_idx = node_idx_map[int(orig)]   # dest is plain int (no prime yet)
    orig_coord = tNet_coords[orig_idx]   # [lat, lon]

    # nearest MATLAB node --> index into closest_nodes
    mat_idx = mat_tree.query(dest_coord)[1]
    mat_idx_orig = mat_tree.query(orig_coord)[1]
    dest_new = closest_nodes[int(mat_idx)]
    orig_new = closest_nodes[int(mat_idx_orig)]

    new_g[(orig_new, dest_new)] = q             # origin stays unchanged

# ------------------------------------------------------------------
# 4. add layer suffixes & pickle -----------------------------------
# ------------------------------------------------------------------
tNet.g = {(f"{k[0]}'", f"{k[1]}''"): v for k, v in new_g.items()}

with open("data/gml/NYC_small_demands.gpickle", "wb") as f:
    pickle.dump(tNet.g, f)
with open("data/gml/NYC_small_roadgraph.gpickle", "wb") as f:
    pickle.dump(roadGraph, f)
    
