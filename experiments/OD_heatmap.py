import experiments.build_NYC_subway_net as nyc
import pickle
import re
import matplotlib.pyplot as plt
import networkx as nx
from pyproj import Transformer
def s2int(node: str) -> int:
    num = int(re.sub(r'\D', '', node))
    return num 

tNet, tstamp, fcoeffs = nyc.build_NYC_net("data/net/NYC/", only_road=True)
with open("results/normal/25000_2/NYC_supergraph_solved.gpickle", 'rb') as f:
            tNet.G_supergraph = pickle.load(f)
with open("results/normal/25000_2/NYC_roadgraph_solved.gpickle", 'rb') as f:
            tNet.G = pickle.load(f)

nonzero_O = dict.fromkeys(tNet.G.nodes(),0)
nonzero_D = dict.fromkeys(tNet.G.nodes(),0)

for u, v, d in tNet.G_supergraph.edges(data=True):
    if d["type"] == "rp" and d["flowNoRebalancing"] > 0.1:
        nonzero_O[s2int(u)] += round(d["flowNoRebalancing"])
        nonzero_D[s2int(v)] += round(d["flowNoRebalancing"])
    else:
        nonzero_O[s2int(u)] += round(d["flowNoRebalancing"])
        nonzero_D[s2int(v)] += round(d["flowNoRebalancing"])

transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
pos_dict = {node: transformer.transform(lon, lat)
                for node, (lon, lat) in tNet.G.nodes(data="pos")}

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,5))
nx.draw(tNet.G, pos_dict, node_color=nonzero_O.values(), node_size=5, cmap=plt.cm.plasma, arrowsize=0, ax=ax1)
nx.draw(tNet.G, pos_dict, node_color=nonzero_D.values(), node_size=5, cmap=plt.cm.plasma, arrowsize=0, ax=ax2)
plt.tight_layout()
plt.show()
# what do i do with this? compare over iterations? how do i properly show OD's? as two seperate graphs?
# can i show OD's in 1 graph?

