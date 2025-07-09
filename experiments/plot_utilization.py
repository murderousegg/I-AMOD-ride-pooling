from __future__ import annotations
from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
from src.solvers import *
import matplotlib.pyplot as plt
import pickle
from pyproj import Transformer
from datetime import datetime
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.serif": ["Times"],
    "font.size": 13,                     # IEEE style prefers 8–10 pt
    "pdf.fonttype": 42,   # Important: embed fonts correctly in PDF
    "ps.fonttype": 42,
    "text.usetex": True,
    "legend.fontsize": 12,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

def plot_flows(G1: nx.DiGraph, G2: nx.DiGraph) -> None:
    pos_coords = np.array([G1.nodes[i]["pos"] for i in G1.nodes()])
    pos_coords = pos_coords[:, ::-1]
    pos_dict = {}
    for count, (u, v, data) in enumerate(G1.edges(data=True)):
        if data["capacity"] > 0:
            data["utilization"] = data["flow"] / data["capacity"]
        else:
            data["utilization"] = 0
    for count, (u, v, data) in enumerate(G2.edges(data=True)):
        if data["capacity"] > 0:
            data["utilization"] = data["flow"] / data["capacity"]
        else:
            data["utilization"] = 0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    pos_dict = {node: transformer.transform(lon, lat) for node, (lon, lat) in G1.nodes(data="pos")}
    # pos_dict = {node: pos_coords[count, ::-1] for count, node in enumerate(G.nodes())}
    # for count,node in enumerate(G.nodes()):
    #     pos_dict[node] = pos_coords[count,:]
    edge_colors1 = [data["utilization"] for _, _, data in G1.edges(data=True)]
    edge_colors2 = [data["utilization"] for _, _, data in G2.edges(data=True)]
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    edges = nx.draw_networkx_edges(
        G1,
        pos_dict,
        edge_color=edge_colors1,
        edge_cmap=plt.cm.plasma,
        edge_vmin=0.0,
        edge_vmax=max(max(edge_colors1), max(edge_colors2)),
        width=1,
        ax=ax[0],
        arrows=False
    )
    nx.draw_networkx_nodes(G1, pos_dict, node_size=5, node_color='black', ax=ax[0])
    # cbar = plt.colorbar(edges, ax=ax[0])
    # cbar.set_label(r"(flow/capacity)", labelpad=8)
    ax[0].set_title(r"Demand multiplier = 2")
    ax[0].axis("off")

    edges = nx.draw_networkx_edges(
        G2,
        pos_dict,
        edge_color=edge_colors2,
        edge_cmap=plt.cm.plasma,
        edge_vmin=0.0,
        edge_vmax=max(max(edge_colors1), max(edge_colors2)),
        width=1,
        ax=ax[1],
        arrows=False
    )
    nx.draw_networkx_nodes(G1, pos_dict, node_size=5, node_color='black', ax=ax[1])
    cbar = plt.colorbar(edges, ax=ax[1])
    cbar.set_label(r"(flow/capacity)", labelpad=8)
    ax[1].set_title(r"Demand multiplier = 4")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"results/road_usage_heatmap.pdf", format="pdf")
    plt.show()

def main() -> None:
    # sim.run()
    with open("results/NYC_roadgraph_solved.gpickle", "rb") as f:
        rg1 = pickle.load(f)
    with open("results/NYC_roadgraph_solved_2.gpickle", "rb") as f:
        rg2 = pickle.load(f)
    plot_flows(rg1, rg2)


if __name__ == "__main__":
    main()