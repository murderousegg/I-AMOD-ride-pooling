from __future__ import annotations
from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
from src.solvers import *
import matplotlib.pyplot as plt
import pickle
from pyproj import Transformer
from datetime import datetime
from pathlib import Path
from dataclasses import fields
import logging

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

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger('iamod')
logger.setLevel(logging.INFO)

def plot_flows(G: nx.DiGraph, dir):
    pos_coords = np.array([G.nodes[i]["pos"] for i in G.nodes()])
    pos_coords = pos_coords[:, ::-1]
    pos_dict = {}
    for count, (u, v, data) in enumerate(G.edges(data=True)):
        if data["capacity"] > 0:
            data["utilization"] = data["flow"] / data["capacity"]
            data["raw_utilization"] = data["flow"]
        else:
            data["utilization"] = 0
            data["raw_utilization"] = 0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    pos_dict = {node: transformer.transform(lon, lat) for node, (lon, lat) in G.nodes(data="pos")}
    # pos_dict = {node: pos_coords[count, ::-1] for count, node in enumerate(G.nodes())}
    # for count,node in enumerate(G.nodes()):
    #     pos_dict[node] = pos_coords[count,:]
    edge_colors = [data["utilization"] for _, _, data in G.edges(data=True)]
    raw_edge_colors = [data["raw_utilization"] for _, _, data in G.edges(data=True)]
    print(max(edge_colors))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    edges = nx.draw_networkx_edges(
        G,
        pos_dict,
        edge_color=edge_colors,
        edge_cmap=plt.cm.viridis,
        edge_vmin=0.0,
        # edge_vmax=1.0,
        width=1,
        ax=ax[0],
        arrows=False
    )
    nx.draw_networkx_nodes(G, pos_dict, node_size=5, node_color='black', ax=ax[0])
    cbar = plt.colorbar(edges, ax=ax[0])
    cbar.set_label(r"(flow/capacity)", labelpad=8)
    ax[0].set_title(r"Road utilization")
    ax[0].axis("off")
    # plt.savefig("results/road_usage_heatmap.pdf", format="pdf")
    # plt.show(block=False)

    flows = np.array(raw_edge_colors)
    # vmax = np.percentile(flows, 95)  # 99th percentile cap

    edges = nx.draw_networkx_edges(
        G,
        pos_dict,
        edge_color=raw_edge_colors,
        edge_cmap=plt.cm.viridis,
        # edge_vmin=0.0,
        # edge_vmax=vmax,
        width=1,
        ax=ax[1],
        arrows=False
    )
    nx.draw_networkx_nodes(G, pos_dict, node_size=5, node_color='black', ax=ax[1])
    cbar = plt.colorbar(edges, ax=ax[1])
    cbar.set_label(r"Flow (vehicles/h)", labelpad=8)
    ax[1].set_title(r"Flow density in network")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{dir}road_usage_heatmap.pdf", format="pdf")
    plt.show()

def main() -> None:
    cfg = SimulationConfig()

    # create results directory
    now_string = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    cfg.results_dir = f"results/NYC_{now_string}/"
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    ###
    cfg.max_iterations = 1
    cfg.vehicle_limit = 15000
    cfg.mu_initial = 1e-2
    cfg.stable_needed = 3
    cfg.demand_multiplier=1
    cfg.delay_factor=1 / 60 # 1 min
    cfg.waiting_time=1 / 60 # 1 min
    sim = RidePoolingSimulationCore(cfg)
    sim.run()
    # with open("results/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     sim.original_G = pickle.load(f)
    with open("results/NYC_roadgraph_solved.gpickle", "wb") as f:
        pickle.dump(sim.original_G, f)
    with open("results/NYC_supergraph_solved.gpickle", "wb") as f:
        pickle.dump(sim.tNet.G_supergraph, f)
    plot_flows(sim.original_G, cfg.results_dir)
    with open(cfg.results_dir+ "config.txt", "w") as f:
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            f.write(f"{field.name}:{value}\n")


if __name__ == "__main__":
    main()
