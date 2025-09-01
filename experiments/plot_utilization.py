from __future__ import annotations

from src.solvers import *
import matplotlib.pyplot as plt
import pickle
from pyproj import Transformer
import pandas as pd

plt.rcParams.update({
    "font.size": 25,                     # IEEE style prefers 8–10 pt
    "pdf.fonttype": 42,   # Important: embed fonts correctly in PDF
    "ps.fonttype": 42,
    "legend.fontsize": 25,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "text.latex.preamble": r'\usepackage{dsfont}',
    "axes.labelsize": 20,
})

def plot_flows(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    G3: nx.DiGraph,
    G4: nx.DiGraph,
    *,
    vmax: float = 1.2,
    cmap = None,
    savepath: str = "results/road_usage_heatmap.pdf"
) -> None:
    """
    Create a 2x2 panel with consistent color scale and a single colorbar,
    with font sizes and paddings scaled for LaTeX single- or double-column widths.
    """

    fig_w = 4
    base = 10

    fig_h = fig_w * 1.4 # set height

    fs_tick   = base       
    fs_label  = base + 0.5    
    fs_title  = base + 2.0
    lw_edge   = 0.4
    ns_node   = 0.01

    # get utilization
    def add_utilization(G):
        for (_, _, d) in G.edges(data=True):
            cap = d.get("capacity", 0.0)
            flow = d.get("flow", 0.0)
            d["utilization"] = (flow / cap) if cap > 0 else 0.0
        return G

    G1 = add_utilization(G1)
    G2 = add_utilization(G2)
    G3 = add_utilization(G3)
    G4 = add_utilization(G4)

    # transform positions
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    pos_dict = {node: transformer.transform(lon, lat)
                for node, (lon, lat) in G1.nodes(data="pos")}

    # gather data
    edge_colors = [
        [d["utilization"] for _, _, d in G1.edges(data=True)],
        [d["utilization"] for _, _, d in G2.edges(data=True)],
        [d["utilization"] for _, _, d in G3.edges(data=True)],
        [d["utilization"] for _, _, d in G4.edges(data=True)],
    ]
    # set colormap
    if cmap is None:
        cmap = plt.cm.plasma

    # layout
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(fig_w, fig_h),
        layout="constrained"
    )

    Gs = [G1, G2, G3, G4]
    titles = [
        r"$N_{\mathrm{cars}} = 5\times 10^3$" "\n" r"$\phi = 0.5$",
        r"$N_{\mathrm{cars}} = 15\times 10^3$" "\n" r"$\phi = 1$",
        r"$N_{\mathrm{cars}} = 25\times 10^3$" "\n" r"$\phi = 2$",
        r"$N_{\mathrm{cars}} = 40\times 10^3$" "\n" r"$\phi = 3$",
    ]
    mappable = None
    for k, ax in enumerate(axes.ravel()):
        vals = edge_colors[k]
        mappable = nx.draw_networkx_edges(
            Gs[k],
            pos_dict,
            edge_color=vals,
            edge_cmap=cmap,
            edge_vmin=0.0,
            edge_vmax=vmax,
            width=lw_edge,
            arrows=False,
            ax=ax
        )
        nx.draw_networkx_nodes(G1, pos_dict, node_size=ns_node, node_color="black", ax=ax)
        ax.set_axis_off()
        ax.annotate(
            titles[k],
            xy=(0.5, 1.02), xycoords="axes fraction",
            ha="center", va="top", fontsize=fs_title
        )
    # Use the entire 2x2 grid
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        location="bottom",
        orientation="horizontal",
        fraction=0.1,    # width of colorbar relative to axes area
        pad=0.02,          # gap between last axes and colorbar
        shrink=0.96
    )
    cbar.set_label("Utilization", labelpad=8, fontsize=fs_label)
    cbar.ax.tick_params(labelsize=fs_tick)

    fig.savefig(
        savepath,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01
    )
    plt.show(block=False)


def plot_flows_no_lim(G1: nx.DiGraph, G2: nx.DiGraph, G3: nx.DiGraph, G4: nx.DiGraph) -> None:
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
    for count, (u, v, data) in enumerate(G3.edges(data=True)):
        if data["capacity"] > 0:
            data["utilization"] = data["flow"] / data["capacity"]
        else:
            data["utilization"] = 0
    for count, (u, v, data) in enumerate(G4.edges(data=True)):
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
    edge_colors3 = [data["utilization"] for _, _, data in G3.edges(data=True)]
    edge_colors4 = [data["utilization"] for _, _, data in G4.edges(data=True)]
    # vmax = max(max(edge_colors1), max(edge_colors2), max(edge_colors3))
    vmax=2.5
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,15), constrained_layout=True)
    edges = nx.draw_networkx_edges(
        G1,
        pos_dict,
        edge_color=edge_colors1,
        edge_cmap=plt.cm.plasma,
        edge_vmin=0,
        edge_vmax=vmax,
        width=0.5,
        ax=ax[0,0],
        arrows=False
    )
    nx.draw_networkx_nodes(G1, pos_dict, node_size=0.5, node_color='black', ax=ax[0,0])
    # cbar = plt.colorbar(edges, ax=ax[0])
    # cbar.set_label(r"(flow/capacity)", labelpad=8)
    ax[0,0].axis("off")

    edges = nx.draw_networkx_edges(
        G2,
        pos_dict,
        edge_color=edge_colors2,
        edge_cmap=plt.cm.plasma,
        edge_vmin=0,
        edge_vmax=vmax,
        width=0.5,
        ax=ax[0,1],
        arrows=False
    )
    nx.draw_networkx_nodes(G1, pos_dict, node_size=0.5, node_color='black', ax=ax[0,1])
    ax[0,1].axis("off")

    edges = nx.draw_networkx_edges(
        G3,
        pos_dict,
        edge_color=edge_colors3,
        edge_cmap=plt.cm.plasma,
        edge_vmin=0,
        edge_vmax=vmax,
        width=0.5,
        ax=ax[1,0],
        arrows=False
    )
    nx.draw_networkx_nodes(G1, pos_dict, node_size=0.5, node_color='black', ax=ax[1,0])
    ax[1,0].axis("off")
    edges = nx.draw_networkx_edges(
        G4,
        pos_dict,
        edge_color=edge_colors4,
        edge_cmap=plt.cm.plasma,
        edge_vmin=0,
        edge_vmax=vmax,
        width=0.5,
        ax=ax[1,1],
        arrows=False
    )
    nx.draw_networkx_nodes(G1, pos_dict, node_size=0.5, node_color='black', ax=ax[1,1])

    cbar = plt.colorbar(edges, ax=ax, location='right', shrink=0.8)
    cbar.set_label(r"Utilization", labelpad=8, fontsize=20)
    ax[1,1].axis("off")
    # plt.tight_layout()
    # fig.suptitle(r"Road Utilization With Varying Demand Without Vehicle Limit", fontsize=27)
    titles = [r"$\phi = 2$",\
              r"$\phi = 4$",\
              r"$\phi = 6$",\
              r"$\phi = 10$"]

    for i in range(4):
        ax[i//2,i%2].annotate(
            titles[i],
            xy=(0.5, 0.95), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=20
        )
    plt.savefig(f"results/road_usage_heatmap_no_lim.pdf", format="pdf")
    plt.show(block=False)

def bar_plot_final_result(df1:pd.DataFrame, df2:pd.DataFrame, df3:pd.DataFrame, df4:pd.DataFrame, df1_2:pd.DataFrame, df2_2:pd.DataFrame, df3_2:pd.DataFrame,  df4_2:pd.DataFrame, df1_lim:pd.DataFrame, df2_lim:pd.DataFrame, df3_lim:pd.DataFrame, df4_lim:pd.DataFrame) -> None:
    
    df1_last = df1.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df2_last = df2.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df3_last = df3.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df4_last = df4.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]

    df1_2_last = df1_2.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df2_2_last = df2_2.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df3_2_last = df3_2.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df4_2_last = df4_2.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]

    df1_lim_last = df1_lim.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df2_lim_last = df2_lim.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df3_lim_last = df3_lim.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    df4_lim_last = df4_lim.iloc[-1][['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow', 'double_share']]
    #combine to 1 df
    bar_colors = plt.get_cmap("tab10").colors[:4]
    plot_df = pd.DataFrame([df1_last, df1_2_last, df1_lim_last, df2_last, df2_2_last, df2_lim_last, df3_last, df3_2_last, df3_lim_last,  df4_last, df4_2_last, df4_lim_last], \
                           index=['df1', 'df1_2', 'df1_lim', 'df2', 'df2_2', 'df2_lim', 'df3', 'df3_2', 'df3_lim', 'df4', 'df4_2', 'df4_lim'])

    # #create fig
    legend_labels = ['Ride-pooling', 'Public Transit', 'Biking', 'Walking']

    bar_width = 0.1
    group_spacing = 0.2
    x_positions = []
    lim_labels = []
    # limits = [r'5',r'10', r'15', r'15', r'20', r'25', r'25', r'30', r'35', r'35', r'40', r'45']
    limits = [r'2.5', r'5', r'7.5', r'10',r'12.5', r'15', r'20', r'25', r'30', r'35', r'40', r'45']
    group_centers = []
    for i in range(4):
        group_start = i * (2 * bar_width + group_spacing)
        for j in range(3):
            x_positions.append(group_start + j * (bar_width+0.01))
            lim_labels.append(limits[3*i+j])
        group_centers.append(group_start + bar_width)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(x_positions))
    modes = ['rp_flow', 'pt_flow', 'bike_flow', 'ped_flow']
    for mode, label in zip(modes, legend_labels):
        values = plot_df[mode].values / 1e4
        ax.bar(x_positions, values, bar_width, bottom=bottom, label=label, edgecolor='black', linewidth=0.5)
        bottom += values

    ax.set_xticks(x_positions)
    ax.set_xticklabels(lim_labels, rotation=0)
    for i in range(4):
        x_center = group_centers[i]
        # ax.text(x_center, -0.09, [r'$\phi = 1$', r'$\phi = 2$', r'$\phi = 3$', r'$\phi = 4$'][i], ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=20)
        ax.text(x_center, 1.09, [r'$\phi = 0.5$', r'$\phi = 1$', r'$\phi = 2$', r'$\phi = 3$'][i], ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=20)
    ax.set_ylabel(r'Time-Based Modal Share ($\times 10^4 \mathrm{h}$)')
    ax.set_xlabel(r"$N_{\mathrm{cars,max}}$ ($\times 10^3$)", labelpad=0)
    # ax.set_title(r"Modal Share Based on Demand and Vehcle Limit", fontsize=25)
    plt.legend(legend_labels ,loc='upper left', fontsize=16)
    # pooling percentages
    ax2 = ax.twinx()
    ax2.set_ylabel(r"($\mathrm{\%}$) of rp requests pooled", color='tab:purple')
    ax2.plot(x_positions, [i*100 for i in plot_df['double_share'].values], linestyle='', marker='o', markersize=12, markeredgewidth=0.5, markeredgecolor='black', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    # ax2.set_ylim([97,100.2])
    ax2.set_ylim([94,100.2])
    plt.grid(axis='y', alpha = 0.3)

    plt.tight_layout()
    plt.savefig("results/fin_modal_share_bar_plot.pdf", format="pdf")
    plt.show()

def main() -> None:
    # with open("results/NYC_10000_1/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     rg1 = pickle.load(f)
    # with open("results/NYC_20000_2/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     rg2 = pickle.load(f)
    # with open("results/NYC_30000_3/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     rg3 = pickle.load(f)
    # with open("results/NYC_40000_4/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     rg4 = pickle.load(f)
    with open("results/normal/5000_0.5/NYC_roadgraph_solved.gpickle", "rb") as f:
        rg1 = pickle.load(f)
    with open("results/normal/15000_1/NYC_roadgraph_solved.gpickle", "rb") as f:
        rg2 = pickle.load(f)
    with open("results/normal/25000_2/NYC_roadgraph_solved.gpickle", "rb") as f:
        rg3 = pickle.load(f)
    with open("results/normal/40000_3/NYC_roadgraph_solved.gpickle", "rb") as f:
        rg4 = pickle.load(f)

    plot_flows(rg1, rg2, rg3, rg4)

    # with open("results/NYC_inf_1/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     rg1 = pickle.load(f)
    # with open("results/NYC_inf_4/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     rg2 = pickle.load(f)
    # with open("results/NYC_inf_6/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     rg3 = pickle.load(f)
    # with open("results/NYC_inf_10/NYC_roadgraph_solved.gpickle", "rb") as f:
    #     rg4 = pickle.load(f)
    # plot_flows_no_lim(rg1, rg2, rg3, rg4)


    ### load results df
    df1 = pd.read_csv("results/normal/2500_0.5/results_NYC.csv")
    df1_2 = pd.read_csv("results/normal/5000_0.5/results_NYC.csv")
    df1_3 = pd.read_csv("results/normal/7500_0.5/results_NYC.csv")
    df2 = pd.read_csv("results/normal/10000_1/results_NYC.csv")
    df2_2 = pd.read_csv("results/normal/12500_1/results_NYC.csv")
    df2_3 = pd.read_csv("results/normal/15000_1/results_NYC.csv")
    df3 = pd.read_csv("results/normal/20000_2/results_NYC.csv")
    df3_2 = pd.read_csv("results/normal/25000_2/results_NYC.csv")
    df3_3 = pd.read_csv("results/normal/30000_2/results_NYC.csv")
    df4 = pd.read_csv("results/normal/35000_3/results_NYC.csv")
    df4_2 = pd.read_csv("results/normal/40000_3/results_NYC.csv")
    df4_3 = pd.read_csv("results/normal/45000_3/results_NYC.csv")

    bar_plot_final_result(df1, df2, df3, df4, df1_2, df2_2, df3_2,df4_2, df1_3, df2_3, df3_3, df4_3)






if __name__ == "__main__":
    main()