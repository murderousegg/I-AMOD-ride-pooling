import experiments.build_NYC_subway_net as nyc
import pickle
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from pyproj import Transformer
import numpy as np
from src.routeFinder import *
import src.tnet as tnet
import random

def s2int(node: str) -> int:
    num = int(re.sub(r'\D', '', node))
    return num 

### load data
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

origin_idx = 10
o = origins[origin_idx]
xo = {e: float(x_sol[row_idx, origin_idx]) for row_idx, e in enumerate(edges)}

with gzip.open(f'results/routeRecovery/routes_by_od_{origin_idx}.pkl.gz', 'rb') as f:
    routes_by_od = pickle.load(f)
with gzip.open('results/car_routes_1.pkl.gz', 'rb') as f:
    route_1 = pickle.load(f)
with gzip.open('results/car_routes_2.pkl.gz', 'rb') as f:
    route_2 = pickle.load(f)

### plotting functions
def plot_network(G, ax, width=1, cmap=plt.cm.Blues, edge_width=False, 
	edgecolors=False, nodecolors=False, nodesize=False, arrowsize=False,edge_alpha=1,linkstyle='solid' ):
    pos = nx.get_node_attributes(G, 'pos')
    if linkstyle == '-':
        nx.draw(G, pos, 
                width=edge_width,  
                ax=ax, 
                edge_color=edgecolors, 
                node_size=nodesize, 
                node_color=nodecolors,
                connectionstyle='arc3, rad=0.02',
                arrowsize=0.5, 
                arrowstyle='fancy',
                alpha=edge_alpha)
    else:
        nx.draw(G, pos, width=edge_width,  
                ax=ax, edge_color=edgecolors, 
                node_size=nodesize, node_color=nodecolors,
                #connectionstyle='arc3, rad=0.04',
                arrowsize=0.5, arrowstyle='fancy',
                alpha=edge_alpha, style=linkstyle)

def plot_routes(tNet, od, result, ax):
    #fig, ax = plt.subplots()
    cmap2 =  ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(30)]
    cmap = ['b','m','y','c','g', 'r']  
    cmap.extend(cmap2)
    #plot_network(original_G, width=0.3)
    l=0
    for i, dic in result.items():
        if l > 10:
              break
        if dic['p'] < 0.1:
              continue
        print(i, dic['p'])
        r = dic['r']
        r_links = []
        stored=None
        for n in range(len(r)-1):
            if len(r[n+1]) <= 6:
                if not stored:
                    r_links.append((r[n], r[n+1]))
                else:
                    r_links.append((stored, r[n+1]))
            else:
                if stored:
                    continue
                stored = r[n]
                continue
            stored = None
        # r_links = [(r[n],r[n+1]) for n in range(len(r)-1)]
        G_ = tNet.G_supergraph.edge_subgraph(r_links)
        
        edges = ((edge[0],edge[1]) for edge in G_.edges(data=True) if edge[2]['type']=="p")	
        G = G_.edge_subgraph(edges)
        edge_colors = ["tab:red" if e in r_links else 'gray' for e in G.edges()]
        edge_width = [3 if e in r_links else 0 for e in G.edges()]
        arrow_size = [3 if e in r_links else 0 for e in G.edges()]
        if sum(edge_width) > 0:
              print(f"p found in od {od}")
        plot_network(G, ax, edge_width=edge_width, 
            edgecolors=edge_colors, nodecolors='gray', 
            nodesize=0.0, arrowsize=arrow_size,edge_alpha=0.7, 
            linkstyle="solid")

        edges = ((edge[0],edge[1]) for edge in G_.edges(data=True) if edge[2]['type']=="s")
        G = G_.edge_subgraph(edges)
        edge_colors = ["tab:orange" if e in r_links else 'gray' for e in G.edges()]
        edge_width = [3 if e in r_links else 0 for e in G.edges()]
        arrow_size = [3 if e in r_links else 0 for e in G.edges()]
        if sum(edge_width) > 0:
              print(f"pt found in od {od}")
        plot_network(G, ax, edge_width=edge_width, 
            edgecolors=edge_colors, nodecolors='gray', 
            nodesize=0.0, arrowsize=arrow_size,edge_alpha=0.7, 
            linkstyle=(0, (1, 1)))        
        
        edges = ((edge[0],edge[1]) for edge in G_.edges(data=True) if edge[2]['type']=="b")
        G = G_.edge_subgraph(edges)
        edge_colors = ["tab:green" if e in r_links else 'gray' for e in G.edges()]
        edge_width = [3 if e in r_links else 0 for e in G.edges()]
        arrow_size = [3 if e in r_links else 0 for e in G.edges()]
        if sum(edge_width) > 0:
              print(f"b found in od {od}")
        plot_network(G, ax, edge_width=edge_width, 
            edgecolors=edge_colors, nodecolors='gray', 
            nodesize=0.1, arrowsize=arrow_size,edge_alpha=0.7, 
            linkstyle='solid')
        
        # edges = ((edge[0],edge[1]) for edge in G_.edges(data=True) if edge[2]['type']=="p")
        # G = G_.edge_subgraph(edges)
        # edge_colors = [cmap[l] if e in r_links else 'gray' for e in G.edges()]
        # edge_width = [1 if e in r_links else 0 for e in G.edges()]
        # arrow_size = [1 if e in r_links else 0 for e in G.edges()]
        # plot_network(G, ax, edge_width=edge_width, 
        #     edgecolors=edge_colors, nodecolors='gray', 
        #     nodesize=0.0, arrowsize=arrow_size,edge_alpha=0.7, 
        #     linkstyle=(0, (1, 5)))
        edges = ((edge[0],edge[1]) for edge in G_.edges(data=True) if edge[2]['type']=="rp")
        G = G_.edge_subgraph(edges)
        edge_colors = ["tab:blue" if e in r_links else 'gray' for e in G.edges()]
        edge_width = [3 if e in r_links else 0 for e in G.edges()]
        arrow_size = [3 if e in r_links else 0 for e in G.edges()]
        if sum(edge_width) > 0:
              print(f"rp found in od {od}")
        plot_network(G, ax, edge_width=edge_width, 
            edgecolors=edge_colors, nodecolors='gray', 
            nodesize=0.0, arrowsize=arrow_size,edge_alpha=0.7, 
            linkstyle=(0, (1, 5)))
        l+=1
    return ax


def plot_cars(G, result, ax):
    #fig, ax = plt.subplots()
    cmap2 =  ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(30)]
    cmap = ['b','m','y','c','g', 'r']  
    cmap.extend(cmap2)
    #plot_network(original_G, width=0.3)
    l =0
    
    for i, dic in result.items():
        if l > 10:
              break
        if dic['p'] < 5e-2:
              continue
        print(i, dic['p'])
        r = dic['r']
        r_links = [(r[n],r[n+1]) for n in range(len(r)-1)]
        G_ = G.edge_subgraph(r_links)
        
        edge_colors = ['tab:blue' if e in r_links else 'gray' for e in G_.edges()]
        edge_width = [1.5 if e in r_links else 0 for e in G_.edges()]
        arrow_size = [1 if e in r_links else 0 for e in G_.edges()]
        plot_network(G_, ax, edge_width=edge_width, 
            edgecolors=edge_colors, nodecolors='gray', 
            nodesize=0.0, arrowsize=arrow_size,edge_alpha=0.7, 
            linkstyle=(0, (1, 1)))
        l+=1
    return ax

def plot_and_save_all_for_origin(tNet, o, routes_by_od, out_dir='results/routeRecovery/'):
    os.makedirs(out_dir, exist_ok=True)

    # Plot each OD
    for (oo, d), routes in routes_by_od.items():
        if oo != o:   # only plot this origin
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = plot_routes(tNet, (oo, d), routes, ax)
        fig.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(out_dir, f'plot_{oo}_{d}.pdf'))
        plt.close(fig)

    return routes_by_od

def plot_and_save_w_cars(tNet, G, o, d, routes, out_dir='results/routeRecovery/'):
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    node_color = ['red' if n == s2int(o) else ('green' if n in d else 'gray') for n in original_G.nodes()]
    #node_color = ['green' if n in [od[1]] else 'gray' for n in tNet.G_supergraph.nodes()]
    node_size = [25 if n in [s2int(o)] or n in d else 0.2 for n in original_G.nodes()]
    plot_network(original_G, ax, edge_width=0.35,
                        edgecolors='gray', nodecolors=node_color,
                        nodesize=node_size, arrowsize=0.15,edge_alpha=0.7)
    for n, route in enumerate(routes): 
        ax = plot_routes(tNet, (o, d[n]), route, ax)
    handles = [
        # match your styles/colors in plot_routes:
        Line2D([0], [0], color='tab:blue',  lw=2, linestyle=(0, (1, 5)), label='Ride-pooling'),  # rp
        Line2D([0], [0], color='tab:orange',lw=2, linestyle='solid', label='Public Transit'),# s
        Line2D([0], [0], color='tab:green', lw=2, linestyle='solid',     label='Biking'),        # b
        Line2D([0], [0], color='tab:red',   lw=2, linestyle='solid',     label='Walking'),       # p
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=16, frameon=True)
    # ax.set_aspect('equal')
    ax.set_xlim(-74.018, -73.960)
    ax.set_ylim(40.7, 40.784)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'all_plot_routes_{o[0]}.pdf'))
    plt.show()
    plt.close(fig)



for node in tNet.G_supergraph.nodes():
    # if "s" in node:
    #      print(node)
    if s2int(node) in original_G.nodes(): 
        tNet.G_supergraph.nodes[node]['pos'] = original_G.nodes[s2int(node)]['pos']
    else:
        tNet.G_supergraph.nodes[node]['pos'] = -1

# for routes in routes_by_od.values():
#     #   print(routes)
#       for route in routes.values():
#             for node in route['r']:
#                   if "s" in node:
#                         print("subway used")
#                         print(route)
#                         print(routes)

def save_routes(routes_by_od, path='results/routes_by_od.pkl.gz'):
    with gzip.open(path, 'wb') as f:
        pickle.dump(routes_by_od, f, protocol=pickle.HIGHEST_PROTOCOL)

selected_routes_by_od = [
     routes_by_od[("222'", "498''")],
     routes_by_od[("222'", "31''")],
     routes_by_od[("222'", "506''")]
]
plot_and_save_w_cars(tNet, original_G, "222'", [498, 31, 506], selected_routes_by_od, out_dir='results/routeRecovery/')

"""
RP recovery not currently working
Could be fixed
"""
# ori = "222'"
# dest = "1034''"
# selected_route = {(ori, dest):routes_by_od[(ori, dest)]}
# print([i for i in selected_route[(ori, dest)].items() if i[1]['p']>=0.1])
# # # ## 132,478 AND 40,478 AND 81,478 AND 73,478
# # # print(tNet.G_supergraph['73rp']["478''"])

# # ###### Find RP routes for given OD
# rp_o = s2int('222rp')#some origin
# rp_d = s2int('1034rp')#some destination
# D_rp = np.load("full_demand10.npy") #OD matrix for rp demand only
# y_sol = np.load("results/normal/45000_3/y_sol.npy")

# np.fill_diagonal(D_rp, 0)
# G_nodes = list(original_G.nodes())
# G_node_idx = {n:i for i,n in enumerate(G_nodes)}


# # #define od indices
# G_o_idx = G_node_idx[rp_o]
# G_d_idx = G_node_idx[rp_d]

# #find correct y column (as dict)
# yo = {e: float(y_sol[row_idx, G_o_idx]) for row_idx, e in enumerate(original_G.edges())}

# # ##### flipped D
# # # find car demand from D_rp as dict
# car_g = {(rp_o, G_nodes[d]): D_rp[d, G_o_idx] for d in range(len(G_nodes))} #only demand for this O
# # # print(car_g_keys2)
# # # print(origin_to_d)
# # # print(D_rp[:,G_o_idx].sum())
# # # decompose 
# routes = userRouteFinder(original_G, car_g, {rp_o: yo}, eps=1)
# print(routes)
# routes = routes[(rp_o, rp_d)]
# # # yf_by_d = solve_flow_decomposition_D_fast_tol(
# # #     original_G, origin=rp_o, xo=yo, g=car_g,
# # #     link_tol_abs=1e-6, link_tol_rel=1e-6, node_tol_abs=1e-6
# # # )

# # # car_routes = routeFinder_OD(original_G, [(rp_o, rp_d),car_g[(rp_o, rp_d)]], yf_by_d[rp_d], eps=1e-3, max_routes=10)
# save_routes(routes, path='results/car_routes_1.pkl.gz')

# ###### Find RP routes for given OD
# rp_o = s2int('282rp')#some origin
# rp_d = s2int('1034rp')#some destination

# #define od indices
# G_o_idx = G_node_idx[rp_o]
# G_d_idx = G_node_idx[rp_d]

# #find correct y column (as dict)
# yo = {e: (float(y_sol[row_idx, G_o_idx]) if float(y_sol[row_idx, G_o_idx])>=0 else 0) for row_idx, e in enumerate(original_G.edges())}

# # find car demand from D_rp as dict
# car_g = {(rp_o, G_nodes[p]): D_rp[p, G_o_idx] for p in range(len(G_nodes))} #only demand for this O
# car_to_d = {(G_nodes[q], rp_d) : (D_rp[G_d_idx, q] if D_rp[G_d_idx, q] >= 0 else 0) for q in range(len(G_nodes))}
# car_g_keys2 = [i[1] for i in car_g.keys() if car_g[i] > 0.1]
# origin_to_d = [i[0] for i in car_to_d.keys() if i[0] in car_g_keys2 and car_to_d[i]>0.1]
# print(car_g)
# print(D_rp[G_d_idx,:].sum())
# print(car_g_keys2)
# print(origin_to_d)
# sdf
# routes2 = userRouteFinder(original_G, car_g, {rp_o: yo}, eps=1)
# print(routes2)
# routes2 = routes2[(rp_o, rp_d)]
# save_routes(routes2, path='results/car_routes_2.pkl.gz')

# ###### Find RP routes for given OD
# rp_o = s2int('274rp')#some origin
# rp_d = s2int('1034rp')#some destination

# #define od indices
# G_o_idx = G_node_idx[rp_o]
# G_d_idx = G_node_idx[rp_d]

# #find correct y column (as dict)
# yo = {e: (float(y_sol[row_idx, G_o_idx]) if float(y_sol[row_idx, G_o_idx])>=0 else 0) for row_idx, e in enumerate(original_G.edges())}

# # find car demand from D_rp as dict
# car_g = {(rp_o, G_nodes[p]): D_rp[G_o_idx, p] for p in range(len(G_nodes))} #only demand for this O

# routes3 = userRouteFinder(original_G, car_g, {rp_o: yo}, eps=1)
# print(routes3)
# routes3 = routes3[(rp_o, rp_d)]
# save_routes(routes3, path='results/car_routes_3.pkl.gz')
# car_to_d = {(G_nodes[q], rp_d) : (D_rp[q, G_d_idx] if D_rp[q, G_d_idx] >= 0 else 0) for q in range(len(G_nodes))}

# print([i for i in car_g.keys() if car_g[i] != 0.0])
# print(car_g)

# for o,d in car_g.keys():
     
#      if car_g[(o,d)] >= 0.01:
#           for o1,d1 in car_to_d.keys():
#                if car_to_d[(d,d1)] >= 1:
#                     print(o,d)
#                     print(o1,d1)

# print([i for i in car_to_d.keys() if car_to_d[i] >= 1])
# for i in car_to_d.keys():
#      if car_to_d[i] >= 1:
#         fig, ax = plt.subplots(figsize=(8, 8))
#         node_color = ['red' if n == i[0] else ('green' if n == i[1] else ('black' if n == 81 else'gray')) for n in original_G.nodes()]
#         #node_color = ['green' if n in [od[1]] else 'gray' for n in tNet.G_supergraph.nodes()]
#         node_size = [25 if n in [i[0], i[1], 81] else 0.2 for n in original_G.nodes()]
        
#         plot_network(original_G, ax, edge_width=0.35,
#                             edgecolors='gray', nodecolors=node_color,
#                             nodesize=node_size, arrowsize=0.15,edge_alpha=0.7)
#         plt.show()

# # decompose 
# yf_by_d = solve_flow_decomposition_D_fast_tol(
#     original_G, origin=rp_o, xo=yo, g=car_g,
#     link_tol_abs=1e-6, link_tol_rel=1e-6, node_tol_abs=1e-6
# )
# print([i for i in yf_by_d.keys() if not isinstance(yf_by_d[i], float)])
# # car_routes_by_od = {}
# car_routes = routeFinder_OD(original_G, [(rp_o, rp_d),car_g[(rp_o, rp_d)]], yf_by_d[rp_d], eps=1e-3, max_routes=100)
# save_routes(car_routes, path='results/car_routes_2.pkl.gz')
# print(routes_by_od)

# selected_routes_by_od = {
#      ("222'", "1034''"):routes_by_od[("222'", "1034''")]
#      }
# plot_and_save_all_for_origin(tNet, o, selected_routes_by_od, out_dir='results/routeRecovery/')
# plot_and_save_all_for_origin(tNet, o, selected_route, out_dir='results/routeRecovery/')