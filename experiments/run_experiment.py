import src.tnet as tnet
import matplotlib.pyplot as plt
import src.CARS as cars
import networkx as nx
import re
import copy

netFile, gFile, fcoeffs,_,_ = tnet.get_network_parameters('NYC_Uber_small')

tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
tNetExog = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
g_exog = tnet.perturbDemandConstant(tNetExog.g, constant=1.5)
tNetExog.set_g(g_exog)
tNet.build_supergraph()
og_graph = tNet.G_supergraph.copy()
#integer demands:
tNet.g = {key: int(round(value)) for key, value in tNet.g.items()}

# origins in walking layer
new_origins = {(f"{k[0]}'",f"{k[1]}'"): v for k, v in tNet.g.items()}
tNet.g = new_origins

#build other layers
tNet.build_rp_layer(avg_speed=1000)
tNet.build_full_layer()
# tNet.build_pickup_layer()

ridepool = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'rp']
pedestrian = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'p']
connector = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'f']
connector_rp = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'f_rp']

totalCost = []
if __name__ == '__main__':
    for i in range(1):
        sol = cars.solve_cars_rp(tNet,fcoeffs=fcoeffs, rebalancing=False)
        G_final = sol[0].G_supergraph
    # Extract edge labels (flow values)
    for u, v in G_final.edges():
        G_final[u][v]['flow'] = round(G_final[u][v]['flow'])

    for k,v in tNet.q.items():
        if v != 0:
            print(f"{k}:{v}")
    print('\n')
    for k,v in tNet.qrp.items():
        if v != 0:
            print(f"{k}:{v}")

    normal_edges = []
    rp_edges = []
    edge_labels_1 = {}
    edge_labels_2 = {}

    flow = {(i,j):G_final[i][j]['flow'] for i,j in G_final.edges() if G_final[i][j]['type']==0}  # car flow
    rp_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flow'] for i,j in G_final.edges() if G_final[i][j]['type']=='rp'}  # 'rp' flow
    ped_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flow'] for i,j in G_final.edges() if G_final[i][j]['type']=='p'}
    f_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'f'}
    frp_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'f_rp'}
    fu_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'u'}
    print(f"f: {sum(f_flow.values())/2}")
    print(f"frp: {sum(frp_flow.values())/2}")
    print(f"fu: {sum(fu_flow.values())/2}")
    print(sum(tNet.q.values()))
    print(sum(tNet.qrp.values()))
    ped_labels_1 = {}
    ped_labels_2 = {}
    for i,j,t in G_final.edges(data=True):
        if t['type'] == 0:
            if i>j:
                if ped_flow[(i,j)] > 0:
                    ped_labels_1[(i,j)] = f"{i} to {j}\nP: {ped_flow[(i,j)]}"
            if i<j:
                if ped_flow[(i,j)] > 0:
                    ped_labels_2[(i,j)] = f"{i} to {j}\nP: {ped_flow[(i,j)]}"

    for i,j,t in G_final.edges(data=True):
        if t['type'] ==0:
            # Create label with both flow values (if applicable)
            if i>j:
                if flow[(i,j)] > 0 and rp_flow[(i,j)] > 0:
                    edge_labels_1[(i, j)] = f"{i} to {j}\nC: {flow[(i,j)]}\nRP: {rp_flow[(i,j)]}"
                elif flow[(i,j)] > 0:
                    edge_labels_1[(i, j)] = f"{i} to {j}\nC: {flow[(i,j)]}"
                elif rp_flow[(i,j)] > 0:
                    
                    edge_labels_1[(i, j)] = f"{i} to {j}\nRP: {rp_flow[(i,j)]}"
            if i<j:
                if flow[(i,j)] > 0 and rp_flow[(i,j)] > 0:
                    edge_labels_2[(i, j)] = f"{i} to {j}\nC: {flow[(i,j)]}\nRP: {rp_flow[(i,j)]}"
                elif flow[(i,j)] > 0:
                    edge_labels_2[(i, j)] = f"{i} to {j}\nC: {flow[(i,j)]}"
                elif rp_flow[(i,j)] > 0:
                    
                    edge_labels_2[(i, j)] = f"{i} to {j}\nRP: {rp_flow[(i,j)]}"


    ### road map
    pos = nx.nx_agraph.graphviz_layout(og_graph, prog='neato')
    # **Filter edges** where `type == 0`
    filtered_edges = [(u, v) for u, v, d in G_final.edges(data=True) if d['type'] == 0]
    # **Filter nodes** that are part of the filtered edges
    filtered_nodes = set([node for edge in filtered_edges for node in edge])
    # Draw nodes
    nx.draw_networkx_nodes(G_final, pos, nodelist=filtered_nodes, node_color='lightblue', node_size=100)
    # Draw normal flows (black)
    nx.draw_networkx_edges(G_final, pos, edgelist=filtered_edges, edge_color='black', width=2, arrows=True)
    # Draw edge labels with both normal and rp flows
    nx.draw_networkx_edge_labels(G_final, pos, edge_labels=edge_labels_1, font_size=5, verticalalignment='top')
    nx.draw_networkx_edge_labels(G_final, pos, edge_labels=edge_labels_2, font_size=5, verticalalignment='bottom')
    nx.draw_networkx_labels(G_final, pos, labels={node: node for node in filtered_nodes}, font_size=6, font_color='black')
    # Show the plot
    plt.show(block=False)

    ### pedestrian map
    plt.figure()
    # **Filter edges** where `type == 0`
    filtered_edges = [(u, v) for u, v, d in G_final.edges(data=True) if d['type'] == 0]
    # **Filter nodes** that are part of the filtered edges
    filtered_nodes = set([node for edge in filtered_edges for node in edge])

    # Draw nodes
    nx.draw_networkx_nodes(G_final, pos, nodelist=filtered_nodes, node_color='lightblue', node_size=100)

    # Draw normal flows (black)
    nx.draw_networkx_edges(G_final, pos, edgelist=filtered_edges, edge_color='black', width=2, arrows=True)
    nx.draw_networkx_labels(G_final, pos, labels={node: node for node in filtered_nodes}, font_size=6, font_color='black')
    # Draw edge labels with both normal and rp flows
    nx.draw_networkx_edge_labels(G_final, pos, edge_labels=ped_labels_1, font_size=6, verticalalignment='top')
    nx.draw_networkx_edge_labels(G_final, pos, edge_labels=ped_labels_2, font_size=6, verticalalignment='bottom')
    plt.show()