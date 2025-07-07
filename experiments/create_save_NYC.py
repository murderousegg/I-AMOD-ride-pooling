import src.tnet as tnet
import experiments.build_NYC_subway_net as nyc
import pickle

tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
#increase demand
tNet.set_g(tnet.perturbDemandConstant(tNet.g, constant=1))
# build supergraph based on original roadgraph
tNet.build_walking_supergraph()
#build other layers 
tNet.build_layer(one_way=False, avg_speed=13.5, symb="b")
#load subway
layer = tnet.readNetFile(netFile='data/net/NYC/NYC_M_Subway_net.txt')
tNet.add_layer(layer=layer, layer_symb='s')
# switch to smaller roadgraph fro rp layer
with open("data/gml/NYC_small_roadgraph.gpickle", 'rb') as f:
    G_roadgraph = pickle.load(f)
# reduce destination layer
with open("data/gml/NYC_small_demands.gpickle", 'rb') as f:
            small_g = pickle.load(f)
dest_layer = [u for u in tNet.G_supergraph.nodes() if "''" in str(u)] #dest layer
actual_dests = list(set([v for u,v in small_g.keys()]))   # dests from (pruned) demand
pruned_dest_nodes = list(set(dest_layer) - set(actual_dests))   # list of nodes to remove
tNet.G_supergraph.remove_nodes_from(pruned_dest_nodes)  # also removes connections to edges

#TODO: make V_RPd of same size as V_d
tNet.build_rp_layers(G_roadgraph, actual_dests)
# save
with open("data/gml/NYC.gpickle", "wb") as f:
    pickle.dump(tNet.G_supergraph, f)
