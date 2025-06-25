import src.tnet as tnet
import experiments.build_NYC_subway_net as nyc
import pickle

tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
#increase demand
tNet.set_g(tnet.perturbDemandConstant(tNet.g, constant=1))
# build supergraph based on original roadgraph
tNet.build_walking_supergraph()
#build other layers 
tNet.build_layer(one_way=False, avg_speed=6, symb="b")
#load subway
layer = tnet.readNetFile(netFile='data/net/NYC/NYC_M_Subway_net.txt')
tNet.add_layer(layer=layer, layer_symb='s')
# switch to smaller roadgraph fro rp layer
with open("data/gml/NYC_small_roadgraph.gpickle", 'rb') as f:
    G_roadgraph = pickle.load(f)
tNet.build_rp_layers(G_roadgraph)
# save
with open("data/gml/NYC.gpickle", "wb") as f:
    pickle.dump(tNet.G_supergraph, f)
