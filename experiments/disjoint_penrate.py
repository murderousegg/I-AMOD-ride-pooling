#########################################################################################
# This is a python implementation (converted from MATLAB) from the research paper       #
# "A Time-invariant Network Flow Model for Ride-pooling in Mobility-on-Demand Systems"  #
# Paper written by Fabio Paparella, Leonardo Pedroso, Theo Hofman, Mauro Salazar        #
# Python implementation by Frank Overbeeke                                              #
#########################################################################################

import numpy as np
import networkx as nx
from src.rp_utils import *
import copy
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import src.tnet as tnet
import matplotlib.pyplot as plt
import src.CARS as cars
import networkx as nx
import re
import copy
from gurobipy import *
import time
from matplotlib import rc
import pandas as pd
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('axes', labelsize=13)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# tracemalloc.start()
CITY_FOLDER = "NYC_Uber_small"
WAITINGTIME = 2/60
DELAY = 0.1
VEHICLE_LIM = 1500
DEMAND_MULT = 1.5
netFile, gFile, fcoeffs,_,_ = tnet.get_network_parameters(CITY_FOLDER)


tNet_cars = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
tNet_private = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)

#build other layers
tNet.build_walking_supergraph()
tNet.build_layer(one_way=False, avg_speed=6, symb="b")
tNet.build_full_layer() # fully connected rp layer

# origins in walking layer
new_origins = {(f"{k[0]}'",f"{k[1]}'"): v for k, v in tNet.g.items()}
tNet.g = new_origins
iamod_g = copy.deepcopy(tNet.g)
private_g = copy.deepcopy(tNet_private.g)



# edge masks
ridepool = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'rp']
bike = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'b']
pedestrian = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'p']
connector = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'f']
connector_rp = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'frp']
edge_order = list(tNet.G_supergraph.edges())
car_edge_order = list(tNet_cars.G.edges())
# node masks
node_order = list(tNet.G_supergraph.nodes())
car_node_order = list(tNet_cars.G.nodes())
car_node_index_map = {node: i for i, node in enumerate(tNet_cars.G.nodes())}

# IAMoD information
Binc = nx.incidence_matrix(tNet.G_supergraph, nodelist=node_order, edgelist=edge_order, oriented=True)
[N_nodes,N_edges] = Binc.shape
node_index_map = {node: i for i, node in enumerate(tNet.G_supergraph.nodes())}
Binc_road = nx.incidence_matrix(tNet_cars.G)
[N_nodes_road,N_edges_road] = Binc_road.shape

def initialize_networks(pen_rate):
    tNet.set_g(tnet.perturbDemandConstant(iamod_g, constant=DEMAND_MULT*pen_rate/24))
    tNet_private.set_g(tnet.perturbDemandConstant(private_g, constant=DEMAND_MULT*(1-pen_rate)/24))

def IAMoD_loop(FullList, exogenous_G=False, pen_rate=1):
    max_iter = 21
    #### adaptive mu
    mu=0
    prev_x = None
    prev2_x = None
    prev_obj = None
    prev2_obj = None
    reb_cars = 0
    ## cars estimation adjustment
    c_ratio = 1.0 
    rho_cars = 0.5
    r=2
    now = time.time()
    for iamod_iter in range(max_iter):
        # x_rp_mat = cars.solve_matrix_base(tnet=tNet,fcoeffs=fcoeffs, times=i)
        if iamod_iter == 2:
            mu = 0.0001
        _,_,_,x,expected_cars, obj = cars.solve_bush_CARSn(tnet=tNet, fcoeffs=fcoeffs, iteration=iamod_iter, unf=False, r=r, mu=mu, prev_x=prev_x, reb_cars=reb_cars, c_ratio=c_ratio, vehicle_lim=VEHICLE_LIM*pen_rate)

        # # x based mu
        # if prev_x is not None and prev2_x is not None:
        #     s_num = np.linalg.norm(np.array(x)-np.array(prev_x))
        #     s_den = np.linalg.norm(np.array(prev_x)-np.array(prev2_x))
        #     s = s_num / s_den if s_den>0 else 0
        #     if   s > 0.9:  mu = min(mu*2, 10.0)
        #     elif s < 0.6:  mu = max(mu*0.5, 1e-5)
        
        # obj based mu
        if prev_obj is not None and prev2_obj is not None:
            s_num = abs(obj-prev_obj)
            s_den = abs(prev_obj - prev2_obj)
            s = s_num / (s_den+1e-12)
            if   s > 0.7:  mu = min(mu*2, 10.0)
            elif s < 0.4:  mu = max(mu*0.5, 1e-5)
        # update solution storage    
        prev2_x = prev_x
        prev_x  = x
        prev2_obj = prev_obj
        prev_obj  = obj
        # find D_rp, match edges to nodes in tNet.G
        D_rp = np.zeros((N_nodes_road, N_nodes_road))
        for count, (u,v) in enumerate(ridepool):
            # D_rp[car_node_index_map[int(v[:-2])], car_node_index_map[int(u[:-2])]] = sum(x_rp_mat[count])
            D_rp[car_node_index_map[int(v[:-2])], car_node_index_map[int(u[:-2])]] = tNet.G_supergraph[u][v]['flowNoRebalancing']

        y, yr, TrackDems, TotG, gamma_arr = compute_results(FullList=FullList, Delay=DELAY, WaitingTime=WAITINGTIME,\
                                                            Demand=D_rp, tNet=tNet_cars, N_nodes=N_nodes_road, fcoeffs=fcoeffs,\
                                                                idx_map=car_node_index_map)
        reb_cars = 0
        total_cars = 0
        for k in range(N_edges_road):
            u, v = list(tNet_cars.G.edges())[k]
            t0 = tNet_cars.G[u][v]['t_0']
            capacity = tNet_cars.G[u][v]['capacity']
            # Get flow on arc (u,v)
            flow_ij = sum([y[k, j] for j in range(N_nodes_road)]) + yr[k]
            if exogenous_G:
                exo_flow_ij = exogenous_G[u][v]['flow']
            else:
                exo_flow_ij = 0
            # Compute congestion ratio
            ratio = (flow_ij + exo_flow_ij) / (capacity)
            # Apply BPR function
            adjusted_tij = t0 * (1 + 0.15 * ratio**4)
            tNet_cars.G[u][v]['t_1'] = adjusted_tij
            reb_cars += yr[k]*adjusted_tij
            total_cars += flow_ij*adjusted_tij
            tNet_cars.G[u][v]['flow'] = flow_ij
            tNet_cars.G[u][v]['flowRebalancing'] = yr[k]

            
        # print(f"\nrebalancing cars: {reb_cars}")
        # print(expected_cars)
        # print(f"Total number of cars: {total_cars}\n")
        ### adaptive decay mu
        if prev2_x is not None:
            if np.linalg.norm(np.array(prev_x)-np.array(prev2_x)) < 1 and total_cars-10<VEHICLE_LIM:
                print(f"convergence achieved after {iamod_iter+1} iterations")
                break
        if prev2_obj is not None:
            if abs(prev_obj-prev2_obj) < 10 and total_cars-10<VEHICLE_LIM:
                print(f"convergence achieved after {iamod_iter+1} iterations")
                break
        # update cars estimation
        ratio  = total_cars / expected_cars
        c_ratio = rho_cars * c_ratio + (1-rho_cars) * ratio
        OD_delay, Et = update_t(tNet, tNet_cars, D_rp, FullList, gamma_arr, car_node_index_map, car_node_order)
        
        PercNRP = TrackDems[1]/(TrackDems[2] + TrackDems[1])
        # Normalize TotG row-wise and scale it by (1 - PercNRP)
        TotG_normalized = TotG / np.sum(TotG, axis=0, keepdims=True)  # Normalize rows of TotG
        scaled_TotG = TotG_normalized * (1 - PercNRP)  # Scale by (1 - PercNRP)
        # Combine PercNRP and the scaled TotG into a new array
        total_PercNRP = np.hstack([PercNRP, scaled_TotG])
        r = 1 + total_PercNRP[1]/(total_PercNRP[0]+total_PercNRP[1])
        if TrackDems[1]==0 or TrackDems[1] == np.nan:   #case no demand, continue anyway
            r= 1.5
        print(f"IAMoD iteration {iamod_iter+1}: {time.time()-now}")
        now=time.time()
        
    return total_PercNRP

def save_results(results, dir_out):
    labels = ['pedFlow', 'bikeFlow', 'solo_ride', 'shared_ride', 'privateFlow', 'rebFlow', 'IAMoDCosts', 'privateCosts', 'totCost']
    df = pd.DataFrame(list(map(list, zip(*results))), columns=labels)
    df.to_csv(dir_out + '/results_'+time.strftime("%Y-%m-%d_%H_%M_%S")+'.csv')

def main():
    ### run only once ###
    # create costs matrix
    # solPart = A1_SP(N_nodes_road, tNet_cars, car_node_order, CITY_FOLDER)
    # sol2_LC = A2_LinearComb2(solPart, N_nodes_road, car_node_order, CITY_FOLDER)
    ###
    FullList = Generate_Full_List(CITY_FOLDER)

    IAMoDCosts = []
    privateCosts = []
    totCost = []
    solo_ride = []
    shared_ride = []
    privateFlow = []
    pedFlow = []
    bikeFlow = []
    rebFlow = []

    for pen_rate in np.linspace(0.01,0.99, 10):
        initialize_networks(pen_rate)
        for stackelberg in range(10):
            #### solve I-AMoD
            if stackelberg==0:
                total_PercNRP = IAMoD_loop(FullList=FullList, pen_rate=pen_rate)
            else:
                total_PercNRP = IAMoD_loop(FullList=FullList, exogenous_G=tNet_private.G, pen_rate=pen_rate)
            tNet_private.solveMSA(exogenous_G=tNet_cars.G, verbose=0)   #set verbose 1 for console prints
        ### append flows in user travel time       
        pedFlow.append(sum([tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_0'] for i,j in pedestrian]))
        bikeFlow.append(sum([tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_0'] for i,j in bike]))
        solo_ride.append(total_PercNRP[0]*sum([tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_1'] for i,j in ridepool]))
        shared_ride.append(total_PercNRP[1]*sum([tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_1'] for i,j in ridepool]))
        privateFlow.append(sum([tNet_private.G[i][j]['flow'] * tNet_cars.G[i][j]['t_1'] for i,j in tNet_private.G.edges()]))
        rebFlow.append(sum([tNet_cars.G[i][j]['flowRebalancing'] * tNet_cars.G[i][j]['t_1'] for i,j in tNet_cars.G.edges()]))
        IAMoDFlow = sum([tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_1'] for i,j in tNet.G_supergraph.edges()])
        ### append costs
        IAMoDCosts.append(IAMoDFlow/tNet.totalDemand)
        privateCosts.append(privateFlow[-1]/tNet_private.totalDemand)
        totCost.append((IAMoDFlow+privateFlow[-1])/(tNet.totalDemand+tNet_private.totalDemand))  #TODO: calculate the cost with rebalancing
        print(f"penetration rate: {pen_rate}")



    if os.path.isdir('results/penetration') == False:
        os.mkdir('results/penetration')
    print("\n")
    print(IAMoDFlow)
    print(privateFlow)
    print(tNet.totalDemand)
    print(tNet_private.totalDemand)
    print(totCost)
    print(IAMoDCosts)
    print(privateCosts)
    
    plt.figure()
    plt.plot(list(np.linspace(0.01,0.99, len(totCost))), totCost, label='Average all users', marker='o')
    plt.plot(list(np.linspace(0.01,0.99, len(totCost))), IAMoDCosts, label='I-AMoD', marker='^')
    plt.plot(list(np.linspace(0.01,0.99, len(totCost))), privateCosts, label='Private', marker= 'x')
    plt.legend()
    plt.xlabel('Penetration Rate')
    plt.ylabel('Avg. Travel Time [hrs]')
    plt.grid(True, alpha=0.5)
    plt.xlim([0,1])
    plt.tight_layout()
    plt.savefig('results/' + "penetration" +'/costs.png', dpi=300)

    plt.figure()
    width = 0.05
    ind = list(np.linspace(0.01,0.99, len(totCost)))
    p1 = plt.bar(ind, privateFlow, width)
    p2 = plt.bar(ind, solo_ride, width,
                bottom=privateFlow)
    p3 = plt.bar(ind, shared_ride, width,
                 bottom=[x + y for x, y in zip(privateFlow, solo_ride)])
    p4 = plt.bar(ind, bikeFlow, width,
                bottom=[x+y+z for x,y,z in zip(privateFlow, solo_ride, shared_ride)])
    p5 = plt.bar(ind, pedFlow, width,
                bottom=[x+y+z+i for x,y,z,i in zip(privateFlow, solo_ride, shared_ride, bikeFlow)])
    plt.ylabel('Flow')
    plt.xlabel('Penetration rate', fontsize=10)
    plt.legend((p1[0], p2[0], p3[0], p4[0],p5[0]), ('Private', 'Solo AMoD', 'Shared AMoD', 'Bike', 'Pedestrian'), fontsize=12)
    plt.xlim([0,1])
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.5)
    plt.savefig('results/' + "penetration" +'/modal_share.png', dpi=300)
    plt.show()
    tstamp = time.strftime("%Y-%m-%d_%H_%M_%S")
    dir_out = 'results/penetration'
    save_results(results=(pedFlow, bikeFlow, solo_ride, shared_ride, privateFlow, rebFlow, IAMoDCosts, privateCosts, totCost), dir_out=dir_out)

    
    

if __name__ == "__main__":
    main()
