#########################################################################################
# This is a python implementation (converted from MATLAB) from the research paper       #
# "A Time-invariant Network Flow Model for Ride-pooling in Mobility-on-Demand Systems"  #
# Paper written by Fabio Paparella, Leonardo Pedroso, Theo Hofman, Mauro Salazar        #
# Python implementation by Frank Overbeeke                                              #
#########################################################################################

import numpy as np
import networkx as nx
from Utilities.RidePooling.LTIFM_reb import LTIFM_reb
from Utilities.RidePooling.calculate_gamma_k2 import calculate_gamma_k2
import matplotlib.pyplot as plt
import gc
import src.tnet as tnet
import matplotlib.pyplot as plt
import src.CARS as cars
import networkx as nx
import re
from gurobipy import *
from matplotlib import rc
import experiments.build_NYC_subway_net as nyc
import pickle
import pandas as pd
import time

rc('text', usetex=True)
plt.rc('axes', labelsize=13)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# import tracemalloc

# tracemalloc.start()

CITY_FOLDER = "NYC"
WAITINGTIME = 2/60
DELAY = 0.1
VEHICLE_LIM = 200000
tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
#increase demand
tNet.set_g(tnet.perturbDemandConstant(tNet.g, constant=1/24))

with open("data/gml/NYC_small_roadgraph.gpickle", 'rb') as f:
    tNet.G = pickle.load(f)

with open("data/gml/NYC.gpickle", 'rb') as f:
    tNet.G_supergraph = pickle.load(f)

# origins in walking layer
new_origins = {(f"{k[0]}'",f"{k[1]}'"): v for k, v in tNet.g.items()}
tNet.g = new_origins

ridepool = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'rp']
pedestrian = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'p']
connector = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'f']
connector_rp = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'frp']

node_order = list(tNet.G_supergraph.nodes())
edge_order = list(tNet.G_supergraph.edges())

car_node_order = list(tNet.G.nodes())
car_node_index_map = {node: i for i, node in enumerate(tNet.G.nodes())}
car_edge_order = list(tNet.G.edges())


Binc = nx.incidence_matrix(tNet.G_supergraph, nodelist=node_order, edgelist=edge_order, oriented=True)
[N_nodes,N_edges] = Binc.shape
node_index_map = {node: i for i, node in enumerate(tNet.G_supergraph.nodes())}

Binc_road = nx.incidence_matrix(tNet.G)
[N_nodes_road,N_edges_road] = Binc_road.shape


def compute_results(FullList, Delay, WaitingTime, Demand, fcoeffs):
    TotGamma2 = 0
    Cumul_delay2 = 0
    OriginalDemand = Demand.copy()
    DemandS =  Demand.copy()
    Demands_rp = np.zeros([N_nodes_road,N_nodes_road])
    gamma_arr = np.zeros(FullList.shape[0], dtype=np.float16)
    for iii in range(FullList.shape[0]):
        # Seperate function to remove clutter
        Cumul_delay2, TotGamma2, DemandS, Demands_rp, gamma =\
              calculate_gamma_k2(FullList,DemandS, Delay, N_nodes_road, WaitingTime, Cumul_delay2, TotGamma2, iii, Demands_rp, car_node_index_map)
        gamma_arr[iii] = gamma
    
    Demands_rp = Demands_rp - np.diag(np.diag(Demands_rp))
    print(f"pooled demand: {Demands_rp.sum()}")
    print(f"solo demand: {DemandS.sum()}")
    #calculate solution
    # solBase =LTIFM_reb(OriginalDemand,tNet.G, fcoeffs=fcoeffs)
    solNP = LTIFM_reb(DemandS,tNet.G, fcoeffs=fcoeffs)
    solRP = LTIFM_reb(Demands_rp,tNet.G, fcoeffs=fcoeffs)
    y, yr = solRP["x"]+solNP["x"], solRP["xr"]+solNP["xr"]

    #reset diagonals (these are modified in LTIFM_reb)
    DemandS = DemandS - np.diag(np.diag(DemandS))   
    Demands_rp = Demands_rp - np.diag(np.diag(Demands_rp))
    TrackDems_temp = [np.sum(OriginalDemand), np.sum(DemandS),np.sum(Demands_rp)]
    TotGamma = [TotGamma2]
    return y, yr, TrackDems_temp, TotGamma, gamma_arr

def save_results(results, dir_out):
    labels = ['ped_flow', 'b_flow', 'solo_ride', 'shared_ride', 'tot_reb_flow', 'tot_iamod_costs', 'tot_cars']
    df = pd.DataFrame(list(map(list, zip(*results))), columns=labels)
    df.to_csv(dir_out + '/results_'+time.strftime("%Y-%m-%d_%H_%M_%S")+'.csv')

def main():
    tot_rp_flow = []
    tot_frp_flow = []
    tot_ped_flow = []
    tot_b_flow = []
    tot_reb_flow = []
    single_ride = []
    double_ride = []
    triple_ride = []
    tot_iamod_costs = []
    tot_cars = []
    max_iter = 21
    r=2
    #### adaptive mu
    mu=0.0001
    prev_x = prev2_x = prev_obj = prev2_obj = None
    reb_cars = 0
    ## cars estimation adjustment
    c_ratio = 1.0 
    rho_cars = 0.5
    print(f"Total demand: {sum(tNet.g.values())}")
    for i in range(max_iter):
        if i == 2:
            mu = 0.0001
            
        _,_,avg_time,x,expected_cars, obj = cars.solve_bush_CARSn_bundled(tnet=tNet, fcoeffs=fcoeffs, iteration=i, unf=False,rebalancing=False, r=r, mu=mu, prev_x=prev_x, reb_cars=reb_cars, c_ratio=c_ratio, vehicle_lim=VEHICLE_LIM)
        # x based mu
        # if prev_x is not None and prev2_x is not None:
        #     s_num = np.linalg.norm(np.array(x)-np.array(prev_x))
        #     s_den = np.linalg.norm(np.array(prev_x)-np.array(prev2_x))
        #     s = s_num / (s_den+1e-12)
        #     if   s > 0.9:  mu = min(mu*2, 10.0)
        #     elif s < 0.6:  mu = max(mu*0.5, 1e-8)
        # obj based mu
        if prev_obj is not None and prev2_obj is not None:
            s_num = abs(obj-prev_obj)
            s_den = abs(prev_obj - prev2_obj)
            s = s_num / (s_den+1e-12)
            if   s > 0.7:  mu = min(mu*2, 10.0)
            elif s < 0.4:  mu = max(mu*0.5, 1e-5)
        prev2_x = prev_x
        prev_x  = x
        prev2_obj = prev_obj
        prev_obj  = obj
        print(f"mu = {mu}")

        # find D_rp, match edges to nodes in tNet.G
        D_rp = np.zeros((N_nodes_road, N_nodes_road))
        for count, (u,v) in enumerate(ridepool):
            D_rp[car_node_index_map[int(v[:-2])], car_node_index_map[int(u[:-2])]] = tNet.G_supergraph[u][v]['flowNoRebalancing']
        print(f"ridepooling demand: {D_rp.sum()}")
        print(f"iteration: {i+1}")
        with np.load('NYC/MatL2.npz') as data:
            FullList = data[data.files[0]].astype(np.float32)
        y, yr, TrackDems, TotG, gamma_arr = compute_results(FullList=FullList, Delay=DELAY, WaitingTime=WAITINGTIME, Demand=D_rp, fcoeffs=fcoeffs)
        reb_cars = 0
        total_cars = 0
        for k in range(N_edges_road):
            u, v = list(tNet.G.edges())[k]
            t0 = tNet.G[u][v]['t_0']
            capacity = tNet.G[u][v]['capacity']
            # Get flow on arc (u,v)
            flow_ij = sum([y[k, j] for j in range(N_nodes_road)]) + yr[k]
            # Compute congestion ratio
            ratio = flow_ij / (capacity)
            # Apply BPR function
            adjusted_tij = t0 * (1 + 0.15 * ratio**4)
            tNet.G[u][v]['t_1'] = adjusted_tij
            reb_cars += yr[k]*adjusted_tij
            total_cars += flow_ij*adjusted_tij
            
        print(f"\nrebalancing cars: {reb_cars}")
        print(f"expected cars: {expected_cars}")
        print(f"Total number of cars: {total_cars}\n")
        print(f"obj: {prev_obj}")
        print(f"obj_prev: {prev2_obj}")        
        ### adaptive decay mu
        if prev2_x is not None:
            if np.linalg.norm(np.array(prev_x)-np.array(prev2_x)) < 1 and total_cars-10<VEHICLE_LIM:
                print(f"convergence achieved through x after {i+1} iterations")
                stable_obj+=1
            elif abs(prev_obj-prev2_obj) < 10 and total_cars-10<VEHICLE_LIM:
                print(f"convergence achieved though obj after {i+1} iterations")
                stable_obj+=1
            else:
                stable_obj=0
        if stable_obj == 3:
            break
        # update cars estimation
        ratio  = total_cars / expected_cars
        c_ratio = rho_cars * c_ratio + (1-rho_cars) * ratio
        print(f"c_ratio: {c_ratio}")
        #### take delays and waitingtime into account for t_1 ####
        OD_delays = np.zeros((D_rp.shape)) # expected delays
        Et = np.zeros((D_rp.shape)) # expected waiting time
        gamma_count = np.zeros((D_rp.shape))
        for iii in range(len(FullList)):
            if gamma_arr[iii] != 0:
                jj1 = car_node_index_map[int(FullList[iii][3])]
                ii1 = car_node_index_map[int(FullList[iii][4])]
                jj2 = car_node_index_map[int(FullList[iii][5])]
                ii2 = car_node_index_map[int(FullList[iii][6])]
                if np.array_equal(FullList[iii][7:11], [1, 2, 1, 2]):
                    OD_delays[ii1][jj1] += gamma_arr[iii] *(nx.shortest_path_length(tNet.G, source=car_node_order[ii1], target=car_node_order[ii2], weight='t_1')\
                                                        + nx.shortest_path_length(tNet.G, source=car_node_order[ii2], target=car_node_order[jj1], weight='t_1')\
                                                            - nx.shortest_path_length(tNet.G, source=car_node_order[ii1], target=car_node_order[jj1], weight='t_1'))
                    OD_delays[ii2][jj2] += gamma_arr[iii] *(nx.shortest_path_length(tNet.G, source=car_node_order[ii2], target=car_node_order[jj1], weight='t_1')\
                                                        + nx.shortest_path_length(tNet.G, source=car_node_order[jj1], target=car_node_order[jj2], weight='t_1')\
                                                            - nx.shortest_path_length(tNet.G, source=car_node_order[ii2], target=car_node_order[jj2], weight='t_1'))

                elif np.array_equal(FullList[iii][7:11], [1, 2, 2, 1]):
                    OD_delays[ii1][jj1] += gamma_arr[iii] * (nx.shortest_path_length(tNet.G, source=car_node_order[ii1], target=car_node_order[ii2], weight='t_1')\
                                                        + nx.shortest_path_length(tNet.G, source=car_node_order[ii2], target=car_node_order[jj2], weight='t_1')\
                                                            + nx.shortest_path_length(tNet.G, source=car_node_order[jj2], target=car_node_order[jj1], weight='t_1')\
                                                                - nx.shortest_path_length(tNet.G, source=car_node_order[ii1], target=car_node_order[jj1], weight='t_1'))
                # store occurance of sequence
                gamma_count[ii1][jj1] += gamma_arr[iii]
                gamma_count[ii2][jj2] += gamma_arr[iii]
                #expected waiting time
                
                Et[ii1][jj1] += gamma_arr[iii] * (1/D_rp[ii1][jj1]+1/D_rp[ii2][jj2] - 2/(D_rp[ii1][jj1]+D_rp[ii2][jj2]))/2
                Et[ii2][jj2] += gamma_arr[iii] * (1/D_rp[ii1][jj1]+1/D_rp[ii2][jj2] - 2/(D_rp[ii1][jj1]+D_rp[ii2][jj2]))/2
        del FullList, gamma_arr
        gc.collect()

        # weighted delays / total requests
        total_delay = np.divide(OD_delays+Et, D_rp, out=np.zeros_like(D_rp), where=D_rp!=0)
        
        for u,v,d in tNet.G_supergraph.edges(data=True):
            if d['type'] == 'rp':
                tNet.G_supergraph[u][v]['t_1'] = nx.shortest_path_length(tNet.G, source=int(u[:-2]), target=int(v[:-2]), weight='t_1') + total_delay[car_node_index_map[int(v[:-2])], car_node_index_map[int(u[:-2])]]
                tNet.G_supergraph[u][v]['t_cars'] = nx.shortest_path_length(tNet.G, source=int(u[:-2]), target=int(v[:-2]), weight='t_1')
            else:
                tNet.G_supergraph[u][v]['t_1'] = tNet.G_supergraph[u][v]['t_0']
        for u,v in tNet.G.edges():
            if i >= 1:
                tNet.G[u][v]['t_3'] = tNet.G[u][v]['t_2']
            tNet.G[u][v]['t_2'] = tNet.G[u][v]['t_1']
            
        
        rp_flow = {(k,j): tNet.G_supergraph[k][j]['flowNoRebalancing'] for k,j in tNet.G_supergraph.edges() if tNet.G_supergraph[k][j]['type']=='rp'}  # 'rp' flow
        frp_flow = {(k,j): tNet.G_supergraph[k][j]['flowNoRebalancing'] for k,j in tNet.G_supergraph.edges() if tNet.G_supergraph[k][j]['type']=='frp'}  # 'rp' flow
        ped_flow = {(k,j): tNet.G_supergraph[k][j]['flowNoRebalancing'] for k,j in tNet.G_supergraph.edges() if tNet.G_supergraph[k][j]['type']=='p'}
        b_flow = {(k,j): tNet.G_supergraph[k][j]['flowNoRebalancing'] for k,j in tNet.G_supergraph.edges() if tNet.G_supergraph[k][j]['type']=='b'}
        
        # this_rp_flow = [rp_flow[k] * tNet.G_supergraph[k[0]][k[1]]['t_1'] for k in rp_flow.keys()]
        this_rp_flow = [rp_flow[k] * tNet.G_supergraph[k[0]][k[1]]['t_1'] for k in rp_flow.keys()]
        this_rp_flow = sum(this_rp_flow)
        tot_rp_flow.append(this_rp_flow)
        # print(f"sum of t_0: {sum([tNet.G_supergraph[k[0]][k[1]]['t_0'] for k in rp_flow.keys()])}")
        # print(f"sum of t_1: {sum([tNet.G_supergraph[k[0]][k[1]]['t_1'] for k in rp_flow.keys()])}")
        
        this_frp_flow = [frp_flow[k] * tNet.G_supergraph[k[0]][k[1]]['t_1'] for k in frp_flow.keys()]
        this_frp_flow = sum(this_frp_flow)
        tot_frp_flow.append(this_frp_flow)

        # this_ped_flow = [ped_flow[k] * tNet.G_supergraph[k[0]][k[1]]['t_1'] for k in ped_flow.keys()]
        this_ped_flow = [ped_flow[k] * tNet.G_supergraph[k[0]][k[1]]['t_1'] for k in ped_flow.keys()]
        this_ped_flow = sum(this_ped_flow)
        tot_ped_flow.append(this_ped_flow)

        # this_b_flow = [b_flow[k] * tNet.G_supergraph[k[0]][k[1]]['t_1'] for k in b_flow.keys()]
        this_b_flow = [b_flow[k] * tNet.G_supergraph[k[0]][k[1]]['t_1'] for k in b_flow.keys()]
        this_b_flow = sum(this_b_flow)
        tot_b_flow.append(this_b_flow)

        tot_cars.append(total_cars)
        tot_reb_flow.append(reb_cars)
        tot_iamod_costs.append(sum([tNet.G_supergraph[i][j]['flowNoRebalancing'] * tNet.G_supergraph[i][j]['t_1'] for i,j in tNet.G_supergraph.edges()]))

        PercNRP = TrackDems[1]/(TrackDems[2] + TrackDems[1])
        # Normalize TotG row-wise and scale it by (1 - PercNRP)
        TotG_normalized = TotG / np.sum(TotG, axis=0, keepdims=True)  # Normalize rows of TotG
        scaled_TotG = TotG_normalized * (1 - PercNRP)  # Scale by (1 - PercNRP)
        # Combine PercNRP and the scaled TotG into a new array
        total_PercNRP = np.hstack([PercNRP, scaled_TotG])
        single_ride.append(total_PercNRP[0])
        double_ride.append(total_PercNRP[1])
        triple_ride.append(total_PercNRP[2])
        #update ridepooling percentage
        r = 1 + total_PercNRP[1]/(total_PercNRP[0]+total_PercNRP[1])
        print(f"r = {r}")
        if i%5==0:
            G_final = tNet.G_supergraph
            

            # Extract edge labels (flow values)
            normal_edges = []
            rp_edges = []
            edge_labels_1 = {}
            edge_labels_2 = {}

            flow = {(k,j):G_final[k][j]['flowNoRebalancing'] for k,j in G_final.edges() if G_final[k][j]['type']==0}  # car flow
            rp_flow = {(int(re.search(r'\d+', k).group()),int(re.search(r'\d+', j).group())): G_final[k][j]['flowNoRebalancing'] for k,j in G_final.edges() if G_final[k][j]['type']=='rp'}  # 'rp' flow
            ped_flow = {(int(re.search(r'\d+', k).group()),int(re.search(r'\d+', j).group())): G_final[k][j]['flowNoRebalancing'] for k,j in G_final.edges() if G_final[k][j]['type']=='p'}
            b_flow = {(int(re.search(r'\d+', k).group()),int(re.search(r'\d+', j).group())): G_final[k][j]['flowNoRebalancing'] for k,j in G_final.edges() if G_final[k][j]['type']=='b'}
            f_flow = {(k,j): G_final[k][j]['flowNoRebalancing'] for k,j,d in G_final.edges(data=True) if d['type'] == 'f'}
            frp_flow = {(k,j): G_final[k][j]['flowNoRebalancing'] for k,j,d in G_final.edges(data=True) if d['type'] == 'frp'}
            fu_flow = {(k,j): G_final[k][j]['flowNoRebalancing'] for k,j,d in G_final.edges(data=True) if d['type'] == 'u'}
            
            plt.figure()
            average_all = np.array([G_final[u][v]['flowNoRebalancing'] * G_final[u][v]['t_1'] for u,v in G_final.edges()]).sum()/np.array([k for k in tNet.g.values()]).sum()
            plt.hist(avg_time,bins=10 , color='c', edgecolor='k', alpha=0.65)
            plt.axvline(average_all, color='k', linestyle='dashed', linewidth=1)
            plt.axvline(0.5, color='r', linestyle='solid', linewidth=1)
            plt.savefig(f"fairness_{i}_avg_time_dist.png")
            # plt.show(block=False)
        
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)

    print(tot_rp_flow)
    print(tot_frp_flow)
    print(tot_ped_flow)
    print(tot_b_flow)
    # Create bar plot
    plt.figure()
    plt.bar(np.arange(len(tot_rp_flow)), [tot_rp_flow[k] * single_ride[k] for k in range(len(tot_rp_flow))], label='rp flow 1 user')
    plt.bar(np.arange(len(tot_rp_flow)), [tot_rp_flow[k] * double_ride[k] for k in range(len(tot_rp_flow))], bottom=[tot_rp_flow[k] * single_ride[k] for k in range(len(tot_rp_flow))], label='rp flow 2 users')
    if triple_ride[-1]>0.1:
        plt.bar(np.arange(len(tot_rp_flow)), [tot_rp_flow[k] * triple_ride[k] for k in range(len(tot_rp_flow))], bottom=[tot_rp_flow[k] * (single_ride[k]+ double_ride[k]) for k in range(len(tot_rp_flow))], label='rp flow 3 users')
    plt.bar(np.arange(len(tot_rp_flow)), tot_ped_flow, bottom=tot_rp_flow, label='ped flow')
    plt.bar(np.arange(len(tot_rp_flow)), tot_b_flow, bottom=np.array(tot_rp_flow) + np.array(tot_ped_flow), label='bike flow')

    # Customize plot
    plt.xlabel('Iteration')
    plt.ylabel('Total time spent in mode of transport')
    plt.title('Evolution of mode allocation over time')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()


    G_final = tNet.G_supergraph

    normal_edges = []
    rp_edges = []
    edge_labels_1 = {}
    edge_labels_2 = {}

    flow = {(i,j):G_final[i][j]['flowNoRebalancing'] for i,j in G_final.edges() if G_final[i][j]['type']==0}  # car flow
    rp_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flowNoRebalancing'] for i,j in G_final.edges() if G_final[i][j]['type']=='rp'}  # 'rp' flow
    ped_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flowNoRebalancing'] for i,j in G_final.edges() if G_final[i][j]['type']=='p'}
    b_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flowNoRebalancing'] for i,j in G_final.edges() if G_final[i][j]['type']=='b'}
    f_flow = {(i,j): G_final[i][j]['flowNoRebalancing'] for i,j,d in G_final.edges(data=True) if d['type'] == 'f'}
    frp_flow = {(i,j): G_final[i][j]['flowNoRebalancing'] for i,j,d in G_final.edges(data=True) if d['type'] == 'frp'}
    fu_flow = {(i,j): G_final[i][j]['flowNoRebalancing'] for i,j,d in G_final.edges(data=True) if d['type'] == 'u'}
    print(f"f: {sum(f_flow.values())/2}")
    print(f"frp: {sum(frp_flow.values())/2}")
    print(f"fu: {sum(fu_flow.values())/2}")
    solo_ride = [tot_rp_flow[k] * single_ride[k] for k in range(len(tot_rp_flow))]
    shared_ride = [tot_rp_flow[k] * double_ride[k] for k in range(len(tot_rp_flow))]

    save_results((ped_flow, b_flow, solo_ride, shared_ride, tot_reb_flow, tot_iamod_costs, tot_cars), "NYC/results")

    

if __name__ == "__main__":
    main()
