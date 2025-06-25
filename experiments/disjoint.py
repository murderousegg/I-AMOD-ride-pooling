#########################################################################################
# This is a python implementation (converted from MATLAB) from the research paper       #
# "A Time-invariant Network Flow Model for Ride-pooling in Mobility-on-Demand Systems"  #
# Paper written by Fabio Paparella, Leonardo Pedroso, Theo Hofman, Mauro Salazar        #
# Python implementation by Frank Overbeeke                                              #
#########################################################################################

import numpy as np
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import io
from scipy import sparse
import networkx as nx
from Utilities.RidePooling.LTIFM2_SP import LTIFM2_SP
from Utilities.RidePooling.LTIFM3_SP import LTIFM3_SP
from Utilities.RidePooling.LTIFM4_SP import LTIFM4_SP
from Utilities.RidePooling.LTIFM_reb import LTIFM_reb, LTIFM_reb_damp
from Utilities.RidePooling.calculate_gamma import calculate_gamma
from Utilities.RidePooling.probcomb import probcombN
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py
import time

import src.tnet as tnet
import matplotlib.pyplot as plt
import src.CARS as cars
import networkx as nx
import re
import copy
from gurobipy import *
from matplotlib import rc
rc('text', usetex=True)
plt.rc('axes', labelsize=13)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# import tracemalloc

# tracemalloc.start()

CITY_FOLDER = "NYC_Uber_small"
WAITINGTIME = 2/60
DELAY = 0.1
VEHICLE_LIM = 800
netFile, gFile, fcoeffs,_,_ = tnet.get_network_parameters(CITY_FOLDER)

tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
tNetExog = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)

#increase demand
tNet.set_g(tnet.perturbDemandConstant(tNet.g, constant=1.5/24))
tNet.build_walking_supergraph()
og_graph = tNet.G_supergraph.copy()
#integer demands:
tNet.g = {key: int(round(value)) for key, value in tNet.g.items()}

pos = nx.nx_agraph.graphviz_layout(tNet.G_supergraph, prog='neato')

# origins in walking layer
new_origins = {(f"{k[0]}'",f"{k[1]}'"): v for k, v in tNet.g.items()}
tNet.g = new_origins

#build other layers
tNet.build_layer(one_way=False, avg_speed=6, symb="b")
tNet.build_full_layer()
pos = nx.nx_agraph.graphviz_layout(tNet.G_supergraph, prog='neato')

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


def A1_SP():
    """
    Generates a matrix of dictionaries containing the costs of an arc, 
    a sparse matrix with the possible arcs and the times between destinations
    """
    solPart = np.empty((N_nodes_road, N_nodes_road), dtype=object)
    for ii in range(0,N_nodes_road):
        for jj in range(0,N_nodes_road):
            Dems = np.zeros([N_nodes_road, N_nodes_road])
            #generate demand matrix
            Dems[ii,jj] = 1
            Dems[jj,jj] = -1

            if ii==jj:  #special case, set demand to 0 if origin and destination are equal
                Dems[jj,jj] = -0
            
            D = nx.shortest_path_length(tNet.G, source=car_node_order[jj], target=car_node_order[ii], weight="t_0") 
            # Store the results in solPart[jj][ii] as a dictionary
            solPart[jj, ii] = {
                'obj': D,
                'Dem': sparse.csr_matrix(Dems),  # Convert Dems to sparse format
                'IndividualTimes': D
            }
    # convert to dictionary for .mat convertion
    solPart_dic = {"solPart": solPart}
    io.savemat(CITY_FOLDER + "/solPart_pyth_" + CITY_FOLDER+ ".mat", solPart_dic)
    return solPart

def compute_LinearComb2_for_jj1(jj1, N_nodes, solPart):
    """
    This function computes the minimum cost combinations for travel paths within a transportation network.
    This can be wrapped by a parallel processing toolbox such as joblib.
    
    Parameters:
    - jj1 (int): An index representing the starting node in one of the loop structures.
    - N_nodes (int): The total number of nodes in the network.
    - DemandS (ndarray): A matrix where DemandS[ii, jj] > 0 indicates a demand for travel from node ii to node jj.
    - solPart (dict): A dictionary containing precomputed shortest paths and related data for node pairs.
    
    Returns:
    - sol2_LC (ndarray): An array of optimal path combinations with cost, delay, and travel order.
    
    Notes:
    The function writes the computed matrix to a .mat file for future reference.
    """
    sol2_LC = np.zeros([N_nodes*N_nodes*N_nodes,11]);  
    counter=0
    #loops for exploring all combinations
    for ii1 in range(0,N_nodes):
        for ii2 in range(ii1,N_nodes):
            for jj2 in range(jj1,N_nodes):
                # only calculate when start and ends are not the same
                if not np.any([car_node_order[ii2] == car_node_order[jj2], car_node_order[ii1] == car_node_order[jj1]]):
                    # find the minimum cost for combination
                    opti = np.array([LTIFM2_SP(jj1,ii1,jj2,ii2,solPart, car_node_order),
                                     LTIFM2_SP(jj2,ii2,jj1,ii1,solPart, car_node_order)])
                    opti = opti[np.lexsort(opti[:, ::-1].T)]
                    sol2_LC[counter,:] = opti[0,:] # matrix with objective, delays, order
                    counter = counter+1
    sol2_LC = sol2_LC[:counter, :] # trim unused rows
    # remove rows with zero costs and large delays
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,0] == 0 ),0)  #cost
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,2] > 15 ),0)  #delay
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,1] > 15 ),0)  #delay
    #store in .npy file
    np.save(CITY_FOLDER + "/L2/MatL2_" + f"{jj1+1}.npy", sol2_LC)
    return sol2_LC
    

def A2_LinearComb2(solPart):
    """
    This function prepares the directory structure, sets up parallel processing, and calls 
    `compute_LinearComb2_for_jj1` to compute minimum-cost path combinations for each starting node.
    Computes for the linear combination of 2 arcs.
    
    Parameters:
    - solPart (dict): A dictionary with precomputed path data.
    
    Returns:
    - sol2_LC_list (list): A flattened list of results from all starting nodes, saved to a .mat file.
    """
    # create directory
    try:
        os.mkdir(CITY_FOLDER + "/L2")
    except FileExistsError:
        print(f"Directory '{CITY_FOLDER + "/L2"}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{CITY_FOLDER + "/L2"}'.")
    except Exception as e:
        print(f"An error occurred: {e}")    

    #using joblib:
    results = Parallel(n_jobs=4)(
        delayed(compute_LinearComb2_for_jj1)(jj1, N_nodes_road, solPart)
        for jj1 in tqdm(range(N_nodes_road), desc="Processing jj1 values")
    )
    #store total in .mat file
    sol2_LC_arr = np.asarray([val for row in results for val in row])
    np.save(CITY_FOLDER + "/MatL2.npy", sol2_LC_arr)
    return sol2_LC_arr




def Generate_Full_List(ppl):
    if ppl == 2:
        sol2_LC =np.load(f'{CITY_FOLDER}/MatL2.npy')
        sol2_LC[:, 0] /= 2  # Divide first column by 2
        size2 = sol2_LC.shape[0]
        # Create Sol2 with NaN padding similar to MATLAB's code
        # sol2_LC[:, 3:7] -= 1    #subtract 1 from the indexing
        Sol2 = np.hstack([
            sol2_LC[:, :3],
            np.full((size2, 2), np.nan),
            sol2_LC[:, 3:7],
            np.full((size2, 4), np.nan),
            sol2_LC[:, 7:11],
            np.full((size2, 4), np.nan)
        ])
        
        # FullList for ppl == 2
        FullList = Sol2
    

    FullList = FullList[FullList[:, 0].argsort()]
    FullList = FullList[FullList[:, 0] < -0.01]
    return FullList

def compute_results(FullList, Delay, WaitingTime, Demand, fcoeffs):
    TotGamma2 = 0
    TotGamma3 = 0
    TotGamma4 = 0
    Cumul_delay2 = 0
    Cumul_delay3 = 0
    Cumul_delay4 = 0
    OriginalDemand = Demand.copy()
    DemandS =  Demand.copy()
    Demands_rp = np.zeros([N_nodes_road,N_nodes_road])
    gamma_arr = np.zeros(FullList.shape[0])
    for iii in range(FullList.shape[0]):
        # Seperate function to remove clutter
        Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, DemandS, Demands_rp, gamma =\
              calculate_gamma(FullList,DemandS, Delay, N_nodes_road, WaitingTime, Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, iii, Demands_rp, car_node_index_map)
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
    TotGamma = [TotGamma2, TotGamma3,TotGamma4]
    return y, yr, TrackDems_temp, TotGamma, gamma_arr


def main():
    # create costs matrix
    # solPart = A1_SP()
    # # # ### create solutions for different amount of linear combinations
    # sol2_LC = A2_LinearComb2(solPart)
    # sol3_LC = A2_LinearComb3(solPart)

    FullList = Generate_Full_List(2)
    tot_rp_flow = []
    tot_frp_flow = []
    tot_ped_flow = []
    tot_b_flow = []
    tot_waiting = []
    single_ride = []
    double_ride = []
    triple_ride = []
    max_iter = 21
    r=2
    #### adaptive mu
    mu=0.0001
    prev_x = prev2_x = prev_obj = prev2_obj = None
    reb_cars = 0
    ## cars estimation adjustment
    c_ratio = 1.0 
    rho_cars = 0.5
    stable_obj = 0
    for i in range(max_iter):
        if i == 2:
            mu = 0.0001
            
        _,_,avg_time,x,expected_cars, obj = cars.solve_bush_CARSn(tnet=tNet, fcoeffs=fcoeffs, iteration=i, unf=True, r=r, mu=mu, prev_x=prev_x, reb_cars=reb_cars, c_ratio=c_ratio, vehicle_lim=VEHICLE_LIM)
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
            # D_rp[car_node_index_map[int(v[:-2])], car_node_index_map[int(u[:-2])]] = sum(x_rp_mat[count])
            D_rp[car_node_index_map[int(v[:-2])], car_node_index_map[int(u[:-2])]] = tNet.G_supergraph[u][v]['flowNoRebalancing']
        print(f"ridepooling demand: {D_rp.sum()}")
        print(f"iteration: {i+1}")
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
                jj1 = car_node_index_map[int(FullList[iii][5])]
                ii1 = car_node_index_map[int(FullList[iii][6])]
                jj2 = car_node_index_map[int(FullList[iii][7])]
                ii2 = car_node_index_map[int(FullList[iii][8])]
                if np.array_equal(FullList[iii][13:17], [1, 2, 1, 2]):
                    OD_delays[ii1][jj1] += gamma_arr[iii] *(nx.shortest_path_length(tNet.G, source=car_node_order[ii1], target=car_node_order[ii2], weight='t_1')\
                                                        + nx.shortest_path_length(tNet.G, source=car_node_order[ii2], target=car_node_order[jj1], weight='t_1')\
                                                            - nx.shortest_path_length(tNet.G, source=car_node_order[ii1], target=car_node_order[jj1], weight='t_1'))
                    OD_delays[ii2][jj2] += gamma_arr[iii] *(nx.shortest_path_length(tNet.G, source=car_node_order[ii2], target=car_node_order[jj1], weight='t_1')\
                                                        + nx.shortest_path_length(tNet.G, source=car_node_order[jj1], target=car_node_order[jj2], weight='t_1')\
                                                            - nx.shortest_path_length(tNet.G, source=car_node_order[ii2], target=car_node_order[jj2], weight='t_1'))

                elif np.array_equal(FullList[iii][13:17], [1, 2, 2, 1]):
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
        # weighted delays / total requests
        total_delay = np.divide(OD_delays+Et, D_rp, out=np.zeros_like(D_rp), where=D_rp!=0)
        # print(f"\nDelays: {OD_delays.sum()}")
        # print(f"Total delays: {total_delay.sum()}")
        # print(f"waiting times: {Et.sum()}\n")
        # delay_cars = np.divide(OD_delays, D_rp, out=np.zeros_like(D_rp), where=D_rp!=0)
        
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
            
        
        rp_flow = {(k,j): tNet.G_supergraph[k][j]['flow'] for k,j in tNet.G_supergraph.edges() if tNet.G_supergraph[k][j]['type']=='rp'}  # 'rp' flow
        frp_flow = {(k,j): tNet.G_supergraph[k][j]['flow'] for k,j in tNet.G_supergraph.edges() if tNet.G_supergraph[k][j]['type']=='frp'}  # 'rp' flow
        ped_flow = {(k,j): tNet.G_supergraph[k][j]['flow'] for k,j in tNet.G_supergraph.edges() if tNet.G_supergraph[k][j]['type']=='p'}
        b_flow = {(k,j): tNet.G_supergraph[k][j]['flow'] for k,j in tNet.G_supergraph.edges() if tNet.G_supergraph[k][j]['type']=='b'}
        
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
            for u, v in G_final.edges():
                G_final[u][v]['flow'] = round(G_final[u][v]['flow'])

            normal_edges = []
            rp_edges = []
            edge_labels_1 = {}
            edge_labels_2 = {}

            flow = {(k,j):G_final[k][j]['flow'] for k,j in G_final.edges() if G_final[k][j]['type']==0}  # car flow
            rp_flow = {(int(re.search(r'\d+', k).group()),int(re.search(r'\d+', j).group())): G_final[k][j]['flow'] for k,j in G_final.edges() if G_final[k][j]['type']=='rp'}  # 'rp' flow
            ped_flow = {(int(re.search(r'\d+', k).group()),int(re.search(r'\d+', j).group())): G_final[k][j]['flow'] for k,j in G_final.edges() if G_final[k][j]['type']=='p'}
            b_flow = {(int(re.search(r'\d+', k).group()),int(re.search(r'\d+', j).group())): G_final[k][j]['flow'] for k,j in G_final.edges() if G_final[k][j]['type']=='b'}
            f_flow = {(k,j): G_final[k][j]['flow'] for k,j,d in G_final.edges(data=True) if d['type'] == 'f'}
            frp_flow = {(k,j): G_final[k][j]['flow'] for k,j,d in G_final.edges(data=True) if d['type'] == 'frp'}
            fu_flow = {(k,j): G_final[k][j]['flow'] for k,j,d in G_final.edges(data=True) if d['type'] == 'u'}
            # print(f"f: {sum(f_flow.values())/2}")
            # print(f"frp: {sum(frp_flow.values())/2}")
            # print(f"fu: {sum(fu_flow.values())/2}")
            # print(f"rp flow: {sum(rp_flow.values())}")


            # ped_labels_1 = {}
            # ped_labels_2 = {}
            # for k,j,t in G_final.edges(data=True):
            #     if t['type'] == 0:
            #         if k>j:
            #             if ped_flow[(k,j)] > 0 and b_flow[(k,j)] > 0:
            #                 ped_labels_1[(k,j)] = rf"{k} to {j}\nP: {ped_flow[(k,j)]}\nb: {b_flow[(k,j)]}"
            #             elif ped_flow[(k,j)] > 0:
            #                 ped_labels_1[(k,j)] = rf"{k} to {j}\nP: {ped_flow[(k,j)]}"
            #             elif b_flow[(k,j)] > 0:
            #                 ped_labels_1[(k,j)] = rf"{k} to {j}\nb: {b_flow[(k,j)]}"
                            
            #         if k<j:
            #             if ped_flow[(k,j)] > 0 and b_flow[(k,j)] > 0:
            #                 ped_labels_2[(k,j)] = rf"{k} to {j}\nP: {ped_flow[(k,j)]}\nb: {b_flow[(k,j)]}"
            #             elif ped_flow[(k,j)] > 0:
            #                 ped_labels_2[(k,j)] = rf"{k} to {j}\nP: {ped_flow[(k,j)]}"
            #             elif b_flow[(k,j)] > 0:
            #                 ped_labels_2[(k,j)] = rf"{k} to {j}\nb: {b_flow[(k,j)]}"
                            


            # for k,j,t in G_final.edges(data=True):
            #     if t['type'] == 'rp':
            #         # Create label with both flow values (if applicable)
            #         if k>j:
            #             if t['flow'] > 0:
            #                 edge_labels_1[(k, j)] = rf"{k} to {j}\nC: {t['flow']}"
            #         if k<j:
            #             if t['flow'] > 0:
            #                 edge_labels_2[(k, j)] = rf"{k} to {j}\nC: {t['flow']}"

            # # **Filter edges** where `type == 0`
            # filtered_edges = [(u, v) for u, v, d in G_final.edges(data=True) if d['type'] == "rp" and d['flow']>0]
            # # **Filter nodes** that are part of the filtered edges
            # filtered_nodes = set([node for edge in filtered_edges for node in edge])
            # plt.figure()
            # # Draw nodes
            # nx.draw_networkx_nodes(G_final, pos, nodelist=filtered_nodes, node_color='lightblue', node_size=10)
            # # Draw normal flows (black)
            # nx.draw_networkx_edges(G_final, pos, edgelist=filtered_edges, edge_color='black', width=0.2, arrows=True)
            # # Draw edge labels with both normal and rp flows
            # nx.draw_networkx_edge_labels(G_final, pos, edge_labels=edge_labels_1, font_size=5, verticalalignment='top')
            # nx.draw_networkx_edge_labels(G_final, pos, edge_labels=edge_labels_2, font_size=5, verticalalignment='bottom')
            # nx.draw_networkx_labels(G_final, pos, labels={node: node for node in filtered_nodes}, font_size=6, font_color='black')
            # # Show the plot
            # plt.savefig(f"fairness_{i}_network_graph.png")
            # plt.show(block=False)
            
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
        

    # Extract edge labels (flow values)
    for u, v in G_final.edges():
        G_final[u][v]['flow'] = round(G_final[u][v]['flow'])

    normal_edges = []
    rp_edges = []
    edge_labels_1 = {}
    edge_labels_2 = {}

    flow = {(i,j):G_final[i][j]['flow'] for i,j in G_final.edges() if G_final[i][j]['type']==0}  # car flow
    rp_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flow'] for i,j in G_final.edges() if G_final[i][j]['type']=='rp'}  # 'rp' flow
    ped_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flow'] for i,j in G_final.edges() if G_final[i][j]['type']=='p'}
    b_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flow'] for i,j in G_final.edges() if G_final[i][j]['type']=='b'}
    f_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'f'}
    frp_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'frp'}
    fu_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'u'}
    print(f"f: {sum(f_flow.values())/2}")
    print(f"frp: {sum(frp_flow.values())/2}")
    print(f"fu: {sum(fu_flow.values())/2}")


    ped_labels_1 = {}
    ped_labels_2 = {}
    for i,j,t in G_final.edges(data=True):
        if t['type'] == 0:
            if i>j:
                if ped_flow[(i,j)] > 0 and b_flow[(i,j)] > 0:
                    ped_labels_1[(i,j)] = f"{i} to {j}\nP: {ped_flow[(i,j)]}\nb: {b_flow[(i,j)]}"
                elif ped_flow[(i,j)] > 0:
                    ped_labels_1[(i,j)] = f"{i} to {j}\nP: {ped_flow[(i,j)]}"
                elif b_flow[(i,j)] > 0:
                    ped_labels_1[(i,j)] = f"{i} to {j}\nb: {b_flow[(i,j)]}"
                    
            if i<j:
                if ped_flow[(i,j)] > 0 and b_flow[(i,j)] > 0:
                    ped_labels_2[(i,j)] = f"{i} to {j}\nP: {ped_flow[(i,j)]}\nb: {b_flow[(i,j)]}"
                elif ped_flow[(i,j)] > 0:
                    ped_labels_2[(i,j)] = f"{i} to {j}\nP: {ped_flow[(i,j)]}"
                elif b_flow[(i,j)] > 0:
                    ped_labels_2[(i,j)] = f"{i} to {j}\nb: {b_flow[(i,j)]}"
                    


    for i,j,t in G_final.edges(data=True):
        if t['type'] == 'rp':
            # Create label with both flow values (if applicable)
            if i>j:
                if t['flow'] > 0:
                    edge_labels_1[(i, j)] = f"{i} to {j}\nC: {t['flow']}"
            if i<j:
                if t['flow'] > 0:
                    edge_labels_2[(i, j)] = f"{i} to {j}\nC: {t['flow']}"

    # **Filter edges** where `type == 0`
    filtered_edges = [(u, v) for u, v, d in G_final.edges(data=True) if d['type'] == "rp" and d['flow']>0]
    # **Filter nodes** that are part of the filtered edges
    filtered_nodes = set([node for edge in filtered_edges for node in edge])
    # Draw nodes
    nx.draw_networkx_nodes(G_final, pos, nodelist=filtered_nodes, node_color='lightblue', node_size=10)
    # Draw normal flows (black)
    nx.draw_networkx_edges(G_final, pos, edgelist=filtered_edges, edge_color='black', width=0.2, arrows=True)
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
    # rp_list = create_directional_list(sol2_LC)
    # rp_list_np = np.load("rp_list.npy")
    # rp_list = rp_list_np
    

    # sol3_LC = A2_LinearComb3(solPart)
    # sol4_LC = A2_LinearComb4(solPart)

    # ## Solve for up to ppl combinations
    # A3_Main_K_general(ppl=2)
    # print("Solved A3 for ppl = 2")
    # A3_Main_K_general(ppl=3)
    # print("Solved A3 for ppl = 3")
    # A3_Main_K_general(ppl=4)
    # print("Solved A3 for ppl = 4")

    # # Group results together
    # Group_Together()
    # # Plot
    # plot_groups()



    

if __name__ == "__main__":
    main()
