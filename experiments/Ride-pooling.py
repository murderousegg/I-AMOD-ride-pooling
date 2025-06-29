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
from Utilities.RidePooling.LTIFM_reb import LTIFM_reb
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
netFile, gFile, fcoeffs,_,_ = tnet.get_network_parameters('NYC_small')

tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
tNetExog = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
g_exog = tnet.perturbDemandConstant(tNetExog.g, constant=1.5)
tNetExog.set_g(g_exog)
tNet.build_supergraph()
og_graph = tNet.G_supergraph.copy()
#integer demands:
tNet.g = {key: int(round(value)) for key, value in tNet.g.items()}

pos = nx.nx_agraph.graphviz_layout(tNet.G_supergraph, prog='neato')

# origins in walking layer
new_origins = {(f"{k[0]}'",f"{k[1]}'"): v for k, v in tNet.g.items()}
tNet.g = new_origins

#build other layers
# tNet.build_rp_layer(avg_speed=1000)
tNet.build_layer(one_way=False, avg_speed=10, symb="b")
tNet.build_full_layer()
# tNet.build_pickup_layer()

# ridepool = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'rp']
pedestrian = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'p']
connector = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'f']
connector_rp = [(u, v) for (u, v, d) in tNet.G_supergraph.edges(data=True) if d['type'] == 'f_rp']


node_order = list(tNet.G_supergraph.nodes())
edge_order = list(tNet.G_supergraph.edges())


Binc = nx.incidence_matrix(tNet.G_supergraph, nodelist=node_order, edgelist=edge_order, oriented=True)
[N_nodes,N_edges] = Binc.shape

node_index_map = {node: i for i, node in enumerate(tNet.G_supergraph.nodes())}
demand_matrix = np.zeros((N_nodes, N_nodes))  # Square matrix

for (origin, destination), demand in tNet.g.items():
    if origin in node_index_map and destination in node_index_map:
        i = node_index_map[origin]
        j = node_index_map[destination]
        demand_matrix[i, j] = demand  # Assign demand from origin to destination
# for ii in range(N_nodes):
#     demand_matrix[ii][ii] = -np.sum(demand_matrix[:][ii]) - demand_matrix[ii][ii]

# loading mat files
CITY_FOLDER = "NYC_small"
DemandS = demand_matrix             #store demand

# Generating the diagraph
edgeList = edge_order

# for the main_K_general function
OriginalDemand= np.array(DemandS)
DemandS = np.array(DemandS)
TotDems = np.sum(DemandS)   

# normal parameters
WAITINGTIMES = [2, 5, 10, 15]
DELAYS = [2, 5, 10]
MULTIPLIER = [0.0078, 0.0156, 0.0312, 0.0625, 0.125, 0.25, 0.5, 1, 2]

# Copy for replicating matlab figures
# MULTIPLIER = [0.0156, 0.0312, 0.0625, 0.125, 0.25, 0.5, 1, 2]

Binc_road = nx.incidence_matrix(tNet.G)
[N_nodes_road,N_edges_road] = Binc_road.shape

def A1_SP():
    """
    Generates a matrix of dictionaries containing the costs of an arc, 
    a sparse matrix with the possible arcs and the times between destinations
    """
    solPart = [[{} for _ in range(N_nodes_road)] for _ in range(N_nodes_road)]
    for ii in range(0,N_nodes_road):
        for jj in range(0,N_nodes_road):
            Dems = np.zeros([N_nodes_road, N_nodes_road])
            #generate demand matrix
            Dems[ii,jj] = 1
            Dems[jj,jj] = -1

            if ii==jj:  #special case, set demand to 0 if origin and destination are equal
                Dems[jj,jj] = -0
            
            D = nx.shortest_path_length(tNet.G, source=node_order[jj], target=node_order[ii], weight="t_0")            
            # Store the results in solPart[jj][ii] as a dictionary
            solPart[jj][ii] = {
                'obj': D,
                'Dem': sparse.csr_matrix(Dems),  # Convert Dems to sparse format
                'IndividualTimes': D
            }
    # convert to dictionary for .mat convertion
    solPart_dic = {"solPart": np.array(solPart)}
    io.savemat(CITY_FOLDER + "/solPart_pyth_" + CITY_FOLDER+ ".mat", solPart_dic)
    return solPart

def compute_LinearComb2_for_jj1(jj1, N_nodes, DemandS, solPart):
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
                if not np.any([node_order[ii2] == node_order[jj2], node_order[ii1] == node_order[jj1],\
                                node_order[ii1] == node_order[jj2], node_order[ii2] == node_order[jj1],\
                                      node_order[jj1] == node_order[jj2]]):
                    # find the minimum cost for combination
                    opti = np.array([LTIFM2_SP(jj1,ii1,jj2,ii2,solPart, node_order),
                                     LTIFM2_SP(jj2,ii2,jj1,ii1,solPart, node_order)])
                    opti = opti[np.lexsort(opti[:, ::-1].T)]
                    sol2_LC[counter,:] = opti[0,:] # matrix with objective, delays, order
                    counter = counter+1
    sol2_LC = sol2_LC[:counter, :] # trim unused rows
    # remove rows with zero costs and large delays
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,0] == 0 ),0)  #cost
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,2] > 5 ),0)  #delay
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,1] > 5 ),0)  #delay
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
        delayed(compute_LinearComb2_for_jj1)(jj1, N_nodes_road, DemandS, solPart)
        for jj1 in tqdm(range(N_nodes_road), desc="Processing jj1 values")
    )
    #store total in .mat file
    sol2_LC_arr = np.asarray([val for row in results for val in row])
    np.save(CITY_FOLDER + "/MatL2.npy", sol2_LC_arr)
    return sol2_LC_arr


def compute_LinearComb3_for_jj1(ii1,jj1, N_nodes, DemandS, solPart):
    """
    This function computes the minimum cost combinations for travel paths within a transportation network.
    This can be wrapped by a parallel processing toolbox such as joblib.
    
    Parameters:
    - ii1 (int): An index representing the starting node in one of the loop structures.
    - jj1 (int): An index representing the starting node in one of the loop structures.
    - N_nodes (int): The total number of nodes in the network.
    - DemandS (ndarray): A matrix where DemandS[ii, jj] > 0 indicates a demand for travel from node ii to node jj.
    - solPart (dict): A dictionary containing precomputed shortest paths and related data for node pairs.
    
    Returns:
    - sol3_LC (ndarray): An array of optimal path combinations with cost, delay, and travel order.
    
    Notes:
    The function writes the computed matrix to a .mat file for future reference.
    """
    # large empty array for storing data
    sol3_LC = np.zeros([100000,16]);  
    counter=0
    for jj2 in range(jj1,N_nodes_road):
        for ii2 in range(ii1,N_nodes_road):
            for jj3 in range(jj2,N_nodes_road):
                for ii3 in range(ii2,N_nodes_road):
                    # only calculate when start and ends are not the same, and demands nonzero
                    if not np.any([ii1 == jj1, ii1 == jj2, ii1 == jj3, ii2 == jj1, ii2 == jj2, ii2 == jj3,
                            ii3 == jj1, ii3 == jj2, ii3 == jj3, DemandS[ii1, jj1] == 0, DemandS[ii2, jj2] == 0, DemandS[ii3, jj3] == 0]):
                        # find the minimum cost for combination
                        a = LTIFM3_SP(jj1,ii1,jj2,ii2,jj3,ii3,solPart)
                        b = LTIFM3_SP(jj2,ii2,jj1,ii1,jj3,ii3,solPart)
                        c = LTIFM3_SP(jj3,ii3,jj2,ii2,jj1,ii1,solPart)
                        opti = np.array([a,b,c])
                        opti = opti[np.lexsort(opti[:, ::-1].T)]
                        sol3_LC[counter,:] = opti[0,:] # matrix with objective, delays, order
                        counter = counter+1
    sol3_LC = sol3_LC[:counter, :] # trim unused rows
    # remove rows with zero costs and large delays
    sol3_LC = np.delete(sol3_LC,np.argwhere(sol3_LC[:,0] == 0 ),0)  #cost
    sol3_LC = np.delete(sol3_LC,np.argwhere(sol3_LC[:,1] > 20 ),0)  #delay
    sol3_LC = np.delete(sol3_LC,np.argwhere(sol3_LC[:,2] > 20 ),0)  #delay
    sol3_LC = np.delete(sol3_LC,np.argwhere(sol3_LC[:,3] > 20 ),0)  #delay
    #store in .mat file
    np.save(CITY_FOLDER + "/L3/MatL3_" + f"{jj1+1}_{ii1+1}.npy", sol3_LC)
    return sol3_LC


def A2_LinearComb3(solPart):
    """
    This function prepares the directory structure, sets up parallel processing, and calls 
    `compute_LinearComb3_for_jj1` to compute minimum-cost path combinations for each starting node.
    Computes for the linear combination of 3 arcs.
    
    Parameters:
    - solPart (dict): A dictionary with precomputed path data.
    
    Returns:
    - sol3_LC_list (list): A flattened list of results from all starting nodes, saved to a .mat file.
    """
    # create directory
    try:
        os.mkdir(CITY_FOLDER + "/L3")
    except FileExistsError:
        print(f"Directory '{CITY_FOLDER + "/L3"}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{CITY_FOLDER + "/L3"}'.")
    except Exception as e:
        print(f"An error occurred: {e}")    
    # parallel processing
    # with Pool() as pool:
    #     results = pool.starmap(compute_LinearComb3_for_jj1, [(jj1, N_nodes, DemandS, solPart) for jj1 in range(N_nodes)])
    
    #using joblib:
    sol3_LC_list = []
    for jj1 in tqdm(range(N_nodes_road), desc="Processing jj1 values", ncols=100):
        sol3_LC = Parallel(n_jobs=6)(
            delayed(compute_LinearComb3_for_jj1)(ii1, jj1, N_nodes, DemandS, solPart)
            for ii1 in tqdm(range(N_nodes_road), desc="Processing ii1 values", leave=False)
        ) 
        sol3_LC_list.append(np.vstack(sol3_LC))
    sol3_LC_arr = np.vstack(sol3_LC_list)
    sol3_LC_arr = sol3_LC_arr[np.lexsort(sol3_LC_arr[:, ::-1].T)]
    #store total in .npy file
    np.save(CITY_FOLDER + "/MatL3.npy", sol3_LC_arr)


def compute_LinearComb4_for_jj1(ii1, jj1, N_nodes, DemandS, solPart):
    """
    This function computes the minimum cost combinations for travel paths within a transportation network.
    This can be wrapped by a parallel processing toolbox such as joblib.
    
    Parameters:
    - ii1 (int): An index representing the starting node in one of the loop structures.
    - jj1 (int): An index representing the starting node in one of the loop structures.
    - N_nodes (int): The total number of nodes in the network.
    - DemandS (ndarray): A matrix where DemandS[ii, jj] > 0 indicates a demand for travel from node ii to node jj.
    - solPart (dict): A dictionary containing precomputed shortest paths and related data for node pairs.
    
    Returns:
    - sol4_LC (ndarray): An array of optimal path combinations with cost, delay, and travel order.
    
    Notes:
    The function writes the computed matrix to a .mat file for future reference.
    """
    total_sol4_LC = []
    for jj2 in range(jj1,N_nodes):
        sol4_LC = np.zeros([700000,21]);  
        counter=0
        #loops
        for ii2 in range(ii1,N_nodes_road):
            for jj3 in range(jj2,N_nodes_road):
                for ii3 in range(ii2,N_nodes_road):
                    for jj4 in range(jj3,N_nodes_road):
                        for ii4 in range(ii3,N_nodes_road):
                            # only calculate when start and ends are not the same, and demands nonzero
                            if not any([ii1==jj1,ii1==jj2,ii1==jj3,ii1==jj4,ii2==jj1,ii2==jj2,ii2==jj3,ii2==jj4,ii3==jj1,ii3==jj2,
                                        ii3==jj3,ii3==jj4,ii4==jj1,ii4==jj2,ii4==jj3,ii4==jj4,
                                        DemandS[ii1,jj1]==0,DemandS[ii2,jj2]==0,DemandS[ii3,jj3]==0,DemandS[ii4,jj4]==0]):
                                # find the minimum cost for combination
                                opti = np.array([LTIFM4_SP(jj1,ii1,jj2,ii2,jj3,ii3,jj4,ii4,solPart),
                                                LTIFM4_SP(jj2,ii2,jj1,ii1,jj3,ii3,jj4,ii4,solPart),
                                                LTIFM4_SP(jj3,ii3,jj2,ii2,jj1,ii1,jj4,ii4,solPart),
                                                LTIFM4_SP(jj4,ii4,jj2,ii2,jj3,ii3,jj1,ii1,solPart)])
                                opti = opti[np.lexsort(opti[:, ::-1].T)]
                                sol4_LC[counter,:] = opti[0,:] # matrix with objective, delays, order
                                counter = counter+1
        sol4_LC = sol4_LC[:counter, :] # trim unused rows
        # remove rows with zero costs and large delays
        sol4_LC = np.delete(sol4_LC,np.argwhere(sol4_LC[:,0] == 0 ),0)  #cost
        sol4_LC = np.delete(sol4_LC,np.argwhere(sol4_LC[:,1] > 20 ),0)  #delay
        sol4_LC = np.delete(sol4_LC,np.argwhere(sol4_LC[:,2] > 20 ),0)  #delay
        sol4_LC = np.delete(sol4_LC,np.argwhere(sol4_LC[:,3] > 20 ),0)  #delay
        sol4_LC = np.delete(sol4_LC,np.argwhere(sol4_LC[:,4] > 20 ),0)  #delay

        #store in .mat file
        np.save(CITY_FOLDER + "/L4/MatL4_" + f"{jj1+1}_{ii1+1}_{jj2+1}.npy", sol4_LC)
        total_sol4_LC.append(sol4_LC)
    total_sol4_LC = np.vstack(total_sol4_LC) if total_sol4_LC else np.empty((0, 21))  # Handle empty case
    return total_sol4_LC

def A2_LinearComb4(solPart):
    """
    This function prepares the directory structure, sets up parallel processing, and calls 
    `compute_LinearComb4_for_jj1` to compute minimum-cost path combinations for each starting node.
    Computes for the linear combination of 4 arcs.
    
    Parameters:
    - solPart (dict): A dictionary with precomputed path data.
    
    Returns:
    - sol4_LC_list (list): A flattened list of results from all starting nodes, saved to a .mat file.
    """
    # create directory
    try:
        os.mkdir(CITY_FOLDER + "/L4")
    except FileExistsError:
        print(f"Directory '{CITY_FOLDER + "/L4"}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{CITY_FOLDER + "/L4"}'.")
    except Exception as e:
        print(f"An error occurred: {e}")    
    
    # parallel processing
    # with Pool() as pool:
    #     results = pool.starmap(compute_LinearComb4_for_jj1, [(jj1, N_nodes, DemandS, solPart) for jj1 in range(N_nodes)])
    #using joblib:
    sol4_LC_list = []
    for jj1 in tqdm(range(N_nodes_road), desc="Processing jj1 values", ncols=100):
        sol4_LC = Parallel(n_jobs=6)(
            delayed(compute_LinearComb4_for_jj1)(ii1, jj1, N_nodes, DemandS, solPart)
            for ii1 in tqdm(range(N_nodes_road), desc="Processing ii1 values", leave=False)
        ) 
        sol4_LC_list.append(np.vstack(sol4_LC))
    sol4_LC_arr = np.vstack(sol4_LC_list)
    sol4_LC_arr = sol4_LC_arr[np.lexsort(sol4_LC_arr[:, ::-1].T)]
    #store total in .npy file
    np.save(CITY_FOLDER + "/MatL4.npy", sol4_LC_arr)

def Generate_Full_List(ppl):
    if ppl == 2:
        sol2_LC =np.load(f'{CITY_FOLDER}/MatL2.npy')
        # sol2_LC[:, 0] /= 2  # Divide first column by 2
        size2 = sol2_LC.shape[0]
        # Create Sol2 with NaN padding similar to MATLAB's code
        sol2_LC[:, 3:7] -= 1    #subtract 1 from the indexing
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
    
    elif ppl ==3:
        # Load MatL2.mat and MatL3.mat
        sol2_LC =np.load(f'{CITY_FOLDER}/MatL2.npy')
        sol2_LC[:, 3:7] -= 1    #subtract 1 from the indexing
        
        sol3_LC =np.load(f'{CITY_FOLDER}/MatL3.npy')
        sol3_LC[:, 4:10] -= 1    #subtract 1 from the indexing
        
        # Process sol2_LC
        sol2_LC[:, 0] /= 2
        size2 = sol2_LC.shape[0]
        Sol2 = np.hstack([
            sol2_LC[:, :3],
            np.full((size2, 2), np.nan),
            sol2_LC[:, 3:7],
            np.full((size2, 4), np.nan),
            sol2_LC[:, 7:11],
            np.full((size2, 4), np.nan)
        ])
        
        # Process sol3_LC
        sol3_LC[:, 0] /= 3
        size3 = sol3_LC.shape[0]
        Sol3 = np.hstack([
            sol3_LC[:, :4],
            np.full((size3, 1), np.nan),
            sol3_LC[:, 4:10],
            np.full((size3, 2), np.nan),
            sol3_LC[:, 10:16],
            np.full((size3, 2), np.nan)
        ])
        
        # FullList for ppl == 3
        FullList = np.vstack([Sol2, Sol3])
    elif ppl == 4:
        # Load MatL2.mat, MatL3.mat, and MatL4.mat
        sol2_LC =np.load(f'{CITY_FOLDER}/MatL2.npy')
        sol2_LC[:, 3:7] -= 1    #subtract 1 from the indexing
        
        sol3_LC =np.load(f'{CITY_FOLDER}/MatL3.npy')
        sol3_LC[:, 4:10] -= 1    #subtract 1 from the indexing
        
        sol4_LC =np.load(f'{CITY_FOLDER}/MatL4.npy')
        sol4_LC[:, 5:13] -= 1    #subtract 1 from the indexing
        
        # Process sol2_LC
        sol2_LC[:, 0] /= 2
        size2 = sol2_LC.shape[0]
        Sol2 = np.hstack([
            sol2_LC[:, :3],
            np.full((size2, 2), np.nan),
            sol2_LC[:, 3:7],
            np.full((size2, 4), np.nan),
            sol2_LC[:, 7:11],
            np.full((size2, 4), np.nan)
        ])
        
        # Process sol3_LC
        sol3_LC[:, 0] /= 3
        size3 = sol3_LC.shape[0]
        Sol3 = np.hstack([
            sol3_LC[:, :4],
            np.full((size3, 1), np.nan),
            sol3_LC[:, 4:10],
            np.full((size3, 2), np.nan),
            sol3_LC[:, 10:16],
            np.full((size3, 2), np.nan)
        ])
        
        # Process sol4_LC
        sol4_LC[:, 0] /= 4

        # FullList for ppl == 4
        FullList = np.vstack([Sol2, Sol3, sol4_LC])

    else:
        raise ValueError("ppl should be 2, 3, or 4")
    FullList = FullList[FullList[:, 0].argsort()]

    # Iterate through each row of FullList
    for iiii in range(FullList.shape[0]):
        vect = np.transpose(FullList[iiii, 5:13])  # Adjusted for 0-based indexing
        vectR = vect[~np.isnan(vect)]  # Remove NaN values
        vectR = vectR.reshape(-1, 2)
        num = vectR.shape[0]
        # Check demand condition and modify FullList if necessary
        for iii in range(num):
            if DemandS[int(vectR[iii][1])][int(vectR[iii][0])] == 0:  # DemandS index adjusted for 0-based
                FullList[iiii, 0] = 0  # Set the first column to 0
    FullList = FullList[FullList[:, 0] < -0.01]
    FullList[:, 0] /= ppl

    return FullList

def create_pickup_dropoff_list(sol_LC):
    rp_list = []
    edge_list = []
    for i in range(sol_LC.shape[0]):
        node_idx = sol_LC[i,3:7]
        order_idx = sol_LC[i,7:11]
        part_1 = node_idx[order_idx == 1]
        part_2 = node_idx[order_idx == 2]
        edge_list.append([part_1,part_2])
        edge_1=0
        edge_2=0
        for j in range(len(edge_order)):
            if edge_order[j] == (str(int(part_1[0]))+"q",str(int(part_1[1])) + "q"):
                edge_1 = j
            elif edge_order[j] == (str(int(part_2[0]))+"q", str(int(part_2[1])) + "q"):
                edge_2 = j
        rp_list.append([edge_1,edge_2])
    rp_list_np = np.asarray(rp_list)
    np.save("rp_list.npy", rp_list_np)
    np.save("rp_edge_list.npy", edge_list)
    return rp_list

def create_directional_list(sol_LC):
    """
    Creates a list with all possible sequences in sol_LC
    TODO: implement similar to generate_Full_List
    """
    lst = []
    for i in range(sol_LC.shape[0]):
        lst.append(sol_LC[i,3:7])
    return lst

def create_rp_demand(sol_LC):
    """
    Creates the demand matrix as a linear combination of indices
    corresponding to the selected sequences.
    """
    #get list of sequences
    lst = create_directional_list(sol_LC)
    #initialize empty array of with objects
    D_rp = np.empty((N_nodes_road, N_nodes_road), dtype=object)

    # Initialize each element with an empty list
    for i in range(D_rp.shape[0]):
        for j in range(D_rp.shape[1]):
            D_rp[i, j] = []
    # fill elements with list of tours belonging to said demand
    for i, tour in enumerate(lst):
        if tour[0] != tour[1]:  # skip if same origin
            D_rp[node_index_map[int(tour[0])], node_index_map[int(tour[1])]].append(i)
            D_rp[node_index_map[int(tour[0])], node_index_map[int(tour[0])]].append(i)# -outgoing demand

        D_rp[node_index_map[int(tour[1])], node_index_map[int(tour[2])]].append(i)
        D_rp[node_index_map[int(tour[1])], node_index_map[int(tour[1])]].append(i) # -outgoing demand

        if tour[2] != tour[3]: #skip if same destination
            D_rp[node_index_map[int(tour[2])], node_index_map[int(tour[3])]].append(i)
            D_rp[node_index_map[int(tour[2])], node_index_map[int(tour[2])]].append(i) # -outgoing demand
    return D_rp

def main():
    # create costs matrix
    solPart = A1_SP()

    # ### create solutions for different amount of linear combinations
    sol2_LC = A2_LinearComb2(solPart)
    rp_list = create_pickup_dropoff_list(sol2_LC)
    D_rp = create_rp_demand(sol2_LC)
    # rp_list = create_directional_list(sol2_LC)
    # rp_list_np = np.load("rp_list.npy")
    # rp_list = rp_list_np
    cost_1 = []
    cost_2 = []
    for i in range(sol2_LC.shape[0]):
        arc_1 = rp_list[i][0]
        arc_2 = rp_list[i][1]
        cost_1.append(solPart[node_index_map[int(edge_order[arc_1][0][:-1])]][node_index_map[int(edge_order[arc_1][1][:-1])]]["obj"] - sol2_LC[i,1])
        cost_2.append(solPart[node_index_map[int(edge_order[arc_2][0][:-1])]][node_index_map[int(edge_order[arc_2][1][:-1])]]["obj"] - sol2_LC[i,2])
    sol_cost = np.array([-sol2_LC[:,0], cost_1, cost_2])

    beta_lim = np.ones((len(rp_list), 2))
    beta_lim[:, 0] = 0  
    beta_lim[:, 1] = GRB.INFINITY  

    fixed_indices = []
    for i in range(10):
        selected_tours, tour_count = cars.solve_cars_matrix_rp_reduced(tNet,fcoeffs=fcoeffs, rebalancing=False, rp_list=rp_list, sol_costs=sol_cost, D_rp=D_rp, beta_lim=beta_lim)
        
        filtered_tours = np.delete(tour_count,fixed_indices)
        top_tours = np.argsort(filtered_tours)[-3:]
        Pt = np.zeros((len(tour_count)))
        for j, i in enumerate(tour_count):
            Pt[j] = probcombN([i, i],5)
        beta_lim[top_tours,1] = Pt[top_tours]*tour_count[top_tours]
        beta_lim[top_tours,0] = Pt[top_tours]*tour_count[top_tours]
        fixed_indices.append(top_tours.tolist())
    
    G_final = tNet.G_supergraph

    real_tours = []
    for i in selected_tours:
        sub_tours = []
        for j in i:
            sub_tours.append(edge_order[j])
        real_tours.append(sub_tours)
    print(f"real_tours: {real_tours}")
        

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
    q_flow = {(int(re.search(r'\d+', i).group()),int(re.search(r'\d+', j).group())): G_final[i][j]['flow'] for i,j in G_final.edges() if G_final[i][j]['type']=='q' if G_final[i][j]['flow'] != 0}
    qf_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'fq' if G_final[i][j]['flow'] != 0} 
    f_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'f'}
    frp_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'f_rp'}
    fu_flow = {(i,j): G_final[i][j]['flow'] for i,j,d in G_final.edges(data=True) if d['type'] == 'u'}
    print(f"f: {sum(f_flow.values())/2}")
    print(f"frp: {sum(frp_flow.values())/2}")
    print(f"fu: {sum(fu_flow.values())/2}")
    print(f"q flow: {q_flow}")
    print(f"qf flow: {qf_flow}")

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
        if t['type'] ==0:
            # Create label with both flow values (if applicable)
            if i>j:
                if flow[(i,j)] > 0:
                    edge_labels_1[(i, j)] = f"{i} to {j}\nC: {flow[(i,j)]}"
            if i<j:
                if flow[(i,j)] > 0:
                    edge_labels_2[(i, j)] = f"{i} to {j}\nC: {flow[(i,j)]}"

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