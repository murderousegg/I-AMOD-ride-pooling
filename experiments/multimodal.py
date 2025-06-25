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
netFile, gFile, fcoeffs,_,_ = tnet.get_network_parameters('NYC_Uber_small')

tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
tNetExog = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)

#increase demand
tNet.set_g(tnet.perturbDemandConstant(tNet.g, constant=0.12))

g_exog = tnet.perturbDemandConstant(tNetExog.g, constant=500)
tNetExog.set_g(g_exog)

tNet.build_walking_supergraph()
og_graph = tNet.G_supergraph.copy()
#integer demands:
tNet.g = {key: int(round(value)) for key, value in tNet.g.items()}

pos = nx.nx_agraph.graphviz_layout(tNet.G_supergraph, prog='neato')

# origins in walking layer
new_origins = {(f"{k[0]}'",f"{k[1]}'"): v for k, v in tNet.g.items()}
tNet.g = new_origins

#build other layers
tNet.build_layer(one_way=False, avg_speed=10, symb="b")
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
demand_matrix = np.zeros((N_nodes, N_nodes))  # Square matrix

for (origin, destination), demand in tNet.g.items():
    if origin in node_index_map and destination in node_index_map:
        i = node_index_map[origin]
        j = node_index_map[destination]
        demand_matrix[i, j] = demand  # Assign demand from origin to destination
# for ii in range(N_nodes):
#     demand_matrix[ii][ii] = -np.sum(demand_matrix[:][ii]) - demand_matrix[ii][ii]

# loading mat files
CITY_FOLDER = "NYC_Uber_small"
DemandS = demand_matrix             #store demand

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
            
            D = nx.shortest_path_length(tNet.G, source=car_node_order[jj], target=car_node_order[ii], weight="t_0") 
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
                if not np.any([car_node_order[ii2] == car_node_order[jj2], car_node_order[ii1] == car_node_order[jj1],\
                                car_node_order[ii1] == car_node_order[jj2], car_node_order[ii2] == car_node_order[jj1],\
                                      car_node_order[jj1] == car_node_order[jj2]]):
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
                        a = LTIFM3_SP(jj1,ii1,jj2,ii2,jj3,ii3,solPart,car_node_order)
                        b = LTIFM3_SP(jj2,ii2,jj1,ii1,jj3,ii3,solPart,car_node_order)
                        c = LTIFM3_SP(jj3,ii3,jj2,ii2,jj1,ii1,solPart,car_node_order)
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
    
    elif ppl ==3:
        # Load MatL2.mat and MatL3.mat
        sol2_LC =np.load(f'{CITY_FOLDER}/MatL2.npy')
        # sol2_LC[:, 3:7] -= 1    #subtract 1 from the indexing
        
        sol3_LC =np.load(f'{CITY_FOLDER}/MatL3.npy')
        # sol3_LC[:, 4:10] -= 1    #subtract 1 from the indexing
        
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
    # for iiii in range(FullList.shape[0]):
    #     vect = np.transpose(FullList[iiii, 5:13])  # Adjusted for 0-based indexing
    #     vectR = vect[~np.isnan(vect)]  # Remove NaN values
    #     vectR = vectR.reshape(-1, 2)
    #     num = vectR.shape[0]
        # Check demand condition and modify FullList if necessary
        # for iii in range(num):
        #     if DemandS[car_node_index_map[int(vectR[iii][0])]][car_node_index_map[int(vectR[iii][1])]] == 0:  # DemandS index adjusted for 0-based
        #         FullList[iiii, 0] = 0  # Set the first column to 0
    FullList = FullList[FullList[:, 0] < -0.01]
    # FullList[:, 0] /= ppl
    return FullList

def compute_results(ppl, FullList, Delay, WaitingTime, Demand, fcoeffs):
    TotGamma2 = 0
    TotGamma3 = 0
    TotGamma4 = 0
    Cumul_delay2 = 0
    Cumul_delay3 = 0
    Cumul_delay4 = 0
    OriginalDemand = Demand.copy()
    DemandS =  Demand.copy()
    Demands_rp = np.zeros([N_nodes_road,N_nodes_road])
    print(Delay)
    print(WaitingTime)
    for iii in range(FullList.shape[0]):
        # Seperate function to remove clutter
        Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, DemandS, Demands_rp =\
              calculate_gamma(FullList,DemandS, Delay, N_nodes_road, WaitingTime, Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, iii, Demands_rp, car_node_index_map)
    
    Demands_rp = Demands_rp - np.diag(np.diag(Demands_rp))
    #calculate solution
    solBase =LTIFM_reb(OriginalDemand,tNet.G, fcoeffs=fcoeffs)
    solNP = LTIFM_reb(DemandS,tNet.G, fcoeffs=fcoeffs)
    solRP = LTIFM_reb(Demands_rp,tNet.G, fcoeffs=fcoeffs)

    #reset diagonals (these are modified in LTIFM_reb)
    DemandS = DemandS - np.diag(np.diag(DemandS))   
    Demands_rp = Demands_rp - np.diag(np.diag(Demands_rp))
    TrackDems_temp = [np.sum(OriginalDemand), np.sum(DemandS),np.sum(Demands_rp)]
    TotGamma = [TotGamma2, TotGamma3,TotGamma4]
    Cumul_delay = [Cumul_delay2, Cumul_delay3, Cumul_delay4]
    #Prepare for storing
    solutions_data = {
        "x": np.stack([solBase["x"], solNP["x"], solRP["x"]], axis=0),           # Shape: (3, N_edges * N_nodes)
        "xr": np.stack([solBase["xr"], solNP["xr"], solRP["xr"]], axis=0),       # Shape: (3, N_edges)
        "IndividualTimes": np.stack([solBase["IndividualTimes"], solNP["IndividualTimes"], solRP["IndividualTimes"]], axis=0),  # Shape: (3, N_nodes)
        "obj": np.array([solBase["obj"], solNP["obj"], solRP["obj"]]),           # Shape: (3,)
        "Dem": np.stack([solBase["Dem"], solNP["Dem"], solRP["Dem"]], axis=0)  # Shape: (3, 24, 24)
        }
    # Additional tracking variables
    additional_data = {
        "TrackDems_temp": TrackDems_temp,
        "Cumul_delay": Cumul_delay,
        "TotGamma": TotGamma
    }
    # Save to an HDF5 file
    with h5py.File(f"{CITY_FOLDER}/Results/Delay{Delay}WTime{WaitingTime}.h5", "w") as h5file:
        # Save solution arrays in a flat structure
        for key, value in solutions_data.items():
            h5file.create_dataset(f"solutions/{key}", data=value)
        
        # Save additional tracking variables
        for key, value in additional_data.items():
            h5file.create_dataset(f"additional/{key}", data=value)
    return solRP["x"], solRP["xr"], TrackDems_temp, TotGamma


def main():
    # create costs matrix
    solPart = A1_SP()
    # ### create solutions for different amount of linear combinations
    # sol2_LC = A2_LinearComb2(solPart)
    # sol3_LC = A2_LinearComb3(solPart)

    FullList = Generate_Full_List(2)
    tot_rp_flow = []
    tot_frp_flow = []
    tot_ped_flow = []
    tot_b_flow = []
    single_ride = []
    double_ride = []
    triple_ride = []
    max_iter = 10
    for i in range(max_iter):
        x_rp_mat = cars.solve_matrix_base(tnet=tNet,fcoeffs=fcoeffs, times=i)
        # find D_rp, match edges to nodes in tNet.G
        D_rp = np.zeros((N_nodes_road, N_nodes_road))
        for count, (u,v) in enumerate(ridepool):
            D_rp[car_node_index_map[int(v[:-2])], car_node_index_map[int(u[:-2])]] = sum(x_rp_mat[count])
        print(f"ridepooling demand: {D_rp.sum()}")
        x, xr, TrackDems, TotG = compute_results(2, FullList=FullList, Delay=10, WaitingTime=10, Demand=D_rp, fcoeffs=fcoeffs)

        for i in range(N_edges_road):
            u, v = list(tNet.G.edges())[i]
            t0 = tNet.G[u][v]['t_0']
            capacity = tNet.G[u][v]['capacity']

            # Get flow on arc (u,v)
            flow_ij = sum([x[i, j] for j in range(N_nodes_road)]) + xr[i]
            # Compute congestion ratio
            ratio = flow_ij / (capacity)

            # Apply BPR function
            adjusted_tij = t0 * (1 + 0.15 * ratio**4)
            tNet.G[u][v]['t_1'] = adjusted_tij
        
        
        for u,v,d in tNet.G_supergraph.edges(data=True):
            if d['type'] == 'rp':
                #### maybe don't recalculate shortest path based on t_1?
                ### cuz that just incentivices everyone going around the fastest path
                ### should do shortest path with t_0, but keep t_1 time?
                tNet.G_supergraph[u][v]['t_1'] = nx.shortest_path_length(tNet.G, source=int(u[:-2]), target=int(v[:-2]), weight='t_1')
            else:
                tNet.G_supergraph[u][v]['t_1'] = tNet.G_supergraph[u][v]['t_0']
        
        rp_flow = {(i,j): tNet.G_supergraph[i][j]['flow'] for i,j in tNet.G_supergraph.edges() if tNet.G_supergraph[i][j]['type']=='rp'}  # 'rp' flow
        frp_flow = {(i,j): tNet.G_supergraph[i][j]['flow'] for i,j in tNet.G_supergraph.edges() if tNet.G_supergraph[i][j]['type']=='frp'}  # 'rp' flow
        ped_flow = {(i,j): tNet.G_supergraph[i][j]['flow'] for i,j in tNet.G_supergraph.edges() if tNet.G_supergraph[i][j]['type']=='p'}
        b_flow = {(i,j): tNet.G_supergraph[i][j]['flow'] for i,j in tNet.G_supergraph.edges() if tNet.G_supergraph[i][j]['type']=='b'}
        
        # this_rp_flow = [rp_flow[i] * tNet.G_supergraph[i[0]][i[1]]['t_1'] for i in rp_flow.keys()]
        this_rp_flow = [rp_flow[i] * tNet.G_supergraph[i[0]][i[1]]['t_1'] for i in rp_flow.keys()]
        this_rp_flow = sum(this_rp_flow)
        tot_rp_flow.append(this_rp_flow)
        
        this_frp_flow = [frp_flow[i] * tNet.G_supergraph[i[0]][i[1]]['t_1'] for i in frp_flow.keys()]
        this_frp_flow = sum(this_frp_flow)
        tot_frp_flow.append(this_frp_flow)

        # this_ped_flow = [ped_flow[i] * tNet.G_supergraph[i[0]][i[1]]['t_1'] for i in ped_flow.keys()]
        this_ped_flow = [ped_flow[i] * tNet.G_supergraph[i[0]][i[1]]['t_1'] for i in ped_flow.keys()]
        this_ped_flow = sum(this_ped_flow)
        tot_ped_flow.append(this_ped_flow)

        # this_b_flow = [b_flow[i] * tNet.G_supergraph[i[0]][i[1]]['t_1'] for i in b_flow.keys()]
        this_b_flow = [b_flow[i] * tNet.G_supergraph[i[0]][i[1]]['t_1'] for i in b_flow.keys()]
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
    print(tot_rp_flow)
    print(tot_frp_flow)
    print(tot_ped_flow)
    print(tot_b_flow)
    # Create bar plot
    plt.bar(np.arange(max_iter), [tot_rp_flow[i] * single_ride[i] for i in range(len(tot_rp_flow))], label='rp flow 1 user')
    plt.bar(np.arange(max_iter), [tot_rp_flow[i] * double_ride[i] for i in range(len(tot_rp_flow))], bottom=[tot_rp_flow[i] * single_ride[i] for i in range(len(tot_rp_flow))], label='rp flow 2 users')
    plt.bar(np.arange(max_iter), [tot_rp_flow[i] * triple_ride[i] for i in range(len(tot_rp_flow))], bottom=[tot_rp_flow[i] * (single_ride[i]+ double_ride[i]) for i in range(len(tot_rp_flow))], label='rp flow 3 users')
    plt.bar(np.arange(max_iter), tot_ped_flow, bottom=tot_rp_flow, label='ped flow')
    plt.bar(np.arange(max_iter), tot_b_flow, bottom=np.array(tot_rp_flow) + np.array(tot_ped_flow), label='bike flow')

    # Customize plot
    plt.xlabel('Iteration')
    plt.ylabel('Total time spent in mode of transport')
    plt.title('Evolution of mode allocation over time')
    plt.legend()
    plt.tight_layout()
    plt.show()


    print(x_rp_mat)
    print(x_rp_mat.shape)
    print(np.nonzero(x_rp_mat))

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


"""
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
"""