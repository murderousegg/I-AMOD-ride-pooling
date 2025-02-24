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


node_order = list(tnet.G_supergraph.nodes())
edge_order = list(tnet.G_supergraph.edges())


Binc = nx.incidence_matrix(tnet.G_supergraph, nodelist=node_order, edgelist=edge_order, oriented=True)
[N_nodes,N_edges] = Binc.shape


node_index_map = {node: i for i, node in enumerate(tnet.G_supergraph.nodes())}
demand_matrix = np.zeros((N_nodes, N_nodes))  # Square matrix

for (origin, destination), demand in tnet.g.items():
    if origin in node_index_map and destination in node_index_map:
        i = node_index_map[origin]
        j = node_index_map[destination]
        demand_matrix[i, j] = demand  # Assign demand from origin to destination
for ii in range(N_nodes):
    demand_matrix[ii][ii] = -np.sum(demand_matrix[:][ii]) - demand_matrix[ii][ii]

# loading mat files
CITY_FOLDER = "NYC_small"
DemandS = demand_matrix             #store demand

# Generating the diagraph
edgeList = edge_order

#convert for sioux falls
if CITY_FOLDER == 'SF':
    DemandS = DemandS/24

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



def A1_SP():
    """
    Generates a matrix of dictionaries containing the costs of an arc, 
    a sparse matrix with the possible arcs and the times between destinations
    """
    solPart = [[{} for _ in range(N_nodes)] for _ in range(N_nodes)]
    for ii in range(0,N_nodes):
        for jj in range(0,N_nodes):
            Dems = np.zeros([N_nodes, N_nodes])
            #generate demand matrix
            Dems[ii,jj] = 1
            Dems[jj,jj] = -1

            if ii==jj:  #special case, set demand to 0 if origin and destination are equal
                Dems[jj,jj] = -0
            
            D = nx.shortest_path_length(tNet.G_supergraph, source=jj+1, target=ii+1, weight='weight')
            
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
                # only calculate when start and ends are not the same, and demands nonzero
                if not np.any([ii2 == jj2, ii1 == jj2, ii2 == jj1, ii1 == jj1, DemandS[ii1, jj1] == 0, DemandS[ii2, jj2] == 0]):
                    # find the minimum cost for combination
                    opti = np.array([LTIFM2_SP(jj1,ii1,jj2,ii2,solPart),
                                     LTIFM2_SP(jj2,ii2,jj1,ii1,solPart)])
                    opti = opti[np.lexsort(opti[:, ::-1].T)]
                    sol2_LC[counter,:] = opti[0,:] # matrix with objective, delays, order
                    counter = counter+1
    sol2_LC = sol2_LC[:counter, :] # trim unused rows
    # remove rows with zero costs and large delays
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,0] == 0 ),0)  #cost
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,2] > 20 ),0)  #delay
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,1] > 20 ),0)  #delay
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
        delayed(compute_LinearComb2_for_jj1)(jj1, N_nodes, DemandS, solPart)
        for jj1 in tqdm(range(N_nodes), desc="Processing jj1 values")
    )
    #store total in .mat file
    sol2_LC_arr = np.asarray([val for row in results for val in row])
    np.save(CITY_FOLDER + "/MatL2.npy", sol2_LC_arr)


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
    for jj2 in range(jj1,N_nodes):
        for ii2 in range(ii1,N_nodes):
            for jj3 in range(jj2,N_nodes):
                for ii3 in range(ii2,N_nodes):
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
    for jj1 in tqdm(range(N_nodes), desc="Processing jj1 values", ncols=100):
        sol3_LC = Parallel(n_jobs=6)(
            delayed(compute_LinearComb3_for_jj1)(ii1, jj1, N_nodes, DemandS, solPart)
            for ii1 in tqdm(range(N_nodes), desc="Processing ii1 values", leave=False)
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
        for ii2 in range(ii1,N_nodes):
            for jj3 in range(jj2,N_nodes):
                for ii3 in range(ii2,N_nodes):
                    for jj4 in range(jj3,N_nodes):
                        for ii4 in range(ii3,N_nodes):
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
    for jj1 in tqdm(range(N_nodes), desc="Processing jj1 values", ncols=100):
        sol4_LC = Parallel(n_jobs=6)(
            delayed(compute_LinearComb4_for_jj1)(ii1, jj1, N_nodes, DemandS, solPart)
            for ii1 in tqdm(range(N_nodes), desc="Processing ii1 values", leave=False)
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

def compute_results(ppl, FullList, Delay, WaitingTime):
    for mult in MULTIPLIER:
        TotGamma2 = 0
        TotGamma3 = 0
        TotGamma4 = 0
        Cumul_delay2 = 0
        Cumul_delay3 = 0
        Cumul_delay4 = 0
        DemandS =  mult* OriginalDemand
        Demands_rp = np.zeros([N_nodes,N_nodes])

        for iii in range(FullList.shape[0]):
            # Seperate function to remove clutter
            Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3,Cumul_delay4, TotGamma4, DemandS, Demands_rp = calculate_gamma(FullList,DemandS, Delay, N_nodes, WaitingTime, Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, iii, Demands_rp)
        
        Demands_rp = Demands_rp - np.diag(np.diag(Demands_rp))

        #calculate solutions
        solBase =LTIFM_reb(mult*OriginalDemand,CITY_FOLDER)
        solNP = LTIFM_reb(DemandS,CITY_FOLDER)
        solRP = LTIFM_reb(Demands_rp,CITY_FOLDER)
        #reset diagonals (these are modified in LTIFM_reb)
        DemandS = DemandS - np.diag(np.diag(DemandS))
        Demands_rp = Demands_rp - np.diag(np.diag(Demands_rp))
        TrackDems_temp = [np.sum(mult*OriginalDemand), np.sum(DemandS),np.sum(Demands_rp)]
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
        with h5py.File(f"{CITY_FOLDER}/Results/Ppl_{ppl}Delay{Delay}WTime{WaitingTime}Dem{mult}.h5", "w") as h5file:
            # Save solution arrays in a flat structure
            for key, value in solutions_data.items():
                h5file.create_dataset(f"solutions/{key}", data=value)
            
            # Save additional tracking variables
            for key, value in additional_data.items():
                h5file.create_dataset(f"additional/{key}", data=value)

def A3_Main_K_general(ppl):
    #create list with the minimum path costs up to ppl depth
    FullList = Generate_Full_List(ppl)    
    for WaitingTime in tqdm(WAITINGTIMES, desc="Iterating through Waitingtimes", ncols=100):
        Parallel(n_jobs=4)(
            delayed(compute_results)(ppl, FullList, Delay, WaitingTime)
            for Delay in tqdm(DELAYS, desc="Iterating through Delays")
        )

def load_h5_grouping(ppl, Delay, WaitingTime, multiplicator):
    with h5py.File(f"{CITY_FOLDER}/Results/Ppl_{ppl}Delay{Delay}WTime{WaitingTime}Dem{multiplicator}.h5", "r") as h5file:
        # Load solution arrays
        x_vals = h5file["solutions/x"][:]
        xr_vals = h5file["solutions/xr"][:]
        individual_times = h5file["solutions/IndividualTimes"][:]
        obj_vals = h5file["solutions/obj"][:]

        # Separate out each solution, if needed
        solBase = {"x": x_vals[0], "xr": xr_vals[0], "IndividualTimes": individual_times[0], "obj": obj_vals[0], "Dem": h5file["solutions/Dem"][0]}
        solNP = {"x": x_vals[1], "xr": xr_vals[1], "IndividualTimes": individual_times[1], "obj": obj_vals[1], "Dem": h5file["solutions/Dem"][1]}
        solRP = {"x": x_vals[2], "xr": xr_vals[2], "IndividualTimes": individual_times[2], "obj": obj_vals[2], "Dem": h5file["solutions/Dem"][2]}

        # Load additional tracking data
        TrackDems_temp = h5file["additional/TrackDems_temp"][:]
        Cumul_delay = h5file["additional/Cumul_delay"][:]
        TotGamma = h5file["additional/TotGamma"][:]
    return solBase, solNP, solRP, TrackDems_temp, Cumul_delay, TotGamma

def Group_Together():
    ### grouping together
    for ppl in [2, 3, 4]:
        for Delay in DELAYS:
            for WaitingTime in WAITINGTIMES:
                Improv = []
                objs = []
                TrackDems = []
                TotG = []
                Del = []
                for multiplicator in MULTIPLIER:
                    solBase, solNP, solRP, TrackDems_temp, Cumul_delay, TotGamma = load_h5_grouping(ppl, Delay, WaitingTime, multiplicator)
                    # combine into 1
                    Improv.append(1-((solNP['obj']+solRP['obj'])/solBase['obj']))
                    objs.append([solBase['obj'],solNP['obj'],solRP['obj']])
                    TrackDems.append(TrackDems_temp)
                    TotG.append(TotGamma)
                    Del.append(Cumul_delay)
                np.savez(f"{CITY_FOLDER}/Results/Ppl{ppl}New_Delay{Delay}WTime{WaitingTime}.npz", TrackDems=TrackDems, objs=objs, Improv=Improv, TotG=TotG, Del=Del)     

def plot_pooling():
    """
    Plots percentage of pooled riders and average delay.
    saves in city/Result_plots/Pooled_riders.
    plots for all Ppls.
    """

    # Define a colormap and create a color iterator
    colors = plt.cm.tab10  # 'tab10' is similar to 'lines' in MATLAB
    color_indices = [0, 0.1, 0.2, 0.4]  # Example normalized indices
    CM = [colors(idx) for idx in color_indices]
    color_cycle = itertools.cycle(CM)

    # Define markers and create a marker iterator
    markers = ['o', '+', '*']  # Circle, plus, star
    marker_cycle = itertools.cycle(markers)

    # Track unique WaitingTimes and Delays for the legend
    
    for Ppl in [2,3,4]:
        fig, axs = plt.subplots(2, 1, layout='constrained')
        ax1 = axs[0]
        ax2 = axs[1]
        waitingtime_legend = []
        delay_legend = []
        for Delay, marker in zip(DELAYS, marker_cycle):
            delay_legend.append(Line2D([0], [0], color='black', marker=marker, linestyle='None', label=r'$\bar{\delta}$'+f'={Delay}'))        
            for WaitingTime, color in zip(WAITINGTIMES, color_cycle):
                if len(waitingtime_legend) < len(WAITINGTIMES):  # Ensure Delay legend is created only once
                    waitingtime_legend.append(Line2D([0], [0], color=color, lw=2, label=r'$\bar{t}$'+f'={WaitingTime}'))
                file = np.load(f"{CITY_FOLDER}/Results/Ppl{Ppl}New_Delay{Delay}WTime{WaitingTime}.npz")
                TrackDems = file['TrackDems']
                Del = file['Del']
                TotG = file['TotG']
                ax1.plot(TrackDems[:,0], 100- 100*TrackDems[:,1]/TrackDems[:,0], color=color, marker=marker)
                ax2.plot(TrackDems[:,0], np.sum(Del/np.sum(TotG,axis=1, keepdims=True),1), color=color, marker=marker)
                
        ax1.set_xscale('log')
        ax1.set_ylabel('Pooled Riders [%]')
        ax1.grid()
        # Combine legends into two columns
        legend_elements = waitingtime_legend + delay_legend
        fig.legend(handles=legend_elements, ncol=2)
        ax2.set_xscale('log')
        ax2.set_ylabel('Average Delay [mins]')
        ax2.set_xlabel('Demands Per Hour')
        ax2.grid()
        plt.savefig(f"{CITY_FOLDER}/Result_plots/Pooled_riders/Ppl{Ppl}.png", dpi=150)
        plt.close()


def plot_groups():
    plot_pooling()
    colors = plt.cm.tab10  # 'tab10' is similar to 'lines' in MATLAB
    color_indices = [0, 0.1, 0.2, 0.4]  # Example normalized indices
    CM = [colors(idx) for idx in color_indices]
    for Ppl in [2, 3, 4]:
        for Delay in DELAYS:
            for WaitingTime in WAITINGTIMES:
                file = np.load(f"{CITY_FOLDER}/Results/Ppl{Ppl}New_Delay{Delay}WTime{WaitingTime}.npz")
                TrackDems = file['TrackDems']
                TotG = file['TotG']
                PercNRP = TrackDems[:,1]/(TrackDems[:,2] + TrackDems[:,1])
                # Normalize TotG row-wise and scale it by (1 - PercNRP)
                TotG_normalized = TotG / np.sum(TotG, axis=1, keepdims=True)  # Normalize rows of TotG
                scaled_TotG = TotG_normalized * (1 - PercNRP[:, np.newaxis])  # Scale by (1 - PercNRP)
                # Combine PercNRP and the scaled TotG into a new array
                total_PercNRP = np.hstack([PercNRP[:, np.newaxis], scaled_TotG])

                fig, ax = plt.subplots()
                for i in range(total_PercNRP.shape[1]):  # Iterate over each column
                    if i == 0:
                        ax.bar(np.arange(total_PercNRP.shape[0]), total_PercNRP[:, i], color=CM[i], label=f'1 person')
                    else:
                        ax.bar(np.arange(total_PercNRP.shape[0]), total_PercNRP[:, i], bottom=np.sum(total_PercNRP[:, :i], axis=1), color=CM[i], label=f'{i+1} people')
                ax.legend()
                
                x_labels = np.round(TrackDems[:, 0], -1).astype(int)  # Round to nearest 10
                ax.set_xticks(np.arange(len(x_labels)))  # Set positions for x-ticks
                ax.set_xticklabels(x_labels.astype(str),rotation=45)
                plt.ylabel("Composition")
                plt.title(r'$\bar{\delta}$'+ f" = {Delay} min, " + r'$\bar{t}$' + f" = {WaitingTime} min")
                plt.savefig(f"{CITY_FOLDER}/Result_plots/Composition/Ppl{Ppl}Delay{Delay}WTime{WaitingTime}.png", dpi=150)
                plt.close()



def main():
    # create costs matrix
    solPart = A1_SP()

    # ### create solutions for different amount of linear combinations
    # sol2_LC = A2_LinearComb2(solPart)
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