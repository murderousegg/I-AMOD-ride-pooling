#########################################################################################
# This is a python implementation (converted from MATLAB) from the research paper       #
# "A Time-invariant Network Flow Model for Ride-pooling in Mobility-on-Demand Systems"  #
# Paper written by Fabio Paparella, Leonardo Pedroso, Theo Hofman, Mauro Salazar        #
# Python implementation by Frank Overbeeke                                              #
#########################################################################################

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import io
import networkx as nx
from Utilities.RidePooling.LTIFM2_SP import LTIFM2_SP
import os
import src.tnet as tnet
from gurobipy import *
import experiments.build_NYC_subway_net as nyc
import pickle
import time

CITY_FOLDER = "NYC"
WAITINGTIME = 2/60
DELAY = 0.1
tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)


with open("data/gml/NYC_small_roadgraph.gpickle", 'rb') as f:
    G_roadgraph = pickle.load(f)

car_node_order = list(G_roadgraph.nodes())
Binc_road = nx.incidence_matrix(G_roadgraph)
[N_nodes_road,N_edges_road] = Binc_road.shape

def compute_sol_row(ii, G, car_node_order):
    row = [None] * len(car_node_order)
    for jj in range(len(car_node_order)):
        try:
            D = nx.shortest_path_length(
                G,
                source=car_node_order[jj],
                target=car_node_order[ii],
                weight="t_0"
            )
        except nx.NetworkXNoPath:
            D = float('inf')  # or some fallback
        row[jj] = {'obj': D}
    return ii, row


def A1_SP():
    """
    Generates a matrix of dictionaries containing the costs of an arc, 
    a sparse matrix with the possible arcs and the times between destinations
    """
    solPart = np.empty((N_nodes_road, N_nodes_road), dtype=object)
    print('start')
    results_nested = Parallel(n_jobs=-1, backend='loky')(
    delayed(compute_sol_row)(ii, tNet.G, car_node_order)
    for ii in tqdm(range(N_nodes_road), desc="Processing ii values")
)
    for ii, row in results_nested:
        solPart[ii, :] = row
    np.save(f"{CITY_FOLDER}/solPart_pyth_{CITY_FOLDER}.npy", solPart)
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
    sol2_LC = np.zeros([75*N_nodes*N_nodes,11]);  
    counter=0
    #loops for exploring all combinations
    for ii1 in range(0,N_nodes):
        for ii2 in range(ii1,N_nodes):
            for jj2 in range(jj1,N_nodes):
                # only calculate when start and ends are not the same
                if not np.any([ii2 == jj2, ii1 == jj1, ii1==jj2, ii2==jj1]):
                    # find the minimum cost for combination
                    opti = np.array([LTIFM2_SP(jj1,ii1,jj2,ii2,solPart, car_node_order),
                                    LTIFM2_SP(jj2,ii2,jj1,ii1,solPart, car_node_order)])
                    opti = opti[np.lexsort(opti[:, ::-1].T)]
                    if opti[0,0] != 0 and opti[0,2]<=0.15 and opti[0,1]<=0.15:
                        sol2_LC[counter,:] = opti[0,:]
                        counter = counter+1
    sol2_LC = sol2_LC[:counter, :] # trim unused rows
    # remove rows with zero costs and large delays
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,0] == 0 ),0)  #cost
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,2] > 15 ),0)  #delay
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,1] > 15 ),0)  #delay
    #store in .npy file
    # np.savez_compressed(CITY_FOLDER + "/L2/MatL2_" + f"{jj1+1}.npz", sol2_LC)
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
        print(f"Directory '{CITY_FOLDER}/L2' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{CITY_FOLDER}/L2'.")
    except Exception as e:
        print(f"An error occurred: {e}")    

    #using joblib:
    results = Parallel(n_jobs=-1)(
        delayed(compute_LinearComb2_for_jj1)(jj1, N_nodes_road, solPart)
        for jj1 in tqdm(range(N_nodes_road), desc="Processing jj1 values")
    )
    #store total in .mat file
    sol2_LC_arr = np.asarray([val for row in results for val in row], dtype=np.float32)
    sol2_LC_arr[:, 0] /= 2
    sol2_LC_arr = sol2_LC_arr[sol2_LC_arr[:, 0].argsort()]
    sol2_LC_arr = sol2_LC_arr[sol2_LC_arr[:, 0] < -0.001]
    np.savez_compressed(CITY_FOLDER + "/MatL2.npz", sol2_LC_arr)

def main():
    # create costs matrix
    solPart = A1_SP()
    # # # ### create solutions for different amount of linear combinations
    # solPart = np.load(f"{CITY_FOLDER}/solPart_pyth_{CITY_FOLDER}.npy", allow_pickle=True)
    A2_LinearComb2(solPart)
   

if __name__ == "__main__":
    main()
    
