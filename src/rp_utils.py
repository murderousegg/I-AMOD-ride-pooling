import os
import networkx as nx
import numpy as np
from scipy import sparse
from tqdm import tqdm
from scipy import io
from Utilities.RidePooling.LTIFM2_SP import LTIFM2_SP
from Utilities.RidePooling.LTIFM_reb import LTIFM_reb
from Utilities.RidePooling.calculate_gamma import calculate_gamma
from joblib import Parallel, delayed


def A1_SP(N_nodes, tNet, node_order, city_folder):
    """
    Generates a matrix of dictionaries containing the costs of an arc, 
    a sparse matrix with the possible arcs and the times between destinations
    """
    solPart = np.empty((N_nodes, N_nodes), dtype=object)
    for ii in range(0,N_nodes):
        for jj in range(0,N_nodes):
            Dems = np.zeros([N_nodes, N_nodes])
            #generate demand matrix
            Dems[ii,jj] = 1
            Dems[jj,jj] = -1

            if ii==jj:  #special case, set demand to 0 if origin and destination are equal
                Dems[jj,jj] = -0
            
            D = nx.shortest_path_length(tNet.G, source=node_order[jj], target=node_order[ii], weight="t_0") 
            # Store the results in solPart[jj][ii] as a dictionary
            solPart[jj, ii] = {
                'obj': D,
                'Dem': sparse.csr_matrix(Dems),  # Convert Dems to sparse format
                'IndividualTimes': D
            }
    # convert to dictionary for .mat convertion
    solPart_dic = {"solPart": np.array(solPart)}
    io.savemat(city_folder + "/solPart_pyth_" + city_folder+ ".mat", solPart_dic)
    return solPart

def compute_LinearComb2_for_jj1(jj1, N_nodes, solPart, node_order, city_folder):
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
                if not np.any([node_order[ii2] == node_order[jj2], node_order[ii1] == node_order[jj1]]):
                    # find the minimum cost for combination
                    opti = np.array([LTIFM2_SP(jj1,ii1,jj2,ii2,solPart, node_order),
                                     LTIFM2_SP(jj2,ii2,jj1,ii1,solPart, node_order)])
                    opti = opti[np.lexsort(opti[:, ::-1].T)]
                    sol2_LC[counter,:] = opti[0,:] # matrix with objective, delays, order
                    counter = counter+1
    sol2_LC = sol2_LC[:counter, :] # trim unused rows
    # remove rows with zero costs and large delays
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,0] == 0 ),0)  #cost
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,2] > 15 ),0)  #delay
    sol2_LC = np.delete(sol2_LC,np.argwhere(sol2_LC[:,1] > 15 ),0)  #delay
    #store in .npy file
    np.save(city_folder + "/L2/MatL2_" + f"{jj1+1}.npy", sol2_LC)
    return sol2_LC
    

def A2_LinearComb2(solPart, N_nodes, node_order, city_folder):
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
        os.mkdir(city_folder + "/L2")
    except FileExistsError:
        print(f"Directory '{city_folder + "/L2"}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{city_folder + "/L2"}'.")
    except Exception as e:
        print(f"An error occurred: {e}")    

    #using joblib:
    results = Parallel(n_jobs=4)(
        delayed(compute_LinearComb2_for_jj1)(jj1, N_nodes, solPart, node_order, city_folder)
        for jj1 in tqdm(range(N_nodes), desc="Processing jj1 values")
    )
    #store total in .mat file
    sol2_LC_arr = np.asarray([val for row in results for val in row])
    np.save(city_folder + "/MatL2.npy", sol2_LC_arr)
    return sol2_LC_arr


def Generate_Full_List(city_folder):
    sol2_LC =np.load(f'{city_folder}/MatL2.npy')
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

def compute_results(FullList, Delay, WaitingTime, Demand, tNet, N_nodes, fcoeffs, idx_map):
    TotGamma2 = 0
    TotGamma3 = 0
    TotGamma4 = 0
    Cumul_delay2 = 0
    Cumul_delay3 = 0
    Cumul_delay4 = 0
    OriginalDemand = Demand.copy()
    DemandS =  Demand.copy()
    Demands_rp = np.zeros([N_nodes,N_nodes])
    gamma_arr = np.zeros(FullList.shape[0])
    for iii in range(FullList.shape[0]):
        # Seperate function to remove clutter
        Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, DemandS, Demands_rp, gamma =\
              calculate_gamma(FullList,DemandS, Delay, N_nodes, WaitingTime, Cumul_delay2, TotGamma2, Cumul_delay3, TotGamma3, Cumul_delay4, TotGamma4, iii, Demands_rp, idx_map)
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

def update_t(tNet, tNet_cars, D_rp, FullList, gamma_arr, idx_map, node_order):
    #### take delays and waitingtime into account for t_1 ####
    OD_delays = np.zeros((D_rp.shape)) # expected delays
    Et = np.zeros((D_rp.shape)) # expected waiting time
    gamma_count = np.zeros((D_rp.shape))
    for iii in range(len(FullList)):
        if gamma_arr[iii] != 0:
            jj1 = idx_map[int(FullList[iii][5])]
            ii1 = idx_map[int(FullList[iii][6])]
            jj2 = idx_map[int(FullList[iii][7])]
            ii2 = idx_map[int(FullList[iii][8])]
            if np.array_equal(FullList[iii][13:17], [1, 2, 1, 2]):
                OD_delays[ii1][jj1] += gamma_arr[iii] *(nx.shortest_path_length(tNet_cars.G, source=node_order[ii1], target=node_order[ii2], weight='t_1')\
                                                    + nx.shortest_path_length(tNet_cars.G, source=node_order[ii2], target=node_order[jj1], weight='t_1')\
                                                        - nx.shortest_path_length(tNet_cars.G, source=node_order[ii1], target=node_order[jj1], weight='t_1'))
                OD_delays[ii2][jj2] += gamma_arr[iii] *(nx.shortest_path_length(tNet_cars.G, source=node_order[ii2], target=node_order[jj1], weight='t_1')\
                                                    + nx.shortest_path_length(tNet_cars.G, source=node_order[jj1], target=node_order[jj2], weight='t_1')\
                                                        - nx.shortest_path_length(tNet_cars.G, source=node_order[ii2], target=node_order[jj2], weight='t_1'))

            elif np.array_equal(FullList[iii][13:17], [1, 2, 2, 1]):
                OD_delays[ii1][jj1] += gamma_arr[iii] * (nx.shortest_path_length(tNet_cars.G, source=node_order[ii1], target=node_order[ii2], weight='t_1')\
                                                    + nx.shortest_path_length(tNet_cars.G, source=node_order[ii2], target=node_order[jj2], weight='t_1')\
                                                        + nx.shortest_path_length(tNet_cars.G, source=node_order[jj2], target=node_order[jj1], weight='t_1')\
                                                            - nx.shortest_path_length(tNet_cars.G, source=node_order[ii1], target=node_order[jj1], weight='t_1'))
            # store occurance of sequence
            gamma_count[ii1][jj1] += gamma_arr[iii]
            gamma_count[ii2][jj2] += gamma_arr[iii]
            #expected waiting time
            
            Et[ii1][jj1] += gamma_arr[iii] * (1/D_rp[ii1][jj1]+1/D_rp[ii2][jj2] - 2/(D_rp[ii1][jj1]+D_rp[ii2][jj2]))/2
            Et[ii2][jj2] += gamma_arr[iii] * (1/D_rp[ii1][jj1]+1/D_rp[ii2][jj2] - 2/(D_rp[ii1][jj1]+D_rp[ii2][jj2]))/2
    # weighted delays / total requests
    total_delay = np.divide(OD_delays+Et, D_rp, out=np.zeros_like(D_rp), where=D_rp!=0)
    # delay_cars = np.divide(OD_delays, D_rp, out=np.zeros_like(D_rp), where=D_rp!=0)
    
    for u,v,d in tNet.G_supergraph.edges(data=True):
        if d['type'] == 'rp':
            tNet.G_supergraph[u][v]['t_1'] = nx.shortest_path_length(tNet_cars.G, source=int(u[:-2]), target=int(v[:-2]), weight='t_1') + total_delay[idx_map[int(v[:-2])], idx_map[int(u[:-2])]]
            tNet.G_supergraph[u][v]['t_cars'] = nx.shortest_path_length(tNet_cars.G, source=int(u[:-2]), target=int(v[:-2]), weight='t_1')
        else:
            tNet.G_supergraph[u][v]['t_1'] = tNet.G_supergraph[u][v]['t_0']
    return OD_delays, Et