import numpy as np

def LTIFM2_SP(jj1, ii1, jj2, ii2, solPart, node_order):
    sol1 = {}
    sol1["obj"] = 0 #%solPart{jj1,ii1}.obj + solPart{jj2,ii2}.obj;  difference wrt no rp 
    sol1["order"]= [1, 1, 2, 2]
    sol1["Delay"] = [0,0];  #solPart{jj1,ii1}.obj

    sol2 = {}
    sol2["obj"] = solPart[jj1, jj2]["obj"] + solPart[jj2, ii1]["obj"] + solPart[ii1, ii2]["obj"] - solPart[jj1, ii1]["obj"] - solPart[jj2, ii2]["obj"]
    sol2["order"]= [1, 2, 1, 2]
    sol2["Delay"] = [solPart[jj1, jj2]["obj"] + solPart[jj2, ii1]["obj"] - solPart[jj1, ii1]["obj"], solPart[jj2, ii1]["obj"] + solPart[ii1, ii2]["obj"] - solPart[jj2, ii2]["obj"]]

    sol3 = {}
    sol3["obj"] = solPart[jj1, jj2]["obj"] + solPart[jj2, ii2]["obj"] + solPart[ii2, ii1]["obj"] - solPart[jj1, ii1]["obj"] - solPart[jj2, ii2]["obj"]
    sol3["order"]=[1, 2, 2, 1]
    sol3["Delay"] = [solPart[jj1, jj2]["obj"] + solPart[jj2, ii2]["obj"] + solPart[ii2, ii1]["obj"] - solPart[jj1, ii1]["obj"],0]

    # if jj1 == 3 and jj2 == 1 and ii1 == 24 and ii2 == 21:
    #     print(sol1["Delay"])
    #     print(sol2["Delay"])
    #     print(sol3["Delay"])
    #     print(sol2["obj"])
    #     print(sol3["obj"])
    #     asdfs
    # iterators = [jj1+1, ii1+1, jj2+1, ii2+1]
    iterators = [node_order[jj1], node_order[ii1], node_order[jj2], node_order[ii2]]
    if sol2["obj"] < 0:
        sol = [sol2["obj"]]
        sol.extend(sol2["Delay"])
        sol.extend(iterators)
        sol.extend(sol2["order"])
    elif  sol3["obj"] < 0:
        sol = [sol3["obj"]]
        sol.extend(sol3["Delay"])
        sol.extend(iterators)
        sol.extend(sol3["order"])
    else:
        sol = [sol1["obj"]]
        sol.extend(sol1["Delay"])
        sol.extend(iterators)
        sol.extend(sol1["order"])
    return sol

def LTIFM2_SP_all(solPart, N_nodes):
    """
    Same but using broadcasting to speed up computation times
    not tested
    """
    # Convert solPart into NumPy arrays for faster indexing and operations
    obj_array = np.array([[solPart[j][i]["obj"] for i in range(N_nodes)] for j in range(N_nodes)])

    results = []
    indices = np.arange(N_nodes)
    
    # Creating index grids using broadcasting-friendly indices
    J1, I1, J2, I2 = np.ix_(indices, indices, indices, indices)

    # Compute intermediate terms as arrays
    sol1_obj = np.zeros((N_nodes, N_nodes, N_nodes, N_nodes))
    sol1_order = [1, 1, 2, 2]
    sol1_delay = [0, 0]

    sol2_obj = obj_array[J1, J2] + obj_array[J2, I1] + obj_array[I1, I2] - obj_array[J1, I1] - obj_array[J2, I2]
    sol2_order = [1, 2, 1, 2]
    sol2_delay = np.stack([
        obj_array[J1, J2] + obj_array[J2, I1] - obj_array[J1, I1],
        obj_array[J2, I1] + obj_array[I1, I2] - obj_array[J2, I2]
    ], axis=-1)

    sol3_obj = obj_array[J1, J2] + obj_array[J2, I2] + obj_array[I2, I1] - obj_array[J1, I1] - obj_array[J2, I2]
    sol3_order = [1, 2, 2, 1]
    sol3_delay = np.stack([
        obj_array[J1, J2] + obj_array[J2, I2] + obj_array[I2, I1] - obj_array[J1, I1],
        np.zeros_like(obj_array[J1, J2])
    ], axis=-1)

    # Iterate over valid index combinations (J2 >= J1, I2 >= I1) only
    for j1 in range(N_nodes):
        for i1 in range(N_nodes):
            for j2 in range(j1, N_nodes):
                for i2 in range(i1, N_nodes):
                    iterators = [j1 + 1, i1 + 1, j2 + 1, i2 + 1]
                    
                    # Apply conditions and construct sol based on sol2_obj and sol3_obj
                    if sol2_obj[j1, i1, j2, i2] < 0:
                        sol = [sol2_obj[j1, i1, j2, i2]]
                        sol.extend(sol2_delay[j1, i1, j2, i2].tolist())
                        sol.extend(iterators)
                        sol.extend(sol2_order)
                    elif sol3_obj[j1, i1, j2, i2] < 0:
                        sol = [sol3_obj[j1, i1, j2, i2]]
                        sol.extend(sol3_delay[j1, i1, j2, i2].tolist())
                        sol.extend(iterators)
                        sol.extend(sol3_order)
                    else:
                        sol = [sol1_obj[j1, i1, j2, i2]]
                        sol.extend(sol1_delay)
                        sol.extend(iterators)
                        sol.extend(sol1_order)

                    results.append(sol)

    return results