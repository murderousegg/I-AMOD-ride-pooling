import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import kron, eye, csr_matrix, hstack
import time
import src.pwapprox as pw


def eval_travel_time(x, fcoeffs):
    return sum([fcoeffs[i]*x**i for i in range(len(fcoeffs))])

def get_approx_fun(fcoeffs, range_=[0,2], nlines=3, theta=False):
    # Generate data
    x = [i  for i in list(np.linspace(range_[0], range_[1], 100))]
    y = [eval_travel_time(i, fcoeffs) for i in x]
    if theta==False:
        pws = pw.pwapprox(x, y, k=nlines)
        pws.fit_convex_boyd(N=30, L=30)
        rms = min(pws.rms_vec)
        i = pws.rms_vec.index(rms)
        a = pws.a_list[i]
        b = pws.b_list[i]
        theta = pws.thetas[i]
        theta.insert(0,0)
        theta.append(range_[1])
    else:
        pws = pw.pwapprox(x, y, k=nlines)
        pws.fit_convex_with_theta(theta)
        theta = theta
        a = pws.a
        rms = 0
    return  theta, a, rms

def LTIFM_reb(Demands, G, fcoeffs, n=3, theta_n=3, a=False, theta=False, exogenous_G=False):
    '''
    DEPRECATED
    Solve congestion aware vehicle routing problem.
    
    Parameters
    ----------
    Demands: demand matrix
    G: roadgraph
    fcoeffs: coeffficients for bpr function
    n: Number of sections in piecewise aprox
    theta_n: max expected overcapacity, used for calculating piecewise slopes
    
    Returns
    --------
    sol: dictionary with x, xr, objective and demands

    '''
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    start = time.time()
    # For the digraph
    
    weights = [G[u][v]['t_0'] for u,v in G.edges()]

    # Binc = nx.incidence_matrix(G)
    # Explicitly set the order of nodes and edges if needed
    node_order = list(G.nodes())
    edge_order = list(G.edges())

    Binc = nx.incidence_matrix(G, nodelist=node_order, edgelist=edge_order, oriented=True)
    [N_nodes,N_edges] = Binc.shape
    # Binc.sort_indices()

    fc = fcoeffs.copy()
    if (theta==False) or (a==False):
        theta, a, rms  = get_approx_fun(fcoeffs=fc, nlines=n, range_=[0,theta_n])

    for ii in range(N_nodes):
         Demands[ii][ii] = -np.sum(Demands[:][ii]) - Demands[ii][ii]
    
    # Initialize Gurobi model and variables
    m = gp.Model("LTIFM_reb")
    m.setParam('OutputFlag',0 )
    x = m.addVars(N_edges, N_nodes, vtype=GRB.CONTINUOUS, name="x")
    x_r = m.addVars(N_edges, vtype=GRB.CONTINUOUS, name="x_r")
    e = m.addVars(n, N_edges, vtype=GRB.CONTINUOUS, lb=0, name="e")

    # Set up objective
    # obj = gp.quicksum(weights[i] * x.sum(i,'*') for i in range(N_edges)) + gp.quicksum(weights[j] * x_r[j] for j in range(N_edges))
    if not exogenous_G:
        obj = gp.quicksum(\
                gp.quicksum(G[edge_order[i][0]][edge_order[i][1]]['t_0'] * a[l]/G[edge_order[i][0]][edge_order[i][1]]['capacity'] *( \
                e[l,i] * (0+gp.quicksum(((theta[k + 1] - theta[k])*G[edge_order[i][0]][edge_order[i][1]]['capacity']) for k in range(0,l))) \
                + e[l,i] * ((theta[l + 1] - theta[l])*G[edge_order[i][0]][edge_order[i][1]]['capacity'] ) \
                + (theta[l+1] - theta[l])*G[edge_order[i][0]][edge_order[i][1]]['capacity']*(0+gp.quicksum(e[k,i] for k in range(l+1, len(theta)-1))) \
                ) for l in range(len(theta)-1))  \
                + (G[edge_order[i][0]][edge_order[i][1]]['t_0']) * x_r[i]\
                for i in range(N_edges))
    elif exogenous_G:
        obj = gp.quicksum(\
                gp.quicksum(G[edge_order[i][0]][edge_order[i][1]]['t_0'] * a[l]/G[edge_order[i][0]][edge_order[i][1]]['capacity'] *( \
                e[l,i] * (0+gp.quicksum(((theta[k + 1] - theta[k])*G[edge_order[i][0]][edge_order[i][1]]['capacity']) for k in range(0,l))) \
                + e[l,i] * ((theta[l + 1] - theta[l])*G[edge_order[i][0]][edge_order[i][1]]['capacity'] ) \
                + (theta[l+1] - theta[l])*G[edge_order[i][0]][edge_order[i][1]]['capacity']*(0+gp.quicksum(e[k,i] for k in range(l+1, len(theta)-1))) \
                - e[l,i] * exogenous_G[edge_order[i][0]][edge_order[i][1]]['flow'] \
                ) for l in range(len(theta)-1))  \
                + (G[edge_order[i][0]][edge_order[i][1]]['t_0']) * x_r[i]\
                for i in range(N_edges))
    
    #quicksum(quicksum(
    # weights[i] * a[l]/cap[i] *
    #  (e[l,i] * 
    # (quicksum((theta[k+1]-theta[k])*cap[i] for k in range(l)))
    #  + e[l,i] * ((theta[l+1]-theta[l])*cap[i])+(theta[l+1] - theta[l])*cap[i]*(quicksum(e[l,i] for k in range(l+1, len(theta)-1)))-e[l,i])for l in range(len(theta)-1) + weights[i] *x_r[i] for i in range(N_edges)
    m.setObjective(obj, GRB.MINIMIZE)

    if not exogenous_G:
        m.addConstrs(e[l,i]\
                    >=  x.sum(i,'*') \
                    +  x_r[i] \
                    - theta[l]*G[edge_order[i][0]][edge_order[i][1]]['capacity'] \
                    - gp.quicksum(e[l+k+1,i] for k in range(n-l-1)) for i in range(N_edges) for l in range(n))
    elif exogenous_G:
        m.addConstrs(e[l,i]\
                    >=  x.sum(i,'*') \
                    +  x_r[i] \
                    + exogenous_G[edge_order[i][0]][edge_order[i][1]]['flow'] \
                    - theta[l]*G[edge_order[i][0]][edge_order[i][1]]['capacity'] \
                    - gp.quicksum(e[l+k+1,i] for k in range(n-l-1)) for i in range(N_edges) for l in range(n))

    # Demand reshaped to 1D array
    b = Demands.flatten()

    B_kron = kron(np.eye(N_nodes), Binc, format='csr')  # Sparse matrix

    # Use sparse matrix multiplication for the demand balance constraint
    # print(f"start = {time.time()-start}")
    for i in range(len(b)):
        lhs = gp.LinExpr()  # Initialize linear expression for constraint
        row_start = B_kron.indptr[i]  # Start of row i in CSR format
        row_end = B_kron.indptr[i + 1]  # End of row i
        row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values
        edge_indices = [j % N_edges for j in row_indices]  # Extract corresponding edge index
        node_indices = [j // N_edges for j in row_indices]  # Extract node index
        # Add terms corresponding to non-zero values in the row
        lhs.addTerms(row_data, [x[edge_indices[idx], node_indices[idx]] for idx in range(len(row_indices))])
        
        # Add constraint for row i
        m.addConstr(lhs == b[i], name=f"DemandBalance_{i}")

    # print(f"end = {time.time()-start}")

    m.addConstrs((x[i,j] >= 0 for i in range(N_edges) for j in range(N_nodes)), name="NonNegativeX")
    m.addConstrs((x_r[j] >= 0 for j in range(N_edges)), name="NonNegativeXR")

    m.addConstrs(
        (gp.quicksum(Binc[i, j] * (x.sum(j,'*') + x_r[j])
        for j in range(N_edges))
         == 0 for i in range(N_nodes)),
        name="Incidence")
    

    # Solve the model
    print("optimizing Y")
    m.optimize()

    # Extract solution
    x_mat = np.zeros((N_edges, N_nodes))
    for i in range(N_edges):
        for j in range(N_nodes):
            x_mat[i,j] = x[i,j].X
    xr_vals = np.array([v.X for v in m.getVars() if "x_r[" in v.varName])

    # Package solution
    sol = {
        "x": x_mat,
        "xr": xr_vals,
        "obj": obj.getValue()
    }
    

    # Reshape x to matrix and calculate individual times
    sol["IndividualTimes"] = 0
    sol["Dem"] = Demands
    m.close()
    env.close()
    
    return sol


def LTIFM_reb_sparse(Demands, G, fcoeffs, n=3, theta_n=3, a=False, theta=False, nash=False):
    '''
    Solve congestion aware vehicle routing problem. Implementation using sparse matrices
    for optimization. 
    
    Parameters
    ----------
    Demands: demand matrix
    G: roadgraph
    fcoeffs: coeffficients for bpr function
    n: Number of sections in piecewise aprox
    theta_n: max expected overcapacity, used for calculating piecewise slopes
    nash: check if we are doing penetration rate
    
    Returns
    --------
    sol: dictionary with x, xr, objective and demands

    '''
    node_order = list(G.nodes())
    edge_order = list(G.edges())

    Binc = nx.incidence_matrix(G, nodelist=node_order, edgelist=edge_order, oriented=True)
    [N_nodes,N_edges] = Binc.shape
    # Binc.sort_indices()
    edge_times = [G[u][v].get("t_0") for u, v in edge_order]
    capacities = [G[u][v].get("capacity") for u, v in edge_order]    
    cap_arr = np.asarray(capacities)

    # recalculate theta_n based on exogenous flow (if exists)
    if nash:
        exo_flow = [G[u][v]['exo_flow'] for u, v in edge_order]
        max_ratio = np.max(np.array(exo_flow) / np.maximum(cap_arr, 1e-9))
        theta_n = max(theta_n, 1.2*max_ratio)  # 20% headroom

    # create piecewise curve
    fc = fcoeffs.copy()
    if (theta==False) or (a==False):
        theta, a, rms  = get_approx_fun(fcoeffs=fc, nlines=n, range_=[0,theta_n])

    # precompute widths of sections in piecewise approx
    widths = np.diff(theta)
    seg_w = np.outer(widths, cap_arr)

    # make sure demands match expected shape (and - sum on diagonals)
    Dem = np.array(Demands, copy=True, dtype=float)
    row_sums = Dem.sum(axis=1)
    diag = np.diag(Dem)
    Dem[np.arange(N_nodes), np.arange(N_nodes)] = -(row_sums - diag)    

    # Initialize Gurobi model and variables
    m = gp.Model("LTIFM_reb")
    m.setParam('OutputFlag',0)
    m.setParam("Method", 2)
    m.setParam("Crossover", 0)
    x = m.addMVar((N_edges, N_nodes), lb=0, name="x")
    xr = m.addMVar(N_edges, lb=0, name="xr")
    e = m.addVars(range(n), range(N_edges), lb=0.0, ub={(l,i): (seg_w[l,i] if l < n-1 else GRB.INFINITY) for l in range(n) for i in range(N_edges)}, name="e")
    m.update()
    
    if not nash:    #check if exo flow should be present
        # Objective with congestion
        obj = gp.quicksum(\
                gp.quicksum(edge_times[i] * a[l]/capacities[i] *( \
                e[l,i] * (0+gp.quicksum(((theta[k + 1] - theta[k])*capacities[i]) for k in range(0,l))) \
                + e[l,i] * ((theta[l + 1] - theta[l])*capacities[i] ) \
                + (theta[l+1] - theta[l])*capacities[i]*(0+gp.quicksum(e[k,i] for k in range(l+1, len(theta)-1))) \
                ) for l in range(len(theta)-1))  \
                + (edge_times[i]) * xr[i]\
                for i in range(N_edges))
        m.setObjective(obj, GRB.MINIMIZE)
    elif nash:
        # same objectie, but broken up in parts
        exo_flow = [G[u][v]['exo_flow'] for u, v in edge_order]
        obj_terms = []
        for i in range(N_edges):
            t0 = edge_times[i]
            cap = capacities[i]
            xp  = exo_flow[i]
            for l in range(n):
                prefix = gp.quicksum((theta[k+1] - theta[k]) * cap for k in range(l))
                width_l = (theta[l+1] - theta[l]) * cap
                suffix = gp.quicksum(e[k,i] for k in range(l+1, n))
                obj_terms.append(
                    t0 * a[l]/cap * (
                        e[l,i] * (prefix) +
                        e[l,i] * (width_l) +
                        width_l * suffix -
                        e[l,i] * xp
                    )
                )
            # rebalancing penalty
            obj_terms.append(t0 * xr[i]) 
        obj = gp.quicksum(obj_terms)
        m.setObjective(obj, GRB.MINIMIZE)

    # epsilon constraints
    if not nash:
        m.addConstrs(e[l,i]\
                    >=  x[i,:].sum() \
                    +  xr[i] \
                    - theta[l]*capacities[i] \
                    - gp.quicksum(e[l+k+1,i] for k in range(n-l-1)) for i in range(N_edges) for l in range(n))
    elif nash:
        exo_flow = [G[u][v]['flow'] for u, v in edge_order]
        m.addConstrs(e[l,i]\
                    >=  x[i,:].sum() \
                    +  xr[i] \
                    + exo_flow[i] \
                    - theta[l]*capacities[i] \
                    - gp.quicksum(e[l+k+1,i] for k in range(n-l-1)) for i in range(N_edges) for l in range(n))
        
    # Demand reshaped to 1D array
    B = Binc.toarray().astype(float)
    for i in range(N_nodes):
        m.addMConstr(B, x[:,i], sense='=', b=Dem[i,:], name=f"DemandBalance_{i}")
    
    #start rebalancing constraints
    total_flow = gp.MLinExpr.zeros(N_edges)
    for c in range(N_nodes):
        total_flow += x[:, c]
    total_flow += xr
    for n_idx in range(N_nodes):
        expr = gp.LinExpr()
        for e_idx, coeff in enumerate(B[n_idx, :]):
            if coeff != 0:
                expr += coeff * total_flow[e_idx]
        m.addConstr(expr == 0.0, name=f"rebal_{n_idx}")

    # Solve the model
    m.update()
    m.printStats()
    m.optimize()
    # Extract solution
    x_mat   = x.X
    xr_vals = xr.X
    # Package solution
    sol = {
        "x": x_mat,
        "xr": xr_vals,
        "obj": obj.getValue()
    }
    
    # Reshape x to matrix and calculate individual times
    sol["IndividualTimes"] = 0
    sol["Dem"] = Dem
    m.close()
    
    return sol