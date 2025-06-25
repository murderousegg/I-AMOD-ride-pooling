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

def LTIFM_reb_sparse(Demands, G, fcoeffs, n=3, theta_n=3, a=False, theta=False, exogenous_G=False):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    start = time.time()
    # For the digraph

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
    x = m.addMVar((N_edges, N_nodes), lb=0, name="x")
    xr = m.addMVar(N_edges, lb=0, name="xr")
    e = m.addVars(n, N_edges, vtype=GRB.CONTINUOUS, lb=0, name="e")
    # Set up objective
    if not exogenous_G:
        obj = gp.quicksum(\
                gp.quicksum(G[edge_order[i][0]][edge_order[i][1]]['t_0'] * a[l]/G[edge_order[i][0]][edge_order[i][1]]['capacity'] *( \
                e[l,i] * (0+gp.quicksum(((theta[k + 1] - theta[k])*G[edge_order[i][0]][edge_order[i][1]]['capacity']) for k in range(0,l))) \
                + e[l,i] * ((theta[l + 1] - theta[l])*G[edge_order[i][0]][edge_order[i][1]]['capacity'] ) \
                + (theta[l+1] - theta[l])*G[edge_order[i][0]][edge_order[i][1]]['capacity']*(0+gp.quicksum(e[k,i] for k in range(l+1, len(theta)-1))) \
                ) for l in range(len(theta)-1))  \
                + (G[edge_order[i][0]][edge_order[i][1]]['t_0']) * xr[i]\
                for i in range(N_edges))
    elif exogenous_G:
        obj = gp.quicksum(\
                gp.quicksum(G[edge_order[i][0]][edge_order[i][1]]['t_0'] * a[l]/G[edge_order[i][0]][edge_order[i][1]]['capacity'] *( \
                e[l,i] * (0+gp.quicksum(((theta[k + 1] - theta[k])*G[edge_order[i][0]][edge_order[i][1]]['capacity']) for k in range(0,l))) \
                + e[l,i] * ((theta[l + 1] - theta[l])*G[edge_order[i][0]][edge_order[i][1]]['capacity'] ) \
                + (theta[l+1] - theta[l])*G[edge_order[i][0]][edge_order[i][1]]['capacity']*(0+gp.quicksum(e[k,i] for k in range(l+1, len(theta)-1))) \
                - e[l,i] * exogenous_G[edge_order[i][0]][edge_order[i][1]]['flow'] \
                ) for l in range(len(theta)-1))  \
                + (G[edge_order[i][0]][edge_order[i][1]]['t_0']) * xr[i]\
                for i in range(N_edges))
    
    m.setObjective(obj, GRB.MINIMIZE)
    if not exogenous_G:
        m.addConstrs(e[l,i]\
                    >=  x[i,:].sum() \
                    +  xr[i] \
                    - theta[l]*G[edge_order[i][0]][edge_order[i][1]]['capacity'] \
                    - gp.quicksum(e[l+k+1,i] for k in range(n-l-1)) for i in range(N_edges) for l in range(n))
    elif exogenous_G:
        m.addConstrs(e[l,i]\
                    >=  x[i,:].sum() \
                    +  xr[i] \
                    + exogenous_G[edge_order[i][0]][edge_order[i][1]]['flow'] \
                    - theta[l]*G[edge_order[i][0]][edge_order[i][1]]['capacity'] \
                    - gp.quicksum(e[l+k+1,i] for k in range(n-l-1)) for i in range(N_edges) for l in range(n))
    # Demand reshaped to 1D array
    b = Demands.flatten()
    I_n  = eye(N_nodes, format="csr")
    A_x   = kron(I_n,  Binc, format="csc")
    x_flat = x.reshape(-1)
    m.addMConstr(A_x, x_flat, '=', b, name="FlowConservation")
    one_row = csr_matrix(np.ones((1, N_nodes)))
    A_sum   = kron(one_row, Binc, format="csc")    
    A_inc   = hstack([A_sum, Binc], format="csc")

    xr_vec   = xr.reshape(-1, 1) # (E,   1)
    x_column = x_flat.reshape(-1, 1) # (E·N, 1)
    vars_inc = gp.vstack([x_column, xr_vec]).reshape(-1)

    zero_rhs = np.zeros(N_nodes)
    m.addMConstr(A_inc, vars_inc, '=', zero_rhs, name="Incidence")
    
    
    # Solve the model
    m.optimize()

    # Extract solution
    x_mat = np.zeros((N_edges, N_nodes))
    for i in range(N_edges):
        for j in range(N_nodes):
            x_mat[i,j] = x[i,j].X
    xr_vals = np.array([v.X for v in m.getVars() if "xr[" in v.varName])

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