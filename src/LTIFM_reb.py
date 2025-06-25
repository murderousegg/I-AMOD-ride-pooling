from scipy import io
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import kron, csr_matrix, hstack, eye
import time
import pwapprox as pw


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

def LTIFM_reb_sparse_eff(Demands, G, fcoeffs, n=3, theta_n=3, a=False, theta=False, exogenous_G=False):
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
    f   = m.addMVar(N_edges,            lb=0.0, name="f")
    t   = m.addMVar(N_edges,            lb=0.0, name="t")  
    # e = m.addVars(n, N_edges, vtype=GRB.CONTINUOUS, lb=0, name="e")

    m.addConstrs((f[i] == x[i, :].sum() + xr[i]
                  for i in range(N_edges)), name="LinkFlow")
    
    for i, (u, v) in enumerate(edge_order):
        t0   = G[u][v]['t_0']
        cap  = G[u][v]['capacity']
        bp_x = (cap * np.array(theta, dtype=float)).tolist()
        bp_y = [t0]
        for k in range(len(a)):
            delta_theta = theta[k+1] - theta[k]
            bp_y.append(bp_y[-1] + t0 * a[k] * delta_theta)

        # bp_y now has length n+1, aligned with bp_x
        m.addGenConstrPWL(f[i], t[i], bp_x, bp_y, name=f"PWL_{i}")
        if exogenous_G:
            t_exo = exogenous_G[u][v]['flow']
            m.addConstr(f[i] >= t_exo, name=f"ExoCap_{i}")
    
    obj = (f @ t).sum()
    
    m.setObjective(obj, GRB.MINIMIZE)
    A_flow = kron(eye(N_nodes, format="csr"), Binc, format="csc")
    b = Demands.flatten()
    m.addMConstr(A_flow, x.reshape(-1), '=', b, name="ODbalance")
    # Demand reshaped to 1D array
    

    I_n  = eye(N_nodes, format="csr")
    one  = csr_matrix(np.ones((N_nodes, 1)))
    A_x   = kron(I_n,  Binc, format="csc")
    A_xr  = kron(one, Binc, format="csc")
    A_inc = hstack([A_x, A_xr], format="csc")
    x_col  = x.reshape(-1, 1)
    xr_col = xr.reshape(-1, 1)
    vars_inc = gp.vstack([x_col, xr_col]).reshape(-1)
    m.addMConstr(A_inc, vars_inc, '=', b, name="Incidence")
    
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
    for i in range(N_nodes):                            # each network node
        m.addConstr(
            gp.quicksum(Binc[i, e] * (x[e,:].sum() + xr[e])
                        for e in range(N_edges)) == 0,
            name=f"Incidence_{i}"
        )
    
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

def LTIFM_reb_damp(Demands, G, fcoeffs, n=3, theta_n=3, a=False, theta=False, prev_x=None, prev_xr=None, labda=0.0):
    """Ride-pooling QP with proximal regularisation.
       If labda>0 and previous flows are supplied, solves

           min  G_cong(x,x_r) +
                λ/2 (‖x-prev_x‖² + ‖x_r-prev_xr‖²)
    """

    # ---------------- initial boiler-plate (unchanged) -------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    node_order = list(G.nodes())
    edge_order = list(G.edges())
    Binc = nx.incidence_matrix(G, nodelist=node_order,
                               edgelist=edge_order, oriented=True)
    [N_nodes, N_edges] = Binc.shape

    # make θ–a break-point vectors if the caller did not provide them
    fc = fcoeffs.copy()
    if (theta is False) or (a is False):
        theta, a, _ = get_approx_fun(fcoeffs=fc, nlines=n,
                                     range_=[0, theta_n])

    # balance demand matrix
    for ii in range(N_nodes):
        Demands[ii][ii] = -np.sum(Demands[:][ii]) - Demands[ii][ii]

    # ---------------- build Gurobi model ---------------------------
    m = gp.Model("LTIFM_reb")
    m.setParam("OutputFlag", 0)

    x   = m.addVars(N_edges, N_nodes, vtype=GRB.CONTINUOUS, name="x")
    x_r = m.addVars(N_edges,          vtype=GRB.CONTINUOUS, name="x_r")
    e   = m.addVars(n, N_edges,       vtype=GRB.CONTINUOUS,
                    lb=0, name="e")

    # ---------- original congestion / travel-time objective --------
    obj = gp.QuadExpr()          # declare once (can hold linear terms)

    
    for i in range(N_edges):
        t0 = G[edge_order[i][0]][edge_order[i][1]]['t_0']
        cap = G[edge_order[i][0]][edge_order[i][1]]['capacity']

        # linear cost of empty rebalancing
        obj.add(t0 * x_r[i])

        # piece-wise-linear approximation (same loops as your code)
        obj += gp.quicksum(
                t0 * a[l] / cap *
                (e[l, i] *
                 (gp.quicksum(((theta[k + 1] - theta[k]) * cap)
                      for k in range(0,l))) +
                e[l,i] * \
                  ((theta[l + 1] - theta[l]) * cap) +
                 (theta[l + 1] - theta[l]) * cap *
                 gp.quicksum(e[k, i] for k in range(l + 1, len(theta) - 1)))
                 for l in range(len(theta)-1)
            )

    # ---------- proximal regularisation (new lines) ----------------
    if labda > 0 and prev_x is not None and prev_xr is not None:
        prox = gp.QuadExpr()
        half_labda = 0.5 * float(labda)

        # ‖x - prev_x‖² term
        for i in range(N_edges):
            for j in range(N_nodes):
                diff = x[i, j] - float(prev_x[i, j])
                prox.add(diff * diff)

        # ‖x_r - prev_xr‖² term
        for i in range(N_edges):
            diff_r = x_r[i] - float(prev_xr[i])
            prox.add(diff_r * diff_r)
        obj.add(half_labda * prox)

    m.setObjective(obj, GRB.MINIMIZE)

    # ---------------- constraints (unchanged) ----------------------
    # piece-wise linear surplus e ≥ flow − θ Cap …
    m.addConstrs(
        e[l, i] >=
        x.sum(i, '*') + x_r[i] -
        theta[l] * G[edge_order[i][0]][edge_order[i][1]]['capacity'] -
        gp.quicksum(e[l+k+1,i] for k in range(n-l-1)) for i in range(N_edges) for l in range(n)
    )

    # demand balance via Kronecker incidence
    b = Demands.flatten()
    B_kron = kron(np.eye(N_nodes), Binc, format='csr')
    for i in range(len(b)):
        lhs = gp.LinExpr()
        row_start = B_kron.indptr[i]
        row_end   = B_kron.indptr[i + 1]
        row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values
        edge_indices = [j % N_edges for j in row_indices]  # Extract corresponding edge index
        node_indices = [j // N_edges for j in row_indices]  # Extract node index
        # Add terms corresponding to non-zero values in the row
        lhs.addTerms(row_data, [x[edge_indices[idx], node_indices[idx]] for idx in range(len(row_indices))])
        m.addConstr(lhs == b[i], name=f"DemandBalance_{i}")

    # non-negativity and incidence (station balance)
    m.addConstrs(x[i, j]  >= 0 for i in range(N_edges)
                                for j in range(N_nodes))
    m.addConstrs(x_r[j]   >= 0 for j in range(N_edges))

    m.addConstrs((
        gp.quicksum(Binc[i, j] * (x.sum(j, '*') + x_r[j])
                    for j in range(N_edges)) == 0
        for i in range(N_nodes)), name="Incidence")

    # ---------------- solve & package solution --------------------
    m.optimize()
    x_mat = np.array([[x[i, j].X
                       for j in range(N_nodes)]
                      for i in range(N_edges)])
    xr_vals = np.array([x_r[i].X for i in range(N_edges)])
    sol = dict(
        x   = x_mat,
        xr  = xr_vals,
        obj = m.ObjVal,
        IndividualTimes = 0,     # placeholder (unchanged)
        Dem = Demands
    )

    m.close()
    env.close()
    return sol
