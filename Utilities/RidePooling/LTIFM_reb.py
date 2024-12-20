from scipy import io
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import kron, csr_matrix
import time

def LTIFM_reb(Demands, city):    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    start = time.time()
    mat_digraph = io.loadmat(city+'/digraph_data.mat', squeeze_me=True)

    # For the digraph
    adjMatrix = mat_digraph.get('adj_matrix')
    edgeList = mat_digraph.get('edge_list')

    # Create the graph using networkx
    if edgeList is not None:
        G_road = nx.DiGraph()
        G_road.add_weighted_edges_from(edgeList, weight='weight')
    elif adjMatrix is not None:
        G_road = nx.from_numpy_matrix(adjMatrix, create_using=nx.DiGraph)

    # Binc = nx.incidence_matrix(G_road)
    # Explicitly set the order of nodes and edges if needed
    node_order = sorted(list(G_road.nodes()))
    edge_order = sorted(list(G_road.edges()))

    Binc = nx.incidence_matrix(G_road, nodelist=node_order, edgelist=edge_order, oriented=True)

    [N_nodes,N_edges] = Binc.shape
    Binc.sort_indices()

    # Convert Binc to a dense matrix (numpy array)
    # Binc = Binc.toarray()

    for ii in range(N_nodes):
         Demands[ii][ii] = -np.sum(Demands[:][ii]) - Demands[ii][ii]
    weights = edgeList[:,2]

    
    # Initialize Gurobi model and variables
    m = gp.Model("LTIFM_reb")
    x = m.addVars(N_edges * N_nodes, vtype=GRB.CONTINUOUS, name="x")
    x_r = m.addVars(N_edges, vtype=GRB.CONTINUOUS, name="x_r")

    # Set up objective
    # FFT = np.kron(np.ones((1, N_nodes)), weights)
    FFT = np.tile(weights, (1, N_nodes))
    
    m.setObjective(
        gp.quicksum(FFT[0, i] * x[i] for i in range(N_edges * N_nodes)) +
        gp.quicksum(weights[j] * x_r[j] for j in range(N_edges)),
        GRB.MINIMIZE
    )

    # Demand reshaped to 1D array
    b = Demands.reshape(-1, 1)

    B_kron = kron(np.eye(N_nodes), Binc, format='csr')  # Sparse matrix

    # Use sparse matrix multiplication for the demand balance constraint
    # print(f"start = {time.time()-start}")

    for i in range(len(b)):
        lhs = gp.LinExpr()  # Initialize linear expression for constraint
        row_start = B_kron.indptr[i]  # Start of row i in CSR format
        row_end = B_kron.indptr[i + 1]  # End of row i
        row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values
        
        # Add terms corresponding to non-zero values in the row
        lhs.addTerms(row_data, [x[j] for j in row_indices])
        
        # Add constraint for row i
        m.addConstr(lhs == b[i], name=f"DemandBalance_{i}")

    # print(f"end = {time.time()-start}")

    m.addConstrs((x[i] >= 0 for i in range(N_edges * N_nodes)), name="NonNegativeX")
    m.addConstrs((x_r[j] >= 0 for j in range(N_edges)), name="NonNegativeXR")

    # Reshape x variables for incidence matrix constraint
    m.addConstrs(
        (gp.quicksum(Binc[i, j] * (gp.quicksum(x[k] for k in range(j, N_edges * N_nodes, N_edges)) + x_r[j])
         for j in range(N_edges)) == 0 for i in range(N_nodes)),
        name="Incidence"
    )

    # Solve the model
    m.optimize()

    # Extract solution
    x_vals = np.array([v.X for v in m.getVars() if "x[" in v.varName])
    xr_vals = np.array([v.X for v in m.getVars() if "x_r[" in v.varName])

    # Package solution
    sol = {
        "x": x_vals,
        "xr": xr_vals,
        "obj": np.dot(FFT.flatten(), x_vals) + np.dot(weights, xr_vals)
    }

    # Reshape x to matrix and calculate individual times
    x_mat = np.reshape(sol['x'], (N_edges, N_nodes))
    x_mat[x_mat != 0] = 1
    sol["IndividualTimes"] = np.transpose(weights[:,np.newaxis]) @ x_mat
    sol["Dem"] = Demands
    return sol