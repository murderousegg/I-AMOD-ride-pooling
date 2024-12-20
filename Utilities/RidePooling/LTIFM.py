import numpy as np
from multiprocessing import Pool
from scipy import io
from scipy import sparse
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

def LTIFM(Demands, CITY_FOLDER):
        # initialize
        mat_graphs = io.loadmat(CITY_FOLDER+'/Graphs.mat', squeeze_me=True)
        mat_digraph = io.loadmat(CITY_FOLDER+'/digraph_data.mat', squeeze_me=True)
        DemandS = mat_graphs['DemandS']             #store demand
        FreeFlowTime = mat_graphs['FreeFlowTime']   #store freeflowtime
        NodesLoc = mat_graphs['NodesLoc']           #store nodesLoc

        # For the digraph
        adjMatrix = mat_digraph.get('adj_matrix')
        edgeList = mat_digraph.get('edge_list')

        # Create the graph using networkx
        if edgeList is not None:
                G_road = nx.DiGraph()
                G_road.add_edges_from(edgeList)
        elif adjMatrix is not None:
                G_road = nx.from_numpy_matrix(adjMatrix, create_using=nx.DiGraph)

        Adj = nx.Graph.adjacency(G_road)
        Binc = nx.incidence_matrix(G_road)
        [N_nodes,N_edges] = Binc.shape

        for ii in range(1,N_nodes):
                if Demands[ii,ii] >= 0:
                        Demands[ii,ii] = sum(Demands[:,ii]) - Demands[ii,ii]

        # create variables

        m = gp.Model("LTIFM") 
        x = m.addVars(N_edges*N_nodes,1,)
        x_r = m.addVars(N_edges,1)

        B_kron = np.kron(np.eye(N_nodes),Binc)
        FFT=np.kron(np.ones(1,N_nodes),FreeFlowTime) 
        b=np.reshape(Demands,[],1)

        m.setObjective(FFT*x, GRB.MINIMIZE)
        Cons = []
        m.addConstrs(B_kron*x == b)
        m.addConstrs(x >=  0 )
        m.addConstrs(x_r >= 0)
        m.addConstrs(Binc * ( (sum ( np.reshape(x,N_edges,[]) ,2) + x_r))  == 0)

        m.optimize()
        
        x = []
        xr = []
        for v in m.getVars():
                if "x" in v.VarName:
                        x.append(v.X)
                if "xr" in v.VarName:
                        xr.append(v.X)

        sol = {}
        sol["x"]=np.array(x)
        sol["xr"]=np.array(x_r)
        sol["obj"] = FFT*sol.x
        x_mat = np.reshape(sol.x,N_edges,N_nodes)
        x_mat[x_mat!=0] = 1
        sol["IndividualTimes"] = FreeFlowTime*x_mat
        sol["Dem"] = Demands
        return sol

