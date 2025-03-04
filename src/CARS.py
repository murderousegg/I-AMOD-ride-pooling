
from gurobipy import *
#import pwlf as pw
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import json
from src.utils import *
import src.pwapprox as pw
from scipy.sparse import kron


def get_theta(fun):
    return fun.fit_breaks[1:-1]

def get_beta(fun):
    return fun.slopes[1:]


def eval_travel_time(x, fcoeffs):
    return sum([fcoeffs[i]*x**i for i in range(len(fcoeffs))])


def eval_pw(a, b, theta, x):
    theta2  = theta.copy()
    theta2.append(1000)
    for i in range(len(theta2)-1):
        if (theta2[i]<=x) and (theta2[i+1]>x):
            y =  b[i]+a[i]*x
    return  y



def get_approx_fun(fcoeffs, range_=[0,2], nlines=3, theta=False, plot=False):
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

    if plot == True:
        fig, ax = plt.subplots(2)
        ax[0].plot(x, y , label = 'Original', color='k')
        ypws = [eval_pw(a,b, theta[0:-1], i) for i in x]
        ax[0].plot(x, ypws, label='pwlinear', color='red')
        for th in theta:
            ax[0].axvline(x=th, linestyle=':')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('t(x)')
        plt.legend()
        plt.tight_layout()
        pws.plot_rms(ax=ax[1])
        plt.show()
    return  theta, a, rms


@timeit
def add_demand_cnstr(m, tnet, x, bush=False):
    # Set Constraints
    if bush==False:
        for j in tnet.G_supergraph.nodes():
            for w, d in tnet.g.items():
                if j == w[0]:
                    m.addConstr(quicksum(m.getVarByName('x^'+str(w)+'_'+str(i)+'_'+str(j)) for i,l in tnet.G_supergraph.in_edges(nbunch=j)) + d == quicksum(m.getVarByName('x^'+str(w)+'_'+str(j)+'_'+str(k)) for l,k in tnet.G_supergraph.out_edges(nbunch=j)))
                elif j == w[1]:
                    m.addConstr(quicksum(m.getVarByName('x^' + str(w) + '_' + str(i) + '_' + str(j)) for i,l in tnet.G_supergraph.in_edges(nbunch=j)) == quicksum(m.getVarByName('x^' + str(w) + '_' + str(j) + '_' + str(k)) for l,k in tnet.G_supergraph.out_edges(nbunch=j)) + d)
                else:
                    m.addConstr(quicksum(m.getVarByName('x^' + str(w) + '_' + str(i) + '_' + str(j)) for i,l in tnet.G_supergraph.in_edges(nbunch=j)) == quicksum(m.getVarByName('x^' + str(w) + '_' + str(j) + '_' + str(k)) for l,k in tnet.G_supergraph.out_edges(nbunch=j)))
    else:
        p = {j:0  for j in tnet.G_supergraph.nodes()}
        for j in tnet.O:
            p[j] = sum([tnet.g[(s,t)] for s,t in tnet.g.keys() if t==j]) - sum([tnet.g[(s,t)] for s,t in tnet.g.keys() if s==j])

        # Global
        #[m.addConstr(quicksum([x[(i,j)] for i,l in tnet.G_supergraph.in_edges(nbunch=j)]) - quicksum([x[(j,k)] for l,k in tnet.G_supergraph.out_edges(nbunch=j)]) == p[j] ) for j in tnet.G_supergraph.nodes()]
        # Local
        #'''
        dsum = {s:sum([v for k,v in tnet.g.items() if k[0]==s]) for s in tnet.O}
        D = {s:list([d for o, d in tnet.g.keys() if o == s]) for s in tnet.O}
        #l = {s: [j for j in tnet.G_supergraph.nodes() if j not in set(D[s]) if j != s] for s in tnet.O}
        [m.addConstr(quicksum(m.getVarByName('x^' + str(s) + '_' + str(i) + '_' + str(j)) for i, l in tnet.G_supergraph.in_edges(nbunch=j)) - tnet.g[(s,j)] \
                        == quicksum(m.getVarByName('x^' + str(s) + '_' + str(j) + '_' + str(k)) for l, k in tnet.G_supergraph.out_edges(nbunch=j))) for s in tnet.O for j in D[s]]

        [m.addConstr(quicksum(m.getVarByName('x^' + str(s) + '_' + str(i) + '_' + str(s)) for i, l in tnet.G_supergraph.in_edges(nbunch=s)) \
                        == quicksum(m.getVarByName('x^' + str(s) + '_' + str(s) + '_' + str(k)) for l, k in tnet.G_supergraph.out_edges(nbunch=s)) - dsum[s]) for s in tnet.O]

        [m.addConstr(quicksum(m.getVarByName('x^' + str(s) + '_' + str(i) + '_' + str(j)) for i, l in tnet.G_supergraph.in_edges(nbunch=j)) \
                        == quicksum(m.getVarByName('x^' + str(s) + '_' + str(j) + '_' + str(k)) for l, k in tnet.G_supergraph.out_edges(nbunch=j))) \
                        for s in tnet.O for j in [j for j in tnet.G_supergraph.nodes() if j not in set(D[s]) if j != s]]
    m.update()

@timeit
def add_rebalancing_cnstr(m, tnet, xu):
    [m.addConstr(quicksum(m.getVarByName('x^R' + str(i) + '_' + str(j)) + xu[(i, j)] for i, l in tnet.G.in_edges(nbunch=j)) \
        == quicksum(m.getVarByName('x^R' + str(j) + '_' + str(k)) + xu[j, k] for l, k in tnet.G.out_edges(nbunch=j))) for j in tnet.G.nodes()]

    #[m.addConstr(m.getVarByName('x^R'+str(i)+'_'+str(j))==0) for i,j in tnet.G_supergraph.edges() if (type(i)!=int) or (type(j)!=int)]

    m.update()

@timeit
def set_optimal_flows(m , tnet, G_exogenous=False, bush=False):
    if bush:
        for i,j in tnet.G_supergraph.edges():
            tnet.G_supergraph[i][j]['flowNoRebalancing'] = sum(m.getVarByName('x^' + str(s) + '_' + str(i) + '_' + str(j)).X for s in tnet.O)
            tnet.G_supergraph[i][j]['flow'] = tnet.G_supergraph[i][j]['flowNoRebalancing']
            if isinstance(i, int) and isinstance(j, int):
                tnet.G_supergraph[i][j]['flowRebalancing'] = m.getVarByName('x^R' + str(i) + '_' + str(j)).X
                tnet.G_supergraph[i][j]['flow'] += tnet.G_supergraph[i][j]['flowRebalancing']
            #else:
            #tnet.G_supergraph[i][j]['flow'] = tnet.G_supergraph[i][j]['flowRebalancing'] + tnet.G_supergraph[i][j]['flowNoRebalancing']
            tnet.G_supergraph[i][j]['t_k'] = travel_time(tnet, i, j, G_exo=G_exogenous)
    else:
        for i,j in tnet.G_supergraph.edges():
            tnet.G_supergraph[i][j]['flowNoRebalancing'] = sum(m.getVarByName('x^' + str(w) + '_' + str(i) + '_' + str(j)).X for w in tnet.g.keys())
            tnet.G_supergraph[i][j]['flow'] = tnet.G_supergraph[i][j]['flowNoRebalancing']
            if isinstance(i, int) and isinstance(j, int):
                tnet.G_supergraph[i][j]['flowRebalancing'] = m.getVarByName('x^R' + str(i) + '_' + str(j)).X
                tnet.G_supergraph[i][j]['flow'] += tnet.G_supergraph[i][j]['flowRebalancing']

            #else:
            #    tnet.G_supergraph[i][j]['flow'] = m.getVarByName('x^R' + str(i) + '_' + str(j)).X + sum(m.getVarByName('x^' + str(w) + '_' + str(i) + '_' + str(j)).X for w in tnet.g.keys())
            #tnet.G_supergraph[i][j]['flowRebalancing'] = m.getVarByName('x^R' + str(i) + '_' + str(j)).X
            #tnet.G_supergraph[i][j]['flowNoRebalancing'] = sum(m.getVarByName('x^' + str(w) + '_' + str(i) + '_' + str(j)).X for w in tnet.g.keys())
            tnet.G_supergraph[i][j]['t_k'] = travel_time(tnet, i, j, G_exo=G_exogenous)

    for w,d in tnet.q.items():
        tnet.q[w] = sum([m.getVarByName(f"q^{z}_{w[0]}_{w[1]}").X for z,_ in tnet.g.items()])
        tnet.qrp[w] = sum([m.getVarByName(f"qrp^{z}_{w[0]}_{w[1]}").X for z,_ in tnet.g.items()])



@timeit
def set_optimal_rebalancing_flows(m,tnet):
    for i,j in tnet.G.edges():
        tnet.G_supergraph[i][j]['flow'] += m.getVarByName('x^R' + str(i) + '_' + str(j)).X
        tnet.G_supergraph[i][j]['flowRebalancing'] = m.getVarByName('x^R' + str(i) + '_' + str(j)).X
        tnet.G_supergraph[i][j]['flowNoRebalancing'] = tnet.G[i][j]['flow'] - tnet.G[i][j]['flowRebalancing']


def eval_obj_funct(tnet, G_exogenous):
    Vt, Vd, Ve = set_CARS_par(tnet)
    obj = Vt * get_totalTravelTime_without_Rebalancing(tnet, G_exogenous=G_exogenous)
    obj = obj + sum([(Vd*tnet.G_supergraph[i][j]['t_0'] + Ve *tnet.G_supergraph[i][j]['e']) * (tnet.G_supergraph[i][j]['flow']-tnet.G_supergraph[i][j]['flowNoRebalancing']) \
               for i,j in tnet.G.edges()])
    return obj/tnet.totalDemand

def set_CARS_par(tnet):
    # Set obj func parameters
    Vt = 24.4
    Vd = 0.286
    Ve = 0.247
    # Set the electricity constant
    ro = 1.25
    Af = 0.4
    cd = 1
    cr = 0.008
    mv = 750
    g = 9.81
    nu = 0.72

    for i,j in tnet.G_supergraph.edges():
        tnet.G_supergraph[i][j]['e'] =  (ro/2 *Af*cd * (tnet.G_supergraph[i][j]['t_0']/tnet.G_supergraph[i][j]['length'])**2 *cr * mv * g)* tnet.G_supergraph[i][j]['length']/nu
    return Vt, Vd, Ve

@timeit
def set_exogenous_flow(tnet, exogenous_G):
    # Set exogenous flow
    exo_G = tnet.G_supergraph.copy()
    for i, j in tnet.G_supergraph.edges():
        exo_G[i][j]['flow'] = 0
    if exogenous_G != False:
        for i,j in exogenous_G.edges():
            exo_G[i][j]['flow'] = exogenous_G[i][j]['flow']
    return  exo_G

@timeit
def get_obj_CARSn(m, tnet, xu,  theta, a, exogenous_G, linear=False):#, userCentric=False):
    #TODO: this could be written more efficiently, include user-centric approach
    #if linear:
    #userCentric = False
    #if userCentric != True:
    Vt, Vd, Ve = set_CARS_par(tnet)
    if linear == True:
        obj = quicksum(Vt * tnet.G_supergraph[i][j]['t_0'] * xu[(i, j)] for i,j in tnet.G_supergraph.edges())
        obj += quicksum(\
                quicksum( Vt * tnet.G_supergraph[i][j]['t_0'] * a[l]/tnet.G_supergraph[i][j]['capacity'] *( \
                m.getVarByName('e^'+str(l)+'_'+str(i)+'_'+str(j)) * (0+quicksum(((theta[k + 1] - theta[k])*tnet.G_supergraph[i][j]['capacity']) for k in range(0,l))) \
                + m.getVarByName('e^'+str(l)+'_'+str(i)+'_'+str(j)) * ( (theta[l + 1] - theta[l])*tnet.G_supergraph[i][j]['capacity'] ) \
                + (theta[l+1] - theta[l])*tnet.G_supergraph[i][j]['capacity']*(0+quicksum(m.getVarByName('e^'+str(k)+'_'+str(i)+'_'+str(j)) for k in range(l+1, len(theta)-1))) \
                - m.getVarByName('e^'+str(l)+'_'+str(i)+'_'+str(j)) * exogenous_G[i][j]['flow'] \
                ) for l in range(len(theta)-1))  \
                + (Vd * tnet.G_supergraph[i][j]['t_0'] + Ve * tnet.G_supergraph[i][j]['e']) * m.getVarByName('x^R' + str(i) + '_' + str(j))\
                for i,j in tnet.G.edges())
    else:
        obj = quicksum(Vt * tnet.G_supergraph[i][j]['t_0'] * xu[(i, j)] for i,j in tnet.G_supergraph.edges())
        obj = obj+ quicksum(\
                quicksum( Vt * tnet.G_supergraph[i][j]['t_0'] * a[l]/tnet.G_supergraph[i][j]['capacity'] *  m.getVarByName('e^'+str(l)+'_'+str(i)+'_'+str(j)) * (quicksum(((theta[k + 1] - theta[k])*tnet.G_supergraph[i][j]['capacity']) for k in range(0,l))) \
                + Vt * tnet.G_supergraph[i][j]['t_0'] * a[l]/tnet.G_supergraph[i][j]['capacity'] * m.getVarByName('e^'+str(l)+'_'+str(i)+'_'+str(j)) * ( m.getVarByName('e^'+str(l)+'_'+str(i)+'_'+str(j))) \
                + Vt * tnet.G_supergraph[i][j]['t_0'] * a[l]/tnet.G_supergraph[i][j]['capacity'] *(theta[l+1] - theta[l])*tnet.G_supergraph[i][j]['capacity']*(quicksum(m.getVarByName('e^'+str(k)+'_'+str(i)+'_'+str(j)) for k in range(l+1, len(theta)-1))) \
                - Vt * tnet.G_supergraph[i][j]['t_0'] * a[l]/tnet.G_supergraph[i][j]['capacity'] * m.getVarByName('e^'+str(l)+'_'+str(i)+'_'+str(j)) * exogenous_G[i][j]['flow'] \
                for l in range(len(theta)-1)) \
                + (Vd * tnet.G_supergraph[i][j]['t_0'] + Ve * tnet.G_supergraph[i][j]['e']) * m.getVarByName('x^R' + str(i) + '_' + str(j))\
                for i,j in tnet.G.edges())

    return obj

@timeit
def add_epsilon_cnstr(m, tnet, xu, n, theta, exogenous_G):
    [m.addConstr(m.getVarByName('e^'+str(l)+'_'+str(i)+'_'+str(j))\
                      >=  xu[(i,j)] \
                      +  m.getVarByName('x^R' + str(i) + '_' + str(j)) \
                      + exogenous_G[i][j]['flow'] \
                      - theta[l]*tnet.G_supergraph[i][j]['capacity'] \
                      - quicksum(m.getVarByName('e^'+str(l+k+1)+'_'+str(i)+'_'+str(j)) for k in range(n-l-1))) for i,j in tnet.G.edges() for l in range(n) ]


def add_bike_cnstr(m, tnet, xu):
    [m.addConstr(
    	quicksum(xu[(i,j)] for i,l in tnet.G_supergraph.in_edges(nbunch=j)) == quicksum(xu[(j,k)] for l,k in tnet.G_supergraph.out_edges(nbunch=j)))
    		for j in tnet.G_supergraph.nodes() if 'b' in str(j)]         
    m.update()

def solve_multimodal(tnet, sol_costs, fcoeffs, n=3, exogenous_G=False, rebalancing=True, linear=False, LP_method=-1,\
                       QP_method=-1, a=False, rp_list=[]):
    
    ridepool = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'rp']
    pedestrian = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'p']
    connector = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'f']
    connector_rp = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'fq']
    car = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 0]
    full_rp = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == "q"]

    ridepool_nodes = [node for node in tnet.G_supergraph.nodes() if "rp" in str(node)]
    car_nodes = [node for node in tnet.G_supergraph.nodes() if isinstance(node,int)]
    
    node_order = list(tnet.G_supergraph.nodes())
    edge_order = list(tnet.G_supergraph.edges())

    Binc = nx.incidence_matrix(tnet.G_supergraph, nodelist=node_order, edgelist=edge_order, oriented=True)
    [N_nodes,N_edges] = Binc.shape
    B_kron = kron(np.eye(N_nodes), Binc, format='csr')

    valid_pairs = set() # used to shape x
    for i in range(B_kron.shape[0]):
        row_start = B_kron.indptr[i]
        row_end = B_kron.indptr[i + 1]
        row_indices = B_kron.indices[row_start:row_end]

        edge_indices = [j % N_edges for j in row_indices]
        node_indices = [j // N_edges for j in row_indices]

        valid_pairs.update(zip(edge_indices, node_indices))  # Store unique (edge, node) pairs
    

    Binc_car = nx.incidence_matrix(tnet.G, nodelist=tnet.G.nodes(), edgelist=tnet.G.edges(), oriented=True)
    [N_car_nodes,N_car_edges] = Binc_car.shape
    

    node_index_map = {node: i for i, node in enumerate(tnet.G_supergraph.nodes())}
    w_node_index_map = {}
    count=0
    for node in tnet.G_supergraph.nodes():
        if "'" in str(node):
            w_node_index_map[node] = count
            count+=1
    
    demand_matrix = np.zeros((N_car_nodes, N_nodes))  # Square matrix
    for (origin, destination), demand in tnet.g.items():
        if origin in node_index_map and destination in node_index_map:
            i = w_node_index_map[origin]
            j = node_index_map[destination]
            demand_matrix[i, j] = demand  # Assign demand from origin to destination
    
    for ii in range(N_nodes):
        if "'" in str(node_order[ii]):
            demand_matrix[w_node_index_map[node_order[ii]],ii] =\
                -np.sum(demand_matrix[w_node_index_map[node_order[ii]],:]) - demand_matrix[w_node_index_map[node_order[ii]],ii]
            
    FFT = np.array([tnet.G_supergraph[u][v].get('t_0') for u, v in tnet.G_supergraph.edges()])

    # Start model
    m = Model('CARS')
    m.setParam('OutputFlag',0 )
    m.setParam('BarHomogeneous', 1)
    #m.setParam("LogToConsole", 0)
    #m.setParam("CSClientLog", 0)
    if linear:
        m.setParam('Method', LP_method)
    else:
        m.setParam('Method', QP_method)
    m.update()

    # x = m.addVars(N_edges, N_nodes, vtype=GRB.CONTINUOUS, lb=0, name="x")
    # x = m.addVars(valid_pairs, vtype=GRB.CONTINUOUS, lb=0, name="x")
    x = m.addVars(N_edges, N_car_nodes, vtype=GRB.CONTINUOUS, lb=0, name="x")
    x_r = m.addVars(N_edges, vtype=GRB.CONTINUOUS, lb=0, name="x_r")
    rp_full = m.addVars(len(rp_list), lb=0, vtype=GRB.CONTINUOUS, name="rp_full")
    psi = m.addVars(3, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="psi")
    m.update()

    m.setObjective(
        quicksum(FFT[i] * x.sum(i,"*") for i in range(N_edges))
        + quicksum(FFT[j] * x_r[j] for j in range(N_edges))
        + quicksum(sol_costs[i] * rp_full[i] for i in range(len(rp_list))),
        GRB.MINIMIZE)
    m.update()

    m.addConstr(psi.sum('*') == 1)
    

    # # Demand reshaped to 1D array
    b = demand_matrix.flatten()

    for i in range(b.shape[0]):
        lhs = LinExpr()  # Initialize linear expression for constraint
        row_start = B_kron.indptr[i]  # Start of row i in CSR format
        row_end = B_kron.indptr[i + 1]  # End of row i
        row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values

        edge_indices = [j % N_edges for j in row_indices]  # Extract corresponding edge index
        node_indices = [j // N_edges for j in row_indices]  # Extract node index
        
        lhs.addTerms(row_data, [x[edge_indices[idx], node_indices[idx]] for idx in range(len(row_indices))])
        # Add constraint for row i
        m.addConstr(lhs == b[i], name=f"DemandBalance_{i}")

    rp_flow = {}
    rp_full_flow = {}
    car_flow = {}
    A_q_up_flow = {}
    A_q_down_flow = {}

    for i, edge in enumerate(edge_order):
        if edge in ridepool:
            rp_flow[edge] = x.sum(i,'*')
        elif edge in full_rp:
            rp_full_flow[i] = x.sum(i,'*')
        elif edge in car:
            car_flow[edge] = x.sum(i,'*')
        elif edge in connector_rp:
            if "q" in edge[1]:
                A_q_up_flow[edge] = x.sum(i,"*")
            elif "q" in edge[0]:
                A_q_down_flow[edge] = x.sum(i,"*")

    for count,(i,var) in enumerate(rp_full_flow.items()):
        m.addConstr(var == quicksum(rp_full[k] for k in range(len(rp_list)) if i in rp_list[k]))
    
    m.addConstrs((x[i,j] >= 0 for i in range(N_edges) for j in range(N_nodes) if (i,j) in x), name="NonNegativeX")
    m.addConstrs((x_r[j] >= 0 for j in range(N_edges)), name="NonNegativeXR")
    
    ## NOTE: t_0 is relatively high ([60-120]), so there are relatively many vehicles per time unit on the road
    # maybe t_0 should be scaled?    
    vehicle_lim = 9000
    m.addConstr(quicksum(tnet.G_supergraph[edge[0]][edge[1]]['t_0']
                         *(x.sum(j,'*') + x_r[j]) 
                         for j,edge in enumerate(edge_order) 
                         if (isinstance(edge[0],int) and isinstance(edge[1],int)))
                         + quicksum(sol_costs[i] * rp_full[i] for i in range(len(rp_list)))
                              <= vehicle_lim)

    #### bad vehicle limit
    # m.addConstr(quicksum(x.sum(i,"*") + x_r[i] for i,edge in enumerate(edge_order) if edge in As_c_u_flow.keys())
    #             + quicksum(x.sum(i,'*')/2 for i,edge in enumerate(edge_order) if edge in As_rp_u_flow) <= vehicle_lim)

    # Reshape x variables for incidence matrix constraint
    m.addConstrs(
        (quicksum(Binc[i, j] * (x.sum(j,'*') + x_r[j])
         for j,edge in enumerate(edge_order) if isinstance(edge[0],int) and isinstance(edge[1],int))
         == 0 for i in range(N_nodes)),
        name="Incidence")
    
    m.update()
    m.optimize()
    # Extract solution
    x_mat = np.zeros((N_edges, N_nodes))
    for i in range(N_edges):
        for j in range(N_nodes):
            if (i,j) in x:
                x_mat[i,j] = x[i,j].X

    x_r_mat = np.zeros((N_edges))
    for i in range(N_edges):
        x_r_mat[i] = x_r[i].X
    
    selected_tours = [rp_list[i] for i in range(len(rp_list)) if rp_full[i].X!=0]
    arc_flows = x_mat.sum(axis=1) + x_r_mat
    for i in range(N_edges):
        tnet.G_supergraph[edge_order[i][0]][edge_order[i][1]]['flow'] = arc_flows[i]
    print(m)
    print(len(rp_list))
    return selected_tours



@timeit
def solve_cars_matrix_rp_reduced(tnet, sol_costs, fcoeffs, n=3, exogenous_G=False, rebalancing=True, linear=False, LP_method=-1,\
                       QP_method=-1, a=False, rp_list=[]):
    
    ridepool = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'rp']
    pedestrian = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'p']
    connector = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'f']
    connector_rp = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'fq']
    car = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 0]
    full_rp = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == "q"]

    ridepool_nodes = [node for node in tnet.G_supergraph.nodes() if "rp" in str(node)]
    car_nodes = [node for node in tnet.G_supergraph.nodes() if isinstance(node,int)]
    
    node_order = list(tnet.G_supergraph.nodes())
    edge_order = list(tnet.G_supergraph.edges())

    Binc = nx.incidence_matrix(tnet.G_supergraph, nodelist=node_order, edgelist=edge_order, oriented=True)
    [N_nodes,N_edges] = Binc.shape
    B_kron = kron(np.eye(N_nodes), Binc, format='csr')

    valid_pairs = set() # used to shape x
    for i in range(B_kron.shape[0]):
        row_start = B_kron.indptr[i]
        row_end = B_kron.indptr[i + 1]
        row_indices = B_kron.indices[row_start:row_end]

        edge_indices = [j % N_edges for j in row_indices]
        node_indices = [j // N_edges for j in row_indices]

        valid_pairs.update(zip(edge_indices, node_indices))  # Store unique (edge, node) pairs
    

    Binc_car = nx.incidence_matrix(tnet.G, nodelist=tnet.G.nodes(), edgelist=tnet.G.edges(), oriented=True)
    [N_car_nodes,N_car_edges] = Binc_car.shape
    

    node_index_map = {node: i for i, node in enumerate(tnet.G_supergraph.nodes())}
    w_node_index_map = {}
    count=0
    for node in tnet.G_supergraph.nodes():
        if "'" in str(node):
            w_node_index_map[node] = count
            count+=1
    
    demand_matrix = np.zeros((N_car_nodes, N_nodes))  # Square matrix
    for (origin, destination), demand in tnet.g.items():
        if origin in node_index_map and destination in node_index_map:
            i = w_node_index_map[origin]
            j = node_index_map[destination]
            demand_matrix[i, j] = demand  # Assign demand from origin to destination
    
    for ii in range(N_nodes):
        if "'" in str(node_order[ii]):
            demand_matrix[w_node_index_map[node_order[ii]],ii] =\
                -np.sum(demand_matrix[w_node_index_map[node_order[ii]],:]) - demand_matrix[w_node_index_map[node_order[ii]],ii]
            
    FFT = np.array([tnet.G_supergraph[u][v].get('t_0') for u, v in tnet.G_supergraph.edges()])

    # Start model
    m = Model('CARS')
    m.setParam('OutputFlag',0 )
    m.setParam('BarHomogeneous', 1)
    #m.setParam("LogToConsole", 0)
    #m.setParam("CSClientLog", 0)
    if linear:
        m.setParam('Method', LP_method)
    else:
        m.setParam('Method', QP_method)
    m.update()
    x = m.addVars(N_edges, N_car_nodes, vtype=GRB.CONTINUOUS, lb=0, name="x")
    x_r = m.addVars(N_edges, vtype=GRB.CONTINUOUS, lb=0, name="x_r")
    rp_full = m.addVars(len(rp_list), lb=0, vtype=GRB.CONTINUOUS, name="rp_full")
    m.update()

    m.setObjective(
        quicksum(FFT[i] * x.sum(i,"*") for i in range(N_edges))
        + quicksum(FFT[j] * x_r[j] for j in range(N_edges))
        + quicksum(sol_costs[i] * rp_full[i] for i in range(len(rp_list))),
        GRB.MINIMIZE)
    m.update()

    # # Demand reshaped to 1D array
    b = demand_matrix.flatten()

    for i in range(b.shape[0]):
        lhs = LinExpr()  # Initialize linear expression for constraint
        row_start = B_kron.indptr[i]  # Start of row i in CSR format
        row_end = B_kron.indptr[i + 1]  # End of row i
        row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values

        edge_indices = [j % N_edges for j in row_indices]  # Extract corresponding edge index
        node_indices = [j // N_edges for j in row_indices]  # Extract node index
        
        lhs.addTerms(row_data, [x[edge_indices[idx], node_indices[idx]] for idx in range(len(row_indices))])
        # Add constraint for row i
        m.addConstr(lhs == b[i], name=f"DemandBalance_{i}")

    rp_flow = {}
    rp_full_flow = {}
    car_flow = {}
    A_q_up_flow = {}
    A_q_down_flow = {}

    for i, edge in enumerate(edge_order):
        if edge in ridepool:
            rp_flow[edge] = x.sum(i,'*')
        elif edge in full_rp:
            rp_full_flow[i] = x.sum(i,'*')
        elif edge in car:
            car_flow[edge] = x.sum(i,'*')
        elif edge in connector_rp:
            if "q" in edge[1]:
                A_q_up_flow[edge] = x.sum(i,"*")
            elif "q" in edge[0]:
                A_q_down_flow[edge] = x.sum(i,"*")

    for count,(i,var) in enumerate(rp_full_flow.items()):
        m.addConstr(var == quicksum(rp_full[k] for k in range(len(rp_list)) if i in rp_list[k]))

    for count,pool in enumerate(rp_list):
        edge_1 = [int(str(i)[:-1]) for i in edge_order[pool[0]]]
        edge_2 = [int(str(i)[:-1]) for i in edge_order[pool[1]]]
        print(edge_1)
        print(edge_2)
        sdf

        a = count // N_car_nodes
        b = count % N_car_nodes

        # if b != a:
        #     m.addConstr(rp_demand[a,b] == var)
        # elif b == a:
        #     m.addConstr(rp_demand[b,b] + quicksum(rp_demand[p,b] for p in range(N_nodes) if p!=b) == 0)

        ### reconstruct routes (NOTE: OD's are reconstructed individually instead of as a pair)
        # a = count // N_car_nodes
        # b = count % N_car_nodes
        
        # if b != a:
        #     m.addConstr(rp_demand[a,b] == var)
        # elif b == a:
        #     m.addConstr(rp_demand[b,b] + quicksum(rp_demand[p,b] for p in range(N_nodes) if p!=b) == 0)

        # lhs = LinExpr()  # Initialize linear expression for constraint
        # row_start = B_kron.indptr[count]  # Start of row i in CSR format
        # row_end = B_kron.indptr[count + 1]  # End of row i
        # row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        # row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values
        # edge_indices = [j % N_edges for j in row_indices if edge_order[j%N_edges] in car]  # Extract corresponding edge index
        # node_indices = [j // N_edges for j in row_indices]  # Extract node index
        # row_data = [row_data[i] for i in range(len(row_indices)) if edge_order[row_indices[i]%N_edges] in car]
        # lhs.addTerms(row_data, [rp[edge_indices[idx], node_indices[idx]] for idx in range(len(edge_indices))])
        # m.addConstr(lhs == rp_demand[a,b], name=f"reconstruct_rp{a}_{b}")

    #### force going back to walking layer at the end of rp trip
    # m.addConstrs(quicksum(rp_full_flow[j] for j in rp_full_flow.keys() if edge_order[j][1]==i[0])
    #               == A_q_down_flow[i] for i in A_q_down_flow.keys())
    ### limit rp flow to outgoing demand
    # m.addConstrs(quicksum(x[i,j] for i,edge in enumerate(edge_order) if edge in full_rp) <= -b[j + j*N_nodes] for j in range(N_nodes))
    
    m.addConstrs((x[i,j] >= 0 for i in range(N_edges) for j in range(N_nodes) if (i,j) in x), name="NonNegativeX")
    m.addConstrs((x_r[j] >= 0 for j in range(N_edges)), name="NonNegativeXR")
    
    ## NOTE: t_0 is relatively high ([60-120]), so there are relatively many vehicles per time unit on the road
    # maybe t_0 should be scaled?    
    vehicle_lim = 9000
    m.addConstr(quicksum(tnet.G_supergraph[edge[0]][edge[1]]['t_0']
                         *(x.sum(j,'*') + x_r[j]) 
                         for j,edge in enumerate(edge_order) 
                         if (isinstance(edge[0],int) and isinstance(edge[1],int)))
                         + quicksum(sol_costs[i] * rp_full[i] for i in range(len(rp_list)))
                              <= vehicle_lim)

    #### bad vehicle limit
    # m.addConstr(quicksum(x.sum(i,"*") + x_r[i] for i,edge in enumerate(edge_order) if edge in As_c_u_flow.keys())
    #             + quicksum(x.sum(i,'*')/2 for i,edge in enumerate(edge_order) if edge in As_rp_u_flow) <= vehicle_lim)

    # Reshape x variables for incidence matrix constraint
    m.addConstrs(
        (quicksum(Binc[i, j] * (x.sum(j,'*') + x_r[j])
         for j,edge in enumerate(edge_order) if isinstance(edge[0],int) and isinstance(edge[1],int))
         == 0 for i in range(N_nodes)),
        name="Incidence")
    
    m.update()
    m.optimize()
    # Extract solution
    x_mat = np.zeros((N_edges, N_nodes))
    for i in range(N_edges):
        for j in range(N_nodes):
            if (i,j) in x:
                x_mat[i,j] = x[i,j].X

    x_r_mat = np.zeros((N_edges))
    for i in range(N_edges):
        x_r_mat[i] = x_r[i].X
    
    selected_tours = [rp_list[i] for i in range(len(rp_list)) if rp_full[i].X!=0]
    arc_flows = x_mat.sum(axis=1) + x_r_mat
    for i in range(N_edges):
        tnet.G_supergraph[edge_order[i][0]][edge_order[i][1]]['flow'] = arc_flows[i]
    print(m)
    print(len(rp_list))
    return selected_tours


@timeit
def solve_cars_matrix_rp(tnet, sol_costs, fcoeffs, n=3, exogenous_G=False, rebalancing=True, linear=False, LP_method=-1,\
                       QP_method=-1, a=False, rp_list=[]):
    
    ridepool = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'rp']
    pedestrian = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'p']
    connector = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'f']
    connector_rp = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 'fq']
    car = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == 0]
    full_rp = [(u, v) for (u, v, d) in tnet.G_supergraph.edges(data=True) if d['type'] == "q"]

    ridepool_nodes = [node for node in tnet.G_supergraph.nodes() if "rp" in str(node)]
    car_nodes = [node for node in tnet.G_supergraph.nodes() if isinstance(node,int)]
    
    node_order = list(tnet.G_supergraph.nodes())
    edge_order = list(tnet.G_supergraph.edges())

    Binc = nx.incidence_matrix(tnet.G_supergraph, nodelist=node_order, edgelist=edge_order, oriented=True)
    [N_nodes,N_edges] = Binc.shape

    Binc_car = nx.incidence_matrix(tnet.G, nodelist=tnet.G.nodes(), edgelist=tnet.G.edges(), oriented=True)
    [N_car_nodes,N_car_edges] = Binc_car.shape
    

    node_index_map = {node: i for i, node in enumerate(tnet.G_supergraph.nodes())}
    demand_matrix = np.zeros((N_nodes, N_nodes))  # Square matrix
    

    for (origin, destination), demand in tnet.g.items():
        if origin in node_index_map and destination in node_index_map:
            i = node_index_map[origin]
            j = node_index_map[destination]
            demand_matrix[i, j] = demand  # Assign demand from origin to destination
    for ii in range(N_nodes):
        demand_matrix[ii][ii] = -np.sum(demand_matrix[:][ii]) - demand_matrix[ii][ii]

    FFT = np.array([tnet.G_supergraph[u][v].get('t_0') for u, v in tnet.G_supergraph.edges()])

    # Start model
    m = Model('CARS')
    m.setParam('OutputFlag',0 )
    m.setParam('BarHomogeneous', 1)
    #m.setParam("LogToConsole", 0)
    #m.setParam("CSClientLog", 0)
    if linear:
        m.setParam('Method', LP_method)
    else:
        m.setParam('Method', QP_method)
    m.update()

    x = m.addVars(N_edges, N_nodes, vtype=GRB.CONTINUOUS, lb=0, name="x")
    x_r = m.addVars(N_edges, vtype=GRB.CONTINUOUS, lb=0, name="x_r")
    rp_full = m.addVars(len(rp_list), lb=0, vtype=GRB.CONTINUOUS, name="rp_full")
    rp = m.addVars(N_edges, N_nodes, vtype=GRB.CONTINUOUS, lb=0, name="rp")
    rp_demand = m.addVars(N_nodes,N_nodes, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rp_d")

    m.update()
    m.setObjective(
        quicksum(FFT[i] * x.sum(i,"*") for i in range(N_edges))
        + quicksum(FFT[j] * x_r[j] for j in range(N_edges))
        + quicksum(sol_costs[i] * rp_full[i] for i in range(len(rp_list))),
        GRB.MINIMIZE)
    m.update()


    # # Demand reshaped to 1D array
    b = demand_matrix.reshape(-1)
    B_kron = kron(np.eye(N_nodes), Binc, format='csr')

    for i in range(B_kron.shape[0]):
        lhs = LinExpr()  # Initialize linear expression for constraint
        row_start = B_kron.indptr[i]  # Start of row i in CSR format
        row_end = B_kron.indptr[i + 1]  # End of row i
        row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values

        edge_indices = [j % N_edges for j in row_indices]  # Extract corresponding edge index
        node_indices = [j // N_edges for j in row_indices]  # Extract node index
        
        lhs.addTerms(row_data, [x[edge_indices[idx], node_indices[idx]] for idx in range(len(row_indices))])
        # Add constraint for row i
        m.addConstr(lhs == b[i], name=f"DemandBalance_{i}")

    rp_flow = {}
    rp_full_flow = {}
    car_flow = {}
    As_rp_u_flow = {}
    As_rp_d_flow = {}
    A_q_up_flow = {}
    A_q_down_flow = {}

    for i, edge in enumerate(edge_order):
        if edge in ridepool:
            rp_flow[edge] = x.sum(i,'*')
        elif edge in full_rp:
            rp_full_flow[i] = x.sum(i,'*')
        elif edge in car:
            car_flow[edge] = x.sum(i,'*')
        elif edge in connector_rp:
            if "q" in edge[1]:
                A_q_up_flow[edge] = x.sum(i,"*")
            elif "q" in edge[0]:
                A_q_down_flow[edge] = x.sum(i,"*")

    for count,(i,var) in enumerate(rp_full_flow.items()):
        m.addConstr(var == quicksum(rp_full[k] for k in range(len(rp_list)) if i in rp_list[k]))

        ### reconstruct routes (NOTE: OD's are reconstructed individually instead of as a pair)
        # a = count // N_car_nodes
        # b = count % N_car_nodes
        
        # if b != a:
        #     m.addConstr(rp_demand[a,b] == var)
        # elif b == a:
        #     m.addConstr(rp_demand[b,b] + quicksum(rp_demand[p,b] for p in range(N_nodes) if p!=b) == 0)

        # lhs = LinExpr()  # Initialize linear expression for constraint
        # row_start = B_kron.indptr[count]  # Start of row i in CSR format
        # row_end = B_kron.indptr[count + 1]  # End of row i
        # row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        # row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values
        # edge_indices = [j % N_edges for j in row_indices if edge_order[j%N_edges] in car]  # Extract corresponding edge index
        # node_indices = [j // N_edges for j in row_indices]  # Extract node index
        # row_data = [row_data[i] for i in range(len(row_indices)) if edge_order[row_indices[i]%N_edges] in car]
        # lhs.addTerms(row_data, [rp[edge_indices[idx], node_indices[idx]] for idx in range(len(edge_indices))])
        # m.addConstr(lhs == rp_demand[a,b], name=f"reconstruct_rp{a}_{b}")

    # rp_list[k] = [edge 1, edge 2]
    # edge_list[i][0] for i in rp_list[k] ## where edge_list[i] = (node 1q, node 2q)

    #### force going back to walking layer at the end of rp trip
    # m.addConstrs(quicksum(rp_full_flow[j] for j in rp_full_flow.keys() if edge_order[j][1]==i[0])
    #               == A_q_down_flow[i] for i in A_q_down_flow.keys())

    m.addConstrs(quicksum(x[i,j] for i,edge in enumerate(edge_order) if edge in full_rp) <= -b[j + j*N_nodes] for j in range(N_nodes))
    
    m.addConstrs((x[i,j] >= 0 for i in range(N_edges) for j in range(N_nodes)), name="NonNegativeX")
    m.addConstrs((x_r[j] >= 0 for j in range(N_edges)), name="NonNegativeXR")
    m.addConstrs((rp[i,j] >= 0 for i in range(N_edges) for j in range(N_nodes)), name="NonNegativeRP")
    
    ## NOTE: t_0 is relatively high ([60-120]), so there are relatively many vehicles per time unit on the road
    # maybe t_0 should be scaled?    
    vehicle_lim = 9000
    m.addConstr(quicksum(tnet.G_supergraph[edge[0]][edge[1]]['t_0']
                         *(x.sum(j,'*') + x_r[j]) 
                         for j,edge in enumerate(edge_order) 
                         if (isinstance(edge[0],int) and isinstance(edge[1],int)))
                         + quicksum(sol_costs[i] * rp_full[i] for i in range(len(rp_list)))
                              <= vehicle_lim)

    #### bad vehicle limit
    # m.addConstr(quicksum(x.sum(i,"*") + x_r[i] for i,edge in enumerate(edge_order) if edge in As_c_u_flow.keys())
    #             + quicksum(x.sum(i,'*')/2 for i,edge in enumerate(edge_order) if edge in As_rp_u_flow) <= vehicle_lim)

    # Reshape x variables for incidence matrix constraint
    m.addConstrs(
        (quicksum(Binc[i, j] * (x.sum(j,'*') + x_r[j])
         for j,edge in enumerate(edge_order) if isinstance(edge[0],int) and isinstance(edge[1],int))
         == 0 for i in range(N_nodes)),
        name="Incidence")
    
    m.update()
    m.optimize()
    # Extract solution
    x_mat = np.zeros((N_edges, N_nodes))
    for i in range(N_edges):
        for j in range(N_nodes):
            x_mat[i,j] = x[i,j].X

    x_r_mat = np.zeros((N_edges))
    for i in range(N_edges):
        x_r_mat[i] = x_r[i].X
    
    rp_mat = np.zeros((N_edges, N_nodes))
    for i in range(N_edges):
        for j in range(N_nodes):
            rp_mat[i,j] = rp[i,j].X
    
    rp_d = np.zeros((N_nodes,N_nodes))
    for i in range(N_nodes):
        for j in range(N_nodes):
            rp_d[i,j] = rp_demand[i,j].X
    
    selected_tours = [rp_list[i] for i in range(len(rp_list)) if rp_full[i].X!=0]
    # for constr in m.getConstrs():
    #     if "reconstruct_rp1_1" in constr.ConstrName:
    #         print(f"Constraint: {constr.ConstrName}")
    #         expr = m.getRow(constr)
    #         terms = []
    #         for i in range(expr.size()):
    #             var = expr.getVar(i)
    #             coeff = expr.getCoeff(i)
    #             value = var.X  # Get the value of the variable
    #             terms.append(f"{coeff} * {var.VarName} (Value: {value})")
    #         print(f"Expression: {' + '.join(terms)} = {constr.RHS}")
    arc_flows = x_mat.sum(axis=1) + x_r_mat + rp_mat.sum(axis=1)
    for i in range(N_edges):
        tnet.G_supergraph[edge_order[i][0]][edge_order[i][1]]['flow'] = arc_flows[i]
    print(m)
    return selected_tours


def get_CARS_obj_val(tnet, G_exogenous):
    Vt, Vd, Ve = set_CARS_par(tnet)
    tt = get_totalTravelTime_without_Rebalancing(tnet, G_exogenous=G_exogenous)
    reb = get_rebalancing_total_cost(tnet)
    obj = Vt * tt + reb
    return obj

def get_rebalancing_total_cost(tnet):
    Vt, Vd, Ve = set_CARS_par(tnet)
    reb = get_rebalancing_flow(tnet)
    obj = sum((Vd * tnet.G_supergraph[i][j]['t_0'] + Ve * tnet.G_supergraph[i][j]['e']) * tnet.G_supergraph[i][j]['flowRebalancing']  for i,j in tnet.G_supergraph.edges())
    return obj

@timeit
def get_OD_result_flows(m, tnet, bush=False):
    dic = {}
    if bush==True:
        for s in tnet.O:
            dic[s] = {}
            for i,j in tnet.G_supergraph.edges():
                dic[s][(i,j)] = m.getVarByName('x^' + str(s) + '_' + str(i) + '_' + str(j)).X
    else:
        for w in tnet.g.keys():
            dic[w] = {}
            for i,j in tnet.G_supergraph.edges():
                dic[w][(i,j)] = m.getVarByName('x^' + str(w) + '_' + str(i) + '_' + str(j)).X
    return dic


def solve_rebalancing(tnet, exogenous_G=0):
    Vt, Vd, Ve = set_CARS_par(tnet)
    # Set exogenous flow
    if exogenous_G == 0:
        exogenous_G = tnet.G.copy()
        for i, j in tnet.G.edges():
            exogenous_G[i][j]['flow'] = 0

    m = Model('QP')
    m.setParam('OutputFlag', 1)
    m.setParam('BarHomogeneous', 0)
    m.setParam('Method', 1)
    m.update()

    # Define variables
    [m.addVar(lb=0, name='x^R' + str(i) + '_' + str(j)) for i, j in tnet.G.edges()]
    m.update()

    # Set objective
    #obj = quicksum((Vd * tnet.G[i][j]['t_0'] + Ve * tnet.G[i][j]['e']) * m.getVarByName('x^R' + str(i) + '_' + str(j)) for i,j in tnet.G.edges())
    obj = quicksum((Vt * tnet.G[i][j]['t_k']) * m.getVarByName('x^R' + str(i) + '_' + str(j)) for i, j in
        tnet.G.edges())
    m.update()

    # Set Constraints
    for j in tnet.G.nodes():
        m.addConstr(quicksum(m.getVarByName('x^R' + str(i) + '_' + str(j)) + tnet.G[i][l]['flow'] for i, l in
                             tnet.G.in_edges(nbunch=j)) \
                    == quicksum(m.getVarByName('x^R' + str(j) + '_' + str(k)) + tnet.G[j][k]['flow'] for l, k in
                                tnet.G.out_edges(nbunch=j)))
    m.update()
    m.update()

    m.setObjective(obj, GRB.MINIMIZE)
    m.update()
    m.optimize()
    # saving  results
    set_optimal_rebalancing_flows(m,tnet)
    return m.Runtime

def get_totalTravelTime_approx(tnet, fcoeffs, xa):
    fun = get_approx_fun(fcoeffs, xa=xa, nlines=2)
    beta = get_beta(fun)
    theta = get_theta(fun)
    print(beta)
    obj=0
    for i,j in tnet.G_supergraph.edges():
        if tnet.G_supergraph[i][j]['flow']/tnet.G_supergraph[i][j]['capacity'] <= xa:
            obj += tnet.G_supergraph[i][j]['flow'] * tnet.G_supergraph[i][j]['t_0']
        else:
            obj += tnet.G_supergraph[i][j]['flow'] * \
                   (tnet.G_supergraph[i][j]['t_0'] + (beta[0] *tnet.G_supergraph[i][j]['flow'] /tnet.G_supergraph[i][j]['capacity']))
    return obj


def travel_time(tnet, i, j, G_exo=False):
    """
    evalute the travel time function for edge i->j

    Parameters
    ----------
    tnet: transportation network object
    i: starting node of edge
    j: ending node of edge

    Returns
    -------
    float

    """
    if G_exo == False:
        return sum(
            [tnet.fcoeffs[n] * (tnet.G_supergraph[i][j]['flow'] / tnet.G_supergraph[i][j]['capacity']) ** n for n in
             range(len(tnet.fcoeffs))])
    else:
        return sum([tnet.fcoeffs[n] * ((tnet.G_supergraph[i][j]['flow'] + G_exo[i][j]['flow'])/ tnet.G_supergraph[i][j]['capacity']) ** n for n in range(len(tnet.fcoeffs))])


def get_totalTravelTime(tnet, G_exogenous=False):
    """
    evalute the travel time function on the SuperGraph level

    Parameters
    ----------

    tnet: transportation network object

    Returns
    -------
    float

    """
    if G_exogenous == False:
        return sum([tnet.G_supergraph[i][j]['flow'] * tnet.G_supergraph[i][j]['t_0'] * travel_time(tnet, i, j) for i, j in tnet.G_supergraph.edges()])
    else:
        ret = 0
        for i,j in tnet.G_supergraph.edges():
            if isinstance(tnet.G_supergraph[i][j]['type'], float)==True:
                ret += tnet.G_supergraph[i][j]['flow'] * tnet.G_supergraph[i][j]['t_0'] * travel_time_without_Rebalancing(tnet, i, j, G_exogenous[i][j]['flow'])
            else:
                ret += tnet.G_supergraph[i][j]['flow'] * tnet.G_supergraph[i][j]['t_0'] * travel_time_without_Rebalancing(tnet, i, j)
        return ret


def travel_time_without_Rebalancing(tnet, i, j, exo=0):
    """
    evalute the travel time function for edge i->j

    Parameters
    ----------
    tnet: transportation network object
    i: starting node of edge
    j: ending node of edge

    Returns
    -------
    float

    """
    return sum(
        [tnet.fcoeffs[n] * ((tnet.G_supergraph[i][j]['flowNoRebalancing'] +exo )/ tnet.G_supergraph[i][j]['capacity']) ** n for n in range(len(tnet.fcoeffs))])

def get_totalTravelTime_without_Rebalancing(tnet, G_exogenous=False):
    """
    evalute the travel time function on the SuperGraph level

    Parameters
    ----------

    tnet: transportation network object

    Returns
    -------
    float

    """
    if G_exogenous==False:
        return sum([tnet.G_supergraph[i][j]['flowNoRebalancing'] * tnet.G_supergraph[i][j][
            't_0'] * travel_time(tnet, i, j) for i, j in
                    tnet.G_supergraph.edges()])
    else:
        return sum([tnet.G_supergraph[i][j]['flowNoRebalancing'] * tnet.G_supergraph[i][j][
            't_0'] * travel_time(tnet, i, j, G_exo=G_exogenous) for i, j in
                    tnet.G_supergraph.edges()])



def get_pedestrian_flow(tnet):
    """
    get pedestrian flow in a supergraph

    Parameters
    ----------

    tnet: transportation network object

    Returns
    -------
    float

    """
    return sum([tnet.G_supergraph[i][j]['flow']*tnet.G_supergraph[i][j]['length'] for i,j in tnet.G_supergraph.edges() if tnet.G_supergraph[i][j]['type']=='p'])


def get_layer_flow(tnet, symb="'"):
    """
    get  flow in a layer of supergraph

    Parameters
    ----------

    tnet: transportation network object

    Returns
    -------
    float

    """
    return sum([tnet.G_supergraph[i][j]['flow']*tnet.G_supergraph[i][j]['length'] for i,j in tnet.G_supergraph.edges() if tnet.G_supergraph[i][j]['type']==symb])


def get_amod_flow(tnet):
    """
    get amod flow in a supergraph

    Parameters
    ----------

    tnet: transportation network object

    Returns
    -------
    float

    """
    return sum([tnet.G_supergraph[i][j]['flowNoRebalancing']*tnet.G_supergraph[i][j]['length'] for i,j in tnet.G.edges()])

def get_rebalancing_flow(tnet):
    """
    get rebalancing flow in a supergraph

    Parameters
    ----------

    tnet: transportation network object

    Returns
    -------
    float

    """
    return sum([(tnet.G_supergraph[i][j]['flow']-tnet.G_supergraph[i][j]['flowNoRebalancing'])*tnet.G_supergraph[i][j]['length'] for i,j in tnet.G.edges()])


def UC_fcoeffs(fcoeffs):
    f = [fcoeffs[i]/(i+1) for i in range(len(fcoeffs))]
    return f

def plot_supergraph_car_flows(tnet, weight='flow', width=3, cmap=plt.cm.Blues):
    #TODO: add explaination
    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(tnet.G, 'pos')
    d = {(i,j): tnet.G_supergraph[i][j][weight] for i,j in tnet.G.edges()}
    edges, weights = zip(*d.items())
    labels =  {(i,j): int(tnet.G_supergraph[i][j][weight]) for i,j in tnet.G.edges()}
    nx.draw(tnet.G, pos, node_color='b', edgelist=edges, edge_color=weights, width=width, edge_cmap=cmap)
    nx.draw_networkx_edge_labels(tnet.G, pos=pos, edge_labels=labels)
    return fig, ax

def plot_supergraph_pedestrian_flows(G, weight='flow', width=3, cmap=plt.cm.Blues):
	#TODO: add explaination
	fig, ax = plt.subplots()
	pos = nx.get_node_attributes(G, 'pos')
	edges, weights = zip(*nx.get_edge_attributes(G, weight).items())
	nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=width, edge_cmap=cmap)
	return fig, ax

def supergraph2G(tnet):
    # TODO: add explaination
    tnet.G = tnet.G_supergraph.subgraph(list([i for i in tnet.G.nodes()]))

def G2supergraph(tnet):
    # TODO: add explaination
    #tnet.G_supergraph = tnet.G
    for i,j in tnet.G_supergraph.edges():
        try:
            tnet.G_supergraph[i][j]['flow'] = tnet.G[i][j]['flow']
        except:
            tnet.G_supergraph[i][j]['flow'] = 0
        tnet.G_supergraph[i][j]['flowNoRebalancing'] = tnet.G_supergraph[i][j]['flow']

def add_G_flows_no_rebalancing(array):
    # TODO: add description

    G = array[0].copy()
    for tn in array[1:]:
        for i, j in G.edges():
            G[i][j]['flow'] += tn[i][j]['flowNoRebalancing']
    return G


def solveMSAsocialCARS(tnet, exogenous_G=False):
    runtime = time.process_time()
    if exogenous_G == False:
        tnet.solveMSAsocial_supergraph()
    else:
        tnet.solveMSAsocial_supergraph(exogenous_G=exogenous_G)
    t = time.process_time() - runtime
    G2supergraph(tnet)
    return t, tnet.TAP.RG


def nx2json(G, fname, exo=False):
    if exo==False:
        D = G.copy()
        for i,j in D.edges():
            D[i][j]['flow'] = 0
        with open(fname, 'w') as outfile1:
            outfile1.write(json.dumps(json_graph.node_link_data(D)))
    else:
        with open(fname, 'w') as outfile1:
            outfile1.write(json.dumps(json_graph.node_link_data(exo)))

'''
def solve_social_Julia(tnet, exogenous_G=False):

    # Save to json files
    nx.write_gml(tnet.G, "tmp/G.txt")
    nx.write_graphml(tnet.G_supergraph, "tmp/G_supergraph.txt")
    if exogenous_G != False:
        nx.write_graphml(exogenous_G, "tmp/exogenous_G.txt")
    
    f = open("tmp/g.txt","w")
    f.write( str(tnet.g) )
    f.close()

    f = open("tmp/fcoeffs.txt","w")
    f.write( str(tnet.fcoeffs) )
    f.close()
'''


def juliaJson2nx(tnet, dict, exogenous_G=False):
    d = {}
    for key in dict.keys():
        orig, dest = key.split(',')
        orig = orig.split("(")[1].replace('"', '').replace(' ', '')
        dest = dest.split(")")[0].replace('"', '').replace(' ', '')
        if "'" in orig:
            s = orig
        else:
            s = int(orig)
        if "'" in dest:
            t = dest
        else:
            t = int(dest)
        tnet.G_supergraph[s][t]['flow'] = dict[key]['flow']
        tnet.G_supergraph[s][t]['flowNoRebalancing'] = dict[key]['flow']
        if exogenous_G==False:
            tnet.G_supergraph[s][t]['t_k'] = travel_time(tnet,s,t, G_exo=exogenous_G)
        else:
            tnet.G_supergraph[s][t]['t_k'] = travel_time(tnet, s, t, G_exo=exogenous_G)


def solve_social_Julia(tnet, exogenous_G=False):
    # Save to json files
    nx2json(tnet.G, "tmp/G.json")
    nx2json(tnet.G_supergraph, "tmp/G_supergraph.json")
    nx2json(tnet.G_supergraph, "tmp/exogenous_G.json", exo=exogenous_G)

    js = json.dumps({str(k):v for k,v in tnet.g.items()})
    f = open("tmp/g.json", "w")
    f.write(js)
    f.close()

    f = open("tmp/fcoeffs.json", "w")
    f.write(str(tnet.fcoeffs))
    f.close()

    # Solve system-centric in julia
    shell("julia src/CARS.jl", printOut=True)

    # Parse results back
    dict_G = json2dict("tmp/out.json")
    juliaJson2nx(tnet, dict_G, exogenous_G=exogenous_G)
    # Get solve time
    f = open('tmp/solvetime.txt')
    line = f.readline()
    f.close()
    solvetime = float(line)
    shell("rm tmp/out.json", printOut=False)
    shell("rm tmp/G.json", printOut=False)
    shell("rm tmp/G_supergraph.json", printOut=False)
    shell("rm tmp/exogenous_G.json", printOut=False)
    shell("rm tmp/g.jsonn", printOut=False)
    shell("rm tmp/fcoeffs.json", printOut=False)
    shell("rm tmp/solvetime.txt", printOut=False)

    return solvetime
    #TODO: add delete funtion of out json




def solve_social_altruistic_Julia(tnet, exogenous_G=False):
    # Save to json files
    nx2json(tnet.G, "tmp/G.json")
    nx2json(tnet.G_supergraph, "tmp/G_supergraph.json")
    if exogenous_G != False:
        nx2json(exogenous_G, "tmp/exogenous_G.json", exo=True)
    else:
        nx2json(tnet.G, "tmp/exogenous_G.json", exo=False)

    js = json.dumps({str(k):v for k,v in tnet.g.items()})
    f = open("tmp/g.json", "w")
    f.write(js)
    f.close()

    f = open("tmp/fcoeffs.json", "w")
    f.write(str(tnet.fcoeffs))
    f.close()

    # Solve system-centric in julia
    shell("julia src/CARS_altruistic.jl", printOut=False)

    # Parse results back
    dict_G = json2dict("tmp/out.json")
    juliaJson2nx(tnet, dict_G)



'''
import cvxpy as cp
def solve_social_NLP(tnet, exogenous_G=False):

    # Build variables
    xc = {}
    for i,j in tnet.G_supergraph.edges():
        xc[(i,j)] = cp.Variable(name='xc('+str(i) + ','+ str(j) + ")")

    # objective
    if exogenous_G != False:
        obj = 0
        for i,j in tnet.G_supergraph.edges():
            for n in range(len(tnet.fcoeffs)):
                obj += tnet.G_supergraph[i][j]['t_0']*xc[(i,j)]*tnet.fcoeffs[n]*cp.power(xc[(i,j)]+exogenous_G[i][j], n)
    else:
        obj = 0
        for i,j in tnet.G_supergraph.edges():
            for n in range(len(tnet.fcoeffs)):
                obj += tnet.G_supergraph[i][j]['t_0']*xc[(i,j)]*tnet.fcoeffs[n]*cp.power(xc[(i,j)], n)

    cp.Minimize(obj)
    # constraints


    cp.Problem(cp.Minimize(obj)).solve(verbose=True)

    print(xc.values())

'''

def hist_flows(G, G_exo=True):
    if G_exo:
        norm_flows = [(G[i][j]['flow'] + G_exo[i][j]['flow']) / G[i][j]['capacity'] for i,j in G.edges()]
    else:
        norm_flows = [G[i][j]['flow'] / G[i][j]['capacity'] for i, j in G.edges()]
    #_ = plt.hist(norm_flows, bins='auto')
    #count, bins = np.histogram(norm_flows, bins=5)
    #print('bins:' + str(bins))
    print('max flow:' + str(round(max(norm_flows),2)))

    #fig, axs = plt.subplots(1, 1)
    #axs[0].hist(norm_flows, bins=5)
    #plt.show()