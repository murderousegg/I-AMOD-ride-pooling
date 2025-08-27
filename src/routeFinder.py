from  gurobipy import *
import networkx as nx
import numpy as np
from itertools import islice
import src.tnet as tnet
from src.utils import *

def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def solve_flow_finder(G, gw, R, xw):
    """
    G : DiGraph
    gw: [(o,d), demand]  (same as your caller)
    R : list of routes, each route is a list of nodes [o,...,d]
    xw: dict {(i,j): flow}   per-OD link flows (can be sparse / subset of G.edges)
    """
    demand = gw[1]

    # --- 1) Build the unified edge set E = support(xw) ∪ edges used by routes
    def route_edges(r):
        return [(r[i], r[i+1]) for i in range(len(r)-1)]

    E = set(xw.keys())
    for r in R:
        E.update(route_edges(r))
    links = list(E)                         # fixed ordering
    idx   = {e:i for i,e in enumerate(links)}
    nLinks = len(links)
    nRoutes = len(R)

    # Target vector aligned with 'links'
    xw_vec = np.zeros(nLinks, dtype=float)
    for e, v in xw.items():
        xw_vec[idx[e]] = float(v)

    # --- 2) Build sparse incidence info: for each edge, which routes use it
    # inc[e] = list of route indices that contain edge e
    inc = {e: [] for e in links}
    for r_id, r in enumerate(R):
        for e in route_edges(r):
            if e in inc:                    # only edges in 'links'
                inc[e].append(r_id)

    # --- 3) QP: min  Σ_e ( xw_e - demand * Σ_{r∋e} p_r )^2  s.t. Σ p = 1, p>=0
    m = Model('QP')
    m.setParam('OutputFlag', 0)
    m.setParam('BarHomogeneous', 1)

    p = [m.addVar(lb=0.0, ub=1.0, name=f"p{r}") for r in range(nRoutes)]
    m.update()

    # xhat_e = demand * Σ_{r in inc[e]} p_r
    # Objective = Σ_e (xw_vec[i] - xhat_e)^2
    obj_terms = []
    for i, e in enumerate(links):
        if inc[e]:
            xhat_e = demand * quicksum(p[r] for r in inc[e])
        else:
            xhat_e = 0.0
        diff = xw_vec[i] - xhat_e
        obj_terms.append(diff * diff)

    m.setObjective(quicksum(obj_terms), GRB.MINIMIZE)
    m.addConstr(quicksum(p) == 1.0)
    m.optimize()

    # Extract non-negligible routes
    sol = {i: {'p': p[i].X, 'r': R[i]} for i in range(nRoutes) if p[i].X > 1e-12}
    return sol, m.objVal


def get_sol(m, R):
    sol_dic = {}
    j = 0
    for r in range(len(R)):
        p = m.getVarByName('p' + str(r)).X
        if p>0.001:
            sol_dic[j] = {}
            sol_dic[j]['p'] = p
            sol_dic[j]['r'] = R[r]
            j+=1
    return  sol_dic

@timeit
def routeFinder_OD(G, gw, od_flows_w, eps, max_routes=100):
    i, j = gw[0]
    k = 0
    err = 999999
    while (err >= eps) and (k<max_routes):
        k += 1
        R = k_shortest_paths(G, source=i, target=j, k=k, weight='t_1')
        sol, obj = solve_flow_finder(G, gw, R, od_flows_w)
        err = obj
        #print(str(k) + ' : ' + str(err))
    return sol


def routeFinder(G, g, od_flows,  eps):
    routes = {}
    for gi in g.items():
        routes[gi[0]] = routeFinder_OD(G, gi, od_flows[gi[0]], eps=eps)
    return routes


## CODE FOR REBALANCING FINDER


def get_node_potentials(G):
    for n in G.nodes():
        in_flow = sum([G[i][j]['flowRebalancing'] for i, j in G.in_edges(n)])
        out_flow = sum([G[i][j]['flowRebalancing'] for i, j in G.out_edges(n)])
        G[n]['potential'] = in_flow - out_flow

def get_rebalancing_ods(G, eps = 0.0):
    o = []
    d = []
    for n in G.nodes():
        in_flow = sum([G[i][j]['flowRebalancing'] for i,j in G.in_edges(n)])
        out_flow  =  sum([G[j][k]['flowRebalancing'] for j,k in G.out_edges(n)])
        G.nodes[n]['potential'] = in_flow - out_flow
        if in_flow - out_flow >= eps:
            d.append(n)
        elif in_flow - out_flow <= -eps:
            o.append(n)
    return o,d

def get_sorted_ods(G, o, d, astar=False):
    r = {origin:{dest: nx.shortest_path_length(G, origin, dest, 't_0') for dest in d} for origin in o}
    o_list = {}
    for origin in o:
        o_list[origin] = [k for k,v in sorted(r[origin].items(), key=lambda item: item[1])]
    d_list = {}
    for dest in d:
        d_list[dest] = [a[1] for a in sorted([[r[origin][dest],origin] for origin in o])]
    return o_list, d_list

@timeit
def solve_rebalancing_O(G, O):
    m = Model()
    m.setParam('OutputFlag', 0)
    m.setParam('BarHomogeneous', 1)
    m.setParam('Method', 1)
    # Define vars
    [m.addVar(lb=0, name='x^'+str(o)+'_'+str(i)+'_'+str(j)) for i,j in G.edges() for o in O]
    m.update()
    x = {}
    for i,j in G.edges():
        x[(i,j)] = quicksum([m.getVarByName('x^'+str(o)+'_'+str(i)+'_'+str(j)) for o in O])
    # Add Obj
    obj = quicksum(G[i][j]['t_1']*x[(i,j)] for i,j in G.edges())
    m.setObjective(obj)
    m.update()
    # Add constraints
    [m.addConstr(quicksum(x[(i,j)] for i,j in G.in_edges(nbunch=n)) - quicksum(x[(j,k)] for j,k in G.out_edges(nbunch=n)) == G.nodes[n]['potential']) for n in G.nodes()]
    [m.addConstr(x[(i, j)]-G[i][j]['flowRebalancing'] == 0) for i,j in G.edges()]
    m.update()
    for o in O:
        for n in G.nodes() :
            if n == o :
                m.addConstr(quicksum(m.getVarByName('x^' + str(o) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                            - quicksum(m.getVarByName('x^' + str(o) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) == G.nodes[n]['potential'])
            else:
                m.addConstr(quicksum(m.getVarByName('x^' + str(o) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                            - quicksum(m.getVarByName('x^' + str(o) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) >= 0)
    m.update()
    m.optimize()
    return {o:{(i,j): m.getVarByName('x^'+str(o)+'_'+str(i)+'_'+str(j)).X for i,j in G.edges()} for o in O}

@timeit
def solve_rebalancing_D(G, origin, D, xo, potential):
    m = Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    # Define vars
    [m.addVar(lb=0, name='x^'+str(d)+'_'+str(i)+'_'+str(j)) for i,j in G.edges() for d in D]
    m.update()
    x = {}
    for i,j in G.edges():
        x[(i,j)] = quicksum([m.getVarByName('x^'+str(d)+'_'+str(i)+'_'+str(j)) for d in D])
    m.update()
    # Add Obj
    obj = quicksum(G[i][j]['t_1']*x[(i,j)] for i,j in G.edges())
    m.setObjective(obj)
    m.update()
    # Add constraints
    [m.addConstr(x[(i, j)] - xo[(i, j)] == 0) for i, j in G.edges()]
    m.addConstr(quicksum(x[(i,j)] for i,j in G.in_edges(nbunch=origin)) - quicksum(x[(j,k)] for j,k in G.out_edges(nbunch=origin)) == potential[origin])
    m.update()
    for d in D:
        for n in G.nodes() :
            if n == origin:
                m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                            - quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) <= 0)
            elif n == d:
                m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                            - quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) >= 0)
            else:
                m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                            - quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) == 0)

    m.update()
    m.optimize()
    m.update()
    return {d: {(i, j): m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)).X for i, j in G.edges()} for d in D}



@timeit
def solve_flow_decomposition_D(G, origin, xo, g, L=1):
    m = Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    # Define vars
    D = [d for o,d in g.keys() if o==origin]
    [m.addVar(lb=0, name='x^'+str(d)+'_'+str(i)+'_'+str(j)) for i,j in G.edges() for d in D]
    m.update()
    x = {}
    for i,j in G.edges():
        x[(i,j)] = quicksum([m.getVarByName('x^'+str(d)+'_'+str(i)+'_'+str(j)) for d in D])
    m.update()
    # Add Obj
    obj = quicksum(L*G[i][j]['t_1']*x[(i,j)] for i,j in G.edges())
    m.setObjective(obj)
    m.update()
    # Add constraints
    [m.addConstr(x[(i, j)] - xo[(i, j)] == 0) for i, j in G.edges()]
    #m.addConstr(quicksum(x[(i,j)] for i,j in G.in_edges(nbunch=origin)) - quicksum(x[(j,k)] for j,k in G.out_edges(nbunch=origin)) == potential[origin])
    m.update()
    for d in D:
        for n in G.nodes() :
            if origin != d:
                if n == origin:
                    m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                                 - quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) + g[(origin, d)] == 0)
                                #- quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) <= 0)
                elif n == d:
                    m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                                - g[(origin, d)]- quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) ==0)
                else:
                    m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                                - quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n))==0)
            else:
                m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=n))
                            - quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=n)) == 0)

    m.update()
    m.optimize()
    m.update()
    return {d: {(i, j): m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)).X for i, j in G.edges()} for d in D}

def solve_flow_decomposition_D_fast_tol(G, origin, xo, g, cost_key='t_1',
                                        tol_edge=1e-12, link_tol_abs=1e-8, link_tol_rel=1e-9,
                                        node_tol_abs=1e-8):
    # 1) Work on positive-flow subgraph
    Epos = [(i,j) for (i,j),v in xo.items() if v > tol_edge]
    H = G.edge_subgraph(Epos).copy()
    if origin not in H:
        return {}

    D = [d for (o,d),val in g.items() if o==origin and val>node_tol_abs and d in H]
    fwd = set(nx.descendants(H, origin)) | {origin}
    Hr = H.reverse(copy=False)
    back = {d: (set(nx.descendants(Hr, d)) | {d}) for d in D}
    D = [d for d in D if d in fwd]

    Ed = {d: [(i,j) for (i,j) in H.edges() if (i in fwd) and (j in back[d])] for d in D}
    if not any(Ed[d] for d in D):
        return {d: {e:0.0 for e in H.edges()} for d in D}

    m = Model()
    m.Params.OutputFlag = 0
    m.Params.Presolve   = 2
    m.Params.Method     = 1   # dual simplex

    # 2) Variables only where usable
    x = {}
    for d in D:
        for (i,j) in Ed[d]:
            x[(d,i,j)] = m.addVar(lb=0.0, name=f"x[{d},{i},{j}]")
    m.update()

    # 3) Soft linking:  sum_d x_d(e) ≈ xo_e  within ±tau_e
    for (i,j) in H.edges():
        terms = [x[(d,i,j)] for d in D if (d,i,j) in x]
        tau_e = max(link_tol_abs, link_tol_rel*max(1.0, xo[(i,j)]))
        if terms:
            m.addConstr(quicksum(terms) >= max(0.0, xo[(i,j)] - tau_e))
            m.addConstr(quicksum(terms) <= xo[(i,j)] + tau_e)
        else:
            # if no destination can use this edge, its xo must be tiny
            m.addConstr(0.0 <= xo[(i,j)] + tau_e)
            m.addConstr(0.0 >= max(0.0, xo[(i,j)] - tau_e))

    # 4) Node balance per destination with small tolerance
    for d in D:
        dem = g[(origin,d)]
        nodes_d = (fwd & back[d])
        for n in nodes_d:
            infl  = quicksum(x[(d,i,n)] for (i,_) in H.in_edges(n)  if (d,i,n) in x)
            outfl = quicksum(x[(d,n,j)] for (_,j) in H.out_edges(n) if (d,n,j) in x)
            if n == origin and origin != d:
                m.addConstr(infl - outfl + dem >= -node_tol_abs)
                m.addConstr(infl - outfl + dem <=  node_tol_abs)
            elif n == d:
                m.addConstr(infl - outfl - dem >= -node_tol_abs)
                m.addConstr(infl - outfl - dem <=  node_tol_abs)
            else:
                m.addConstr(infl - outfl >= -node_tol_abs)
                m.addConstr(infl - outfl <=  node_tol_abs)

    # 5) Tie-break objective (travel time)
    m.setObjective(quicksum(G[i][j][cost_key] * x[(d,i,j)]
                            for d in D for (i,j) in Ed[d] if (d,i,j) in x), GRB.MINIMIZE)
    m.optimize()

    return {d: {e: (x[(d,e[0],e[1])].X if (d,e[0],e[1]) in x else 0.0) for e in H.edges()} for d in D}


@timeit
def solve_flow_decomposition_D_fast(G, origin, xo, g, cost_key='t_1', tol=1e-12):
    # Subgraph of edges with positive bundled flow from this origin
    Epos = [(i,j) for (i,j),v in xo.items() if v > tol]
    H = G.edge_subgraph(Epos).copy()
    if origin not in H:  # no positive flow at all
        return {}

    # Destinations with demand and present in H
    D = [d for (o,d),val in g.items() if o==origin and val>tol and d in H]

    # Forward reachability from origin within H
    fwd = set(nx.descendants(H, origin)) | {origin}
    # Reverse graph once (for back-reachability to each destination)
    Hr = H.reverse(copy=False)

    # Precompute back-reachable sets per destination
    back = {d: (set(nx.descendants(Hr, d)) | {d}) for d in D}

    # Keep only destinations actually reachable in H
    D = [d for d in D if d in fwd]

    # Candidate edges per destination (edge usable if tail ∈ fwd and head ∈ back[d])
    Ed = {
        d: [(i,j) for (i,j) in H.edges()
            if (i in fwd) and (j in back[d])]
        for d in D
    }

    # If nothing to decompose, return empties
    if not any(Ed[d] for d in D):
        return {d: {e:0.0 for e in H.edges()} for d in D}

    m = Model()
    m.Params.OutputFlag = 0
    m.Params.Presolve = 2
    m.Params.Method   = 1  # dual simplex is usually faster here

    # Variables: only where an edge can plausibly be on some o→d path
    x = {}
    for d in D:
        for (i,j) in Ed[d]:
            x[(d,i,j)] = m.addVar(lb=0.0, name=f"x[{d},{i},{j}]")
    m.update()

    # Linking constraints: sum_d x_d(e) == xo_e  for each edge e in support
    # Only sum over d where we actually created a var for that edge
    for (i,j) in H.edges():
        terms = []
        for d in D:
            if (d,i,j) in x:
                terms.append(x[(d,i,j)])
        if terms:
            m.addConstr(quicksum(terms) == xo[(i,j)])
        else:
            # No destination can use this edge → its xo must be ~0
            # If xo>tol here, the original data is inconsistent; to be safe:
            m.addConstr(0.0 == xo[(i,j)])

    # Flow conservation per destination, restricted to nodes that can lie on o→d paths
    for d in D:
        nodes_d = (fwd & back[d])  # only these can appear on a valid o→d path
        dem = g[(origin,d)]
        for n in nodes_d:
            infl  = quicksum(x[(d,i,n)] for (i,_) in H.in_edges(n)  if (d,i,n) in x)
            outfl = quicksum(x[(d,n,j)] for (_,j) in H.out_edges(n) if (d,n,j) in x)
            if n == origin and origin != d:
                m.addConstr(infl - outfl + dem == 0)
            elif n == d:
                m.addConstr(infl - outfl - dem == 0)
            else:
                m.addConstr(infl - outfl == 0)

    # Objective: any feasible split is fine; use a tiny cost to help tie-break
    m.setObjective(quicksum(G[i][j][cost_key] * x[(d,i,j)]
                            for d in D for (i,j) in Ed[d] if (d,i,j) in x), GRB.MINIMIZE)

    m.optimize()

    # Extract solution (zero if var not created)
    sol = {d: {e: 0.0 for e in H.edges()} for d in D}
    for d in D:
        for (i,j) in Ed[d]:
            sol[d][(i,j)] = x[(d,i,j)].X if (d,i,j) in x else 0.0
    return sol


@timeit
def solve_decomposition_dest(G, D, xo, potential, origin, eps=0.001):
    m = Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Method', 1)
    # Define vars
    [m.addVar(lb=0, name='x^'+str(d)+'_'+str(i)+'_'+str(j)) for i,j in G.edges() for d in D]
    m.update()
    x = {}
    for i,j in G.edges():
        x[(i,j)] = quicksum([m.getVarByName('x^'+str(d)+'_'+str(i)+'_'+str(j)) for d in D])
    m.update()
    # Add Obj
    obj = quicksum(G[i][j]['t_1']*x[(i,j)] for i,j in G.edges())
    m.setObjective(obj)
    m.update()
    # Add constraints
    [m.addConstr(x[(i, j)] - xo[(i, j)] >= -eps) for i, j in G.edges()]
    [m.addConstr(x[(i, j)] - xo[(i, j)] <= eps) for i, j in G.edges()]

    for d in D:
        for j in G.nodes():
            if j == origin:
                m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(origin)) for i, j in G.in_edges(nbunch=origin))
                - quicksum(m.getVarByName('x^' + str(d) + '_' + str(origin) + '_' + str(k)) for j, k in G.out_edges(nbunch=origin))
                == -potential[d])
            elif j ==d:
                m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=j))
                            - quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=j))
                            == potential[d])
            else:
                m.addConstr(quicksum(m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)) for i, j in G.in_edges(nbunch=j))
                            - quicksum(m.getVarByName('x^' + str(d) + '_' + str(j) + '_' + str(k)) for j, k in G.out_edges(nbunch=j))
                            <= 0)
    m.update()
    m.optimize()
    m.update()
    return {d: {(i, j): m.getVarByName('x^' + str(d) + '_' + str(i) + '_' + str(j)).X for i, j in G.edges()} for d in D}



@timeit
def get_destinations(G,x, eps=0.01):
    potential = {}
    o = []
    d = []
    for n in G.nodes():
        in_flow = sum([x[(i,j)] for i, j in G.in_edges(n)])
        out_flow = sum([x[(j,k)] for j, k in G.out_edges(n)])
        potent = in_flow - out_flow
        if potent >= eps:
            potential[n] = potent
            d.append(n)
        elif potent <= -eps:
            o.append(n)
            potential[n] = potent
        else:
            potential[n] = 0
    return o, d, potential

@timeit
def rebRouteFinder(G, eps, print_=False):
    routes_dic = {}
    O,D = get_rebalancing_ods(G, eps=0)
    xo = solve_rebalancing_O(G, O)
    if print_:
        print('Origin-flows have been found!')
    for origin, x in xo.items():
        Oo, Do, potential = get_destinations(G,x, eps=0.001)
        #xf = solve_rebalancing_D(G, origin, Do, x, potential)
        xf = solve_decomposition_dest(G, Do, x, potential, origin)
        if print_:
            print('OD-flows have been found!')
        xf = {(origin, d):v for d, v in xf.items()}
        for o,d in xf.keys():
            gw = [(o,d), sum([v for k,v in xf[(o,d)].items() if k[0]==o])]
            if gw[1] > eps:
                routes = routeFinder_OD(G, gw, xf[(o,d)], eps=eps, max_routes=20)
                routes_dic[(o,d)] = routes
    if print_:
        print('Route-flows have been found!')
    return routes_dic



def userRouteFinder(G, g, s_flows, eps):
    routes_dic = {}
    for origin, x in s_flows.items():
        xf = solve_flow_decomposition_D(G, origin, x, g, L=1)
        xf = {(origin, d):v for d, v in xf.items()}
        for o,d in xf.keys():
            gw = [(o,d), sum([v for k,v in xf[(o,d)].items() if k[0]==o])]
            if gw[1] > eps:
                routes = routeFinder_OD(G, gw, xf[(o,d)], eps=20, max_routes=20)
                routes_dic[(o,d)] = routes
    return routes_dic


#xf = solve_flow_decomposition_D(G, origin, x, g, L=1)

def RouteFinder(G, g, s_flows, eps, od):
    o,d = od
    x = s_flows[o]
    xf = solve_flow_decomposition_D(G, o, x, g, L=1)
    xf = {(o, dest):v for dest, v in xf.items()}
    gw = [(o,d), sum([v for k,v in xf[(o,d)].items() if k[0]==o])]
    routes = routeFinder_OD(G, gw, xf[(o,d)], eps=eps, max_routes=10)
    return routes

def RouteTravelTime(G,path):
    return sum([G[path[i]][path[i+1]]['t_1'] for i in range(len(path)-1)])