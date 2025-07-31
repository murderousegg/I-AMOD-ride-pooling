import time
import gc
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Any
import logging
import functools

import networkx as nx
import numpy as np
from scipy.sparse import kron
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB, LinExpr, QuadExpr, quicksum
from src.LTIFM_reb import LTIFM_reb_sparse
import warnings

warnings.filterwarnings("ignore", message=".*Chained matrix multiplications of MVars is inefficient.*")
logging.getLogger("gurobipy").setLevel(logging.WARNING)
logger = logging.getLogger("iamod")

def timeit(func):
    log = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        log.info("time spent in %s: %.3f s", func.__qualname__, dt)
        return res

    return wrapper

def s2int(node: str) -> int:
    num = int(re.sub(r'\D', '', node))
    return num 

def dict_to_lookup(d: dict[int, int]) -> np.ndarray:
    """Return 1-D array s.t. lookup[raw_id] -> compact index,  -1 if absent."""
    max_id = max(d)                    # largest raw node id
    lut = np.full(max_id + 1, -1, dtype=np.int32)
    for raw_id, compact in d.items():
        lut[raw_id] = compact
    return lut

def probcombN(a, waiting):
    waiting = waiting
    n = len(a)
    prob = 0
    if np.any(a==0):
        print('prob0')
    for ii in range(0,n):
        a_temp = np.array(a)
        a_temp = np.delete(a_temp, ii)
        prob = prob +(a[ii]/sum(a)) * np.prod(1 - np.exp(-a_temp*waiting))
    if np.isnan(prob):
        prob = 0
    if prob < 10e-10:
        prob = 0
    return prob


@dataclass
class SolverParams:
    iteration: int
    mu: float
    prev_x: np.ndarray | None = None
    rebalancing: bool = True
    vehicle_limit: int = 50000
    c_ratio: float = 1.0
    reb_cars: float = 0.0
    method: int = -1  # Gurobi Method param
    threads: int | None = None
    r: float = 2
    Tmax: float = 0.5
    rho_time: float = 1e-6

@dataclass
class NetSnapshot:
    Binc: Any  # scipy.sparse.csr_matrix
    edge_order: list
    node_order: list
    origins: list
    demand_matrix: np.ndarray  # shape = (|origins| , |nodes|)
    alpha_o: np.ndarray
    @property
    def N_edges(self) -> int:  # noqa: N802 (want to match paper notation)
        return self.Binc.shape[1]

    @property
    def N_nodes(self) -> int:  # noqa: N802
        return self.Binc.shape[0]

@dataclass
class CARSResult:
    avg_time: list[float]
    x_vec: np.ndarray  # length = N_edges
    expected_cars: float
    obj_val: float

# ---------------------------------------------------------------------------
# Pre-processing helper: build snapshot once and reuse every iteration
# ---------------------------------------------------------------------------

def _build_snapshot(tnet) -> NetSnapshot:
    node_order = list(tnet.G_supergraph.nodes())
    edge_order = list(tnet.G_supergraph.edges())
    Binc = nx.incidence_matrix(tnet.G_supergraph, nodelist=node_order, edgelist=edge_order, oriented=True).tocsr()
    origins = sorted({o for o, _ in tnet.g.keys()})
    origin_idx = {o: i for i, o in enumerate(origins)}
    node_idx = {n: i for i, n in enumerate(node_order)}
    alpha_o = np.zeros(len(origins))
    demand = np.zeros((len(origins), len(node_order)))
    for (o, d), q in tnet.g.items():
        demand[origin_idx[o], node_idx[d]] += q
        alpha_o[origin_idx[o]] += q
    # ensure row-sum zero (supply = –demand) per origin
    for o, i in origin_idx.items():
        demand[i, node_idx[o]] = -demand[i].sum()
        
    return NetSnapshot(Binc=Binc, edge_order=edge_order, node_order=node_order, origins=origins, demand_matrix=demand, alpha_o=alpha_o)

@timeit
def _configure_gurobi(m: gp.Model, params: SolverParams) -> None:
    m.setParam("OutputFlag", 0)
    if params.threads:
        m.setParam("Threads", params.threads)
    m.setParam("Method", params.method)

@timeit
def _add_flow_conservation_M(m: gp.Model, snap: NetSnapshot, x, xr, params: SolverParams) -> LinExpr:
    for i in range(len(snap.origins)):
        m.addMConstr(snap.Binc, x[:,i], sense='=', b=snap.demand_matrix[i,:], name=f"DemandBalance_{i}")

@timeit
def _add_vehicle_cap_M(m: gp.Model, tnet, snap: NetSnapshot, x, xr, params: SolverParams):
    tvals = []
    idxs = []
    for j, (u, v) in enumerate(snap.edge_order):
        if "rp" in str(u):
            t = tnet.G_supergraph[u][v]["t_0"] if params.iteration == 0 else tnet.G_supergraph[u][v]["t_cars"]
            tvals.append(t)
            idxs.append(j)

    t_array = np.array(tvals)
    x_rp = x[idxs, :]
    expr = (t_array @ x_rp).sum()

    if xr is not None and params.iteration == 0:
        expr += (t_array @ xr[idxs])
        
    if params.iteration == 0:
        m.addConstr(expr <= params.vehicle_limit * params.r)  # r default 2
    else:
        expr = expr / params.r + params.reb_cars
        m.addConstr(expr * params.c_ratio <= params.vehicle_limit)
    return expr

@timeit
def _add_modal_constraints_M(m: gp.Model, snap: NetSnapshot, x):
    """Implements the zero-flow rules for walk→bike connectors."""
    switching_idx = []
    o = []
    for i, (u, v) in enumerate(snap.edge_order):
        u_str, v_str = str(u), str(v)
        if "'" in u_str and "b" in v_str:
            switching_idx.append(i)
            o.append(s2int(u_str))
    switching_idx = np.array(switching_idx, dtype=int)
    o = np.array(o, dtype=int)
    #TODO: pre-store boolean array for biking constraint, calcuate once
    for i, u in enumerate(snap.origins):
        not_o = np.nonzero(o != s2int(u))
        true_switching_idx = switching_idx[not_o[0]]
        x_switch = x[true_switching_idx,i]
        m.addConstr(x_switch == 0,
                    name=f"ModalConstraints_{i}")  # zero flow for all non-matching origins
        

@timeit
def _build_objective_M(tnet, snap: NetSnapshot, x: gp.MVar, params: SolverParams):
    edge_times = np.array([
        tnet.G_supergraph[u][v].get("t_0" if params.iteration == 0 else "t_1")
        for u, v in snap.edge_order
    ])
    base_obj = edge_times @ x.sum(axis=1)

    if params.mu <= 0 or params.prev_x is None:
        return base_obj, 0

    n_o = len(snap.origins)
    prev = params.prev_x.reshape(snap.N_edges, n_o)

    prox_expr = QuadExpr()

    # Flatten x and prev to 1D arrays
    x_flat = x.reshape(-1).tolist()
    prev_flat = prev.reshape(-1)

    # Quadratic terms: (x_ij)^2
    prox_expr.addTerms([1.0] * len(x_flat), x_flat, x_flat)

    # Linear terms: -2 * prev_ij * x_ij
    prox_expr.addTerms((-2.0 * prev_flat).tolist(), x_flat)

    # Constant term: sum(prev_ij^2)
    prox_expr.addConstant((prev_flat ** 2).sum())

    # Scale the entire expression
    prox_expr *= 0.5 * params.mu

    return base_obj, prox_expr

@timeit
def _solve_cars_gurobi_M(tnet, snap: NetSnapshot, params: SolverParams) -> CARSResult:
    '''
    Vehicle routing solver using MVars and matrix operations for performance.
    '''
    m = gp.Model(f"CARS{params.iteration}")
    _configure_gurobi(m, params)

    # variables -------------------------------------------------------
    x = m.addMVar((snap.N_edges, len(snap.origins)), name="x", lb=0)
    xr = None
    if params.rebalancing and params.iteration == 0:
        xr = m.addMVar(snap.N_edges, name="xr", lb=0)

    # constraints -----------------------------------------------------
    _add_flow_conservation_M(m, snap, x, xr, params)
    expr = _add_vehicle_cap_M(m, tnet, snap, x, xr, params)
    _add_modal_constraints_M(m, snap, x)

    # objective -------------------------------------------------------
    base_obj, mu_obj = _build_objective_M(tnet, snap, x, params)
    m.setObjective(base_obj + mu_obj, GRB.MINIMIZE)
    
    @timeit
    def _perform_opt():
        m.optimize()
    _perform_opt()

    x_mat  = x.X

    flows = x_mat.sum(axis=1)
    prev_mat = x_mat
    obj = base_obj.getValue()
    cars_expected = expr.getValue()
    # write flows back for downstream code
    for i, (u, v) in enumerate(snap.edge_order):
        tnet.G_supergraph[u][v]["flowNoRebalancing"] = flows[i]

    m.dispose()
    return CARSResult(avg_time=[0.0], x_vec=prev_mat, expected_cars=cars_expected, obj_val=obj)

@timeit
def _build_objective_fair(tnet, snap: NetSnapshot, x, eps, params: SolverParams):
    '''
    Build objective function with fairness added.
    '''
    edge_times = np.array([
        tnet.G_supergraph[u][v].get("t_0" if params.iteration == 0 else "t_1")
        for u, v in snap.edge_order
    ])
    base_obj = edge_times @ x.sum(axis=1)

    #sufficiency
    tot_alpha = snap.alpha_o.sum()
    suff_obj = quicksum(eps[i]*snap.alpha_o[i] for i in range(len(snap.origins)))
    suff_obj = suff_obj / tot_alpha

    # no prox term in first iter
    if params.mu <= 0 or params.prev_x is None:
        return suff_obj, base_obj, 0
    
    n_o = len(snap.origins)
    prev = params.prev_x.reshape(snap.N_edges, n_o)

    prox_expr = QuadExpr()

    # Flatten x and prev to 1D arrays
    x_flat = x.reshape(-1).tolist()
    prev_flat = prev.reshape(-1)
    # Quadratic terms: (x_ij)^2
    prox_expr.addTerms([1.0] * len(x_flat), x_flat, x_flat)
    # Linear terms: -2 * prev_ij * x_ij
    prox_expr.addTerms((-2.0 * prev_flat).tolist(), x_flat)
    # Constant term: sum(prev_ij^2)
    prox_expr.addConstant((prev_flat ** 2).sum())
    # Scale the entire expression
    prox_expr *= 0.5 * params.mu

    return suff_obj, base_obj, prox_expr

def _add_sufficiency_constraints(tnet, m, x, eps, Tmax, iter, snap: NetSnapshot):
    '''
    build epsilon constraints.
    '''
    expr = []
    if iter == 0:
        times = np.array([float(tnet.G_supergraph[u][v]['t_0']) for u, v in tnet.G_supergraph.edges()])
    else:
        times = np.array([float(tnet.G_supergraph[u][v]['t_1']) for u, v in tnet.G_supergraph.edges()])
    
    for idx, o in enumerate(snap.origins):
        avg_time_expr = (times @ x[:, idx]) / snap.alpha_o[idx]  # matrix-vector product
        expr.append(avg_time_expr)
        m.addConstr(eps[idx] >= avg_time_expr - Tmax)
    return expr
    

@timeit
def _solve_cars_gurobi_fair(tnet, snap: NetSnapshot, params: SolverParams) -> tuple[CARSResult, Dict]:
    '''
    Ride-pooling solver for the commute sufficiency objective. Tune rho_time for tradeoff
    minimum time and commute sufficiency.
    '''
    m = gp.Model(f"CARS{params.iteration}")
    _configure_gurobi(m, params)
    # variables -------------------------------------------------------
    x = m.addMVar((snap.N_edges, len(snap.origins)), name="x", lb=0)
    xr = None
    if params.rebalancing and params.iteration == 0:
        xr = m.addMVar(snap.N_edges, name="xr", lb=0)
    eps = m.addMVar(len(snap.origins), lb=0, vtype=GRB.CONTINUOUS, name='eps')
    # constraints -----------------------------------------------------
    _add_flow_conservation_M(m, snap, x, xr, params)
    expr = _add_vehicle_cap_M(m, tnet, snap, x, xr, params)
    _add_modal_constraints_M(m, snap, x)
    expr_suff = _add_sufficiency_constraints(tnet, m, x, eps, params.Tmax, params.iteration, snap)

    # objective -------------------------------------------------------
    suff_obj, base_obj, mu_obj = _build_objective_fair(tnet, snap, x, eps, params)
    base_scale = params.rho_time
    m.setObjective(base_scale*(base_obj+mu_obj) + suff_obj, GRB.MINIMIZE)
    # move mu_obj out of brackets, try different scales for rho (start = 1e-4)
    
    @timeit
    def _perform_opt():
        m.optimize()
    _perform_opt()

    x_mat  = x.X
    flows = x_mat.sum(axis=1)
    prev_mat = x_mat
    obj_dict = {}
    
    obj = suff_obj.getValue()
    obj_dict["suff"] = obj
    obj_dict["base"] = base_scale*base_obj.getValue()
    if mu_obj:
        obj_dict["mu"] = base_scale*mu_obj.getValue()
    else:
        obj_dict["mu"] = 0

    logger.info(f"Fairness objective: {obj}")
    cars_expected = expr.getValue()
    avg_time_suff = []
    for i in range(len(snap.origins)):
        avg_time_suff.append(expr_suff[i].getValue())
    # write flows back for downstream code
    for i, (u, v) in enumerate(snap.edge_order):
        tnet.G_supergraph[u][v]["flowNoRebalancing"] = flows[i]

    m.dispose()
    return CARSResult(avg_time=avg_time_suff, x_vec=prev_mat, expected_cars=cars_expected, obj_val=obj), obj_dict


#### Helpers for preprocessing D^RP ####
def _pad_to_full(mat_small: np.ndarray,
                 idx_small: Dict[int, int],
                 idx_full: Dict[int, int]) -> np.ndarray:
    """Return |full|x|full| matrix with `mat_small` dropped in the right block."""
    full = np.zeros((len(idx_full), len(idx_full)), dtype=mat_small.dtype)
    rows = np.array([idx_full[n]                    # position in big ordering
                     for n, _ in sorted(idx_small.items(),
                                        key=lambda kv: kv[1])])
    full[np.ix_(rows, rows)] = mat_small
    return full

def _lut_from_dict(d: dict[int, int]) -> np.ndarray:
    max_id = max(d)                       # biggest raw node ID
    lut = -np.ones(max_id + 1, dtype=np.int32)
    for ridx, pos in d.items():
        lut[ridx] = pos
    return lut

def compute_results(
    full_list: np.ndarray,
    delay: float,
    waiting_time: float,
    demand: np.ndarray,
    fcoeffs: np.ndarray,
    car_node_index_map: Dict[int, int],
    original_index_map: Dict[int, int],
    n_nodes_road: int,
    road_graph: nx.DiGraph,
    nash: int,
    verbose: bool=True
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], np.ndarray]:
    """
    Convert the ride-pooling requests into a vehicle demand matrix, and run
    LTIFM_reb to optimise vehicle routing.
    
    Returns:
    y: vehicle flow
    yr: rebalncing flow
    demand_stats: stats on ride-pooling percentage
    gamma_arr: array containing indices of selected rp tours.
    """

    cumul_delay = total_gamma = 0.0
    original_demand = demand.copy()
    solo_demand = demand.copy()
    pooled_demand = np.zeros_like(demand)
    gamma_arr = np.zeros(full_list.shape[0], dtype=np.float32)

    ### create lookup atbles
    delay1   = full_list[:, 1]                # view, no copy
    delay2   = full_list[:, 2]
    o1_raw   = full_list[:, 3].astype(int)
    d1_raw   = full_list[:, 4].astype(int)
    o2_raw   = full_list[:, 5].astype(int)
    d2_raw   = full_list[:, 6].astype(int)
    pattern  = full_list[:, 7:11]             # shape (N, 4)

    lut = _lut_from_dict(car_node_index_map)  # ndarray for O(1) mapping
    jj1_arr = lut[o1_raw]                     # ndarray of row indices
    ii1_arr = lut[d1_raw]
    jj2_arr = lut[o2_raw]
    ii2_arr = lut[d2_raw]
    del full_list
    gc.collect()
    mask0 = (solo_demand[ii1_arr, jj1_arr] >= 1e-3) & \
        (solo_demand[ii2_arr, jj2_arr] >= 1e-3)
    mask1 = (delay1 < delay) & (delay2 < delay)
    keep  = mask0 & mask1                      # Boolean length-N array
    delay1   = delay1[keep]
    delay2   = delay2[keep]
    jj1_arr  = jj1_arr[keep];  ii1_arr = ii1_arr[keep]
    jj2_arr  = jj2_arr[keep];  ii2_arr = ii2_arr[keep]
    is_1212 = (pattern[:, 2] == 1)   #  boolean view, no copy
    del pattern
    is_1212  = is_1212[keep]
    gamma_compact = np.zeros(len(jj1_arr), dtype=np.float32)
    for idx in tqdm(range(delay1.shape[0]), desc="gamma-updates", unit="pair", mininterval=10, disable=logger.level>=20):
        jj1, ii1 = jj1_arr[idx], ii1_arr[idx]
        jj2, ii2 = jj2_arr[idx], ii2_arr[idx]

        if (delay1[idx] < delay   and delay2[idx] < delay   and
            solo_demand[ii1, jj1] >= 1e-3               and
            solo_demand[ii2, jj2] >= 1e-3):

            prob = probcombN([solo_demand[ii1, jj1], solo_demand[ii2, jj2]],
                             waiting_time)
            gamma = min(solo_demand[ii1, jj1],
                         solo_demand[ii2, jj2]) * prob * 0.5
            multip = 1 if (jj1 == jj2 and ii1 == ii2) else 2
            if is_1212[idx]:
                pooled_demand[jj2, jj1] += multip*gamma
                pooled_demand[ii1, jj2] += multip*gamma
                pooled_demand[ii2, ii1] += multip*gamma
            else:
                pooled_demand[jj2, jj1] += multip*gamma
                pooled_demand[ii2, jj2] += multip*gamma
                pooled_demand[ii1, ii2] += multip*gamma
            solo_demand[ii1, jj1] -= multip*gamma
            solo_demand[ii2, jj2] -= multip*gamma

            cumul_delay += multip*gamma * (delay1[idx] + delay2[idx])
            total_gamma += multip*gamma
            gamma_compact[idx] = gamma
    gamma_arr[keep] = gamma_compact
    del delay1, delay2, o1_raw, d1_raw, o2_raw, d2_raw,\
        lut, jj1_arr, jj2_arr, ii1_arr, ii2_arr
    gc.collect()
    # remove self-loops (LTIFM expects zeros on diagonal)
    solo_demand -= np.diag(np.diag(solo_demand))
    pooled_demand -= np.diag(np.diag(pooled_demand))

    #### map onto the original full roadgraph
    full_solo_demand = _pad_to_full(solo_demand, car_node_index_map, original_index_map)
    full_pooled_demand = _pad_to_full(pooled_demand, car_node_index_map, original_index_map)

    np.fill_diagonal(full_solo_demand,   0)
    np.fill_diagonal(full_pooled_demand, 0)
    full_demand = full_solo_demand + full_pooled_demand
    # np.save("full_demand.npy", full_demand)
    # LTIFM per class ----------------------------------------------------
    sol_np = LTIFM_reb_sparse(full_demand, road_graph, fcoeffs=fcoeffs, nash=nash, n=5)

    full_demand -= np.diag(np.diag(full_demand))
    
    y = sol_np["x"] 
    yr = sol_np["xr"]
    non_pooled_perc = np.sum(full_solo_demand)/(np.sum(original_demand))
    demand_stats = (
        float(original_demand.sum()),
        float(solo_demand.sum()),
        float(pooled_demand.sum()),
        non_pooled_perc,
        1-non_pooled_perc
    )
    return y, yr, demand_stats, gamma_arr