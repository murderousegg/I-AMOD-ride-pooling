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

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
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
###############################################################################
# 3.  Refactored Gurobi solver                                                 #
###############################################################################

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

@dataclass
class NetSnapshot:
    Binc: Any  # scipy.sparse.csr_matrix
    edge_order: list
    node_order: list
    origins: list
    demand_matrix: np.ndarray  # shape = (|origins| , |nodes|)

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
    demand = np.zeros((len(origins), len(node_order)))
    origin_idx = {o: i for i, o in enumerate(origins)}
    node_idx = {n: i for i, n in enumerate(node_order)}
    for (o, d), q in tnet.g.items():
        demand[origin_idx[o], node_idx[d]] += q
    # ensure row-sum zero (supply = –demand) per origin
    for o, i in origin_idx.items():
        demand[i, node_idx[o]] = -demand[i].sum()
    return NetSnapshot(Binc=Binc, edge_order=edge_order, node_order=node_order, origins=origins, demand_matrix=demand)

# ---------------------------------------------------------------------------
# Core model builder
# ---------------------------------------------------------------------------
@timeit
def _configure_gurobi(m: gp.Model, params: SolverParams) -> None:
    m.setParam("OutputFlag", 0)
    if params.threads:
        m.setParam("Threads", params.threads)
    m.setParam("Method", params.method)

@timeit
def _add_flow_conservation(m: gp.Model, snap: NetSnapshot, x, xr, params: SolverParams) -> LinExpr:
    B = snap.Binc
    b = snap.demand_matrix.flatten()
    B_kron = kron(np.eye(snap.N_nodes), B, format='csr') 
    for i in range(b.shape[0]):
        lhs = LinExpr()  # Initialize linear expression for constraint
        row_start = B_kron.indptr[i]  # Start of row i in CSR format
        row_end = B_kron.indptr[i + 1]  # End of row i
        row_data = B_kron.data[row_start:row_end]  # Non-zero values in row i
        row_indices = B_kron.indices[row_start:row_end]  # Column indices of non-zero values

        edge_indices = [j % snap.N_edges for j in row_indices]  # Extract corresponding edge index
        node_indices = [j // snap.N_edges for j in row_indices]  # Extract node index
        
        lhs.addTerms(row_data, [x[edge_indices[idx], node_indices[idx]] for idx in range(len(row_indices))])
        # Add constraint for row i
        m.addConstr(lhs == b[i], name=f"DemandBalance_{i}")

@timeit
def _add_vehicle_cap(m: gp.Model, tnet, snap: NetSnapshot, x, xr, params: SolverParams):
    expr = LinExpr()
    for j, (u, v) in enumerate(snap.edge_order):
        if "rp" in str(u) and "rp" in str(v):
            coeff = tnet.G_supergraph[u][v]["t_0"] if params.iteration == 0 else tnet.G_supergraph[u][v]["t_cars"]
            expr.add(coeff * x.sum(j, "*"))
            if xr is not None and params.iteration == 0:
                expr.add(coeff * xr[j])
    if params.iteration == 0:
        m.addConstr(expr <= params.vehicle_limit * params.r)  # r default 2
    else:
        expr = expr / params.r
        expr.add(params.reb_cars)
        m.addConstr(expr * params.c_ratio <= params.vehicle_limit)
    return expr

@timeit
def _add_modal_constraints(m: gp.Model, snap: NetSnapshot, x):
    """Implements the zero-flow rules for walk→bike connectors."""
    for i, (u, v) in enumerate(snap.edge_order):
        u_str, v_str = str(u), str(v)
        if "'" in u_str and "b" in v_str:  # walk → bike layer
            for j, origin in enumerate(snap.origins):
                if str(s2int(origin)) + "b" != v_str:
                    m.addConstr(x[i, j] == 0)

@timeit
def _build_objective(tnet, snap: NetSnapshot, x, params: SolverParams):
    edge_times = [
        tnet.G_supergraph[u][v].get("t_0" if params.iteration == 0 else "t_1")
        for u, v in snap.edge_order
    ]
    base_obj = quicksum(edge_times[i] * x.sum(i, "*") for i in range(snap.N_edges))

    # --- remove prox if not requested ---------------------------------
    if params.mu <= 0 or params.prev_x is None:
        return base_obj

    # --- diagonal (sparse) prox term ----------------------------------
    prox = QuadExpr()
    half_mu = 0.5 * params.mu
    n_o = len(snap.origins)
    prev = params.prev_x.reshape(snap.N_edges, n_o)   # ensure 2-D

    for i in range(snap.N_edges):
        for j in range(n_o):
            diff = x[i, j] - float(prev[i, j])
            prox.add(diff * diff)

    return base_obj + half_mu * prox

@timeit
def _solve_cars_gurobi(tnet, snap: NetSnapshot, params: SolverParams) -> CARSResult:
    m = gp.Model(f"CARS{params.iteration}")
    _configure_gurobi(m, params)

    # variables -------------------------------------------------------
    x = m.addVars(snap.N_edges, len(snap.origins), name="x", lb=0)
    xr = None
    if params.rebalancing and params.iteration == 0:
        xr = m.addVars(snap.N_edges, name="xr", lb=0)

    # constraints -----------------------------------------------------
    _add_flow_conservation(m, snap, x, xr, params)
    expr = _add_vehicle_cap(m, tnet, snap, x, xr, params)
    _add_modal_constraints(m, snap, x)

    # objective -------------------------------------------------------
    m.setObjective(_build_objective(tnet, snap, x, params), GRB.MINIMIZE)
    
    @timeit
    def _perform_opt():
        m.optimize()
    _perform_opt()

    x_vars = list(x.values())
    x_vals = m.getAttr("X", x_vars)

    x_mat  = np.asarray(x_vals, dtype=float)\
                .reshape(snap.N_edges, len(snap.origins), order="C")

    flows = x_mat.sum(axis=1)
    prev_mat = x_mat
    obj = m.ObjVal
    cars_expected = expr.getValue()
    # write flows back for downstream code
    for i, (u, v) in enumerate(snap.edge_order):
        tnet.G_supergraph[u][v]["flowNoRebalancing"] = flows[i]

    m.dispose()
    return CARSResult(avg_time=[0.0], x_vec=prev_mat, expected_cars=cars_expected, obj_val=obj)

###############################################################################
# 0. Functional helper: compute_results                                        #
###############################################################################
def _pad_to_full(mat_small: np.ndarray,
                 idx_small: Dict[int, int],
                 idx_full: Dict[int, int]) -> np.ndarray:
    """Return |full|×|full| matrix with `mat_small` dropped in the right block."""
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
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], np.ndarray]:
    """Split ride-pool OD matrix into solo / pooled parts & run LTIFM."""

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
    
    for idx in tqdm(range(delay1.shape[0]), desc="γ-updates", unit="pair", mininterval=5):
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

    # LTIFM per class ----------------------------------------------------
    sol_np = LTIFM_reb_sparse(full_solo_demand, road_graph, fcoeffs=fcoeffs)
    sol_rp = LTIFM_reb_sparse(full_pooled_demand, road_graph, fcoeffs=fcoeffs)

    full_solo_demand -= np.diag(np.diag(full_solo_demand))
    full_pooled_demand -= np.diag(np.diag(full_pooled_demand))
    
    y = sol_np["x"] + sol_rp["x"]
    yr = sol_np["xr"] + sol_rp["xr"]

    ### determine pooling percentage
    TrackDems = [np.sum(demand), np.sum(solo_demand),np.sum(pooled_demand)]
    PercNRP = TrackDems[1]/(TrackDems[2] + TrackDems[1])
    # Normalize TotG row-wise and scale it by (1 - PercNRP)
    TotG_normalized = total_gamma / np.sum(total_gamma, axis=0, keepdims=True)  # Normalize rows of TotG
    scaled_TotG = TotG_normalized * (1 - PercNRP)  # Scale by (1 - PercNRP)
    # Combine PercNRP and the scaled TotG into a new array
    total_PercNRP = np.hstack([PercNRP, scaled_TotG])

    demand_stats = (
        float(original_demand.sum()),
        float(solo_demand.sum()),
        float(pooled_demand.sum()),
        total_PercNRP[0],
        total_PercNRP[1]
    )
    return y, yr, demand_stats, gamma_arr
