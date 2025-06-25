"""
Ride-pooling network-flow simulation (refactored)
=================================================
A **clean**, modular rewrite of the original monolithic prototype for the paper
"A Time-invariant Network Flow Model for Ride-pooling in Mobility-on-Demand
Systems".  All domain mathematics is preserved, but the architecture now follows
best practices:
"""

from __future__ import annotations

###############################################################################
# 0.  Imports                                                                  #
###############################################################################

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re
import gc

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import kron
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

import gurobipy as gp
from gurobipy import GRB, LinExpr, QuadExpr, quicksum

# heavy project-specific modules ---------------------------------------------
import src.tnet as tnet
import src.CARS as cars  # legacy import kept for compatibility
import experiments.build_NYC_subway_net as nyc
from Utilities.RidePooling.LTIFM_reb import LTIFM_reb, LTIFM_reb_sparse
from Utilities.RidePooling.calculate_gamma_k2 import calculate_gamma_k2
from Utilities.RidePooling.probcomb import probcombN

###############################################################################
# 1.  Global logging setup                                                     #
###############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Silence duplicate Gurobi banner
logging.getLogger("gurobipy").setLevel(logging.WARNING)

###############################################################################
# 2.  Utility: @timeit decorator                                               #
###############################################################################

def timeit(func):
    def wrapper(*args, **kwargs):
        bt = time.time()
        res = func(*args, **kwargs)
        et = time.time()
        logger.info(f"time spent on {func.__name__}: {et - bt:.2f}s")
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
        m.addConstr(expr <= params.vehicle_limit * 2)  # r default 2
    else:
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
            prox.add(diff * diff)                     # no cross-products

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

    # gather results --------------------------------------------------
    flows = np.array([
        sum(x[i, j].X for j in range(len(snap.origins)))
        for i in range(snap.N_edges)
    ])
    prev_mat = np.array([[x[i, j].X for j in range(len(snap.origins))]
                     for i in range(snap.N_edges)])
    cars_expected = expr.getValue() 
    avg_time = [0.0]
    obj = m.ObjVal

    # write flows back for downstream code
    for i, (u, v) in enumerate(snap.edge_order):
        tnet.G_supergraph[u][v]["flowNoRebalancing"] = flows[i]

    m.dispose()
    return CARSResult(avg_time=avg_time, x_vec=prev_mat, expected_cars=cars_expected, obj_val=obj)

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

###############################################################################
# 1. CONFIGURATION                                                            #
###############################################################################

@dataclass(slots=True)
class SimulationConfig:
    """All tweakable parameters in *one* place."""

    city_root: Path = Path("data")
    city_tag: str = "NYC"
    waiting_time: float = 10 / 60  # h
    delay_factor: float = 10/60
    vehicle_limit: int = 10_000
    max_iterations: int = 21

    # adaptive-µ controls
    mu_initial: float = 1e-4
    mu_min: float = 1e-5
    mu_max: float = 10.0
    s_high: float = 0.7
    s_low: float = 0.4

    # convergence & smoothing
    gamma_cars: float = 0.5  # EMA smoothing factor
    tol_x: float = 1.0       # ‖xₖ − xₖ₋₁‖
    tol_obj: float = 10.0
    stable_needed: int = 3   # consecutive hits

    # output
    results_dir: Path = Path("results")

    # ------------------------------------------------------------------
    # derived paths
    # ------------------------------------------------------------------
    @property
    def gml_small(self) -> Path:
        return self.city_root / "gml" / f"{self.city_tag}_small_roadgraph.gpickle"

    @property
    def gml_super(self) -> Path:
        return self.city_root / "gml" / f"{self.city_tag}.gpickle"

    @property
    def mat_fulllist(self) -> Path:
        return Path(self.city_tag) / "MatL2.npz"

    @property
    def results_csv(self) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir / f"results_{self.city_tag}"

###############################################################################
# 2. CORE SIMULATION                                                          #
###############################################################################


class RidePoolingSimulation:
    """Run the fixed-point mode-allocation simulation."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

        # 2.1 Load base road-only network --------------------------------
        self.tNet, self.tstamp, self.fcoeffs = nyc.build_NYC_net(
            "data/net/NYC/", only_road=True
        )
        with open("data/gml/NYC_small_demands.gpickle", 'rb') as f:
            self.tNet.g = pickle.load(f)
        self.tNet.set_g(tnet.perturbDemandConstant(self.tNet.g, 1/24))
        self.original_G = self.tNet.G
        # 2.2 Replace road graphs with pre-built pickles ------------------
        self._load_gml_graphs()

        # 2.3 Pre-compute helpers ----------------------------------------
        self._car_node_idx: Dict[int, int] = {
            node: i for i, node in enumerate(self.tNet.G.nodes())
        }
        self._ori_car_node_idx: Dict[int, int] = {
            node:i for i, node in enumerate(self.original_G.nodes())
        }
        self._n_nodes = len(self._car_node_idx)
        self._car_node_idx_np = dict_to_lookup(self._car_node_idx)

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def run(self) -> None:
        metrics = self._init_metrics()
        mu, r = self.cfg.mu_initial, 2.0
        prev_x = prev2_x = None
        prev_obj = prev2_obj = None
        reb_cars_est = 0.0
        car_ratio_smoothed = 1.0
        stable_hits = 0

        logger.info("Total initial demand: %.0f", sum(self.tNet.g.values()))

        for it in range(self.cfg.max_iterations):
            mu = self._adapt_mu(it, mu, prev_obj, prev2_obj)
            avg_time, x, expected_cars, obj = self._solve_cars(
                it, mu, prev_x, reb_cars_est, car_ratio_smoothed, r
            )
            logger.info(f"expected number of cars: {expected_cars}")
            prev2_x, prev_x = prev_x, x
            prev2_obj, prev_obj = prev_obj, obj

            D_rp = self._extract_ridepool_od()
            y, yr, demand_split, gamma_arr = self._compute_pooled(D_rp)
            total_cars, reb_cars = self._update_road_edge_costs(y, yr)
            
            if it != 0:
                self._record_metrics(metrics, demand_split, total_cars, reb_cars)

            stable_hits = self._check_convergence(
                it, x, prev2_x, obj, prev2_obj, total_cars, stable_hits
            )
            if stable_hits >= self.cfg.stable_needed:
                logger.info("Converged after %d iterations", it + 1)
                break

            # update EMA car-ratio
            car_ratio_smoothed = (
                self.cfg.gamma_cars * car_ratio_smoothed
                + (1 - self.cfg.gamma_cars) * (total_cars / expected_cars)
            )
            self._update_supergraph_costs(D_rp, gamma_arr)
            if it == 0:
                self._record_metrics(metrics, demand_split, total_cars, reb_cars)
            logger.info("Completed iteration %d", it + 1)
        self._save_metrics_csv(metrics)
        self._plot_mode_share(metrics)
        

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_gml_graphs(self) -> None:
        cfg = self.cfg
        with open(cfg.gml_small, 'rb') as f:
            self.tNet.G = pickle.load(f)
        with open(cfg.gml_super, 'rb') as f:
            self.tNet.G_supergraph = pickle.load(f)
        logger.info("Loaded pickled roadgraph: |V|=%d, |A|=%d", self.tNet.G.number_of_nodes(), self.tNet.G.number_of_edges())
        logger.info("Loaded pickled supergraph: |V|=%d, |A|=%d", self.tNet.G_supergraph.number_of_nodes(), self.tNet.G_supergraph.number_of_edges())

    @staticmethod
    def _init_metrics() -> Dict[str, List[float]]:
        keys = [
            "single_share",
            "double_share",
            "triple_share",
            "total_cars",
            "reb_flow",
            "ped_flow",
            "bike_flow",
            "pt_flow",
            "rp_flow"
        ]
        return {k: [] for k in keys}

    # ---------- adaptive-µ -------------------------------------------

    def _adapt_mu(self, it: int, mu: float, obj: float | None, obj_prev: float | None) -> float:
        if it == 2:
            mu = self.cfg.mu_initial  # reset as in legacy code
        if obj is not None and obj_prev is not None:
            s = abs(obj - obj_prev) / (abs(obj_prev) + 1e-12)
            if s > self.cfg.s_high:
                mu = min(mu * 2, self.cfg.mu_max)
            elif s < self.cfg.s_low:
                mu = max(mu * 0.5, self.cfg.mu_min)
        logger.debug("mu=%.3g", mu)
        return mu

    # ---------- call into CARS solver --------------------------------
    def _solve_cars(self, it, mu, prev_x, reb_cars_est, c_ratio, r):
        params = SolverParams(
            iteration=it,
            mu=mu,
            prev_x=prev_x,
            rebalancing=True,
            vehicle_limit=self.cfg.vehicle_limit,
            c_ratio=c_ratio,
            reb_cars=reb_cars_est,
        )
        snap = getattr(self, "_cars_snapshot", None)
        if snap is None:
            snap = self._cars_snapshot = _build_snapshot(self.tNet)
        res = _solve_cars_gurobi(self.tNet, snap, params)
        return res.avg_time, res.x_vec, res.expected_cars, res.obj_val

    # ---------- OD extraction ----------------------------------------

    def _extract_ridepool_od(self) -> np.ndarray:
        n = self._n_nodes
        D_rp = np.zeros((n, n))
        for u, v, d in self.tNet.G_supergraph.edges(data=True):
            if d.get("type") == "rp":
                ii = self._car_node_idx[s2int(v)]
                jj = self._car_node_idx[s2int(u)]
                D_rp[ii, jj] = d["flowNoRebalancing"]
        logger.info(f"Total rp demand: {D_rp.sum()}")
        return D_rp

    # ---------- LTIFM wrapper ----------------------------------------
    @timeit
    def _compute_pooled(
        self, D_rp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], np.ndarray]:
        with np.load(self.cfg.mat_fulllist) as data:
            full_list = data[data.files[0]].astype(np.float32)
        return compute_results(
            full_list,
            self.cfg.delay_factor,
            self.cfg.waiting_time,
            D_rp,
            self.fcoeffs,
            self._car_node_idx,
            self._ori_car_node_idx,
            self._n_nodes,
            self.original_G,
        )

    # ---------- edge cost update -------------------------------------

    def _update_road_edge_costs(self, y: np.ndarray, yr: np.ndarray) -> Tuple[float, float]:
        total_cars = reb_cars = 0.0
        for k, (u, v) in enumerate(self.original_G.edges()):
            edge = self.original_G[u][v]
            t0, cap = edge["t_0"], edge["capacity"]
            flow = y[k, :].sum() + yr[k]
            edge["t_1"] = t0 * (1 + 0.15 * (flow / cap) ** 4)
            total_cars += flow * edge["t_1"]
            reb_cars += yr[k] * edge["t_1"]
        return total_cars, reb_cars

    # ---------- convergence check ------------------------------------

    def _check_convergence(
        self,
        it: int,
        x: np.ndarray,
        x_prev: np.ndarray | None,
        obj: float,
        obj_prev: float | None,
        total_cars: float,
        stable_hits: int,
    ) -> int:
        if x_prev is not None:
            if np.linalg.norm(x.sum(axis=1) - x_prev.sum(axis=1)) < self.cfg.tol_x and total_cars - 10 < self.cfg.vehicle_limit:
                stable_hits += 1
                logger.info("Δx below tol (hit %d/%d)", stable_hits, self.cfg.stable_needed)
            elif obj_prev is not None and abs(obj - obj_prev) < self.cfg.tol_obj and total_cars - 10 < self.cfg.vehicle_limit:
                stable_hits += 1
                logger.info("Δobj below tol (hit %d/%d)", stable_hits, self.cfg.stable_needed)
            else:
                stable_hits = 0
        return stable_hits

    
    # ---------- multi-layer cost update --------------------------------
    def _chunk_delay(self, chunk_idx,
                 jj1_arr, ii1_arr, jj2_arr, ii2_arr,
                 seq0_arr, gammas, D_rp, Tmat):
        """
        Accumulates OD_detour and waiting-time matrices for a slice `chunk_idx`
        (array of row indices).  Runs in a Joblib worker; uses only NumPy.
        """
        n  = D_rp.shape[0]
        OD = np.zeros((n, n), dtype=np.float64)
        WT = np.zeros((n, n), dtype=np.float64)

        for k in chunk_idx:
            g   = gammas[k]
            jj1 = jj1_arr[k];  ii1 = ii1_arr[k]
            jj2 = jj2_arr[k];  ii2 = ii2_arr[k]

            if seq0_arr[k] == 1:            # pattern [1 2 1 2]
                OD[ii1, jj1] += g * (Tmat[ii1, ii2] + Tmat[ii2, jj1] - Tmat[ii1, jj1])
                OD[ii2, jj2] += g * (Tmat[ii2, jj1] + Tmat[jj1, jj2] - Tmat[ii2, jj2])
            else:                           # pattern [1 2 2 1]
                OD[ii1, jj1] += g * (Tmat[ii1, ii2] + Tmat[ii2, jj2] +
                                    Tmat[jj2, jj1] - Tmat[ii1, jj1])

            d1 = D_rp[ii1, jj1]
            d2 = D_rp[ii2, jj2]
            if d1 > 0.0 and d2 > 0.0:
                wt = 0.5 * (1/d1 + 1/d2 - 2/(d1 + d2))
                WT[ii1, jj1] += g * wt
                WT[ii2, jj2] += g * wt

        return OD, WT


    def _update_supergraph_costs(self, D_rp: np.ndarray, gamma_arr: np.ndarray) -> None:
        # ------------------------------------------------------------------
        # 0)  pre-compute shortest-path travel times once for this iteration
        # ------------------------------------------------------------------
        n = self._n_nodes
        Tmat = np.zeros((n, n), dtype=np.float64)
        G_nodes = list(self.tNet.G.nodes())

        for src_idx, src_raw in enumerate(G_nodes):
            lengths = nx.single_source_dijkstra_path_length(
                self.original_G, source=src_raw, weight="t_1")
            for dst_raw, d in lengths.items():
                if dst_raw in G_nodes:
                    dst_idx = self._car_node_idx[dst_raw]
                    Tmat[src_idx, dst_idx] = d

        # ------------------------------------------------------------------
        # 1)  prepare arrays for Joblib
        # ------------------------------------------------------------------
        with np.load(self.cfg.mat_fulllist) as data:
            full_list = data[data.files[0]].astype(np.float32)

        mask       = gamma_arr > 1e-3
        rows_used  = full_list[mask]
        gammas     = gamma_arr[mask].astype(np.float64)

        jj1_arr = np.take(self._car_node_idx_np, rows_used[:, 3].astype(np.int32))
        ii1_arr = np.take(self._car_node_idx_np, rows_used[:, 4].astype(np.int32))
        jj2_arr = np.take(self._car_node_idx_np, rows_used[:, 5].astype(np.int32))
        ii2_arr = np.take(self._car_node_idx_np, rows_used[:, 6].astype(np.int32))
        seq0_arr = rows_used[:, 7].astype(np.int8)   # 1 or 0  (for [1 2 1 2] vs [1 2 2 1])

        # ------------------------------------------------------------------
        # 2)  parallel accumulation with Joblib
        # ------------------------------------------------------------------
        idx        = np.arange(len(rows_used))
        n_jobs     = min(cpu_count(), max(1, len(idx)//10_000))  # heuristic
        chunks     = np.array_split(idx, n_jobs)

        partials = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(self._chunk_delay)(chunk, jj1_arr, ii1_arr, jj2_arr, ii2_arr,
                                seq0_arr, gammas, D_rp, Tmat)
            for chunk in chunks
        )

        # reduce
        OD_delays = sum(p[0] for p in partials)
        Et        = sum(p[1] for p in partials)

        total_delay = np.divide(OD_delays + Et, D_rp,
                                out=np.zeros_like(D_rp), where=D_rp != 0)
        Gs = self.tNet.G_supergraph
        for u, v, d in Gs.edges(data=True):
            if d["type"] == "rp":
                base = nx.shortest_path_length(self.original_G, source=s2int(u), target=s2int(v), weight="t_1")
                extra = total_delay[self._car_node_idx[s2int(v)], self._car_node_idx[s2int(u)]]
                d["t_1"] = base + extra
                d["t_cars"] = base
            else:
                d["t_1"] = d["t_0"]

        # also keep a history on the road graph (optional diagnostics) --
        for u, v in self.tNet.G.edges():
            e = self.tNet.G[u][v]
            e["t_2"] = e.get("t_1", e["t_0"])  # shift previous value
        del full_list
        gc.collect()

    # ---------- metric book-keeping -----------------------------------

    def _record_metrics(
        self,
        M: Dict[str, List[float]],
        demand_stats: Tuple[float, float, float],
        total_cars: float,
        reb_cars: float,
    ) -> None:
        ped_flow=0
        bike_flow=0
        pt_flow=0
        rp_flow=0
        for u,v,d in self.tNet.G_supergraph.edges(data=True):
            if d['type'] == "'":
                ped_flow += self.tNet.G_supergraph[u][v]['flowNoRebalancing']*self.tNet.G_supergraph[u][v]['t_1']
            elif d['type'] == "b":
                bike_flow += self.tNet.G_supergraph[u][v]['flowNoRebalancing']*self.tNet.G_supergraph[u][v]['t_1']
            elif d['type'] == 's':
                pt_flow += self.tNet.G_supergraph[u][v]['flowNoRebalancing']*self.tNet.G_supergraph[u][v]['t_1']
            elif d['type'] == 'rp':
                rp_flow += self.tNet.G_supergraph[u][v]['flowNoRebalancing']*self.tNet.G_supergraph[u][v]['t_1']
        orig, solo, pooled, solo_perc, pooled_perc = demand_stats
        M["single_share"].append(solo_perc)
        M["double_share"].append(pooled_perc)
        M["triple_share"].append(max(0.0, 1 - solo_perc - pooled_perc))
        M["total_cars"].append(total_cars)
        M["reb_flow"].append(reb_cars)
        M["ped_flow"].append(ped_flow)
        M["bike_flow"].append(bike_flow)
        M["pt_flow"].append(pt_flow)
        M["rp_flow"].append(rp_flow)
        logger.info(f"total cars: {total_cars}, single share: {solo_perc}, double_share: {pooled_perc}")
        logger.info(f"Solo demand: {np.sum(solo)}, Pooled demand: {np.sum(pooled)}")

    # ---------- plotting ----------------------------------------------

    def _plot_mode_share(self, M: Dict[str, List[float]]) -> None:
        plt.figure()
        x = np.arange(len(M["single_share"]))
        solo = [M["single_share"][i]*M['rp_flow'][i] for i in range(len(M["single_share"]))]
        plt.bar(x, solo, label="Solo rides")
        bottom=solo
        duo = [M["double_share"][i]*M['rp_flow'][i] for i in range(len(M["single_share"]))]
        plt.bar(x, duo, bottom=bottom, label="Matched rides")
        bottom=[duo[i] + solo[i] for i in range(len(duo))]
        plt.bar(x, M['pt_flow'], bottom=bottom, label="Public transporation")
        bottom=[duo[i] + solo[i] + M['pt_flow'][i] for i in range(len(duo))]
        plt.bar(x, M['bike_flow'], bottom=bottom, label="Biking")
        bottom=[duo[i] + solo[i] + M['pt_flow'][i] + M['bike_flow'][i] for i in range(len(duo))]
        plt.bar(x, M['ped_flow'], bottom=bottom, label="Walking")        
        plt.xlabel("Iteration")
        plt.ylabel("User hours travelled")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/mode_share.eps", format='eps')
        plt.show()

    # ---------- CSV output --------------------------------------------

    def _save_metrics_csv(self, M: Dict[str, List[float]]) -> None:
        import pandas as pd

        df = pd.DataFrame(M)
        df.to_csv(self.cfg.results_csv + ".csv", index=False)
        logger.info("Metrics saved → %s", self.cfg.results_csv)

###############################################################################
# 3. CLI helper                                                                #
###############################################################################


def main() -> None:
    cfg = SimulationConfig()
    sim = RidePoolingSimulation(cfg)
    try:
        sim.run()
    except NotImplementedError as exc:
        logger.error("%s - fill in the TODO section to run full simulation", exc)


if __name__ == "__main__":
    main()
