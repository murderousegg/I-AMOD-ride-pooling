import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List
from joblib import Parallel, delayed, cpu_count
import os
import pandas as pd

from src.solvers import *
from src.solvers import _build_snapshot, _solve_cars_gurobi_M
from src.simConfig import SimulationConfig
import src.tnet as tnet
import experiments.build_NYC_subway_net as nyc
import logging

logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

plt.rcParams.update({
    "font.size": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.fontsize": 12,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "text.latex.preamble": r'\usepackage{dsfont}',
    "axes.labelsize": 13,
})

class RidePoolingSimulationCore:
    """Run the fixed-point mode-allocation simulation."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.metrics: Dict[str, List[float]]
        # load base network
        self.tNet, self.tstamp, self.fcoeffs = nyc.build_NYC_net(
            "data/net/NYC/", only_road=True
        )
        with open("data/gml/NYC_small_demands.gpickle", 'rb') as f:
            self.tNet.g = pickle.load(f)
        self.tNet.set_g(tnet.perturbDemandConstant(self.tNet.g, self.cfg.demand_multiplier))
        self.tNet.read_node_coordinates("data/pos/NYC.txt")
        self.original_G = self.tNet.G
        # load rest of networks
        self._load_gml_graphs()

        #precompute helper variables
        self._car_node_idx: Dict[int, int] = {
            node: i for i, node in enumerate(self.tNet.G.nodes())
        }
        self._ori_car_node_idx: Dict[int, int] = {
            node:i for i, node in enumerate(self.original_G.nodes())
        }
        self._n_nodes = len(self._car_node_idx)
        self._car_node_idx_np = dict_to_lookup(self._car_node_idx)
        self.nash = 0   #init

    def run(self, warm=0) -> None:
        '''
        Run base optimization loop between I-AMoD and rp
        '''
        self._init_metrics()
        mu, r = self.cfg.mu_initial, 2.0
        prev_x = prev2_x = None
        prev_obj = prev2_obj = None
        reb_cars_est = 0.0
        car_ratio_smoothed = 1.0
        stable_hits = 0

        logger.info("Total initial demand: %.0f", sum(self.tNet.g.values()))

        for it in range(warm, self.cfg.max_iterations+warm):
            # update proximal term
            mu = self._adapt_mu(it, mu, prev_obj, prev2_obj)
            # solve I-AMoD
            avg_time, x, expected_cars, obj = self._solve_cars(
                it, mu, prev_x, reb_cars_est, car_ratio_smoothed, r
            )
            logger.info(f"expected number of cars: {expected_cars}")
            # keep backlog of results
            prev2_x, prev_x = prev_x, x
            prev2_obj, prev_obj = prev_obj, obj
            
            D_rp = self._extract_ridepool_od()

            # np.save("temp.npy", D_rp)
            # D_rp = np.load("temp.npy")
            
            y, yr, demand_split, gamma_arr = self._compute_pooled(D_rp) # Do rp algorithm
            
            total_cars, reb_cars = self._update_road_edge_costs(y, yr)
            
            if it != 0: # record metrics before escape from convergence
                self._record_metrics(demand_split, total_cars, reb_cars)

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
                self._record_metrics(demand_split, total_cars, reb_cars)

            logger.info("Completed iteration %d", it + 1)
            # update rp percentage
            r = 1 + self.metrics["double_share"][-1]
        # save and plot
        self._save_metrics_csv()
        self._plot_mode_share()
        return x

    def _load_gml_graphs(self) -> None:
        '''
        Load both supergraph and (pruned) roadgraph
        '''
        cfg = self.cfg
        with open(cfg.gml_small, 'rb') as f:
            self.tNet.G = pickle.load(f)
        with open(cfg.gml_super, 'rb') as f:
            self.tNet.G_supergraph = pickle.load(f)
        logger.info("Loaded pickled roadgraph: |V|=%d, |A|=%d", self.tNet.G.number_of_nodes(), self.tNet.G.number_of_edges())
        logger.info("Loaded pickled supergraph: |V|=%d, |A|=%d", self.tNet.G_supergraph.number_of_nodes(), self.tNet.G_supergraph.number_of_edges())

    def _init_metrics(self) -> Dict[str, List[float]]:
        keys = [
            "single_share",
            "double_share",
            "triple_share",
            "total_cars",
            "reb_cars",
            "ped_flow",
            "bike_flow",
            "pt_flow",
            "rp_flow",
            "reb_flow"
        ]
        self.metrics = {k: [] for k in keys}

    # ---------- adaptive-µ -------------------------------------------

    def _adapt_mu(self, it: int, mu: float, obj: float | None, obj_prev: float | None) -> float:
        '''
        Algorithm for updating proximal term. Only initialize mu once it==2.
        '''
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

    def _solve_cars(self, it, mu, prev_x, reb_cars_est, c_ratio, r):
        '''
        Prepare for I-AMoD optimziation call. Set solverParams, create network snapshot
        (if not present)
        '''
        params = SolverParams(
            iteration=it,
            mu=mu,
            prev_x=prev_x,
            rebalancing=True,
            vehicle_limit=self.cfg.vehicle_limit,
            c_ratio=c_ratio,
            reb_cars=reb_cars_est,
            r=r,
            rho_time=self.cfg.rho_time
        )
        
        if it == 0 or self.nash == 1:
            snap = self._cars_snapshot = _build_snapshot(self.tNet)
        res = _solve_cars_gurobi_M(self.tNet, snap, params)
        return res.avg_time, res.x_vec, res.expected_cars, res.obj_val

    # ---------- OD extraction ----------------------------------------

    def _extract_ridepool_od(self) -> np.ndarray:
        '''
        Convert flows in X to ride-pooling demand matrix
        '''
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
        '''
        Load in precomputed dataset of possible rp combinations, call 
        rp solver.
        '''
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
            self.nash,
            self.cfg.verbose
        )

    def _update_road_edge_costs(self, y: np.ndarray, yr: np.ndarray) -> Tuple[float, float]:
        '''
        Update costs with BRP based on flow and capacity in real road network. Also set flow
        in roadgraph.
        '''
        total_cars = reb_cars = 0.0
        for k, (u, v) in enumerate(self.original_G.edges()):
            edge = self.original_G[u][v]
            t0, cap = edge["t_0"], edge["capacity"]
            flow = y[k, :].sum() + yr[k]
            edge["t_1"] = t0 * (1 + 0.15 * (flow / cap) ** 4)
            total_cars += flow * edge["t_1"]
            reb_cars += yr[k] * edge["t_1"]
            edge['flow'] = flow
            edge['flowRebalancing'] = yr[k]
        return total_cars, reb_cars


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
        '''
        If N_cars AND (x or obj) within tolerances increment stablility counter.
        '''
        if x_prev is not None:
            # for the cars, take 2% overhead since we are approaching the solution anyway
            if np.linalg.norm(x.sum(axis=1) - x_prev.sum(axis=1)) < self.cfg.tol_x and total_cars - 0.02 * total_cars < self.cfg.vehicle_limit:
                stable_hits += 1
                logger.info("Δx below tol (hit %d/%d)", stable_hits, self.cfg.stable_needed)
            elif obj_prev is not None and abs(obj - obj_prev) < self.cfg.tol_obj*obj and total_cars - 0.02 * total_cars < self.cfg.vehicle_limit:
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
        (array of row indices). 
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

    def _apply_new_times(self, Gs:nx.DiGraph, total_delay):
        '''
        Applies new shortests paths + delays to direct rp arcs in supergraph
        '''
        for u, v, d in Gs.edges(data=True):
            if d["type"] == "rp":
                base = nx.shortest_path_length(self.original_G, source=s2int(u), target=s2int(v), weight="t_1")
                if isinstance(total_delay, np.ndarray):
                    extra = total_delay[self._car_node_idx[s2int(v)], self._car_node_idx[s2int(u)]]
                else:
                    extra=0
                d["t_1"] = base + extra
                d["t_cars"] = base
            else:
                d["t_1"] = d["t_0"]

    def _update_supergraph_costs(self, D_rp: np.ndarray, gamma_arr: np.ndarray) -> None:
        '''
        Update the costs in the supergraph based on rp optimzation. Calculates
        delays (temporal and spatial), calculates new shortest routes and applies in
        't_1'
        '''
        n = self._n_nodes
        Tmat = np.zeros((n, n), dtype=np.float64)
        G_nodes = list(self.tNet.G.nodes())

        # precompute shortest paths through dijkstra algorithm
        for src_idx, src_raw in enumerate(G_nodes):
            lengths = nx.single_source_dijkstra_path_length(
                self.original_G, source=src_raw, weight="t_1")
            for dst_raw, d in lengths.items():
                if dst_raw in G_nodes:
                    dst_idx = self._car_node_idx[dst_raw]
                    Tmat[src_idx, dst_idx] = d
        # load in possible combinations
        with np.load(self.cfg.mat_fulllist) as data:
            full_list = data[data.files[0]].astype(np.float32)

        # mask for unused combinations
        mask       = gamma_arr > 1e-3
        rows_used  = full_list[mask]
        gammas     = gamma_arr[mask].astype(np.float64)

        jj1_arr = np.take(self._car_node_idx_np, rows_used[:, 3].astype(np.int32))
        ii1_arr = np.take(self._car_node_idx_np, rows_used[:, 4].astype(np.int32))
        jj2_arr = np.take(self._car_node_idx_np, rows_used[:, 5].astype(np.int32))
        ii2_arr = np.take(self._car_node_idx_np, rows_used[:, 6].astype(np.int32))
        seq0_arr = rows_used[:, 7].astype(np.int8)   # 1 or 0  (for [1 2 1 2] vs [1 2 2 1])

        # prepare for parallel processing
        idx        = np.arange(len(rows_used))
        n_jobs     = min(cpu_count(), max(1, len(idx)//10_000))  # heuristic
        chunks     = np.array_split(idx, n_jobs)

        partials = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(self._chunk_delay)(chunk, jj1_arr, ii1_arr, jj2_arr, ii2_arr,
                                seq0_arr, gammas, D_rp, Tmat)
            for chunk in chunks
        )

        OD_delays = sum(p[0] for p in partials)
        Et        = sum(p[1] for p in partials)

        total_delay = np.divide(OD_delays + Et, D_rp,
                                out=np.zeros_like(D_rp), where=D_rp != 0)
        Gs = self.tNet.G_supergraph
        self._apply_new_times(Gs, total_delay)

        del full_list
        gc.collect()

    # ---------- metric book-keeping -----------------------------------

    def _record_metrics(
        self,
        demand_stats: Tuple[float, float, float],
        total_cars: float,
        reb_cars: float,
    ) -> None:
        ped_flow=0
        bike_flow=0
        pt_flow=0
        rp_flow=0
        reb_flow=0
        # compute time based modal shares in hours
        for u,v,d in self.tNet.G_supergraph.edges(data=True):
            if "p" in d['type']:
                ped_flow += self.tNet.G_supergraph[u][v]['flowNoRebalancing']*self.tNet.G_supergraph[u][v]['t_1']
            elif "b" in d['type']:
                bike_flow += self.tNet.G_supergraph[u][v]['flowNoRebalancing']*self.tNet.G_supergraph[u][v]['t_1']
            elif 's' in d['type']:
                pt_flow += self.tNet.G_supergraph[u][v]['flowNoRebalancing']*self.tNet.G_supergraph[u][v]['t_1']
            elif 'rp' in d['type']:
                rp_flow += self.tNet.G_supergraph[u][v]['flowNoRebalancing']*self.tNet.G_supergraph[u][v]['t_1']
        # rebalancing flows
        for u,v in self.original_G.edges():
            reb_flow += self.original_G[u][v]['flowRebalancing']*self.original_G[u][v]['t_1']
        orig, solo, pooled, solo_perc, pooled_perc = demand_stats
        # store
        self.metrics["single_share"].append(solo_perc)
        self.metrics["double_share"].append(pooled_perc)
        self.metrics["triple_share"].append(max(0.0, 1 - solo_perc - pooled_perc))
        self.metrics["total_cars"].append(total_cars)
        self.metrics["reb_cars"].append(reb_cars)
        self.metrics["ped_flow"].append(ped_flow)
        self.metrics["bike_flow"].append(bike_flow)
        self.metrics["pt_flow"].append(pt_flow)
        self.metrics["rp_flow"].append(rp_flow)
        self.metrics["reb_flow"].append(reb_flow)
        logger.info(f"total cars: {total_cars}, single share: {solo_perc}, double_share: {pooled_perc}")
        logger.info(f"Solo demand: {np.sum(solo)}, Pooled demand: {np.sum(pooled)}")

    # ---------- plotting ----------------------------------------------

    def _plot_mode_share(self) -> None:
        M = self.metrics
        bar_colors = plt.get_cmap("tab10").colors
        fig, ax1 = plt.subplots()
        x = np.arange(len(self.metrics["single_share"]))
        color = 'tab:blue'
        rp = M["rp_flow"]
        ax1.bar(x, rp, label="ride-pooling", color=bar_colors[0], edgecolor='black', linewidth=0.5)
        bottom=rp
        ax1.bar(x, M['pt_flow'], bottom=bottom, label="Public transporation", color=bar_colors[1], edgecolor='black', linewidth=0.5)
        bottom=[rp[i] + M['pt_flow'][i] for i in range(len(rp))]
        ax1.bar(x, M['bike_flow'], bottom=bottom, label="Biking",color=bar_colors[2], edgecolor='black', linewidth=0.5)
        bottom=[rp[i] + M['pt_flow'][i] + M['bike_flow'][i] for i in range(len(rp))]
        ax1.bar(x, M['ped_flow'], bottom=bottom, label="Walking", color=bar_colors[3], edgecolor='black', linewidth=0.5)  
        bottom = [rp[i] + M['pt_flow'][i] + M['bike_flow'][i] + M['ped_flow'][i] for i in range(len(rp))]
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel(r"Time-based Modal Share ($\mathrm{h}$)", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(r"($\mathrm{\%}$) of rp requests pooled", color=color)
        ax2.plot(x, [i * 100 for i in M["double_share"]], color=color, marker = 'o')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([90,100])
        fig.legend(loc='lower left')
        fig.tight_layout()
        fig.savefig(f"{self.cfg.results_dir}mode_share.pdf", format='pdf')


    # ---------- CSV output --------------------------------------------

    def _save_metrics_csv(self) -> None:
        if not os.path.exists(self.cfg.results_dir):
            os.mkdir(self.cfg.results_dir)
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.cfg.results_dir + f"results_{self.cfg.city_tag}.csv", index=False)
        logger.info("Metrics saved → %s", self.cfg.results_dir)