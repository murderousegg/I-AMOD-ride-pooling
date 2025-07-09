from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
import experiments.build_NYC_subway_net as nyc
import src.tnet as tnet
import pickle
from typing import Dict, List
from src.solvers import *
import pandas as pd
from src.solvers import _build_snapshot, _solve_cars_gurobi_fair
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class sufficiencySimulation(RidePoolingSimulationCore):
    def __init__(self, cfg: SimulationConfig):
        super().__init__(cfg)
        self.Tmax = cfg.Tmax
        self.tNet_private, self.tstamp, self.fcoeffs = nyc.build_NYC_net(
            "data/net/NYC/", only_road=True
        )
        with open("data/gml/NYC_small_demands.gpickle", 'rb') as f:
            self.g = pickle.load(f)
    
    def run(self) -> None:
        self._init_metrics()
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
            # np.save("temp.npy", D_rp)
            # D_rp = np.load("temp.npy")
            y, yr, demand_split, gamma_arr = self._compute_pooled(D_rp)
            total_cars, reb_cars = self._update_road_edge_costs(y, yr)
            
            if it != 0:
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
            r = 1 + self.metrics["double_share"][-1]
            self._plot_sufficiency_share(self, avg_time, it, x)
        self._save_metrics_csv()
        return avg_time, x
    
    def _solve_cars(self, it, mu, prev_x, reb_cars_est, c_ratio, r):
        params = SolverParams(
            iteration=it,
            mu=mu,
            prev_x=prev_x,
            rebalancing=True,
            vehicle_limit=self.cfg.vehicle_limit,
            c_ratio=c_ratio,
            reb_cars=reb_cars_est,
            r=r,
            Tmax=self.Tmax
        )
        snap = getattr(self, "_cars_snapshot", None)
        if snap is None:
            snap = self._cars_snapshot = _build_snapshot(self.tNet, fairness=True)
        res = _solve_cars_gurobi_fair(self.tNet, snap, params)
        return res.avg_time, res.x_vec, res.expected_cars, res.obj_val

    def _bin_results(self, avg_time, x):
        num_bins = 20
        ODs = len(avg_time)
        sorted_idx = np.argsort(avg_time)
        bin_ids = np.empty(ODs, dtype=int)
        bin_size = ODs // num_bins
        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else ODs
            bin_ids[sorted_idx[start:end]] = i
        N_edges = x.shape[0]
        binned_x = np.zeros((N_edges, num_bins))

        alphas = np.array(list(self.tNet.g.values()))
        binned_alphas = np.bincount(bin_ids, weights=alphas, minlength=num_bins)
        for i in range(N_edges):
            np.add.at(binned_x[i], bin_ids, x[i])
        return binned_x, binned_alphas


    def _plot_sufficiency_share(self, avg_time, it, x):
        binned_x, binned_alphas = self._bin_results(avg_time, x)
        ped_flow = np.zeros(binned_x.shape[1])
        bike_flow = np.zeros(binned_x.shape[1])
        pt_flow = np.zeros(binned_x.shape[1])
        rp_flow = np.zeros(binned_x.shape[1])
        for idx, (u,v,d) in enumerate(self.tNet.G_supergraph.edges(data=True)):
            if d['type'] == "'":
                ped_flow += binned_x[idx,:]*self.tNet.G_supergraph[u][v]['t_1']
            elif d['type'] == "b":
                bike_flow += binned_x[idx,:]*self.tNet.G_supergraph[u][v]['t_1']
            elif d['type'] == 's':
                pt_flow += binned_x[idx,:]*self.tNet.G_supergraph[u][v]['t_1']
            elif d['type'] == 'rp':
                rp_flow += binned_x[idx,:]*self.tNet.G_supergraph[u][v]['t_1']
        ped_flow /= binned_alphas
        bike_flow /= binned_alphas
        pt_flow /= binned_alphas
        rp_flow /= binned_alphas
        modal_data = np.vstack([rp_flow, pt_flow, bike_flow, ped_flow])
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = ['ride-pooling', 'Public transporation', 'Biking', 'Walking']
        colors = plt.get_cmap("tab10").colors[0:3]
        bottom = np.zeros(bike_flow.shape[0])
        for i, (data, label, color) in enumerate(zip(modal_data, labels, colors)):
            ax.bar(np.arange(bike_flow.shape[0]), data, bottom=bottom, label=label, color=color)
            bottom += data  
        plt.axvline(self.Tmax, color='r', linestyle='dashed', linewidth=1)
        ax.set_xlabel('Bin index')
        ax.set_ylabel('Average time per request (min)')
        ax.set_title('Mode-specific travel times per bin')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.cfg.results_dir}_fairness_{it}_avg_time_dist.pdf", format='pdf')
        plt.show(block=False)

def main() -> None:
    cfg = SimulationConfig()

    # create results directory
    now_string = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    cfg.results_dir = f"results/Fairness_{now_string}/"
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    ###
    cfg.max_iterations = 1
    cfg.vehicle_limit = 45000
    cfg.mu_initial = 1e-2
    cfg.stable_needed = 3
    cfg.demand_multiplier=4
    cfg.delay_factor=1 / 60
    cfg.waiting_time=1 / 60

    sim = sufficiencySimulation(cfg)
    avg_time, x = sim.run()



if __name__ == "__main__":
    main()