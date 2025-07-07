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
            self._plot_sufficiency_share(self, avg_time, it)
        self._save_metrics_csv()
    
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
            snap = self._cars_snapshot = _build_snapshot(self.tNet)
        res = _solve_cars_gurobi_fair(self.tNet, snap, params)
        return res.avg_time, res.x_vec, res.expected_cars, res.obj_val

    def _plot_sufficiency_share(self, avg_time, it):
        plt.figure()
        average_all = np.array([self.tNet.G_supergraph[u][v]['flowNoRebalancing'] * self.tNet.G_supergraph[u][v]['t_1'] for u,v in self.tNet.G_supergraph.edges()]).sum()/np.array([k for k in self.tNet.g.values()]).sum()
        plt.hist(avg_time, bins=10 , color='c', edgecolor='k', alpha=0.65)
        plt.axvline(average_all, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(self.Tmax, color='r', linestyle='solid', linewidth=1)
        plt.savefig(f"results/fairness/fairness_{it}_avg_time_dist.pdf")
        np.save(f"results/fairness/average_all_{it}.npy", np.array(average_all))
        np.save(f"results/fairness/avg_time_{it}.npy", np.array(avg_time))
            

def main() -> None:
    cfg = SimulationConfig()
    cfg.vehicle_limit = 1200
    sim = sufficiencySimulation(cfg)
    sim.run()


if __name__ == "__main__":
    main()