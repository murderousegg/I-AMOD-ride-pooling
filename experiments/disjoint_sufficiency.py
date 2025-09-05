from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
import experiments.build_NYC_subway_net as nyc
import pickle
from src.solvers import *
from src.solvers import _build_snapshot, _solve_cars_gurobi_fair
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dataclasses import fields
import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger('iamod')
logger.setLevel(logging.INFO)

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
            avg_time, x, expected_cars, obj, obj_dict = self._solve_cars(
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
            self._plot_sufficiency_share(avg_time, it, x, obj_dict)
        self._plot_mode_share()
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
            snap = self._cars_snapshot = _build_snapshot(self.tNet)
        (res, obj_dict) = _solve_cars_gurobi_fair(self.tNet, snap, params)
        return res.avg_time, res.x_vec, res.expected_cars, res.obj_val, obj_dict

    def _weighted_percentile(self, data, weights, percentiles):
        """Return weighted percentiles of 1-D `data`."""
        data = np.asarray(data)
        weights = np.asarray(weights)
        sorter = np.argsort(data)
        data_sorted = data[sorter]
        w_sorted = weights[sorter]

        cdf = np.cumsum(w_sorted)
        cdf /= cdf[-1]
        return np.interp(np.asarray(percentiles) / 100.0, cdf, data_sorted)

    def _bin_results(self, avg_time, x, bin_width_min=1):
        avg_time_min = np.asarray(avg_time) * 60  # Convert from hours to minutes
        alpha_o = self._cars_snapshot.alpha_o

        # stats
        mean_w = np.average(avg_time_min, weights=alpha_o)
        var_w = np.average((avg_time_min - mean_w) ** 2, weights=alpha_o)
        std_w = np.sqrt(var_w)
        cv_w = std_w / mean_w
        p50, p85, p90, p95 = self._weighted_percentile(
            avg_time_min, alpha_o, [50, 85, 90, 95])

        print(f"Current iteration Trip-time stats (min): "
          f"mean={mean_w:.2f}, std={std_w:.2f}, CV={cv_w:.2f}, "
          f"P50={p50:.2f}, P85={p85:.2f}, "
          f"P90={p90:.2f}, P95={p95:.2f}")


        # Define bin edges in minutes
        max_time = np.ceil(avg_time_min.max())
        bin_edges = np.arange(0, max_time + bin_width_min, bin_width_min)
        num_bins = len(bin_edges) - 1

        # Bin assignments
        bin_ids = np.digitize(avg_time_min, bin_edges, right=False) - 1
        bin_ids = np.clip(bin_ids, 0, num_bins - 1)

        # Bin centers for plotting
        binned_times = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        N_edges = x.shape[0]
        binned_x = np.zeros((N_edges, num_bins))
        binned_alphas = np.bincount(bin_ids, weights=alpha_o, minlength=num_bins)

        for i in range(N_edges):
            np.add.at(binned_x[i], bin_ids, x[i])

        return binned_x, binned_alphas, binned_times


    def _plot_sufficiency_share(self, avg_time, it, x, obj_dict):
        binned_x, binned_alphas, binned_times = self._bin_results(avg_time, x, bin_width_min=0.5)

        ped_flow = np.zeros(binned_x.shape[1])
        bike_flow = np.zeros(binned_x.shape[1])
        pt_flow = np.zeros(binned_x.shape[1])
        rp_flow = np.zeros(binned_x.shape[1])

        for idx, (u, v, d) in enumerate(self.tNet.G_supergraph.edges(data=True)):
            t_1 = self.tNet.G_supergraph[u][v]['t_1']
            if "p" in d['type']:
                ped_flow += binned_x[idx, :] * t_1
            elif "b" in d['type']:
                bike_flow += binned_x[idx, :] * t_1
            elif 's' in d['type']:
                pt_flow += binned_x[idx, :] * t_1
            elif 'rp' in d['type']:
                rp_flow += binned_x[idx, :] * t_1

        modal_data = np.vstack([rp_flow, pt_flow, bike_flow, ped_flow]) * 1e-4
        # fig, ax = plt.subplots(figsize=(10, 5))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [5, 1]})
        labels = ['Ride-pooling', 'Public transportation', 'Biking', 'Walking']
        colors = plt.get_cmap("tab10").colors[:4]
        bottom = np.zeros_like(binned_times)

        for data, label, color in zip(modal_data, labels, colors):
            ax1.bar(
                binned_times, data, bottom=bottom, label=label, color=color,
                width=0.4, edgecolor='black', linewidth=0.5
            )
            bottom += data
        ax1.set_xlim(0,25)
        ax1.axvline(self.Tmax * 60, color='r', linestyle='dashed', linewidth=1.5, label=r'$T_{\max}$')
        # plt.text(.01, .3, f'Sufficiency obj: {obj_dict['suff']}\nTime obj: {obj_dict['base']}\nProximal term: {obj_dict['mu']}',\
        #           ha='left', va='top', transform=ax1.transAxes, fontsize=13)
        ax1.set_xlabel(r'Average time per request ($\mathrm{min}$)', fontsize = 20)
        ax1.set_ylabel(r'Time-Based Modal Share ($\times 10^4$ $\mathrm{h}$)', fontsize = 20)
        # ax1.set_title(fr"Commute Sufficiency $T_{{\max}} = {self.cfg.Tmax}$, $\phi = {self.cfg.demand_multiplier}$ and $N_{{\mathrm{{cars,max}}}} = {self.cfg.vehicle_limit/1000} \times 10^3$", fontsize=18)
        avg_all = np.average(binned_times, weights=binned_alphas)
        ax1.axvline(avg_all, color='k', linestyle='dashed', linewidth=1.5, label=r'$T_{\mathrm{avg}}$')
        ax1.legend(fontsize=17)

        total_per_mode = modal_data.sum(axis=1)
        bottom = 0
        for value, label, color in zip(total_per_mode, labels, colors):
            ax2.bar([0], [value], bottom=bottom, color=color, edgecolor='black', linewidth=0.5)
            bottom += value

        # Format the right bar
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xticks([])
        ax2.set_ylabel(r"Total Modal Share ($\times 10^4$ $\mathrm{h}$)", fontsize=20)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=20)

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
    cfg.max_iterations = 15
    cfg.vehicle_limit = 25000
    cfg.mu_initial = 1e-2
    cfg.stable_needed = 2
    cfg.demand_multiplier=2
    cfg.delay_factor=1 / 60
    cfg.waiting_time=1 / 60
    cfg.tol_obj = 0.1
    
    cfg.Tmax = 12/60    #10 mins

    sim = sufficiencySimulation(cfg)
    avg_time, x = sim.run()
    with open(cfg.results_dir+ "config.txt", "w") as f:
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            f.write(f"{field.name}:{value}\n")



if __name__ == "__main__":
    main()