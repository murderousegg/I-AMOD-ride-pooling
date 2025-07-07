from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
import experiments.build_NYC_subway_net as nyc
import src.tnet as tnet
import pickle
from typing import Dict, List
from src.solvers import *
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import fields
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.serif": ["Times"],
    "font.size": 13,                     # IEEE style prefers 8–10 pt
    "pdf.fonttype": 42,   # Important: embed fonts correctly in PDF
    "ps.fonttype": 42,
    "text.usetex": True,
    "legend.fontsize": 12,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

class penrateSimulation(RidePoolingSimulationCore):
    def __init__(self, cfg: SimulationConfig):
        super().__init__(cfg)
        self.stackelberg:int=0
        self.metrics: Dict[str, List[float]]
        self.tNet_private, self.tstamp, self.fcoeffs = nyc.build_NYC_net(
            "data/net/NYC/", only_road=True
        )
        self._init_penrate_metrics()

        with open("data/gml/NYC_small_demands.gpickle", 'rb') as f:
            self.g = pickle.load(f)
        self.g = tnet.perturbDemandConstant(self.g, self.cfg.demand_multiplier/24)
        self.private_g = tnet.perturbDemandConstant(self.tNet_private.g, self.cfg.demand_multiplier/24)

    def _init_penrate_metrics(self):
        keys = [
            "reb_flow",
            "ped_flow",
            "bike_flow",
            "pt_flow",
            "rp_flow",
            "private_flow",
            "IAMoDCosts",
            "privateCosts",
            "totCost"
        ]
        self.penrate_metrics = {k: [] for k in keys}

    def _init_penrate(self, penrate):
        self.cfg.vehicle_limit *= penrate
        self._initialize_networks(pen_rate=penrate)

    def _initialize_networks(self, pen_rate):
        self.tNet.set_g(tnet.perturbDemandConstant(self.g, constant=(1-pen_rate)))
        self.tNet_private.set_g(tnet.perturbDemandConstant(self.private_g, constant=pen_rate))

    def _update_road_edge_costs(self, y: np.ndarray, yr: np.ndarray) -> Tuple[float, float]:
        total_cars = reb_cars = 0.0
        for k, (u, v) in enumerate(self.original_G.edges()):
            edge = self.original_G[u][v]
            t0, cap = edge["t_0"], edge["capacity"]
            if self.stackelberg:
                exo_flow_ij = self.tNet_private.G[u][v]['flow']
            else:
                exo_flow_ij = 0
            flow = y[k, :].sum() + yr[k]
            flow_exo = flow + exo_flow_ij
            edge["t_1"] = t0 * (1 + 0.15 * (flow_exo / cap) ** 4)
            total_cars += flow_exo * edge["t_1"]
            reb_cars += yr[k] * edge["t_1"]
            edge['flow'] = flow
            edge['flowRebalancing'] = yr[k]
        return total_cars, reb_cars
    
    def add_private_to_rg(self) -> None:
        """
        Add private vehicle flows to the original road graph.
        This is done by updating the edge flows in the original_G with the flows from tNet_private.
        """
        for i, j in self.original_G.edges():
            self.original_G[i][j]['flow'] += self.tNet_private.G[i][j]['flow']

    @timeit
    def run_private(self) -> None:
        self.tNet_private.TAP.n_iter_tm = 500    # limit tap iterations
        self.tNet_private.solveMSA(exogenous_G=self.original_G, verbose=0)   #set verbose 1 for console prints

    def log_penrate_results(self, pen_rate)-> None:
        ### append flows in user travel time
        final_metrics = {k: v[-1] for k, v in self.metrics.items()}
        IAMoDFlow = final_metrics["reb_flow"] + final_metrics["ped_flow"] + final_metrics["bike_flow"] + final_metrics["pt_flow"] + final_metrics["rp_flow"]
        IAMoDCosts = IAMoDFlow/sum(self.tNet.g.values())
        privateFlow = sum([self.tNet_private.G[i][j]['flow'] * self.original_G[i][j]['t_1'] for i,j in self.original_G.edges()])
        privateCosts = privateFlow/sum(self.tNet_private.g.values())
        totCost = ((IAMoDFlow+privateFlow)/sum(self.g.values()))
        self.penrate_metrics["reb_flow"].append(final_metrics["reb_flow"])
        self.penrate_metrics["ped_flow"].append(final_metrics["ped_flow"])
        self.penrate_metrics["bike_flow"].append(final_metrics["bike_flow"])
        self.penrate_metrics["pt_flow"].append(final_metrics["pt_flow"])
        self.penrate_metrics["rp_flow"].append(final_metrics["rp_flow"])
        self.penrate_metrics["private_flow"].append(privateFlow)
        self.penrate_metrics["IAMoDCosts"].append(IAMoDCosts)
        self.penrate_metrics["privateCosts"].append(privateCosts)
        self.penrate_metrics["totCost"].append(totCost)
        logger.info(f"penetration rate: {pen_rate}")
    
    def save_penrate_csv(self):
        df = pd.DataFrame(self.penrate_metrics)
        df.to_csv(self.cfg.results_dir + f"penrate_metrics.csv", index=False)
        logger.info("Metrics saved → %s", self.cfg.results_dir)


def plot_penrate(sim: penrateSimulation, dir: str) -> None:
    # load params
    totCost = sim.penrate_metrics["totCost"]
    IAMoDCosts = sim.penrate_metrics["IAMoDCosts"]
    privateCosts = sim.penrate_metrics["privateCosts"]
    privateFlow = sim.penrate_metrics["private_flow"]
    rpFlow = sim.penrate_metrics["rp_flow"]
    pedFlow = sim.penrate_metrics["ped_flow"]
    bikeFlow = sim.penrate_metrics["bike_flow"]
    ptFlow = sim.penrate_metrics["pt_flow"]

    plt.figure()
    plt.plot(list(np.linspace(0.01,0.99, len(totCost))), totCost, label='Average all users', marker='o')
    plt.plot(list(np.linspace(0.01,0.99, len(totCost))), IAMoDCosts, label='I-AMoD', marker='o')
    plt.plot(list(np.linspace(0.01,0.99, len(totCost))), privateCosts, label='Private', marker= 'o')
    plt.legend()
    plt.xlabel(r'Penetration Rate ($\mathrm{\%}$)')
    plt.ylabel(r'Avg. Travel Time ($\mathrm{h}$)')
    plt.grid(True, alpha=0.5)
    plt.xlim([0,1])
    plt.tight_layout()
    plt.savefig(sim.cfg.results_dir + f"costs.pdf")
    plt.figure()
    width = 0.05
    ind = list(np.linspace(0.01,0.99, len(totCost)))
    p1 = plt.bar(ind, privateFlow, width)
    p2 = plt.bar(ind, rpFlow, width,
                bottom=privateFlow)
    p3 = plt.bar(ind, ptFlow, width, bottom=[x+y for x,y in zip(privateFlow, rpFlow)])
    p4 = plt.bar(ind, bikeFlow, width,
                bottom=[x+y+z for x,y,z in zip(privateFlow, rpFlow, ptFlow)])
    p5 = plt.bar(ind, pedFlow, width,
                bottom=[x+y+z+i for x,y,z,i in zip(privateFlow, rpFlow, ptFlow, bikeFlow)])
    plt.ylabel(r"Time-based Modal Share ($\mathrm{h}$)")
    plt.xlabel(r"Penetration Rate ($\mathrm{\%}$)")
    plt.legend((p1[0], p2[0], p3[0], p4[0],p5[0]), ('Private', 'Ride-pooling', 'Public transportation', 'Biking', 'Walking'))
    plt.xlim([0,1])
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.5)
    plt.savefig(sim.cfg.results_dir + "modal_share.pdf")
    plt.show()

            
def main() -> None:
    cfg = SimulationConfig()

    # create results directory
    now_string = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    cfg.results_dir = f"results/penRate_NYC_{now_string}/"
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    ###

    cfg.max_iterations = 1
    cfg.vehicle_limit = 1400
    cfg.mu_initial = 1e-2
    cfg.stable_needed = 3
    cfg.demand_multiplier=2
    cfg.delay_factor= 2 / 60
    cfg.waiting_time= 2 / 60
    sim = penrateSimulation(cfg)
    for pen_rate in np.linspace(0.01,0.99, 4):
        sim._init_penrate(pen_rate)
        for stackelberg in range(1):
            sim.stackelberg = stackelberg
            sim.run()
            sim.run_private()
            sim.add_private_to_rg()
        sim.log_penrate_results(pen_rate)
        sim.save_penrate_csv()
    plot_penrate(sim, cfg.results_dir)
    with open(cfg.results_dir+ "config.txt", "w") as f:
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            f.write(f"{field.name}:{value}\n")
    
            


if __name__ == "__main__":
    main()