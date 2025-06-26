from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
import experiments.build_NYC_subway_net as nyc
import src.tnet as tnet
import pickle
from typing import Dict, List
from src.solvers import *
import pandas as pd

class penrateSimulation(RidePoolingSimulationCore):
    def __init__(self, cfg: SimulationConfig):
        super().__init__(cfg.core)
        self.stackelberg:int=0
        self.metrics: Dict[str, List[float]]
        self.tNet_private, self.tstamp, self.fcoeffs = nyc.build_NYC_net(
            "data/net/NYC/", only_road=True
        )
        self._init_penrate_metrics()

        with open("data/gml/NYC_small_demands.gpickle", 'rb') as f:
            self.g = pickle.load(f)

    def _init_penrate_metrics(self):
        keys = [
            "reb_flow",
            "ped_flow",
            "bike_flow",
            "pt_flow",
            "rp_flow",
            "IAMoDCosts",
            "privateCosts"
            "totCost"
        ]
        self.penrate_metrics = {k: [] for k in keys}

    def _init_penrate(self, penrate):
        self.cfg.vehicle_limit *= penrate
        self._initialize_networks(pen_rate=penrate)

    def _initialize_networks(self, pen_rate):
        self.tNet.set_g(tnet.perturbDemandConstant(self.g, constant=(1-pen_rate)/24))
        self.tNet_private.set_g(tnet.perturbDemandConstant(self.g, constant=pen_rate/24))

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

    def run_private(self) -> None:
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
        self.penrate_metrics["IAMoDCosts"].append(IAMoDCosts)
        self.penrate_metrics["privateCosts"].append(privateCosts)
        self.penrate_metrics["totCost"].append(totCost)
        logger.info(f"penetration rate: {pen_rate}")
    
    def save_penrate_csv(self):
        df = pd.DataFrame(self.penrate_metrics)
        df.to_csv(self.cfg.results_csv / f"results_{self.cfg.city_tag}_penrate_metrics.csv", index=False)
        logger.info("Metrics saved → %s", self.cfg.results_csv)

    
def main() -> None:
    cfg = SimulationConfig()
    cfg.vehicle_limit = 1200
    sim = penrateSimulation(cfg)
    for pen_rate in np.linspace(0.01,0.99, 10):
        sim._init_penrate(pen_rate)
        for stackelberg in range(10):
            sim.stackelberg = stackelberg
            sim.run()
            sim.run_private()
        sim.log_penrate_results(pen_rate)
        sim.save_penrate_csv()
            


if __name__ == "__main__":
    main()