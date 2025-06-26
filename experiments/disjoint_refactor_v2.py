from __future__ import annotations
from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
from src.solvers import *

def main() -> None:
    cfg = SimulationConfig()
    cfg.max_iterations = 7
    cfg.vehicle_limit = 150
    sim = RidePoolingSimulationCore(cfg)
    sim.run()


if __name__ == "__main__":
    main()
