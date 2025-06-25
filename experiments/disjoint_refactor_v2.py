from __future__ import annotations
from src.simulationCore import RidePoolingSimulationCore
from src.simConfig import SimulationConfig
from src.solvers import *

def main() -> None:
    cfg = SimulationConfig()
    sim = RidePoolingSimulationCore(cfg)
    sim.run()


if __name__ == "__main__":
    main()
