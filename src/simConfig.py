from pathlib import Path
from dataclasses import dataclass

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