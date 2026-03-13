import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import ExportConfig
from src.library_generation import generate_library_parallel
from src.simulation_config import SimulationConfig


if __name__ == "__main__":
    workers = max(1, (os.cpu_count() or 2) // 2)

    config = SimulationConfig(
        n_patients=10,
        n_days=3,
        international_unit=True,
        noise_std=0.10,
        noise_autocorr=0.7,
        random_scenarios=True,
        clip_states=True,
        std_patient=False,
        random_seed=999,
        enable_plots=False,
    )

    export_config = ExportConfig(export_to_parquet=True, export_to_csv=True)
    folder = generate_library_parallel(config, export_config, workers=workers)
    print(f"Parallel library generated at: {folder}")
