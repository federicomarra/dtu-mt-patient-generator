# Hovorka Model Monte Carlo Simulation
# Main script for generating a library of patient simulations in parallel.

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
        n_patients=20000,
        n_days=14,           # 2 weeks: gives sequence models a full baseline before anomaly days
        international_unit=True,
        noise_std=0.10,
        noise_autocorr=0.7,
        random_scenarios=True,
        clip_states=True,
        std_patient=False,
        random_seed=42,
        enable_plots=False,
        # Slightly tighter quality thresholds for cleaner ML training data
        quality_max_hypo_pct_threshold=3.0,   # default 4.0
        quality_max_hyper_pct_threshold=10.0, # default 12.0
    )

    export_config = ExportConfig(
        export_to_parquet=True,
        export_to_csv=False
    )
    
    folder = generate_library_parallel(config, export_config, workers=workers)
    print(f"Parallel library generated at: {folder}")
