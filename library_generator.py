# Hovorka Model Monte Carlo Simulation
# Main script for generating a library of patient simulations in parallel.

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import ExportConfig
from src.library_generation import generate_library_parallel
from src.simulation_config import SimulationConfig


if __name__ == "__main__":
    # On DTU HPC (LSF), use the allocated CPU count instead of the node's total.
    # os.cpu_count() returns all CPUs on the physical node (e.g. 64), not your
    # LSF allocation, which would over-subscribe your job and risk getting killed.
    # LSB_MAX_NUM_PROCESSORS is set automatically by LSF to exactly your --ncpus.
    # Locally (no LSF), fall back to half the available cores as before.
    _lsf_cpus = int(os.environ.get("LSB_MAX_NUM_PROCESSORS", 0))
    workers = _lsf_cpus if _lsf_cpus > 0 else max(1, (os.cpu_count() or 2) // 2)

    config = SimulationConfig(
        n_patients=200,
        n_days=14,           # 2 weeks: gives sequence models a full baseline before anomaly days
        international_unit=True,
        noise_std=0.10,
        noise_autocorr=0.7,
        random_scenarios=True,
        clip_states=True,
        std_patient=False,
        random_seed=42,
        enable_plots=False,
    )

    export_config = ExportConfig(
        export_to_parquet=True,
        export_to_csv=False
    )

    t0 = time.perf_counter()
    folder = generate_library_parallel(config, export_config, workers=workers)
    total_s = time.perf_counter() - t0

    mins, secs = divmod(int(total_s), 60)
    print(f"Parallel library generated at: {folder}  (total {mins:02d}:{secs:02d})")
