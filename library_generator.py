# Hovorka Model Monte Carlo Simulation
# Main script for generating a library of patient simulations in parallel.

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import ExportConfig
from src.library_generation import generate_library_parallel
from src.simulation_config import SimulationConfig


if __name__ == "__main__":
    # Realism-knob experiment (ml/docs/SIM_REALISM.md): generate twin cohorts that
    # differ ONLY in the per-patient therapy-heterogeneity knob, identical otherwise.
    #   --knob on  → per-patient glycaemic target + therapy mis-calibration (default)
    #   --knob off → control cohort (every patient calibrated to one target, perfect ICR/ISF)
    # Output filename is tagged knobon/knoboff so the two cohorts never collide.
    ap = argparse.ArgumentParser(description="parallel patient-library generation")
    ap.add_argument("--n_patients", type=int, default=2000)
    ap.add_argument("--n_days", type=int, default=42)
    ap.add_argument("--knob", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="",
                    help="extra filename tag (pass $LSB_JOBID on HPC so concurrent "
                         "jobs that share a timestamp folder never collide)")
    args = ap.parse_args()
    knob_on = args.knob == "on"
    suffix = f"knob{args.knob}" + (f"_{args.tag}" if args.tag else "")
    # On DTU HPC (LSF), use the allocated CPU count instead of the node's total.
    # os.cpu_count() returns all CPUs on the physical node (e.g. 64), not your
    # LSF allocation, which would over-subscribe your job and risk getting killed.
    # LSB_MAX_NUM_PROCESSORS is set automatically by LSF to exactly your --ncpus.
    # Locally (no LSF), fall back to half the available cores as before.
    _lsf_cpus = int(os.environ.get("LSB_MAX_NUM_PROCESSORS", 0))
    workers = _lsf_cpus if _lsf_cpus > 0 else max(1, (os.cpu_count() or 2) // 2)

    config = SimulationConfig(
        n_patients=args.n_patients,
        n_days=args.n_days,  # 42 days: long baseline before anomaly days; matches deliverable horizon
        international_unit=True,
        noise_std=0.10,
        noise_autocorr=0.7,
        random_scenarios=True,
        clip_states=True,
        std_patient=False,
        random_seed=args.seed,
        enable_plots=False,
        # Realism knob #1 (both default ON in SimulationConfig; flipped off for the control cohort)
        personalise_glycemic_target=knob_on,
        therapy_miscalibration=knob_on,
    )

    export_config = ExportConfig(
        export_to_parquet=True,
        export_to_csv=False
    )

    t0 = time.perf_counter()
    print(f"  knob={args.knob}  →  output suffix: {suffix}", flush=True)
    folder = generate_library_parallel(config, export_config, workers=workers,
                                       name_suffix=suffix)
    total_s = time.perf_counter() - t0

    mins, secs = divmod(int(total_s), 60)
    print(f"Parallel library generated at: {folder}  (total {mins:02d}:{secs:02d})")
