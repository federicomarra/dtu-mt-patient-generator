from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

from src.export import ExportConfig, export_to_formats
from src.simulation import run_simulation
from src.simulation_config import SimulationConfig
from src.simulation_utils import create_export_directory


def _worker_run(args: tuple[int, int, SimulationConfig]) -> dict[int, dict[str, object]]:
    worker_idx, n_patients_chunk, base_config = args

    worker_seed = (base_config.random_seed or 0) + (worker_idx * 10000)
    worker_config = replace(
        base_config,
        n_patients=n_patients_chunk,
        random_seed=worker_seed,
        enable_plots=False,
    )

    no_export = ExportConfig(export_to_parquet=False, export_to_csv=False)
    result = run_simulation(
        worker_config,
        no_export,
        return_results=True,
        show_progress=False,
        show_summary=False,
    )

    if result is None:
        return {}
    return {k: dict(v) for k, v in result.items()}


def _merge_results(
    worker_results: list[dict[int, dict[str, object]]],
) -> dict[int, dict[str, object]]:
    merged: dict[int, dict[str, object]] = {}
    next_patient_id = 0

    for block in worker_results:
        for old_patient_id in sorted(block.keys()):
            entry = dict(block[old_patient_id])
            entry["patient_id"] = next_patient_id
            merged[next_patient_id] = entry
            next_patient_id += 1

    return merged


def generate_library_parallel(
    config: SimulationConfig,
    export_config: ExportConfig,
    workers: int,
    output_base_folder: str = "monte_carlo_results_parallel",
) -> Path | None:
    """Generate a large patient library in parallel and export merged results."""
    if workers <= 0:
        raise ValueError("workers must be >= 1")

    target_patients = int(config.n_patients)
    if target_patients <= 0:
        raise ValueError("config.n_patients must be >= 1")

    workers_eff = min(workers, target_patients)
    chunk = int(math.ceil(target_patients / workers_eff))
    chunks = [chunk] * workers_eff
    overflow = (chunk * workers_eff) - target_patients
    for i in range(overflow):
        chunks[-(i + 1)] -= 1
    chunks = [c for c in chunks if c > 0]

    args: list[tuple[int, int, SimulationConfig]] = [
        (idx, n_chunk, config) for idx, n_chunk in enumerate(chunks)
    ]

    if workers_eff == 1:
        blocks = [_worker_run(args[0])]
    else:
        with mp.Pool(processes=workers_eff) as pool:
            blocks = pool.map(_worker_run, args)

    merged = _merge_results(blocks)
    accepted_total = len(merged)

    output_folder = create_export_directory(base_folder=output_base_folder)
    if output_folder is None:
        return None

    metadata: dict[str, object] = {
        "parallel_workers": workers_eff,
        "requested_patients": target_patients,
        "accepted_patients": accepted_total,
        "n_days": config.n_days,
        "random_seed": config.random_seed,
        "enable_plots": False,
    }

    export_to_formats(
        results_dict=merged,
        n_patients=accepted_total,
        n_days=config.n_days,
        output_folder=output_folder,
        export=export_config.to_list(),
        config_metadata=metadata,
    )

    return output_folder
