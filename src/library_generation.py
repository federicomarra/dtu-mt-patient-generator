from __future__ import annotations

import math
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

from src.export import ExportConfig, export_to_formats
from src.simulation import run_simulation
from src.simulation_config import SimulationConfig
from src.simulation_utils import create_export_directory


# Return type includes a stats dict so the coordinator can print live progress
# without any shared memory or locks.
_WorkerResult = tuple[dict[int, dict[str, object]], dict[str, object]]


def _worker_run(args: tuple[int, int, SimulationConfig]) -> _WorkerResult:
    worker_idx, n_patients_chunk, base_config = args

    worker_seed = (base_config.random_seed or 0) + (worker_idx * 10000)
    worker_config = replace(
        base_config,
        n_patients=n_patients_chunk,
        random_seed=worker_seed,
        enable_plots=False,
    )

    no_export = ExportConfig(export_to_parquet=False, export_to_csv=False)
    t0 = time.perf_counter()
    result = run_simulation(
        worker_config,
        no_export,
        return_results=True,
        show_progress=False,
        show_summary=False,
    )
    elapsed = time.perf_counter() - t0

    if result is None:
        results: dict[int, dict[str, object]] = {}
    else:
        results = {k: dict(v) for k, v in result.items()}

    n_accepted = len(results)
    stats: dict[str, object] = {
        "worker_idx": worker_idx,
        "n_requested": n_patients_chunk,
        "n_accepted": n_accepted,
        "elapsed_s": elapsed,
        # Avoid division by zero; use requested as denominator so zero-accepted
        # workers still produce a finite (pessimistic) per-patient estimate.
        "s_per_patient": elapsed / n_patients_chunk,
    }
    return results, stats


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

    print(
        f"\n── Parallel library generation ──────────────────────────────\n"
        f"  target patients : {target_patients}  |  days/patient : {config.n_days}\n"
        f"  workers         : {workers_eff}  |  chunk sizes  : {chunks}\n"
        f"  random_seed     : {config.random_seed}\n"
        f"─────────────────────────────────────────────────────────────"
    )

    args: list[tuple[int, int, SimulationConfig]] = [
        (idx, n_chunk, config) for idx, n_chunk in enumerate(chunks)
    ]

    t_start = time.perf_counter()
    all_stats: list[dict[str, object]] = []
    blocks: list[dict[int, dict[str, object]]] = []

    if workers_eff == 1:
        result_block, stats = _worker_run(args[0])
        blocks.append(result_block)
        all_stats.append(stats)
        _print_worker_done(stats, workers_eff, t_start)
    else:
        with mp.Pool(processes=workers_eff) as pool:
            # imap_unordered lets us print a line as each worker finishes.
            pending_blocks: list[tuple[int, dict[int, dict[str, object]]]] = []
            for result_block, stats in pool.imap_unordered(_worker_run, args):
                all_stats.append(stats)
                worker_idx = int(stats["worker_idx"])  # type: ignore[arg-type]
                pending_blocks.append((worker_idx, result_block))
                _print_worker_done(stats, workers_eff, t_start)

        # Restore deterministic order before merging so patient IDs are stable.
        pending_blocks.sort(key=lambda x: x[0])
        blocks = [b for _, b in pending_blocks]

    merged = _merge_results(blocks)
    accepted_total = len(merged)
    total_elapsed = time.perf_counter() - t_start

    # ── Final summary ──────────────────────────────────────────────
    accepted_per_worker = [int(s["n_accepted"]) for s in all_stats]  # type: ignore[arg-type]
    s_per_patient_vals = [float(s["s_per_patient"]) for s in all_stats]  # type: ignore[arg-type]
    avg_s_per_patient = sum(s_per_patient_vals) / len(s_per_patient_vals) if s_per_patient_vals else 0.0
    acceptance_rate = 100.0 * accepted_total / target_patients if target_patients else 0.0

    print(
        f"\n── Summary ───────────────────────────────────────────────────\n"
        f"  accepted / requested : {accepted_total} / {target_patients}  ({acceptance_rate:.1f}%)\n"
        f"  per-worker accepted  : {accepted_per_worker}\n"
        f"  total elapsed        : {_fmt_elapsed(total_elapsed)}\n"
        f"  avg time / patient   : {avg_s_per_patient:.1f} s  (wall-clock per requested slot)\n"
        f"─────────────────────────────────────────────────────────────"
    )

    if accepted_total < target_patients * 0.8:
        print(
            f"Warning: acceptance rate {acceptance_rate:.1f}% is below 80%. "
            "Consider relaxing quality thresholds or increasing n_patients."
        )

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
        "total_elapsed_s": round(total_elapsed, 1),
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


def _print_worker_done(
    stats: dict[str, object],
    workers_eff: int,
    t_start: float,
) -> None:
    """Print a one-line status update when a worker finishes."""
    wall = time.perf_counter() - t_start
    idx = int(stats["worker_idx"])          # type: ignore[arg-type]
    accepted = int(stats["n_accepted"])     # type: ignore[arg-type]
    requested = int(stats["n_requested"])   # type: ignore[arg-type]
    elapsed = float(stats["elapsed_s"])     # type: ignore[arg-type]
    s_pp = float(stats["s_per_patient"])    # type: ignore[arg-type]
    print(
        f"  worker {idx+1:>2}/{workers_eff}  done  "
        f"accepted {accepted:>5}/{requested:<5}  "
        f"worker elapsed {_fmt_elapsed(elapsed)}  "
        f"({s_pp:.1f} s/patient)  "
        f"wall {_fmt_elapsed(wall)}",
        flush=True,
    )


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as mm:ss or h:mm:ss."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60:02d}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
