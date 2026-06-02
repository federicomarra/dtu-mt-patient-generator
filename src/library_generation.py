from __future__ import annotations

import math
import multiprocessing as mp
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

from src.export import ExportConfig, export_to_formats, write_chunk_to_parquet, merge_parquet_chunks
from src.simulation import run_simulation
from src.simulation_config import SimulationConfig
from src.simulation_utils import create_export_directory


# Return type includes a stats dict so the coordinator can print live progress
# without any shared memory or locks.
_WorkerResult = tuple[dict[int, dict[str, object]], dict[str, object]]


def _worker_run(args: tuple[int, int, SimulationConfig, int]) -> _WorkerResult:
    worker_idx, n_patients_chunk, base_config, patient_offset = args

    # Shift the seed by the cumulative number of patients assigned to prior workers
    # so that worker i's patient j maps to global patient (patient_offset + j) in
    # the meal-schedule RNG space. This avoids the collision where workers with the
    # same (worker_idx + local_patient_id) sum produce identical meal schedules.
    worker_seed = (base_config.random_seed or 0) + (patient_offset * 10000)
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
        return_diagnostics=True,
        show_progress=False,
        show_summary=False,
    )
    elapsed = time.perf_counter() - t0

    diagnostics: dict[str, float | int] = {
        "sampled_patients": 0,
        "accepted_patients": 0,
        "rejected_patients": 0,
        "rejection_rate_percent": 0.0,
    }
    results: dict[int, dict[str, object]] = {}

    if result is None:
        pass
    elif isinstance(result, tuple):
        raw_results_any, diagnostics_any = result
        raw_results = cast(dict[int, object], raw_results_any)
        diagnostics = diagnostics_any
        results = {k: dict(cast(dict[str, object], v)) for k, v in raw_results.items()}
    else:
        raw_results = cast(dict[int, object], result)
        results = {k: dict(cast(dict[str, object], v)) for k, v in raw_results.items()}

    n_accepted = len(results)
    stats: dict[str, object] = {
        "worker_idx": worker_idx,
        "n_requested": n_patients_chunk,
        "n_accepted": n_accepted,
        "n_sampled": int(diagnostics.get("sampled_patients", n_accepted)),
        "n_rejected": int(diagnostics.get("rejected_patients", 0)),
        "rejection_rate_percent": float(diagnostics.get("rejection_rate_percent", 0.0)),
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

    # Cumulative patient offsets so _worker_run can compute a globally unique seed
    # for every local patient ID without overlap between workers.
    cumulative_offsets: list[int] = []
    running = 0
    for c in chunks:
        cumulative_offsets.append(running)
        running += c

    args: list[tuple[int, int, SimulationConfig, int]] = [
        (idx, n_chunk, config, cumulative_offsets[idx]) for idx, n_chunk in enumerate(chunks)
    ]

    t_start = time.perf_counter()
    all_stats: list[dict[str, object]] = []

    # Write each worker result to a temp parquet chunk immediately as it arrives,
    # then free the Python dict. This keeps peak RAM to one worker's data at a time
    # instead of holding all 20k patients simultaneously before the parquet write.
    tmp_dir = Path(tempfile.mkdtemp(prefix="libgen_chunks_"))
    # Maps worker_idx → (temp_parquet_path, n_accepted) for ordered merge later.
    chunk_info: dict[int, tuple[Path, int]] = {}

    try:
        if workers_eff == 1:
            result_block, stats = _worker_run(args[0])
            all_stats.append(stats)
            worker_idx = int(stats["worker_idx"])  # type: ignore[arg-type]
            chunk_path = tmp_dir / f"chunk_{worker_idx:04d}.parquet"
            n_rows = write_chunk_to_parquet(result_block, patient_id_offset=0, n_days=config.n_days, output_path=chunk_path)
            chunk_info[worker_idx] = (chunk_path, len(result_block))
            del result_block
            _print_worker_done(stats, workers_eff, t_start)
        else:
            with mp.Pool(processes=workers_eff) as pool:
                for result_block, stats in pool.imap_unordered(_worker_run, args):
                    all_stats.append(stats)
                    worker_idx = int(stats["worker_idx"])  # type: ignore[arg-type]
                    chunk_path = tmp_dir / f"chunk_{worker_idx:04d}.parquet"
                    # Cumulative patient ID offset for this worker so IDs are globally sequential.
                    pid_offset = cumulative_offsets[worker_idx]
                    write_chunk_to_parquet(result_block, patient_id_offset=pid_offset, n_days=config.n_days, output_path=chunk_path)
                    chunk_info[worker_idx] = (chunk_path, len(result_block))
                    del result_block   # free immediately — dict no longer needed
                    _print_worker_done(stats, workers_eff, t_start)

        accepted_total = sum(n for _, n in chunk_info.values())
        total_elapsed = time.perf_counter() - t_start

        # ── Final summary ──────────────────────────────────────────────
        accepted_per_worker = [int(s["n_accepted"]) for s in all_stats]  # type: ignore[arg-type]
        sampled_per_worker = [int(s["n_sampled"]) for s in all_stats]  # type: ignore[arg-type]
        rejected_per_worker = [int(s["n_rejected"]) for s in all_stats]  # type: ignore[arg-type]
        s_per_patient_vals = [float(s["s_per_patient"]) for s in all_stats]  # type: ignore[arg-type]
        avg_s_per_patient = sum(s_per_patient_vals) / len(s_per_patient_vals) if s_per_patient_vals else 0.0
        acceptance_rate = 100.0 * accepted_total / target_patients if target_patients else 0.0
        sampled_total = sum(sampled_per_worker)
        rejected_total = sum(rejected_per_worker)
        rejection_rate = (100.0 * rejected_total / sampled_total) if sampled_total else 0.0

        print(
            f"\n── Summary ───────────────────────────────────────────────────\n"
            f"  accepted / requested : {accepted_total} / {target_patients}  ({acceptance_rate:.1f}%)\n"
            f"  sampled / rejected   : {sampled_total} / {rejected_total}  (rejection {rejection_rate:.1f}%)\n"
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
            "sampled_patients": sampled_total,
            "accepted_patients": accepted_total,
            "rejected_patients": rejected_total,
            "rejection_rate_percent": round(rejection_rate, 3),
            "n_days": config.n_days,
            "random_seed": config.random_seed,
            "enable_plots": False,
            "total_elapsed_s": round(total_elapsed, 1),
        }

        if export_config.export_to_parquet:
            # Stream-merge the per-worker chunk parquets — no full DataFrame in RAM.
            base_file_name = f"results_{accepted_total}p_{config.n_days}d"
            parquet_path = output_folder / f"{base_file_name}.parquet"
            sorted_chunks = [chunk_info[idx][0] for idx in sorted(chunk_info)]
            expected_rows = accepted_total * config.n_days * 1440
            print(f"Merging {len(sorted_chunks)} chunk parquets → {parquet_path} …")
            merge_parquet_chunks(sorted_chunks, parquet_path, expected_rows)

        if export_config.export_to_csv:
            # CSV export requires loading all results — memory-intensive, avoid for large cohorts.
            # Reconstruct from chunk parquets via pandas rather than re-running the simulation.
            import pandas as pd
            sorted_chunks = [chunk_info[idx][0] for idx in sorted(chunk_info)]
            df_csv = pd.concat([pd.read_parquet(p) for p in sorted_chunks], ignore_index=True)
            base_file_name = f"results_{accepted_total}p_{config.n_days}d"
            csv_tmp = output_folder / f"{base_file_name}.csv.tmp"
            df_csv.to_csv(csv_tmp, index=False)
            csv_tmp.replace(output_folder / f"{base_file_name}.csv")
            print(f"Data successfully exported in csv format to {output_folder / f'{base_file_name}.csv'}")

        return output_folder

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
    sampled = int(stats["n_sampled"])       # type: ignore[arg-type]
    rejection_rate = float(stats["rejection_rate_percent"])  # type: ignore[arg-type]
    elapsed = float(stats["elapsed_s"])     # type: ignore[arg-type]
    s_pp = float(stats["s_per_patient"])    # type: ignore[arg-type]
    print(
        f"  worker {idx+1:>2}/{workers_eff}  done  "
        f"accepted {accepted:>5}/{requested:<5}  sampled {sampled:<5}  "
        f"rej {rejection_rate:4.1f}%  "
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
