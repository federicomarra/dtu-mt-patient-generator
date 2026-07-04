from __future__ import annotations

import math
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

from src.export import ExportConfig, write_chunk_to_parquet, validate_parquet_output
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
    name_suffix: str = "",
) -> Path | None:
    """Generate a large patient library in parallel and export merged results.

    name_suffix: optional tag appended to the parquet filename (e.g. "knobon") so
    cohorts that differ only in config (not patient/day count) get distinct names.
    """
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
        f"\n-- Parallel library generation ------------------------------\n"
        f"  target patients : {target_patients}  |  days/patient : {config.n_days}\n"
        f"  workers         : {workers_eff}  |  chunk sizes  : {chunks}\n"
        f"  random_seed     : {config.random_seed}\n"
        f"-------------------------------------------------------------"
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

    # Pre-create the output folder so we can open the parquet writer before workers start.
    output_folder = create_export_directory(base_folder=output_base_folder)
    if output_folder is None:
        return None

    base_file_name = f"results_{target_patients}p_{config.n_days}d" + (f"_{name_suffix}" if name_suffix else "")
    parquet_path   = output_folder / f"{base_file_name}.parquet"
    parquet_tmp    = output_folder / f"{base_file_name}.parquet.tmp"

    t_start = time.perf_counter()
    all_stats:  list[dict[str, object]] = []
    # Accumulate accepted counts per worker for the summary.
    accepted_per_worker_map: dict[int, int] = {}

    # Stream each worker result into the output parquet one chunk at a time.
    # A single ~400 MB temp file is written per worker, appended to the main
    # parquet writer, then deleted immediately - peak disk = output file + one chunk.
    import pyarrow.parquet as pq

    writer: pq.ParquetWriter | None = None
    chunk_tmp = output_folder / "_chunk.parquet.tmp"

    try:
        def _flush_block(result_block: dict, worker_idx: int) -> None:
            nonlocal writer
            pid_offset = cumulative_offsets[worker_idx]
            write_chunk_to_parquet(result_block, pid_offset, config.n_days, chunk_tmp)
            table = pq.read_table(chunk_tmp)
            if writer is None:
                writer = pq.ParquetWriter(parquet_tmp, table.schema)
            writer.write_table(table)
            chunk_tmp.unlink()   # free disk immediately

        if workers_eff == 1:
            result_block, stats = _worker_run(args[0])
            all_stats.append(stats)
            widx = int(stats["worker_idx"])  # type: ignore[arg-type]
            accepted_per_worker_map[widx] = len(result_block)
            _flush_block(result_block, widx)
            del result_block
            _print_worker_done(stats, workers_eff, t_start)
        else:
            with mp.Pool(processes=workers_eff) as pool:
                for result_block, stats in pool.imap_unordered(_worker_run, args):
                    all_stats.append(stats)
                    widx = int(stats["worker_idx"])  # type: ignore[arg-type]
                    accepted_per_worker_map[widx] = len(result_block)
                    _flush_block(result_block, widx)
                    del result_block
                    _print_worker_done(stats, workers_eff, t_start)

        if writer is not None:
            writer.close()
            writer = None

    except Exception:
        if writer is not None:
            writer.close()
        for p in [parquet_tmp, chunk_tmp]:
            if p.exists():
                p.unlink()
        raise

    accepted_total = sum(accepted_per_worker_map.values())
    total_elapsed  = time.perf_counter() - t_start

    # -- Final summary ----------------------------------------------
    accepted_per_worker = [int(s["n_accepted"]) for s in all_stats]  # type: ignore[arg-type]
    sampled_per_worker  = [int(s["n_sampled"])   for s in all_stats]  # type: ignore[arg-type]
    rejected_per_worker = [int(s["n_rejected"])  for s in all_stats]  # type: ignore[arg-type]
    s_per_patient_vals  = [float(s["s_per_patient"]) for s in all_stats]  # type: ignore[arg-type]
    avg_s_per_patient   = sum(s_per_patient_vals) / len(s_per_patient_vals) if s_per_patient_vals else 0.0
    acceptance_rate     = 100.0 * accepted_total / target_patients if target_patients else 0.0
    sampled_total       = sum(sampled_per_worker)
    rejected_total      = sum(rejected_per_worker)
    rejection_rate      = (100.0 * rejected_total / sampled_total) if sampled_total else 0.0

    print(
        f"\n-- Summary ---------------------------------------------------\n"
        f"  accepted / requested : {accepted_total} / {target_patients}  ({acceptance_rate:.1f}%)\n"
        f"  sampled / rejected   : {sampled_total} / {rejected_total}  (rejection {rejection_rate:.1f}%)\n"
        f"  per-worker accepted  : {accepted_per_worker}\n"
        f"  total elapsed        : {_fmt_elapsed(total_elapsed)}\n"
        f"  avg time / patient   : {avg_s_per_patient:.1f} s  (wall-clock per requested slot)\n"
        f"-------------------------------------------------------------"
    )

    if accepted_total < target_patients * 0.8:
        print(
            f"Warning: acceptance rate {acceptance_rate:.1f}% is below 80%. "
            "Consider relaxing quality thresholds or increasing n_patients."
        )

    if export_config.export_to_parquet and parquet_tmp.exists():
        expected_rows = accepted_total * config.n_days * 1440
        validate_parquet_output(parquet_tmp, expected_rows)
        parquet_tmp.replace(parquet_path)
        print(f"Data successfully exported in parquet format to {parquet_path}")

    if export_config.export_to_csv:
        import pandas as pd
        df_csv = pd.read_parquet(parquet_path)
        csv_tmp = output_folder / f"{base_file_name}.csv.tmp"
        df_csv.to_csv(csv_tmp, index=False)
        csv_tmp.replace(output_folder / f"{base_file_name}.csv")
        print(f"Data successfully exported in csv format to {output_folder / f'{base_file_name}.csv'}")

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
