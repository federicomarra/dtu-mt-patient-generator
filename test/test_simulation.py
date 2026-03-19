"""Run a simulation smoke test with cohort-level analysis metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import numpy as np  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import ExportConfig
from src.simulation import PatientResult, run_simulation
from src.simulation_config import SimulationConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation smoke test with analysis summary")
    parser.add_argument("--patients", type=int, default=10, help="Number of accepted patients to simulate")
    parser.add_argument("--days", type=int, default=3, help="Number of days per patient")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for cohort generation")
    parser.add_argument("--noise-std", type=float, default=0.10, help="CGM noise std in mmol/L")
    parser.add_argument("--noise-autocorr", type=float, default=0.7, help="CGM noise autocorrelation")
    parser.add_argument(
        "--scenario",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5, 6],
        help="Fixed scenario to run when --random-scenarios is disabled",
    )
    parser.add_argument(
        "--random-scenarios",
        action="store_true",
        help="Sample a scenario each day instead of using --scenario",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable CSV and Parquet exports (enabled by default)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plotting",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress simulation progress output from the core runner",
    )
    return parser.parse_args()


def _collect_bg_and_age(results: Mapping[int, PatientResult]) -> tuple[np.ndarray, np.ndarray]:
    bg_points: list[np.ndarray] = []
    ages: list[float] = []

    for patient_data in results.values():
        params = patient_data.get("params")
        age_raw = params.get("age_years")
        if age_raw is not None:
            ages.append(float(age_raw))

        days = patient_data.get("days")
        for day_data in days.values():
            glucose = day_data["blood_glucose"]
            bg_points.append(glucose.astype(np.float64, copy=False))

    bg_all = np.concatenate(bg_points) if bg_points else np.array([], dtype=np.float64)
    ages_all = np.array(ages, dtype=np.float64) if ages else np.array([], dtype=np.float64)
    return bg_all, ages_all


def _print_analysis_summary(bg_all: np.ndarray, ages_all: np.ndarray) -> None:
    if bg_all.size == 0:
        print("No glucose samples found in results.")
        return

    tir_mask = (bg_all >= 3.9) & (bg_all <= 10.0)
    hypo_mask = bg_all < 3.9
    hyper_mask = bg_all > 10.0

    tir_pct = 100.0 * float(np.mean(tir_mask))
    hypo_pct = 100.0 * float(np.mean(hypo_mask))
    hyper_pct = 100.0 * float(np.mean(hyper_mask))

    print("\n=== Cohort Analysis (mmol/L) ===")
    print(f"Samples: {bg_all.size}")
    print(f"Mean glucose: {float(np.mean(bg_all)):.3f}")
    print(f"Std glucose: {float(np.std(bg_all)):.3f}")
    print(f"Min glucose: {float(np.min(bg_all)):.3f}")
    print(f"Max glucose: {float(np.max(bg_all)):.3f}")
    print(f"TIR 3.9-10.0 mmol/L: {tir_pct:.2f}%")
    print(f"Hypoglycemia <3.9 mmol/L: {hypo_pct:.2f}%")
    print(f"Hyperglycemia >10.0 mmol/L: {hyper_pct:.2f}%")

    if ages_all.size > 0:
        print("\n=== Accepted Age Distribution (years) ===")
        print(f"Count: {ages_all.size}")
        print(f"Mean: {float(np.mean(ages_all)):.2f}")
        print(f"Std: {float(np.std(ages_all)):.2f}")
        print(f"Min: {float(np.min(ages_all)):.2f}")
        print(f"Max: {float(np.max(ages_all)):.2f}")


if __name__ == "__main__":
    args = _parse_args()

    test_sim_config = SimulationConfig(
        n_patients=args.patients,
        n_days=args.days,
        international_unit=True,
        noise_std=args.noise_std,
        noise_autocorr=args.noise_autocorr,
        random_scenarios=args.random_scenarios,
        fixed_scenario=args.scenario,
        clip_states=True,
        std_patient=False,
        random_seed=args.seed,
        enable_plots=not args.no_plots,
    )

    export_enabled = not args.no_export
    export_config = ExportConfig(
        export_to_parquet=export_enabled,
        export_to_csv=export_enabled,
    )

    results = run_simulation(
        test_sim_config,
        export_config,
        return_results=True,
        show_progress=not args.summary_only,
        show_summary=not args.summary_only,
    )

    if results is None:
        raise RuntimeError("Simulation did not return results. Expected return_results=True.")

    bg_all, ages_all = _collect_bg_and_age(results)
    _print_analysis_summary(bg_all, ages_all)