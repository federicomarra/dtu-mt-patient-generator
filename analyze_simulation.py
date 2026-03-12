#!/usr/bin/env python3
"""
Analyze Monte Carlo simulation results.

Usage:
    python analyze_simulation.py [path/to/results.csv]
    
If no path provided, finds the most recent results automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]


def find_latest_results() -> Path:
    """Find the most recent PARQUET orCSV results file."""
    results_dir = Path("monte_carlo_results")
    if not results_dir.exists():
        raise FileNotFoundError("No monte_carlo_results directory found")

    # Find all PARQUET files
    parquet_files = list(results_dir.glob("**/results_*.parquet"))
    # Find all CSV files
    csv_files = list(results_dir.glob("**/results_*.csv"))

    if not csv_files and not parquet_files:
        raise FileNotFoundError("No results PARQUET or CSV files found")
    
    # Sort by modification time, get most recent
    latest_csv: Path | None = max(csv_files, key=lambda p: p.stat().st_mtime) if csv_files else None
    latest_parquet: Path | None = max(parquet_files, key=lambda p: p.stat().st_mtime) if parquet_files else None

    print(f"Latest CSV: {latest_csv}")
    print(f"Latest Parquet: {latest_parquet}")

    if latest_csv is not None and latest_parquet is not None:
        return latest_csv if latest_csv.stat().st_mtime >= latest_parquet.stat().st_mtime else latest_parquet
    if latest_parquet is not None:
        return latest_parquet
    if latest_csv is not None:
        return latest_csv
    raise FileNotFoundError("No results PARQUET or CSV files found")

def analyze_results(file_path: Path) -> None:
    """Analyze simulation results and print statistics."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*70}\n")
    
    if file_path.suffix == '.parquet':
        df: pd.DataFrame = pd.read_parquet(file_path)  # type: ignore[assignment]
    else:
        df: pd.DataFrame = pd.read_csv(file_path)  # type: ignore[assignment]
    
    # Auto-detect unit (mg/dL vs mmol/L)
    glucose_max = float(df['blood_glucose'].max())  # type: ignore[arg-type]
    is_mgdl = glucose_max > 30  # mg/dL values typically > 30, mmol/L typically < 25
    unit_str = "mg/dL" if is_mgdl else "mmol/L"
    
    # Set thresholds based on unit
    if is_mgdl:
        hypo_threshold = 70.0
        hyper_threshold = 180.0
        guard_threshold = 75.6  # 4.2 mmol/L
    else:
        hypo_threshold = 3.9  # ~70 mg/dL
        hyper_threshold = 10.0  # ~180 mg/dL
        guard_threshold = 4.2  # mmol/L
    
    # Basic statistics
    print(f'=== Glucose Statistics ({unit_str}) ===')
    print(f'Min:    {df["blood_glucose"].min():.1f}')
    print(f'Max:    {df["blood_glucose"].max():.1f}')
    print(f'Mean:   {df["blood_glucose"].mean():.1f}')
    print(f'Median: {df["blood_glucose"].median():.1f}')
    print(f'Std:    {df["blood_glucose"].std():.1f}')
    
    # Range analysis
    hypo_count: int = int((df['blood_glucose'] < hypo_threshold).sum())  # type: ignore[arg-type]
    hyper_count: int = int((df['blood_glucose'] > hyper_threshold).sum())  # type: ignore[arg-type]
    in_range: int = int(((df['blood_glucose'] >= hypo_threshold) & (df['blood_glucose'] <= hyper_threshold)).sum())  # type: ignore[arg-type]
    total: int = len(df)
    
    print(f'\n=== Glycemic Control ===')
    print(f'Hypoglycemia (<{hypo_threshold:.1f}):   {hypo_count:5d}/{total} ({100*hypo_count/total:5.1f}%)')
    print(f'In Range ({hypo_threshold:.1f}-{hyper_threshold:.1f}):    {in_range:5d}/{total} ({100*in_range/total:5.1f}%)')
    print(f'Hyperglycemia (>{hyper_threshold:.1f}): {hyper_count:5d}/{total} ({100*hyper_count/total:5.1f}%)')
    
    # Per-patient breakdown
    print('\n=== Per-Patient Analysis ===')
    print(f"{'Patient':<8} {'Min':>7} {'Max':>7} {'Mean':>7} {'Hypo%':>7} {'Hyper%':>7}")
    print('-' * 50)
    for patient_id in sorted(df['patient_id'].unique()):  # type: ignore[arg-type]
        patient_df = df[df['patient_id'] == patient_id]  # type: ignore[index]
        min_g = float(patient_df['blood_glucose'].min())  # type: ignore[arg-type]
        max_g = float(patient_df['blood_glucose'].max())  # type: ignore[arg-type]
        mean_g = float(patient_df['blood_glucose'].mean())  # type: ignore[arg-type]
        hypo_pct = 100 * float((patient_df['blood_glucose'] < hypo_threshold).sum()) / len(patient_df)  # type: ignore[arg-type]
        hyper_pct = 100 * float((patient_df['blood_glucose'] > hyper_threshold).sum()) / len(patient_df)  # type: ignore[arg-type]
        print(f'{patient_id:<8} {min_g:7.1f} {max_g:7.1f} {mean_g:7.1f} {hypo_pct:6.1f}% {hyper_pct:6.1f}%')
    
    # Hypo-guard analysis
    below_guard: int = int((df['blood_glucose'] <= guard_threshold).sum())  # type: ignore[arg-type]
    print(f'\n=== Hypo-Guard Analysis ===')
    print(f'Guard threshold: {guard_threshold:.1f} {unit_str}')
    print(f'Time below guard: {below_guard}/{total} ({100*below_guard/total:.1f}%)')
    print('(When guard is active, basal insulin is suspended)')
    
    # Extreme values
    print('\n=== Extreme Values ===')
    print('Lowest 5 readings:')
    print(df.nsmallest(5, 'blood_glucose')[['patient_id', 'day', 'minute', 'time', 'blood_glucose']].to_string(index=False))
    print('\nHighest 5 readings:')
    print(df.nlargest(5, 'blood_glucose')[['patient_id', 'day', 'minute', 'time', 'blood_glucose']].to_string(index=False))
    
    # Time in range by day
    if 'day' in df.columns and df['day'].nunique() > 1:  # type: ignore[operator]
        print('\n=== Time in Range by Day ===')
        print(f"{'Day':<5} {'In Range %':>12} {'Hypo %':>10} {'Hyper %':>10}")
        print('-' * 40)
        for day in sorted(df['day'].unique()):  # type: ignore[arg-type]
            day_df = df[df['day'] == day]  # type: ignore[index]
            tir = 100 * float(((day_df['blood_glucose'] >= hypo_threshold) & (day_df['blood_glucose'] <= hyper_threshold)).sum()) / len(day_df)  # type: ignore[arg-type]
            hypo = 100 * float((day_df['blood_glucose'] < hypo_threshold).sum()) / len(day_df)  # type: ignore[arg-type]
            hyper = 100 * float((day_df['blood_glucose'] > hyper_threshold).sum()) / len(day_df)  # type: ignore[arg-type]
            print(f'{day:<5} {tir:11.1f}% {hypo:9.1f}% {hyper:9.1f}%')
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            file_path = Path(sys.argv[1])
        else:
            file_path = find_latest_results()
        
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        analyze_results(file_path)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUsage: python analyze_simulation.py [path/to/results.csv]")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
