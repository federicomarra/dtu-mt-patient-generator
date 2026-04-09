#!/usr/bin/env python3
"""
Analyze Monte Carlo simulation results.

Usage:
    python analyze_simulation.py [path/to/results.csv]
    
If no path provided, finds the most recent results automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd  # type: ignore[import-untyped]


def pick_input_column(df: pd.DataFrame, primary: str, legacy: str) -> str | None:
    """Pick preferred input column name with legacy fallback."""
    if primary in df.columns:
        return primary
    if legacy in df.columns:
        return legacy
    return None


def find_latest_results() -> Path:
    """Find the most recent PARQUET or CSV results file."""
    candidates: list[Path] = []
    for results_dir in (Path("monte_carlo_results"), Path("monte_carlo_results_parallel")):
        if results_dir.exists():
            candidates.extend(results_dir.glob("**/results_*.parquet"))
            candidates.extend(results_dir.glob("**/results_*.csv"))

    if not candidates:
        raise FileNotFoundError("No results files found in monte_carlo_results/ or monte_carlo_results_parallel/")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Latest file: {latest}")
    return latest

def analyze_results(file_path: Path) -> None:
    """Analyze simulation results and print statistics."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*70}\n")
    
    df: pd.DataFrame
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)  # type: ignore[assignment]
    else:
        df = pd.read_csv(file_path)  # type: ignore[assignment]
    
    # Auto-detect unit (mg/dL vs mmol/L).
    # mmol/L ceiling is ~33 mmol/L (instability threshold 33.3 + CGM noise).
    # mg/dL values are always ≥18× higher; min physiological mg/dL reading >50.
    # Threshold at 50 safely separates the two unit spaces (mirrors analyze_large.py).
    glucose_max = float(df['blood_glucose'].max())  # type: ignore[arg-type]
    is_mgdl = glucose_max > 50
    unit_str = "mg/dL" if is_mgdl else "mmol/L"
    
    # Set thresholds based on unit
    if is_mgdl:
        hypo_threshold = 70.0
        hyper_threshold = 180.0
        guard_threshold = 72.0  # 4.0 mmol/L
    else:
        hypo_threshold = 3.9  # ~70 mg/dL
        hyper_threshold = 10.0  # ~180 mg/dL
        guard_threshold = 4.0  # mmol/L
    
    # Basic statistics
    print(f'=== Glucose Statistics ({unit_str}) ===')
    print(f'Min:    {df["blood_glucose"].min():.1f}')
    print(f'Max:    {df["blood_glucose"].max():.1f}')
    print(f'Mean:   {df["blood_glucose"].mean():.1f}')
    print(f'Median: {df["blood_glucose"].median():.1f}')
    print(f'Std:    {df["blood_glucose"].std():.1f}')
    mean_glucose = float(df["blood_glucose"].mean())
    std_glucose = float(df["blood_glucose"].std())
    cv_pct = (100.0 * std_glucose / mean_glucose) if mean_glucose > 0.0 else 0.0
    p5 = float(df["blood_glucose"].quantile(0.05))
    p95 = float(df["blood_glucose"].quantile(0.95))
    print(f'CV%:    {cv_pct:.1f}')
    print(f'P5/P95: {p5:.2f} / {p95:.2f}')

    insulin_col = pick_input_column(df, "insulin_mU_min", "insulin")
    cho_col = pick_input_column(df, "cho_mg_min", "cho")
    
    # Range analysis
    hypo_count: int = int((df['blood_glucose'] < hypo_threshold).sum())  # type: ignore[arg-type]
    hyper_count: int = int((df['blood_glucose'] > hyper_threshold).sum())  # type: ignore[arg-type]
    in_range: int = int(((df['blood_glucose'] >= hypo_threshold) & (df['blood_glucose'] <= hyper_threshold)).sum())  # type: ignore[arg-type]
    total: int = len(df)
    
    print(f'\n=== Glycemic Control ===')
    print(f'Hypoglycemia (<{hypo_threshold:.1f}):   {hypo_count:5d}/{total} ({100*hypo_count/total:5.1f}%)')
    print(f'In Range ({hypo_threshold:.1f}-{hyper_threshold:.1f}):    {in_range:5d}/{total} ({100*in_range/total:5.1f}%)')
    print(f'Hyperglycemia (>{hyper_threshold:.1f}): {hyper_count:5d}/{total} ({100*hyper_count/total:5.1f}%)')

    if "patient_age_years" in df.columns:
        age_series = pd.to_numeric(df["patient_age_years"], errors="coerce").dropna()
        if not age_series.empty:
            print("\n=== Age Distribution (accepted cohort) ===")
            print(f"Count:  {int(age_series.shape[0])}")
            print(f"Mean:   {float(age_series.mean()):.2f} years")
            print(f"Std:    {float(age_series.std()):.2f} years")
            print(f"Min:    {float(age_series.min()):.2f} years")
            print(f"Max:    {float(age_series.max()):.2f} years")
    
    # Per-patient breakdown (capped at 20 patients; use analyze_large.py for full cohorts)
    all_pids = sorted(df['patient_id'].unique())  # type: ignore[arg-type]
    show_pids = all_pids[:20]
    print('\n=== Per-Patient Analysis ===')
    if len(all_pids) > 20:
        print(f"(showing first 20 of {len(all_pids)} patients — use analyze_large.py for full cohort)")
    print(f"{'Patient':<8} {'Min':>7} {'Max':>7} {'Mean':>7} {'Hypo%':>7} {'Hyper%':>7}")
    print('-' * 50)
    for patient_id in show_pids:
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

    control_cols = [
        "guard_active",
        "rescue_active",
        "iob_guard_active",
        "correction_isf_active",
    ]
    available_control_cols = [col for col in control_cols if col in df.columns]
    if available_control_cols:
        print("\n=== Control Activity Flags ===")
        for col in available_control_cols:
            series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            active_pct = 100.0 * float((series > 0.5).mean())
            print(f"{col}: {active_pct:.2f}% active")

    # Input channel analysis
    print('\n=== Input Signals ===')
    if insulin_col is None:
        print('Insulin column not found (expected insulin_mU_min or insulin).')
    else:
        insulin_mU = pd.to_numeric(df[insulin_col], errors='coerce').fillna(0.0)
        insulin_u = insulin_mU / 1000.0
        total_insulin_u = float(insulin_u.sum())
        print(f'Total insulin delivered: {total_insulin_u:.2f} U  (stored as mU/min)')

        insulin_by_patient = insulin_u.groupby(df['patient_id']).sum()
        if 'day' in df.columns:
            ins_pt_day = insulin_u.groupby([df['patient_id'], df['day']]).sum()  # type: ignore[call-overload]
            ins_cohort_per_day = ins_pt_day.groupby(level=1).mean()
            day_strs_ins: list[str] = [
                f"Day {int(d)}: {float(v):.2f}" for d, v in ins_cohort_per_day.items()  # type: ignore[arg-type]
            ]
            print(f"Cohort avg insulin per day [U]:  {',  '.join(day_strs_ins)}")

        if 'day' in df.columns:
            all_pt_ids = sorted(df['patient_id'].unique())
            show_pt_ids = all_pt_ids[:20]
            truncated = len(all_pt_ids) > 20
            print(f"  Per-patient, per-day insulin [U]{' (first 20 patients)' if truncated else ''}:")
            pt_day = insulin_u.groupby([df['patient_id'], df['day']]).sum()  # type: ignore[call-overload]
            days_present = sorted(df['day'].unique())
            header = f"  {'Patient':<10}" + "".join(f" {'Day '+str(d):>9}" for d in days_present) + f" {'Total':>9}"
            print(header)
            for pid in show_pt_ids:
                row_total = float(insulin_by_patient.get(pid, 0.0))
                parts: list[str] = []
                for d in days_present:
                    val = float(pt_day.get((pid, d), 0.0))
                    parts.append(f'{val:9.2f}')
                print(f"  {str(pid):<10}" + " ".join(parts) + f" {row_total:9.2f}")

    if cho_col is None:
        print('CHO column not found (expected cho_mg_min or cho).')
    else:
        cho_mg = pd.to_numeric(df[cho_col], errors='coerce').fillna(0.0)
        cho_g = cho_mg / 1000.0
        total_cho_g = float(cho_g.sum())
        print(f'Total CHO delivered:     {total_cho_g:.2f} g  (stored as mg/min)')

        cho_by_patient = cho_g.groupby(df['patient_id']).sum()
        if 'day' in df.columns:
            cho_pt_day = cho_g.groupby([df['patient_id'], df['day']]).sum()  # type: ignore[call-overload]
            cho_cohort_per_day = cho_pt_day.groupby(level=1).mean()
            day_strs_cho: list[str] = [
                f"Day {int(d)}: {float(v):.2f}" for d, v in cho_cohort_per_day.items()  # type: ignore[arg-type]
            ]
            print(f"Cohort avg CHO per day [g]:      {',  '.join(day_strs_cho)}")

        if 'day' in df.columns:
            all_pt_ids_cho = sorted(df['patient_id'].unique())
            show_pt_ids_cho = all_pt_ids_cho[:20]
            truncated_cho = len(all_pt_ids_cho) > 20
            print(f"  Per-patient, per-day CHO [g]{' (first 20 patients)' if truncated_cho else ''}:")
            pt_day_cho = cho_g.groupby([df['patient_id'], df['day']]).sum()  # type: ignore[call-overload]
            days_present = sorted(df['day'].unique())
            header = f"  {'Patient':<10}" + "".join(f" {'Day '+str(d):>9}" for d in days_present) + f" {'Total':>9}"
            print(header)
            for pid in show_pt_ids_cho:
                row_total = float(cho_by_patient.get(pid, 0.0))
                parts: list[str] = []
                for d in days_present:
                    val = float(pt_day_cho.get((pid, d), 0.0))
                    parts.append(f'{val:9.2f}')
                print(f"  {str(pid):<10}" + " ".join(parts) + f" {row_total:9.2f}")
    
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
    
    # Per-base-scenario breakdown
    base_scen_col = "base_scenario" if "base_scenario" in df.columns else "scenario_id"
    base_scen_names = {1: "normal", 2: "active aerobic", 3: "sedentary"}
    if base_scen_col in df.columns and df[base_scen_col].notna().any():
        print(f'\n=== Per-Base-Scenario Glycemic Profile (column: {base_scen_col}) ===')
        print(f"{'Scen':<5} {'Name':<18} {'Days':>6} {'TIR%':>7} {'Hypo%':>7} {'Hyper%':>7} {'Mean':>7}")
        print('-' * 58)
        for sid in sorted(df[base_scen_col].dropna().unique()):
            s_df = df[df[base_scen_col] == sid]  # type: ignore[index]
            n_days = int(s_df.groupby(['patient_id', 'day']).ngroups) if 'day' in df.columns else '-'  # type: ignore[call-overload]
            tir = 100 * float(((s_df['blood_glucose'] >= hypo_threshold) & (s_df['blood_glucose'] <= hyper_threshold)).sum()) / len(s_df)  # type: ignore[arg-type]
            hypo = 100 * float((s_df['blood_glucose'] < hypo_threshold).sum()) / len(s_df)  # type: ignore[arg-type]
            hyper = 100 * float((s_df['blood_glucose'] > hyper_threshold).sum()) / len(s_df)  # type: ignore[arg-type]
            mean_g = float(s_df['blood_glucose'].mean())  # type: ignore[arg-type]
            name = base_scen_names.get(int(sid), f"sc{int(sid)}")
            print(f"{int(sid):<5} {name:<18} {n_days:>6} {tir:>6.1f}% {hypo:>6.1f}% {hyper:>6.1f}% {mean_g:>7.2f}")

    # Day-level anomaly overlay rates (one row per patient-day)
    if 'day' in df.columns:
        day_df = df.drop_duplicates(subset=['patient_id', 'day'])
        total_days = len(day_df)
        print(f'\n=== Day-Level Anomaly Overlay Rates (n={total_days} patient-days) ===')
        for col, label in [
            ('had_large_meal',   'Large meal'),
            ('had_missed_bolus', 'Missed bolus'),
            ('n_late_boluses',   'Late bolus (avg count)'),
            ('exercise_overlay', 'Exercise overlay'),
        ]:
            if col not in df.columns:
                continue
            series = day_df[col].dropna()
            if col == 'n_late_boluses':
                print(f"  {label:<28} {float(series.mean()):.3f} avg/day  (total {int(series.sum())})")
            elif col == 'exercise_overlay':
                n_overlay = int((series.notna() & (series != 0)).sum())
                print(f"  {label:<28} {100*n_overlay/total_days:.1f}%  ({n_overlay}/{total_days} days)")
            else:
                n_true = int(series.astype(bool).sum())
                print(f"  {label:<28} {100*n_true/total_days:.1f}%  ({n_true}/{total_days} days)")

    # Per-minute ML label distributions
    for col, label in [
        ('bolus_status', 'Bolus Status'),
        ('meal_size',    'Meal Size'),
        ('exercise_type','Exercise Type'),
    ]:
        if col not in df.columns:
            continue
        vc = df[col].fillna('none').value_counts(dropna=False)
        total_col = int(vc.sum())
        print(f'\n=== Per-Minute ML Labels: {label} ===')
        for val, cnt in vc.items():
            print(f"  {str(val):<20} {int(cnt):>10,}  ({100*int(cnt)/total_col:5.1f}%)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Analyze simulation export file")
        parser.add_argument("file", nargs="?", default=None, help="Path to results CSV/Parquet; latest file if omitted")
        args = parser.parse_args()

        if args.file is not None:
            file_path = Path(args.file)
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
