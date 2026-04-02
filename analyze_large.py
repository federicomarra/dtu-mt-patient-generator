#!/usr/bin/env python3
"""
Memory-safe analysis for large simulation datasets (100M+ rows).

Streams through the Parquet file one row-group at a time (~1M rows each),
accumulating statistics without ever loading the full dataset into RAM.

Usage:
    python analyze_large.py [path/to/results.parquet]
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from collections import defaultdict
from typing import cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import numpy as np


# ── Thresholds (mg/dL; auto-detected below) ──────────────────────────────────
HYPO_MGDL  = 70.0
HYPER_MGDL = 180.0
HYPO_MMOL  = 3.9
HYPER_MMOL = 10.0
GUARD_MGDL = 72.0   # 4.0 mmol/L
GUARD_MMOL = 4.0


def find_latest() -> Path:
    d = Path("monte_carlo_results_parallel")
    files = list(d.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError("No parquet files found in monte_carlo_results_parallel/")
    return max(files, key=lambda p: p.stat().st_mtime)


def _read_seed_from_sidecar(file_path: Path) -> str | None:
    """Read random seed from nearby config_*.txt sidecar if present."""
    sidecars = sorted(file_path.parent.glob("config_*p_*d.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for cfg in sidecars:
        try:
            with cfg.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("random_seed:"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            continue
    return None


class OnlineStats:
    """Welford online mean/variance + running min/max."""
    __slots__ = ("n", "mean", "_m2", "mn", "mx")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.mn = math.inf
        self.mx = -math.inf

    def update_array(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        # Welford batch update via numpy
        chunk_n = arr.size
        chunk_mean = float(arr.mean())
        chunk_m2 = float(arr.var()) * chunk_n
        combined_n = self.n + chunk_n
        delta = chunk_mean - self.mean
        self.mean += delta * chunk_n / combined_n
        self._m2 += chunk_m2 + delta * delta * self.n * chunk_n / combined_n
        self.n = combined_n
        mn = float(arr.min())
        mx = float(arr.max())
        if mn < self.mn:
            self.mn = mn
        if mx > self.mx:
            self.mx = mx

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / self.n) if self.n > 1 else 0.0

    @property
    def cv_pct(self) -> float:
        return 100.0 * self.std / self.mean if self.mean > 0 else 0.0


def analyze(file_path: Path) -> None:
    pf = pq.ParquetFile(file_path)
    # cast() pins the type definitively; pyarrow has no stubs so Pylance
    # would otherwise propagate Unknown through all downstream expressions.
    n_groups    = cast(int, pf.metadata.num_row_groups)  # type: ignore[union-attr]
    total_rows  = cast(int, pf.metadata.num_rows)         # type: ignore[union-attr]
    schema_names = cast(list[str], pf.schema_arrow.names) # type: ignore[union-attr]

    print(f"\n{'='*70}")
    print(f"File:       {file_path}")
    print(f"Total rows: {total_rows:,}   Row groups: {n_groups}")
    seed_value = _read_seed_from_sidecar(file_path)
    if seed_value is not None:
        print(f"Seed:       {seed_value} (from config sidecar)")
    else:
        print("Seed:       n/a (not found in nearby config sidecar)")
    print(f"{'='*70}\n")

    # ── Detect unit from first row group ─────────────────────────────────────
    first_batch = cast(pd.DataFrame, pf.read_row_group(0, columns=["blood_glucose"]).to_pandas())  # type: ignore[union-attr]
    is_mgdl = float(first_batch["blood_glucose"].max()) > 30
    unit = "mg/dL" if is_mgdl else "mmol/L"
    hypo_thr  = HYPO_MGDL  if is_mgdl else HYPO_MMOL
    hyper_thr = HYPER_MGDL if is_mgdl else HYPER_MMOL
    guard_thr = GUARD_MGDL if is_mgdl else GUARD_MMOL
    print(f"Detected unit: {unit}  |  hypo <{hypo_thr}  hyper >{hyper_thr}\n")

    # ── Accumulators ─────────────────────────────────────────────────────────
    glc_stats = OnlineStats()
    glc_sample: list[np.ndarray] = []   # reservoir for percentile estimate
    SAMPLE_TARGET = 500_000             # ~4 MB of floats

    hypo_n = in_range_n = hyper_n = guard_n = 0

    # per-day: {day -> [hypo, in_range, hyper, total]}
    day_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    # per-scenario: {sid -> [hypo, in_range, hyper, total, sum_glc]}
    scen_counts: dict[int, list[float]] = defaultdict(lambda: [0, 0, 0, 0, 0.0])

    patient_ids: set[str] = set()
    ages: list[float] = []

    scenario_names = {
        1: "normal",
        2: "active (aerobic)",
        3: "sedentary",
        4: "restaurant meal",
        5: "missed bolus",
        6: "late bolus",
        7: "prolonged aerobic",
        8: "anaerobic",
        9: "exercise+missed bolus",
    }

    print("Streaming row groups", end="", flush=True)
    sample_stride = max(1, total_rows // SAMPLE_TARGET)

    for rg_idx in range(n_groups):
        cols = ["blood_glucose", "patient_id"]
        if "day" in schema_names:
            cols.append("day")
        if "scenario_id" in schema_names:
            cols.append("scenario_id")
        if "patient_age_years" in schema_names:
            cols.append("patient_age_years")

        chunk = cast(pd.DataFrame, pf.read_row_group(rg_idx, columns=cols).to_pandas())  # type: ignore[union-attr]

        g: np.ndarray = chunk["blood_glucose"].to_numpy(dtype=np.float64)
        glc_stats.update_array(g)

        # reservoir sample for percentiles
        sampled: np.ndarray = g[::sample_stride]
        if len(glc_sample) < 20:   # keep list short, concat at end
            glc_sample.append(sampled)

        h = int((g < hypo_thr).sum())
        hi = int((g > hyper_thr).sum())
        ir = len(g) - h - hi
        hypo_n     += h
        in_range_n += ir
        hyper_n    += hi
        guard_n    += int((g <= guard_thr).sum())

        # per-day
        if "day" in chunk.columns:
            for day_val, grp in chunk.groupby("day", sort=False):
                g2: np.ndarray = grp["blood_glucose"].to_numpy(dtype=np.float64)
                dc = day_counts[cast(int, day_val)]
                dc[0] += int((g2 < hypo_thr).sum())
                dc[1] += int(((g2 >= hypo_thr) & (g2 <= hyper_thr)).sum())
                dc[2] += int((g2 > hyper_thr).sum())
                dc[3] += len(g2)

        # per-scenario
        if "scenario_id" in chunk.columns:
            for sid_val, grp in chunk.groupby("scenario_id", sort=False):
                g3: np.ndarray = grp["blood_glucose"].to_numpy(dtype=np.float64)
                sc = scen_counts[cast(int, sid_val)]
                sc[0] += int((g3 < hypo_thr).sum())
                sc[1] += int(((g3 >= hypo_thr) & (g3 <= hyper_thr)).sum())
                sc[2] += int((g3 > hyper_thr).sum())
                sc[3] += len(g3)
                sc[4] += float(g3.sum())

        # patient ids
        if "patient_id" in chunk.columns:
            patient_ids.update(chunk["patient_id"].unique().tolist())

        # ages — sample a few per row group to keep memory trivial
        if "patient_age_years" in chunk.columns and rg_idx % 10 == 0:
            ages.extend(chunk["patient_age_years"].dropna().iloc[::1000].tolist())

        if (rg_idx + 1) % 20 == 0 or rg_idx == n_groups - 1:
            print(f" {rg_idx+1}/{n_groups}", end="", flush=True)

    print("\n")

    total = glc_stats.n

    # ── Percentile estimate from reservoir sample ─────────────────────────────
    sample_arr = np.concatenate(glc_sample) if glc_sample else np.array([])
    p5  = float(np.percentile(sample_arr, 5))  if sample_arr.size else float("nan")
    p95 = float(np.percentile(sample_arr, 95)) if sample_arr.size else float("nan")

    # ── Print results ─────────────────────────────────────────────────────────
    # Min/max are exact (every value compared); mean/std are exact via Welford.
    # Min can appear below physiological hypo thresholds because blood_glucose
    # is the CGM sensor reading: AR(1) noise + 5-min delay means the signal
    # can transiently undershoot the true interstitial glucose during a rapid
    # fall, producing values lower than the underlying ODE state.
    # CV% = (std / mean) × 100: measures relative glucose variability.
    # In CGM literature CV% > 36% indicates high variability; the cohort
    # target is 20–40% to reflect realistic T1D glycemic fluctuation.
    print(f"=== Glucose Statistics ({unit}) ===")
    print(f"  Patients:  {len(patient_ids):,}")
    print(f"  Min:       {glc_stats.mn:.1f}  (CGM signal — noise+delay can undershoot true glucose)")
    print(f"  Max:       {glc_stats.mx:.1f}")
    print(f"  Mean:      {glc_stats.mean:.1f}")
    print(f"  Std:       {glc_stats.std:.1f}")
    print(f"  CV%:       {glc_stats.cv_pct:.1f}  (std/mean×100; >36% = high variability)")
    print(f"  P5/P95*:   {p5:.2f} / {p95:.2f}  (*sampled estimate)")

    if ages:
        age_arr = np.array(ages)
        print(f"\n=== Age Distribution ===")
        print(f"  Mean: {age_arr.mean():.1f} yr   Std: {age_arr.std():.1f} yr   "
              f"Min: {age_arr.min():.0f}   Max: {age_arr.max():.0f}")

    print(f"\n=== Glycemic Control (total {total:,} min-points) ===")
    print(f"  Hypoglycemia  (<{hypo_thr:.1f}):  {hypo_n:>12,}  ({100*hypo_n/total:5.1f}%)")
    print(f"  In Range ({hypo_thr:.1f}–{hyper_thr:.1f}):  {in_range_n:>12,}  ({100*in_range_n/total:5.1f}%)")
    print(f"  Hyperglycemia (>{hyper_thr:.1f}): {hyper_n:>12,}  ({100*hyper_n/total:5.1f}%)")
    print(f"\n  Guard threshold ({guard_thr:.1f} {unit}):    "
          f"{guard_n:>12,}  ({100*guard_n/total:5.1f}%)")

    if day_counts:
        print(f"\n=== Time in Range by Day ===")
        print(f"  {'Day':<5} {'TIR%':>8} {'Hypo%':>8} {'Hyper%':>8} {'Points':>12}")
        print("  " + "-" * 44)
        for day in sorted(day_counts):
            h, ir, hi, tot = day_counts[day]
            print(f"  {day:<5} {100*ir/tot:7.1f}% {100*h/tot:7.1f}% {100*hi/tot:7.1f}% {tot:>12,}")

    if scen_counts:
        print(f"\n=== Per-Scenario Glycemic Profile ===")
        print(f"  {'Scen':<5} {'Name':<25} {'Points':>12} {'TIR%':>7} {'Hypo%':>7} {'Hyper%':>7} {'Mean':>7}")
        print("  " + "-" * 72)
        for sid in sorted(scen_counts):
            h, ir, hi, tot, gsum = scen_counts[sid]
            name = scenario_names.get(sid, f"scenario {sid}")
            mean_g = gsum / tot if tot else 0
            print(f"  {sid:<5} {name:<25} {tot:>12,} {100*ir/tot:>6.1f}% "
                  f"{100*h/tot:>6.1f}% {100*hi/tot:>6.1f}% {mean_g:>7.2f}")

    # ── ML readiness summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ML READINESS ASSESSMENT")
    print(f"{'='*70}")

    tir_pct   = 100 * in_range_n / total
    hypo_pct  = 100 * hypo_n  / total
    hyper_pct = 100 * hyper_n / total

    # TIR target for closed-loop simulation: the safety stack (hypo guard,
    # rescue bolus, ISF correction) actively maintains glucose, so TIR ≥85%
    # is the expected baseline — real-patient TIR targets (55–80%) do not apply.
    checks: list[tuple[str, bool, str]] = [
        ("Patients ≥ 1000",          len(patient_ids) >= 1000,         f"{len(patient_ids):,}"),
        ("TIR ≥85% (closed-loop)",   tir_pct >= 85,                    f"{tir_pct:.1f}%"),
        ("Hypo < 5%",                hypo_pct < 5,                     f"{hypo_pct:.1f}%"),
        ("Hyper < 30%",              hyper_pct < 30,                   f"{hyper_pct:.1f}%"),
        ("CV% 20–40% (variability)", 20 <= glc_stats.cv_pct <= 40,     f"{glc_stats.cv_pct:.1f}%"),
        ("All 9 scenarios present",  all(i in scen_counts for i in range(1, 10)),
            f"present: {sorted(scen_counts.keys())}"),
    ]

    all_pass = True
    for label, ok, val in checks:
        sym = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{sym}] {label:<35} {val}")

    print(f"\n  Overall: {'READY FOR ML PIPELINE' if all_pass else 'REVIEW FLAGGED ITEMS'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = find_latest()

    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    analyze(path)
