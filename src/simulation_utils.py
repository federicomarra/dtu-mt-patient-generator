from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

from src.model import ParameterSet
def clip_state_trajectory(state_trajectory: np.ndarray) -> np.ndarray:
    """Clip state variables that must remain non-negative."""
    clipped = state_trajectory.copy()
    non_negative_indices = np.array([0, 1, 2, 3, 4, 8, 9], dtype=np.int64)
    clipped[non_negative_indices, :] = np.maximum(clipped[non_negative_indices, :], 0.0)
    return clipped


def generate_autocorrelated_noise(
    n_samples: int,
    noise_std: float,
    autocorr: float,
    rng: np.random.Generator,
    initial_value: float | None = None,
) -> np.ndarray:
    """Generate AR(1) sensor noise, optionally continuous across day boundaries."""
    noise = np.zeros(n_samples, dtype=np.float64)
    innovation_std = noise_std * np.sqrt(1 - autocorr**2)

    for idx in range(n_samples):
        if idx == 0:
            if initial_value is None:
                noise[idx] = float(rng.normal(0, noise_std))
            else:
                innovation = cast(float, rng.normal(0, innovation_std))
                noise[idx] = autocorr * initial_value + innovation
        else:
            innovation = cast(float, rng.normal(0, innovation_std))
            noise[idx] = autocorr * noise[idx - 1] + innovation

    return noise


def get_patient_color(patient_idx: int, n_patients: int) -> tuple[float, float, float, float]:
    """Return RGBA tuple used to plot one patient trajectory."""
    if n_patients <= 10:
        cmap = plt.get_cmap("Blues")  # type: ignore[misc]
        color_val = 0.4 + 0.5 * (patient_idx / max(1, n_patients - 1))
        return (*cmap(color_val)[:3], 0.15)
    if n_patients <= 50:
        cmap = plt.get_cmap("cool")  # type: ignore[misc]
        color_val = patient_idx / max(1, n_patients - 1)
        return (*cmap(color_val)[:3], 0.12)

    cmap = plt.get_cmap("viridis")  # type: ignore[misc]
    color_val = patient_idx / max(1, n_patients - 1)
    return (*cmap(color_val)[:3], 0.08)


def measure_glycemia_day(
    state_trajectory: np.ndarray,
    patient_params: ParameterSet,
    noise_sequence: np.ndarray,
    n_measurements: int,
) -> np.ndarray:
    """Measure glycemia for one day given states and additive noise sequence."""
    available = state_trajectory.shape[1]
    effective = min(n_measurements, available)
    if effective == 0:
        return np.zeros(n_measurements, dtype=np.float64)

    vg = float(patient_params["VG"])
    bw = float(patient_params["BW"])
    denom = vg * bw
    q1 = np.asarray(state_trajectory[0, :effective], dtype=np.float64)
    glycemia_day = np.zeros(n_measurements, dtype=np.float64)
    glycemia_day[:effective] = (q1 / denom) + np.asarray(noise_sequence[:effective], dtype=np.float64)

    if effective < n_measurements:
        glycemia_day[effective:] = glycemia_day[effective - 1]

    return glycemia_day


def create_export_directory(base_folder: str = "monte_carlo_results") -> Path | None:
    """Create timestamped export directory, returning None on failure."""
    folder_path: Path = Path(base_folder)
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Warning: Failed to create export {folder_path} directory: {exc}")
        return None

    today_folder_path = folder_path / datetime.now().strftime("%Y%m%d")
    try:
        today_folder_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Warning: Failed to create export {today_folder_path} directory: {exc}")
        return None

    now_sim_folder_path = today_folder_path / datetime.now().strftime("%H%M%S")
    try:
        now_sim_folder_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Warning: Failed to create export {now_sim_folder_path} directory: {exc}")
        return None

    return now_sim_folder_path
