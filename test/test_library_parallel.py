import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.export import ExportConfig
from src.library_generation import generate_library_parallel
from src.simulation_config import SimulationConfig


if __name__ == "__main__":
    workers = max(1, (os.cpu_count() or 2) // 2)

    config = SimulationConfig(
        n_patients=2,
        n_days=14,           # 2 weeks: gives sequence models a full baseline before anomaly days
        international_unit=True,
        noise_std=0.10,
        noise_autocorr=0.7,
        random_scenarios=True,
        clip_states=True,
        std_patient=False,
        random_seed=42,
        enable_plots=False,  # plotting handled below after merge
    )

    export_config = ExportConfig(export_to_parquet=True, export_to_csv=False)
    t0 = time.perf_counter()
    folder = generate_library_parallel(config, export_config, workers=workers)
    total_s = time.perf_counter() - t0

    mins, secs = divmod(int(total_s), 60)
    print(f"Parallel library generated at: {folder}  (total {mins:02d}:{secs:02d})")

    if folder is None:
        print("Generation failed, no plot.")
        sys.exit(1)

    # Load merged parquet and plot all patient BG trajectories
    parquet_files = list(folder.glob("*.parquet"))
    if not parquet_files:
        print("No parquet file found, skipping plot.")
        sys.exit(1)

    df = pd.read_parquet(parquet_files[0])

    plt.figure(figsize=(16, 5))  # type: ignore[misc]
    patient_ids = df["patient_id"].unique()  # type: ignore[union-attr]
    for pid in patient_ids:
        pdata = df[df["patient_id"] == pid].sort_values("absolute_minute")  # type: ignore[index,union-attr]
        time_h: np.ndarray = pdata["absolute_minute"].to_numpy() / 60.0  # type: ignore[union-attr]
        bg: np.ndarray = pdata["blood_glucose"].to_numpy()  # type: ignore[union-attr]
        plt.plot(time_h, bg, alpha=0.4, linewidth=0.8)  # type: ignore[misc]

    # Mean trajectory across all patients
    mean_bg = df.groupby("absolute_minute")["blood_glucose"].mean().reset_index().sort_values("absolute_minute")  # type: ignore[union-attr]
    plt.plot(  # type: ignore[misc]
        mean_bg["absolute_minute"].to_numpy() / 60.0,  # type: ignore[union-attr]
        mean_bg["blood_glucose"].to_numpy(),  # type: ignore[union-attr]
        color="black", linewidth=2.5, label=f"Mean BG (n={len(patient_ids)})", zorder=100,
    )

    plt.axhline(3.9, color="red", linestyle="--", linewidth=1.5, label="Hypoglycemia (3.9 mmol/L)", alpha=0.7)  # type: ignore[misc]
    plt.axhline(10.0, color="orange", linestyle="--", linewidth=1.5, label="Hyperglycemia (10 mmol/L)", alpha=0.7)  # type: ignore[misc]
    for day in range(1, config.n_days):
        plt.axvline(24 * day, color="gray", linestyle=":", alpha=0.3)  # type: ignore[misc]

    hours_total = 24 * config.n_days
    plt.xlim(0, hours_total)  # type: ignore[misc]
    plt.ylim(3, 16.5)  # type: ignore[misc]
    plt.xticks(np.arange(0, hours_total + 1, 24))  # type: ignore[misc]
    plt.xlabel("Time (hours)", fontsize=12)  # type: ignore[misc]
    plt.ylabel("Blood Glucose (mmol/L)", fontsize=12)  # type: ignore[misc]
    plt.title(  # type: ignore[misc]
        f"Parallel Library — {len(patient_ids)} patients × {config.n_days} days\n"
        f"CGM noise σ={config.noise_std:.2f} mmol/L  |  random_scenarios=True",
        fontsize=13, fontweight="bold",
    )
    plt.legend(loc="best", framealpha=0.9, fontsize=10)  # type: ignore[misc]
    plt.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)  # type: ignore[misc]
    plt.tight_layout()  # type: ignore[misc]
    plt.savefig(folder / "parallel_plot.png", dpi=150, bbox_inches="tight")  # type: ignore[misc]
    print(f"Plot saved to: {folder / 'parallel_plot.png'}")
    plt.show()  # type: ignore[misc]
