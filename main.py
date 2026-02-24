# Hovorka Model Monte Carlo Simulation

# Library Imports
from __future__ import annotations
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
#import pandas as pd

GEMINI = False

# --- 1. The Hovorka Model Equations ---
if GEMINI:
    from src.model import hovorka_equations_gemini as hovorka_equations
else:
    from src.model import hovorka_equations, compute_optimal_steady_state_from_glucose

# --- 2. Simulation Helpers ---
if GEMINI:
    from src.parameters import generate_monte_carlo_patients_gemini as generate_monte_carlo_patients
else:
    from src.parameters import generate_monte_carlo_patients

# --- 3. Scenario Definition ---
if GEMINI:
    from src.input import scenario_inputs_gemini as scenario_inputs
else:
    from src.input import scenario_inputs
from src.input import N_SCENARIOS

from src.sensor import measure_glycemia

from src.export import export_to_formats


# --- 4. Main Simulation Loop ---
def run_simulation(do_export: list[bool] = [True, False], international_unit: bool = True, n_patients: int = 100, n_days: int = 7):
    if any(do_export):
        folder_name = "monte_carlo_results"
        folder_path = Path(folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        now_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        now_sim_folder_path = folder_path / now_string
        now_sim_folder_path.mkdir(parents=True, exist_ok=True)

    patients = generate_monte_carlo_patients(n_patients)

    # Time span for 24 hours
    minutes_per_day = int(24 * 60)
    t_eval = np.linspace(0, minutes_per_day, minutes_per_day + 1)  # 0 to 1440 inclusive

    plt.figure(figsize=(12, 6))

    results_tot: Dict[int, dict] = {}
    results_list: List[np.ndarray] = []

    print(f"Running Monte Carlo Simulation for {n_patients} patients...")
    desc_text = "\033[34mSimulating patients\033[0m"
    for i, p in enumerate(tqdm(patients, desc=desc_text, unit="patient", colour="blue")):
        results_tot[i] = {"patient_id": i, "params": p, "days": {}}
        
        # Initial Conditions for each patient
        # Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2
        if GEMINI:
            x0_curr = [800, 800, 100, 100, 10, 0, 0, 0, 0, 0]
        else:
            x0_curr = compute_optimal_steady_state_from_glucose(100, p, international_units=False, max_iterations=10, print_progress=False)
        
        for day in range(n_days):
            scenario = 1  #TODO: Implement multiple scenarios
            #scenario = np.random.randint(1, N_SCENARIOS + 1) # Randomly select a scenario for this day
            
            # Accumulate time offset for plotting
            t_offset = day * minutes_per_day
            
            fx = lambda t, x: hovorka_equations(t, x, p, scenario_inputs, scenario)
            
            y_day = []
            # Solving step-by-step as per MATLAB code
            for k in range(len(t_eval) - 1):
                t_start = t_eval[k]
                t_end = t_eval[k+1]
                
                sol = solve_ivp(fx, (t_start, t_end), x0_curr, method='RK45')
                x0_curr = sol.y[:, -1]
                
                # Measure glycemia at the end of the step
                y_val = measure_glycemia(x0_curr, p, noise_std=0.01)
                y_day.append(y_val)
            
            y_day = np.array(y_day)
            
            if not international_unit: y_day = y_day * (p['MwG'] / 10)  # Convert to mg/dL
            
            # Plot in hours: (minutes + offset) / 60
            plt.plot((t_eval[1:] + t_offset) / 60, y_day, color='blue', alpha=0.15)

            # Store results for export
            patient_results = results_tot[i]
            patient_days = patient_results["days"]
            patient_days[day] = y_day
            results_list.append(y_day)

    if any(do_export): export_to_formats(results_tot, n_patients, now_sim_folder_path, do_export)

    # Plot Mean
    if results_list:
        mean_bg = np.mean(results_list, axis=0)
        # Note: plotting mean for only the first day if we want to show population average per day
        plt.plot(t_eval[1:] / 60, mean_bg, color='black', linewidth=2, label='Mean Population BG (Day 1)')

    # Formatting
    if international_unit:
        plt.axhline(3.8, color='r', linestyle='--', label='Hypoglycemia Limit')
        plt.axhline(10, color='y', linestyle='--', label='Hyperglycemia Limit')
        plt.ylim(0, 20)
        plt.ylabel("Blood Glucose (mmol/l)")
    else:
        plt.axhline(70, color='r', linestyle='--', label='Hypoglycemia Limit')
        plt.axhline(180, color='y', linestyle='--', label='Hyperglycemia Limit')
        plt.ylim(0, 500)
        plt.ylabel("Blood Glucose (mg/dL)")
    plt.title(f"Hovorka Model: Monte Carlo Simulation (n patients={n_patients}, n days={n_days})")
    plt.xlabel("Time (hours)")
    plt.xticks(np.arange(0, (24) + 1, 24))
    plt.xlim(0, 24)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    skipTerminal = True
    EXPORT = [False, True]  # [export_to_parquet, export_to_csv]
    INTERNATIONAL_UNIT = True
    N_PATIENTS = 10
    N_DAYS = 7

    if not skipTerminal:
        unit_answer = input("Do you prefer international unit: mmol/L instead of mg/dL? (y/n): ").strip().lower()
        INTERNATIONAL_UNIT = not (unit_answer in {"n", "no", "0", "false", "f", "nope", "mgdl", "mg/dl"})
        N_PATIENTS = int(input("How many patients do you want to simulate? "))
        N_DAYS = int(input("How many days do you want to simulate for each patient? "))

    run_simulation(do_export=EXPORT, international_unit=INTERNATIONAL_UNIT, n_patients=N_PATIENTS, n_days=N_DAYS)