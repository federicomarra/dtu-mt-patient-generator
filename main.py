# Hovorka Model Monte Carlo Simulation

# Library Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import pandas as pd

# --- 1. The Hovorka Model Equations ---
from model import hovorka_equations_gemini as hovorka_equations

# --- 2. Simulation Helpers ---
from parameters import generate_monte_carlo_patients_gemini as generate_monte_carlo_patients

# --- 3. Scenario Definition ---
from input import scenario_inputs_gemini as scenario_inputs
from input import N_SCENARIOS

from sensor import measure_glycemia

from export import export_to_parquet

# --- 4. Main Simulation Loop ---
def run_simulation(export=True, international_unit=True, n_patients=100, n_days=7):
    if export:
        folder_name = "monte_carlo_results"
        folder_path = Path(folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        now_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        now_sim_folder_path = folder_path / now_string
        now_sim_folder_path.mkdir(parents=True, exist_ok=True)

    patients = generate_monte_carlo_patients(n_patients)

    # Time span for 24 hours
    minutes_per_day = int(24 * 60)  # Total minutes in a day based on time step
    t_span = (0, minutes_per_day)  # 24 hours
    t_eval = np.linspace(0, minutes_per_day, minutes_per_day)  # 1 point per minute

    # Initial Conditions (Steady State approximations)
    # Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2
    # Assume starting glucose ~ 100 mg/dL -> Q1 = 100 * VG / 10 (approx conversion)
    # This is a rough initialization. Real simulations usually run a "burn-in" period.
    x0 = [800, 800, 100, 100, 10, 0, 0, 0, 0, 0]

    plt.figure(figsize=(12, 6))

    results_tot = {}

    print(f"Running Monte Carlo Simulation for {n_patients} patients...")
    desc_text = "\033[34mSimulating patients\033[0m"
    for i, p in enumerate(tqdm(patients, desc=desc_text, unit="patient", colour="blue")):
    #for i, p in enumerate(patients):
        results_patient = []
        for day in range(n_days):
            # Create lambda for the ODE solver with specific patient params
            scenario = 1  #TODO: Implement multiple scenarios
            #scenario = np.random.uniform(1, N_SCENARIOS)  # Randomly select a scenario for this day
            fx = lambda t, x: hovorka_equations(t, x, p, scenario_inputs, scenario)

            sol = solve_ivp(fx, t_span, x0, t_eval=t_eval, method='RK45')

            print(sol.message)

            # Calculate BG in mg/dL (Q1 / VG) * conversion factor if needed
            # Q1 is mass (usually mmol or mg depending on units used in params).
            # In standard Hovorka, Q is mmol, but here we treat mass generically.
            # Let's assume output needs scaling to mg/dL.
            # BG = Q1 / VG. If Q1 is mmol and VG is L, BG is mmol/L.
            # To get mg/dL, multiply by 18.

            #y = (sol.y[0] / p['VG'])
            y = measure_glycemia(sol, p, noise_std=0.01)  # Add some sensor noise
            if not international_unit: y *= 180.16 / 10  # Convert to mmol/L

            plt.plot(sol.t / 60, y, color='blue', alpha=0.15)
            results_patient.append(y)
        results_tot.append(results_patient)

        if export: export_to_parquet(results_patient, i, now_sim_folder_path)

    # Plot Mean
    mean_bg = np.mean(results_patient, axis=0)
    plt.plot(t_eval/60, mean_bg, color='black', linewidth=2, label='Mean Population BG')

    # Formatting
    if international_unit:
        plt.axhline(3.8, color='r', linestyle='--', label='Hypoglycemia Limit')
        plt.axhline(10, color='y', linestyle='--', label='Hyperglycemia Limit')
        plt.ylabel("Blood Glucose (mmol/l)")
    else:
        plt.axhline(70, color='r', linestyle='--', label='Hypoglycemia Limit')
        plt.axhline(180, color='y', linestyle='--', label='Hyperglycemia Limit')
        plt.ylabel("Blood Glucose (mg/dL)")
    plt.title(f"Hovorka Model: Monte Carlo Simulation (N={n_patients})")
    plt.xlabel("Time (hours)")
    plt.xticks(np.arange(0, 25, 1))
    plt.xlim(0, 24 * n_days)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    skipTerminal = True
    EXPORT = False
    INTERNATIONAL_UNIT = True
    N_PATIENTS = 20
    N_DAYS = 3

    if not skipTerminal:
        unit_answer = input("Do you prefer international unit: mmol/L instead of mg/dL? (y/n): ").strip().lower()
        INTERNATIONAL_UNIT = not (unit_answer in {"n", "no", "0", "false", "f", "nope", "mgdl", "mg/dl"})
        N_PATIENTS = int(input("How many patients do you want to simulate? "))
        N_DAYS = int(input("How many days do you want to simulate for each patient? "))

    run_simulation(export=EXPORT, international_unit=INTERNATIONAL_UNIT, n_patients=N_PATIENTS, n_days=N_DAYS)