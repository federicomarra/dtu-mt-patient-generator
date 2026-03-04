# Hovorka Model Monte Carlo Simulation

# Library Imports
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, cast  # noqa: F401
import numpy as np  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
from matplotlib import cm  # type: ignore[import-untyped]
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# --- Imports from src ---
from src.model import hovorka_equations, compute_optimal_steady_state_from_glucose, ParameterSet
from src.parameters import generate_monte_carlo_patients
from src.input import scenario_with_cached_meals, N_SCENARIOS
from src.sensor import measure_glycemia
from src.export import export_to_formats

# --- LEGACY: Gemini imports (not used) ---
# from src.model import hovorka_equations_gemini
# from src.parameters import generate_monte_carlo_patients_gemini
# from src.input import scenario_inputs_gemini


@dataclass
class ExportConfig:
    """Export configuration with named parameters."""
    export_to_parquet: bool = True
    export_to_csv: bool = False
    
    def to_list(self) -> list[bool]:
        """Convert to list format for backward compatibility."""
        return [self.export_to_parquet, self.export_to_csv]


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    n_patients: int = 100
    n_days: int = 7
    international_unit: bool = True
    noise_std: float = 0.15  # CGM noise in mmol/L (literature: 0.1-0.3)
    noise_autocorr: float = 0.7  # AR(1) autocorrelation coefficient
    random_scenarios: bool = True  # Randomize scenarios per day
    clip_states: bool = True  # Clip negative state values
    random_seed: Optional[int] = None  # For reproducibility


def _clip_state_vector(x: np.ndarray) -> np.ndarray:
    """
    Guard against negative masses/concentrations in state vector.
    
    State variables: [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2]
    Q1, Q2, S1, S2, I, D1, D2 should be non-negative (masses/concentrations)
    x1, x2, x3 can be negative but typically small positive
    """
    x_clipped = x.copy()
    # Clip mass/concentration variables to be non-negative
    for i in [0, 1, 2, 3, 4, 8, 9]:  # Q1, Q2, S1, S2, I, D1, D2
        x_clipped[i] = max(0.0, x_clipped[i])
    return x_clipped


def _generate_autocorrelated_noise(
    n_samples: int, 
    noise_std: float, 
    autocorr: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate AR(1) autocorrelated noise for realistic CGM error.
    
    Real CGM noise is temporally correlated unlike independent draws.
    Uses AR(1) process: ε[t] = ρ*ε[t-1] + √(1-ρ²)*η[t]
    where η[t] ~ N(0, σ²)
    
    Parameters:
    -----------
    n_samples: number of time points
    noise_std: standard deviation in mmol/L
    autocorr: autocorrelation coefficient (0-1), typically 0.7-0.9
    rng: random number generator
    
    Returns:
    --------
    noise array of shape (n_samples,)
    """
    noise: np.ndarray = np.zeros(n_samples, dtype=np.float64)
    innovation_std = noise_std * np.sqrt(1 - autocorr**2)
    
    for t in range(n_samples):
        if t == 0:
            noise[t] = float(rng.normal(0, noise_std))
        else:
            innovation: float = cast(float, rng.normal(0, innovation_std))
            noise[t] = autocorr * noise[t-1] + innovation
    
    return noise


def _get_patient_color(patient_idx: int, n_patients: int) -> tuple[float, float, float, float]:
    """
    Generate aesthetically pleasing color for patient trajectory.
    
    Uses colormap to cycle through blue/cyan shades for small cohorts,
    or full spectrum for large cohorts.
    """
    if n_patients <= 10:
        # Use blue/cyan shades for small cohorts
        cmap = cm.get_cmap('Blues')
        color_val = 0.4 + 0.5 * (patient_idx / max(1, n_patients - 1))
        return (*cmap(color_val)[:3], 0.15)  # RGB + alpha
    elif n_patients <= 50:
        # Use cool colors (blue to green)
        cmap = cm.get_cmap('cool')
        color_val = patient_idx / max(1, n_patients - 1)
        return (*cmap(color_val)[:3], 0.12)
    else:
        # Use full spectrum for large cohorts
        cmap = cm.get_cmap('viridis')
        color_val = patient_idx / max(1, n_patients - 1)
        return (*cmap(color_val)[:3], 0.08)


# --- 4. Main Simulation Loop ---
def run_simulation(
    config: SimulationConfig,
    export_config: ExportConfig,
) -> None:
    """
    Run Monte Carlo simulation of Hovorka model across multiple patients and days.
    
    Parameters:
    -----------
    config: SimulationConfig with all simulation parameters
    export_config: ExportConfig specifying export formats
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(config.random_seed)
    
    # Setup export directory
    now_sim_folder_path = None
    if export_config.export_to_parquet or export_config.export_to_csv:
        folder_name = "monte_carlo_results"
        folder_path = Path(folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        now_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        now_sim_folder_path = folder_path / now_string
        try:
            now_sim_folder_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Warning: Failed to create export directory: {e}")
            now_sim_folder_path = None

    # Generate patient cohort
    patients: list[ParameterSet] = generate_monte_carlo_patients(config.n_patients)

    # Time configuration
    minutes_per_day = int(24 * 60)  # 1440 minutes
    
    # Plotting setup
    plt.figure(figsize=(14, 7))  # type: ignore[misc]
    
    # Storage for results
    results_tot: dict[int, dict[str, Any]] = {}
    all_patient_trajectories: list[np.ndarray] = []

    print(f"Running Monte Carlo Simulation: {config.n_patients} patients × {config.n_days} days")
    print(f"CGM noise: σ={config.noise_std:.2f} mmol/L, autocorr={config.noise_autocorr:.2f}")
    
    # Main simulation loop
    desc_text = "\033[34mSimulating patients\033[0m"
    for patient_idx, patient_params in enumerate(tqdm(patients, desc=desc_text, unit="patient", colour="blue")):
        
        # Initialize patient results
        results_tot[patient_idx] = {
            "patient_id": patient_idx, 
            "params": patient_params, 
            "days": {}
        }
        
        # Compute initial steady state (100 mg/dL glucose)
        x0_initial = compute_optimal_steady_state_from_glucose(
            100, 
            patient_params, 
            international_units=False, 
            max_iterations=100, 
            print_progress=False
        )
        
        # Track state across days
        current_state: np.ndarray = np.array(x0_initial, dtype=np.float64)
        patient_full_trajectory: list[np.ndarray] = []
        
        # Simulate each day
        for day_idx in range(config.n_days):
            
            # Select scenario for this day
            if config.random_scenarios:
                scenario = int(rng.integers(1, N_SCENARIOS + 1))
            else:
                scenario = 1
            
            # Time span for this day
            t_span = (0, minutes_per_day)
            t_eval_day = np.arange(0, minutes_per_day + 1)  # Every minute
            
            # Define ODE function with patient-specific parameters
            # NOTE: Using scenario_with_cached_meals for deterministic meal scheduling
            def ode_func(t: float, x: np.ndarray) -> np.ndarray:
                result = hovorka_equations(
                    int(t),
                    x.tolist(),
                    patient_params,
                    scenario_with_cached_meals,
                    scenario=scenario,
                    patient_id=patient_idx,
                    day=day_idx,
                    basal_hourly=0.5,
                    insulin_sensitivity=2.0,
                    meal_schedule=None,
                    seed=config.random_seed,
                )
                return np.array(result)
            
            # Solve ODE once for entire day with dense output
            # Much more efficient than 1440 separate solve_ivp calls
            sol = solve_ivp(  # type: ignore[misc]
                ode_func,
                t_span,
                current_state,
                method='RK45',
                t_eval=t_eval_day,
                dense_output=False,
                rtol=1e-6,
                atol=1e-8
            )
            
            # Extract state trajectory
            state_trajectory: np.ndarray = np.asarray(sol.y, dtype=np.float64)  # type: ignore[misc]
            
            # Clip states if requested (guard against negative masses)
            if config.clip_states:
                for i in range(state_trajectory.shape[1]):
                    state_trajectory[:, i] = _clip_state_vector(state_trajectory[:, i])
            
            # Update current state for next day (continuity)
            current_state = np.asarray(state_trajectory[:, -1], dtype=np.float64)
            
            # Generate autocorrelated CGM noise for this day
            n_measurements = len(t_eval_day)
            noise_sequence = _generate_autocorrelated_noise(
                n_measurements,
                config.noise_std,
                config.noise_autocorr,
                rng
            )
            
            # Measure glycemia at each timepoint with autocorrelated noise
            glycemia_day: list[float] = []
            for t_idx in range(n_measurements):
                state_at_t: tuple[float, ...] = tuple(np.asarray(state_trajectory[:, t_idx], dtype=np.float64).tolist())
                # Use base measurement without internal noise, add our autocorrelated noise
                g_base = measure_glycemia(
                    state_at_t,  # Pass as list for type compatibility
                    patient_params, 
                    noise_std=0.0,  # No internal noise
                    output_unit='mmol/L'
                )
                g_measured: float = float(g_base) + float(noise_sequence[t_idx])
                glycemia_day.append(g_measured)
            
            glycemia_day_array = np.array(glycemia_day, dtype=np.float64)
            
            # Convert units if needed
            if not config.international_unit:
                glycemia_day_array = glycemia_day_array * (float(patient_params['MwG']) / 10.0)  # mmol/L -> mg/dL
            
            # Store results for this day
            results_tot[patient_idx]["days"][day_idx] = glycemia_day_array
            patient_full_trajectory.append(glycemia_day_array)
        
        # Concatenate all days for this patient
        patient_full_trajectory_concat = np.concatenate(patient_full_trajectory, dtype=np.float64)  # type: ignore[arg-type]
        all_patient_trajectories.append(patient_full_trajectory_concat)
        
        # Plot patient trajectory with aesthetically pleasing colors
        time_hours = np.arange(len(patient_full_trajectory_concat)) / 60.0
        patient_color = _get_patient_color(patient_idx, config.n_patients)
        plt.plot(time_hours, patient_full_trajectory_concat, color=patient_color[:3], alpha=patient_color[3])  # type: ignore[misc]

    # Export results if requested
    if now_sim_folder_path and (export_config.export_to_parquet or export_config.export_to_csv):
        try:
            export_to_formats(
                results_tot, 
                config.n_patients, 
                config.n_days, 
                now_sim_folder_path, 
                export_config.to_list()
            )
        except Exception as e:
            print(f"Warning: Export failed: {e}")

    # Plot mean trajectory across all patients and days
    if all_patient_trajectories:
        mean_trajectory: np.ndarray = np.mean(all_patient_trajectories, axis=0)  # type: ignore[assignment]
        time_hours = np.arange(len(mean_trajectory)) / 60.0
        plt.plot(  # type: ignore[misc]
            time_hours, 
            mean_trajectory, 
            color='black', 
            linewidth=2.5, 
            label=f'Mean Population BG (n={config.n_patients})',
            zorder=100
        )

    # Format plot
    if config.international_unit:
        plt.axhline(3.8, color='red', linestyle='--', linewidth=1.5, label='Hypoglycemia (3.8 mmol/L)', alpha=0.7)  # type: ignore[misc]
        plt.axhline(10.0, color='orange', linestyle='--', linewidth=1.5, label='Hyperglycemia (10 mmol/L)', alpha=0.7)  # type: ignore[misc]
        plt.ylim(0, 20)  # type: ignore[misc]
        plt.ylabel("Blood Glucose (mmol/L)", fontsize=12)  # type: ignore[misc]
    else:
        plt.axhline(70, color='red', linestyle='--', linewidth=1.5, label='Hypoglycemia (70 mg/dL)', alpha=0.7)  # type: ignore[misc]
        plt.axhline(180, color='orange', linestyle='--', linewidth=1.5, label='Hyperglycemia (180 mg/dL)', alpha=0.7)  # type: ignore[misc]
        plt.ylim(0, 400)  # type: ignore[misc]
        plt.ylabel("Blood Glucose (mg/dL)", fontsize=12)  # type: ignore[misc]
    
    plt.title(  # type: ignore[misc]
        f"Hovorka Model Monte Carlo Simulation\n"
        f"{config.n_patients} patients × {config.n_days} days, "
        f"CGM noise σ={config.noise_std:.2f} mmol/L",
        fontsize=13,
        fontweight='bold'
    )
    plt.xlabel("Time (hours)", fontsize=12)  # type: ignore[misc]
    plt.xlim(0, 24 * config.n_days)  # type: ignore[misc]
    
    # Set x-ticks at reasonable intervals
    hours_total = 24 * config.n_days
    if hours_total <= 48:
        tick_interval = 4
    elif hours_total <= 168:  # 1 week
        tick_interval = 12
    else:
        tick_interval = 24
    plt.xticks(np.arange(0, hours_total + 1, tick_interval))  # type: ignore[misc]
    
    plt.legend(loc='best', framealpha=0.9, fontsize=10)  # type: ignore[misc]
    plt.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)  # type: ignore[misc]
    plt.tight_layout()  # type: ignore[misc]
    
    # Save plot if export directory exists
    if now_sim_folder_path:
        try:
            plt.savefig(now_sim_folder_path / "simulation_plot.png", dpi=150, bbox_inches='tight')  # type: ignore[misc]
            print(f"\nResults saved to: {now_sim_folder_path}")
        except Exception as e:
            print(f"Warning: Failed to save plot: {e}")
    
    plt.show()  # type: ignore[misc]


if __name__ == "__main__":
    # Configuration
    skip_terminal_input: bool = True
    
    # Simulation configuration
    sim_config = SimulationConfig(
        n_patients=3,
        n_days=2,
        international_unit=True,
        noise_std=0.15,  # mmol/L (realistic CGM noise from literature)
        noise_autocorr=0.7,  # Temporal correlation
        random_scenarios=True,
        clip_states=True,
        random_seed=42  # For reproducibility
    )
    
    # Export configuration
    export_config = ExportConfig(
        export_to_parquet=False,
        export_to_csv=True
    )
    
    # Interactive input (if enabled)
    if not skip_terminal_input:
        unit_answer: str = input("Use international units (mmol/L) instead of mg/dL? (y/n): ").strip().lower()
        sim_config.international_unit = unit_answer not in {"n", "no", "0", "false", "f"}
        
        sim_config.n_patients = int(input("Number of patients to simulate: "))
        sim_config.n_days = int(input("Number of days per patient: "))
        
        noise_answer = input(f"CGM noise std (mmol/L, default {sim_config.noise_std}): ").strip()
        if noise_answer:
            sim_config.noise_std = float(noise_answer)
    
    # Run simulation
    run_simulation(config=sim_config, export_config=export_config)