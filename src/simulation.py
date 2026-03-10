# Simulation Module

# Library Imports
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, cast  # noqa: F401
import numpy as np  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# --- Imports from src ---
from src.model import hovorka_equations, compute_optimal_steady_state_from_glucose, ParameterSet
from src.parameters import generate_monte_carlo_patients
from src.input import scenario_with_cached_meals, N_SCENARIOS, clear_meal_cache
from src.sensor import measure_glycemia
from src.export import export_to_formats, ExportConfig

@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    n_patients: int = 100
    n_days: int = 7
    international_unit: bool = True
    noise_std: float = 0.10  # CGM noise in mmol/L (realistic mild sensor noise)
    noise_autocorr: float = 0.7  # AR(1) autocorrelation coefficient
    random_scenarios: bool = False  # Use baseline scenario by default (set True for stress scenarios)
    clip_states: bool = True  # Clip negative state values
    random_seed: Optional[int] = None  # For reproducibility
    basal_hourly: float = 0.5  # [U/hr] fallback when not using calibrated basal
    use_calibrated_basal: bool = True  # derive basal per patient from calibrated initial steady-state
    insulin_carbohydrates_ratio: float = 19.3  # [g/U], realistic default range ~8-15
    initial_target_glucose_mgdl: float = 100.0  # safer initialization target
    enable_hypo_guard: bool = True  # dynamically reduce insulin delivery near hypoglycemia
    hypo_guard_mmol: float = 5.0  # guard threshold (~90 mg/dL), anticipatory for 5-meal active schedule
    suppress_meal_bolus_on_guard: bool = False  # if True, also suppress meal bolus when guard is active
    enable_hypo_rescue: bool = True  # optional rescue carbs model (e.g., candy correction)
    hypo_rescue_trigger_mmol: float = 4.5  # trigger rescue below this level (~81 mg/dL)
    hypo_rescue_gain_per_min: float = 25.0  # proportional rescue gain [1/min], higher for active schedule
    solver_method: str = "RK45"  # default non-stiff ODE solver method
    solver_max_step: float = 1.0  # minutes (captures meal/bolus discontinuities)
    derivative_clip: float = 1e5  # hard bound on ODE derivatives
    std_patient: bool = False  # use standard patient parameters
    insulin_sensitivity_factor: float = 3.1  # [mmol/L/U], realistic default range ~2-4


def _clip_state_trajectory(state_trajectory: np.ndarray) -> np.ndarray:
    """Vectorized clipping of non-negative physiological state variables across all timepoints."""
    clipped = state_trajectory.copy()
    non_negative_indices = np.array([0, 1, 2, 3, 4, 8, 9], dtype=np.int64)
    clipped[non_negative_indices, :] = np.maximum(clipped[non_negative_indices, :], 0.0)
    return clipped


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
        cmap = plt.get_cmap('Blues')  # type: ignore[misc]
        color_val = 0.4 + 0.5 * (patient_idx / max(1, n_patients - 1))
        return (*cmap(color_val)[:3], 0.15)  # RGB + alpha
    elif n_patients <= 50:
        # Use cool colors (blue to green)
        cmap = plt.get_cmap('cool')  # type: ignore[misc]
        color_val = patient_idx / max(1, n_patients - 1)
        return (*cmap(color_val)[:3], 0.12)
    else:
        # Use full spectrum for large cohorts
        cmap = plt.get_cmap('viridis')  # type: ignore[misc]
        color_val = patient_idx / max(1, n_patients - 1)
        return (*cmap(color_val)[:3], 0.08)


def _measure_glycemia_day(
    state_trajectory: np.ndarray,
    patient_params: ParameterSet,
    noise_sequence: np.ndarray,
    n_measurements: int,
) -> np.ndarray:
    """Compute glycemia trajectory from state trajectory with additive sensor noise."""
    available_measurements = state_trajectory.shape[1]
    effective_measurements = min(n_measurements, available_measurements)

    glycemia_day: list[float] = []
    for t_idx in range(effective_measurements):
        state_at_t: tuple[float, ...] = tuple(np.asarray(state_trajectory[:, t_idx], dtype=np.float64).tolist())
        g_base = measure_glycemia(
            state_at_t,
            patient_params,
            noise_std=0.0,
            output_unit='mmol/L'
        )
        glycemia_day.append(float(g_base) + float(noise_sequence[t_idx]))

    if effective_measurements < n_measurements:
        fallback_value = glycemia_day[-1] if glycemia_day else 0.0
        glycemia_day.extend([fallback_value] * (n_measurements - effective_measurements))

    return np.array(glycemia_day, dtype=np.float64)


# --- Main Simulation Loop ---
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
    
    # Clear any cached meal schedules from previous runs to ensure fresh state
    clear_meal_cache()
    
    # Setup export directory
    if any(export_config.to_list()):
        folder_string = "monte_carlo_results"
        folder_path = Path(folder_string)
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Warning: Failed to create export {folder_path} directory: {e}")
            folder_path = None
        today_string = datetime.now().strftime("%Y%m%d")
        today_folder_path = folder_path / today_string
        try:
            today_folder_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Warning: Failed to create export {today_folder_path} directory: {e}")
            today_folder_path = None
        now_string = datetime.now().strftime("%H%M%S")
        now_sim_folder_path = today_folder_path / now_string
        try:
            now_sim_folder_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Warning: Failed to create export {now_sim_folder_path} directory: {e}")
            now_sim_folder_path = None

    # Generate an oversized candidate pool; keep first N stable patients
    candidate_multiplier = 5
    candidate_pool_size = max(config.n_patients * candidate_multiplier, config.n_patients)
    patients: list[ParameterSet] = generate_monte_carlo_patients(candidate_pool_size, standard_patient=config.std_patient, seed=config.random_seed)

    # Time configuration
    minutes_per_day = int(24 * 60)  # 1440 minutes
    
    # Plotting setup
    plt.figure(figsize=(14, 7))  # type: ignore[misc]
    
    # Storage for results
    results_tot: dict[int, dict[str, Any]] = {}
    all_patient_trajectories: list[np.ndarray] = []
    sampled_patients = 0
    accepted_patients = 0
    rejected_patients = 0
    rejected_initial_glucose = 0
    rejected_instability = 0
    rejection_bounds_mmol = (5.0, 6.5)
    instability_max_glucose_mmol = 20.0
    instability_hyper_pct = 30.0

    print(f"Running Monte Carlo Simulation: {config.n_patients} patients × {config.n_days} days")
    print(f"CGM noise: σ={config.noise_std:.2f} mmol/L, autocorr={config.noise_autocorr:.2f}")
    
    # Main simulation loop (progress tracks accepted patients)
    desc_text = "\033[34mAccepted patients\033[0m"
    with tqdm(total=config.n_patients, desc=desc_text, unit="patient", colour="blue") as pbar:
        for patient_params in patients:
            if accepted_patients >= config.n_patients:
                break
            sampled_patients += 1

            # Compute initial steady state
            x0_initial = compute_optimal_steady_state_from_glucose(
                config.initial_target_glucose_mgdl,
                patient_params, 
                international_units=False, 
                max_iterations=100, 
                print_progress=False
            )

            initial_glucose_mmol = float(
                measure_glycemia(
                    tuple(float(v) for v in x0_initial),
                    patient_params,
                    noise_std=0.0,
                    output_unit='mmol/L',
                )
            )
            if not (rejection_bounds_mmol[0] <= initial_glucose_mmol <= rejection_bounds_mmol[1]):
                rejected_patients += 1
                rejected_initial_glucose += 1
                continue

            sim_patient_id = accepted_patients

            # Initialize patient results
            results_tot[sim_patient_id] = {
                "patient_id": sim_patient_id,
                "params": patient_params,
                "days": {}
            }

            tau_i = float(patient_params["tauI"])
            us_calibrated_mU_min = float(x0_initial[2]) / tau_i if tau_i > 0 else (config.basal_hourly * 1000.0 / 60.0)
            basal_hourly_patient = (us_calibrated_mU_min * 60.0 / 1000.0) if config.use_calibrated_basal else config.basal_hourly
            si3_ref = 520.0e-4
            si3_ratio_raw = float(patient_params.get("SI3", si3_ref)) / si3_ref if si3_ref > 0 else 1.0
            si3_ratio_limited = float(np.clip(si3_ratio_raw, 0.85, 1.15))
            insulin_sensitivity_patient = float(np.clip(config.insulin_carbohydrates_ratio * si3_ratio_limited, 10.0, 14.0))
        
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
                    x_safe = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
                    x_safe = np.clip(x_safe, -1e6, 1e6)
                    basal_hourly_effective = basal_hourly_patient
                    insulin_sensitivity_effective = insulin_sensitivity_patient
                    g_est = measure_glycemia(tuple(x_safe.tolist()), patient_params, noise_std=0.0, output_unit='mmol/L')
                    if config.enable_hypo_guard:
                        if g_est <= config.hypo_guard_mmol:
                            basal_hourly_effective = 0.0
                            if config.suppress_meal_bolus_on_guard:
                                insulin_sensitivity_effective = 1e6
                    result = hovorka_equations(
                        int(t),
                        x_safe.tolist(),
                        patient_params,
                        scenario_with_cached_meals,
                        scenario=scenario,
                        patient_id=sim_patient_id,
                        day=day_idx,
                        basal_hourly=basal_hourly_effective,
                        insulin_sensitivity=insulin_sensitivity_effective,
                        meal_schedule=None,
                        seed=config.random_seed,
                    )
                    dy = np.asarray(result, dtype=np.float64)
                    if config.enable_hypo_rescue and g_est < config.hypo_rescue_trigger_mmol:
                        vg = float(patient_params["VG"])
                        bw = float(patient_params["BW"])
                        deficit = config.hypo_rescue_trigger_mmol - g_est
                        rescue_q1 = config.hypo_rescue_gain_per_min * deficit * vg * bw
                        dy[0] += rescue_q1
                    dy = np.nan_to_num(dy, nan=0.0, posinf=config.derivative_clip, neginf=-config.derivative_clip)
                    dy = np.clip(dy, -config.derivative_clip, config.derivative_clip)
                    return dy
            
                # Solve ODE once for entire day with dense output
                # Much more efficient than 1440 separate solve_ivp calls
                sol = solve_ivp(  # type: ignore[misc]
                    ode_func,
                    t_span,
                    current_state,
                    method=config.solver_method,
                    t_eval=t_eval_day,
                    dense_output=False,
                    rtol=1e-6,
                    atol=1e-8,
                    max_step=config.solver_max_step,
                )
            
                # Extract state trajectory
                state_trajectory: np.ndarray = np.asarray(sol.y, dtype=np.float64)  # type: ignore[misc]
                state_trajectory = np.nan_to_num(state_trajectory, nan=0.0, posinf=1e6, neginf=0.0)
                if state_trajectory.ndim != 2 or state_trajectory.shape[1] == 0:
                    print(
                        f"Warning: ODE solver returned no valid points for patient {sim_patient_id}, day {day_idx}. "
                        "Using previous state as fallback."
                    )
                    state_trajectory = np.asarray(current_state, dtype=np.float64).reshape(-1, 1)
                sol_any: Any = sol  # type: ignore[misc]
                solver_success = bool(getattr(sol_any, "success", True))  # type: ignore[misc]
                solver_message = str(getattr(sol_any, "message", ""))  # type: ignore[misc]
                if not solver_success:
                    print(
                        f"Warning: ODE solver ended early for patient {sim_patient_id}, day {day_idx}: {solver_message}"
                    )
            
                # Clip states if requested (guard against negative masses)
                if config.clip_states:
                    state_trajectory = _clip_state_trajectory(state_trajectory)
            
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

                glycemia_day_array = _measure_glycemia_day(
                    state_trajectory=state_trajectory,
                    patient_params=patient_params,
                    noise_sequence=noise_sequence,
                    n_measurements=n_measurements,
                )
            
                # Convert units if needed
                if not config.international_unit:
                    glycemia_day_array = glycemia_day_array * (float(patient_params['MwG']) / 10.0)  # mmol/L -> mg/dL
            
                # Store results for this day
                results_tot[sim_patient_id]["days"][day_idx] = glycemia_day_array
                patient_full_trajectory.append(glycemia_day_array)
        
            # Concatenate all days for this patient
            patient_full_trajectory_concat = np.concatenate(patient_full_trajectory, dtype=np.float64)  # type: ignore[arg-type]

            hyper_count = int(np.sum(patient_full_trajectory_concat > 10.0))
            total_count = int(patient_full_trajectory_concat.size)
            hyper_pct = (100.0 * hyper_count / total_count) if total_count > 0 else 0.0
            max_glucose = float(np.max(patient_full_trajectory_concat)) if total_count > 0 else 0.0
            if hyper_pct > instability_hyper_pct or max_glucose > instability_max_glucose_mmol:
                rejected_patients += 1
                rejected_instability += 1
                del results_tot[sim_patient_id]
                continue

            accepted_patients += 1
            all_patient_trajectories.append(patient_full_trajectory_concat)
            pbar.update(1)
        
            # Plot patient trajectory with aesthetically pleasing colors
            time_hours = np.arange(len(patient_full_trajectory_concat)) / 60.0
            patient_color = _get_patient_color(sim_patient_id, max(1, config.n_patients))
            plt.plot(time_hours, patient_full_trajectory_concat, color=patient_color[:3], alpha=patient_color[3])  # type: ignore[misc]

    if accepted_patients < config.n_patients:
        print(
            f"Warning: accepted only {accepted_patients}/{config.n_patients} stable patients "
            f"from {sampled_patients} candidates"
        )

    rejection_rate_pct = (100.0 * rejected_patients / sampled_patients) if sampled_patients > 0 else 0.0
    print(
        "Patient sampling summary: "
        f"accepted={accepted_patients}, rejected={rejected_patients}, "
        f"rejection_rate={rejection_rate_pct:.1f}%"
    )
    print(
        "Rejection reasons: "
        f"initial_glucose={rejected_initial_glucose}, instability={rejected_instability}"
    )

    # Export results if requested
    if now_sim_folder_path and (export_config.export_to_parquet or export_config.export_to_csv):
        try:
            # Prepare config metadata for logging
            config_metadata: dict[str, Any] = {
                "n_patients": config.n_patients,
                "n_days": config.n_days,
                "international_unit": config.international_unit,
                "noise_std_mmol_L": config.noise_std,
                "noise_autocorr": config.noise_autocorr,
                "random_scenarios": config.random_scenarios,
                "random_seed": config.random_seed,
                "basal_hourly_U_hr": config.basal_hourly,
                "use_calibrated_basal": config.use_calibrated_basal,
                "insulin_sensitivity_g_U": config.insulin_carbohydrates_ratio,
                "initial_target_glucose_mgdl": config.initial_target_glucose_mgdl,
                "enable_hypo_guard": config.enable_hypo_guard,
                "hypo_guard_mmol_L": config.hypo_guard_mmol,
                "suppress_meal_bolus_on_guard": config.suppress_meal_bolus_on_guard,
                "enable_hypo_rescue": config.enable_hypo_rescue,
                "hypo_rescue_trigger_mmol_L": config.hypo_rescue_trigger_mmol,
                "hypo_rescue_gain_per_min": config.hypo_rescue_gain_per_min,
                "solver_method": config.solver_method,
                "solver_max_step": config.solver_max_step,
                "effective_insulin_sensitivity_min_g_U": 10.0,
                "effective_insulin_sensitivity_max_g_U": 14.0,
                "si3_ratio_scaling_min": 0.85,
                "si3_ratio_scaling_max": 1.15,
                "sampled_patients": sampled_patients,
                "accepted_patients": accepted_patients,
                "rejected_patients": rejected_patients,
                "rejected_initial_glucose": rejected_initial_glucose,
                "rejected_instability": rejected_instability,
                "rejection_rate_percent": round(rejection_rate_pct, 2),
                "initial_glucose_acceptance_min_mmol_L": rejection_bounds_mmol[0],
                "initial_glucose_acceptance_max_mmol_L": rejection_bounds_mmol[1],
                "instability_max_glucose_mmol_L": instability_max_glucose_mmol,
                "instability_hyper_pct_threshold": instability_hyper_pct,
            }
            
            export_to_formats(
                results_tot, 
                accepted_patients, 
                config.n_days, 
                now_sim_folder_path, 
                export_config.to_list(),
                config_metadata=config_metadata
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
            label=f'Mean Population BG (n={accepted_patients})',
            zorder=100
        )

    # Format plot
    if config.international_unit:
        plt.axhline(3.9, color='red', linestyle='--', linewidth=1.5, label='Hypoglycemia (3.9 mmol/L)', alpha=0.7)  # type: ignore[misc]
        plt.axhline(10.0, color='orange', linestyle='--', linewidth=1.5, label='Hyperglycemia (10 mmol/L)', alpha=0.7)  # type: ignore[misc]
        plt.ylim(3, 16.5)  # type: ignore[misc]
        plt.ylabel("Blood Glucose (mmol/L)", fontsize=12)  # type: ignore[misc]
    else:
        plt.axhline(70, color='red', linestyle='--', linewidth=1.5, label='Hypoglycemia (70 mg/dL)', alpha=0.7)  # type: ignore[misc]
        plt.axhline(180, color='orange', linestyle='--', linewidth=1.5, label='Hyperglycemia (180 mg/dL)', alpha=0.7)  # type: ignore[misc]
        plt.ylim(54, 300)  # type: ignore[misc]
        plt.ylabel("Blood Glucose (mg/dL)", fontsize=12)  # type: ignore[misc]
    
    # Add vertical lines to separate days
    for day in range(1, config.n_days):
        plt.axvline(24 * day, color='gray', linestyle=':', alpha=0.3)  # type: ignore[misc]

    plt.title(  # type: ignore[misc]
        f"Hovorka Model Monte Carlo Simulation\n"
        f"{accepted_patients} accepted / {sampled_patients} sampled patients × {config.n_days} days, "
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
    if now_sim_folder_path and any(export_config.to_list()):
        try:
            plt.savefig(now_sim_folder_path / "simulation_plot.png", dpi=150, bbox_inches='tight')  # type: ignore[misc]
            print(f"\nResults saved to: {now_sim_folder_path}")
        except Exception as e:
            print(f"Warning: Failed to save plot: {e}")
    
    # Show plot if interactive backend is available
    backend_name = plt.get_backend().lower()  # type: ignore[misc]
    if "agg" in backend_name:
        print("Plot saved to file.")
    else:
        plt.show()  # type: ignore[misc]


# =============================================================================
# Short-Duration Simulation & Parameter Identification Utilities
# =============================================================================

def simulate_duration(
    initial_state: np.ndarray,
    params: ParameterSet,
    duration_minutes: int,
    basal_hourly: float,
    bolus_mU: float = 0.0,
    bolus_duration_min: int = 1,
    cho_mg: float = 0.0,
    cho_duration_min: int = 15,
    cho_start_min: int = 0,
    solver_method: str = "RK45",
    solver_max_step: float = 1.0,
    clip_states: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Simulate the Hovorka model for a short duration with explicit inputs.

    Uses a simple inline input function (no meal schedules or scenario caching)
    that delivers basal insulin, an optional bolus, and an optional CHO load.

    Parameters:
    -----------
    initial_state: 10-element state vector [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2]
    params: patient ParameterSet
    duration_minutes: simulation length [min]
    basal_hourly: basal insulin rate [U/hr]
    bolus_mU: total bolus insulin [mU] delivered over bolus_duration_min starting at t=0
    bolus_duration_min: duration of bolus delivery [min] (default 1 = near-instantaneous)
    cho_mg: total carbohydrate load [mg] delivered over cho_duration_min
    cho_duration_min: duration of CHO intake [min] (default 15)
    cho_start_min: when CHO intake starts [min] (default 0)
    solver_method: ODE solver method (default RK45)
    solver_max_step: max ODE step [min] (default 1.0)
    clip_states: clip non-negative states to >= 0

    Returns:
    --------
    (final_state, final_glycemia_mmol): state at end, noise-free glucose in mmol/L
    """
    basal_mU_min = basal_hourly * 1000.0 / 60.0
    bolus_rate = bolus_mU / max(1, bolus_duration_min)  # [mU/min]
    cho_rate = cho_mg / max(1, cho_duration_min)  # [mg/min]

    def input_func(
        t: int,
        **_kwargs: object,
    ) -> tuple[float, float]:
        """Inline input: basal + optional bolus + optional CHO."""
        u = basal_mU_min
        d = 0.0
        # Bolus: delivered during [0, bolus_duration_min)
        if 0 <= t < bolus_duration_min:
            u += bolus_rate
        # CHO: delivered during [cho_start_min, cho_start_min + cho_duration_min)
        if cho_start_min <= t < cho_start_min + cho_duration_min:
            d = cho_rate
        return u, d

    t_eval = np.arange(0, duration_minutes + 1)

    def ode_func(t: float, x: np.ndarray) -> np.ndarray:
        x_safe = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        x_safe = np.clip(x_safe, -1e6, 1e6)
        result = hovorka_equations(
            int(t),
            x_safe.tolist(),
            params,
            input_func,
            scenario=0,
        )
        dy = np.asarray(result, dtype=np.float64)
        dy = np.nan_to_num(dy, nan=0.0, posinf=1e5, neginf=-1e5)
        dy = np.clip(dy, -1e5, 1e5)
        return dy

    sol = solve_ivp(
        ode_func,
        (0, duration_minutes),
        np.asarray(initial_state, dtype=np.float64),
        method=solver_method,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        max_step=solver_max_step,
    )

    state_traj = np.asarray(sol.y, dtype=np.float64)
    state_traj = np.nan_to_num(state_traj, nan=0.0, posinf=1e6, neginf=0.0)
    if clip_states:
        state_traj = _clip_state_trajectory(state_traj)

    final_state = state_traj[:, -1]
    final_glycemia = float(measure_glycemia(
        tuple(final_state.tolist()),
        params,
        noise_std=0.0,
        output_unit='mmol/L',
    ))

    return final_state, final_glycemia


def find_icr(
    params: ParameterSet,
    cho_grams: float = 50.0,
    target_glycemia_mmol: float = 5.5,
    measurement_time_min: int = 180,
    initial_glucose_mmol: float = 5.5,
    tolerance_mmol: float = 0.3,
    max_iterations: int = 40,
    print_progress: bool = False,
) -> dict[str, float]:
    """
    Find the insulin-to-carb ratio (ICR) for a patient via bisection.

    Procedure:
      1. Initialize at steady-state for initial_glucose_mmol
      2. Give a fixed CHO meal (cho_grams) and a trial bolus
      3. Simulate for measurement_time_min (default 3h)
      4. Adjust bolus via bisection until final glycemia ≈ target_glycemia_mmol

    Parameters:
    -----------
    params: patient ParameterSet
    cho_grams: carbohydrate load [g] (default 50)
    target_glycemia_mmol: desired postprandial glucose [mmol/L]
    measurement_time_min: time after meal to measure glucose [min] (default 180)
    initial_glucose_mmol: starting glycemia [mmol/L] (default 5.5)
    tolerance_mmol: convergence tolerance [mmol/L]
    max_iterations: max bisection iterations
    print_progress: print each iteration

    Returns:
    --------
    dict with keys: icr_g_per_U, bolus_U, final_glycemia_mmol, basal_hourly_U
    """
    # 1. Compute steady state at initial_glucose_mmol
    x0 = compute_optimal_steady_state_from_glucose(
        params,
        initial_glucose_mmol,
        international_units=True,
        max_iterations=100,
        print_progress=False,
    )
    x0_arr = np.array(x0, dtype=np.float64)

    # Derive calibrated basal from steady state
    tau_i = float(params["tauI"])
    us_calibrated_mU_min = float(x0_arr[2]) / tau_i if tau_i > 0 else 0.5 * 1000.0 / 60.0
    basal_hourly = us_calibrated_mU_min * 60.0 / 1000.0

    cho_mg = cho_grams * 1000.0  # g -> mg

    # Bisection bounds for bolus: 0 to 30 U (= 30_000 mU)
    bolus_low_mU = 0.0
    bolus_high_mU = 30_000.0

    best_bolus_mU = 0.0
    best_glycemia = float('inf')
    best_err = float('inf')

    for i in range(max_iterations):
        trial_bolus_mU = 0.5 * (bolus_low_mU + bolus_high_mU)

        _, final_g = simulate_duration(
            initial_state=x0_arr,
            params=params,
            duration_minutes=measurement_time_min,
            basal_hourly=basal_hourly,
            bolus_mU=trial_bolus_mU,
            bolus_duration_min=1,
            cho_mg=cho_mg,
            cho_duration_min=15,
            cho_start_min=0,
        )

        err = abs(final_g - target_glycemia_mmol)
        if print_progress:
            trial_U = trial_bolus_mU / 1000.0
            print(f"  ICR iter {i+1}: bolus={trial_U:.3f} U, final_G={final_g:.2f} mmol/L, err={err:.3f}")

        if err < best_err:
            best_err = err
            best_bolus_mU = trial_bolus_mU
            best_glycemia = final_g

        if err < tolerance_mmol:
            break

        # If final glucose is too high → need more insulin → raise lower bound
        if final_g > target_glycemia_mmol:
            bolus_low_mU = trial_bolus_mU
        else:
            bolus_high_mU = trial_bolus_mU

    bolus_U = best_bolus_mU / 1000.0
    icr = cho_grams / bolus_U if bolus_U > 0 else float('inf')

    return {
        "icr_g_per_U": round(icr, 3),
        "bolus_U": round(bolus_U, 4),
        "final_glycemia_mmol": round(best_glycemia, 3),
        "basal_hourly_U": round(basal_hourly, 4),
    }


def find_isf(
    params: ParameterSet,
    initial_glucose_mmol: float = 13.0,
    target_glycemia_mmol: float = 5.5,
    measurement_time_min: int = 120,
    tolerance_mmol: float = 0.3,
    max_iterations: int = 40,
    print_progress: bool = False,
) -> dict[str, float]:
    """
    Find the insulin sensitivity factor (ISF) for a patient via bisection.

    Procedure:
      1. Initialize at steady-state for initial_glucose_mmol (e.g. 13 mmol/L)
      2. Give a correction bolus (no carbs)
      3. Simulate for measurement_time_min (default 2h)
      4. Adjust bolus via bisection until final glycemia ≈ target_glycemia_mmol
      5. ISF = glucose_drop / bolus_U

    Parameters:
    -----------
    params: patient ParameterSet
    initial_glucose_mmol: starting glycemia [mmol/L] (default 13.0)
    target_glycemia_mmol: desired final glucose [mmol/L] (default 5.5)
    measurement_time_min: time after bolus to measure glucose [min] (default 120)
    tolerance_mmol: convergence tolerance [mmol/L]
    max_iterations: max bisection iterations
    print_progress: print each iteration

    Returns:
    --------
    dict with keys: isf_mmol_per_U, bolus_U, final_glycemia_mmol, glucose_drop_mmol, basal_hourly_U
    """
    # 1. Compute steady state at initial_glucose_mmol
    x0 = compute_optimal_steady_state_from_glucose(
        params,
        initial_glucose_mmol,
        international_units=True,
        max_iterations=100,
        print_progress=False,
    )
    x0_arr = np.array(x0, dtype=np.float64)

    # Measure actual initial glycemia (may differ slightly from desired due to bisection tolerance)
    actual_initial_g = float(measure_glycemia(
        tuple(x0_arr.tolist()),
        params,
        noise_std=0.0,
        output_unit='mmol/L',
    ))

    # Derive calibrated basal from steady state
    tau_i = float(params["tauI"])
    us_calibrated_mU_min = float(x0_arr[2]) / tau_i if tau_i > 0 else 0.5 * 1000.0 / 60.0
    basal_hourly = us_calibrated_mU_min * 60.0 / 1000.0

    # Bisection bounds for correction bolus: 0 to 30 U (= 30_000 mU)
    bolus_low_mU = 0.0
    bolus_high_mU = 30_000.0

    best_bolus_mU = 0.0
    best_glycemia = float('inf')
    best_err = float('inf')

    for i in range(max_iterations):
        trial_bolus_mU = 0.5 * (bolus_low_mU + bolus_high_mU)

        _, final_g = simulate_duration(
            initial_state=x0_arr,
            params=params,
            duration_minutes=measurement_time_min,
            basal_hourly=basal_hourly,
            bolus_mU=trial_bolus_mU,
            bolus_duration_min=1,
            cho_mg=0.0,  # No carbs — pure correction
            cho_duration_min=1,
            cho_start_min=0,
        )

        err = abs(final_g - target_glycemia_mmol)
        if print_progress:
            trial_U = trial_bolus_mU / 1000.0
            drop = actual_initial_g - final_g
            print(f"  ISF iter {i+1}: bolus={trial_U:.3f} U, final_G={final_g:.2f} mmol/L, drop={drop:.2f}, err={err:.3f}")

        if err < best_err:
            best_err = err
            best_bolus_mU = trial_bolus_mU
            best_glycemia = final_g

        if err < tolerance_mmol:
            break

        # If final glucose is still too high → need more insulin
        if final_g > target_glycemia_mmol:
            bolus_low_mU = trial_bolus_mU
        else:
            bolus_high_mU = trial_bolus_mU

    bolus_U = best_bolus_mU / 1000.0
    glucose_drop = actual_initial_g - best_glycemia
    isf = glucose_drop / bolus_U if bolus_U > 0 else float('inf')

    return {
        "isf_mmol_per_U": round(isf, 3),
        "bolus_U": round(bolus_U, 4),
        "final_glycemia_mmol": round(best_glycemia, 3),
        "glucose_drop_mmol": round(glucose_drop, 3),
        "basal_hourly_U": round(basal_hourly, 4),
    }