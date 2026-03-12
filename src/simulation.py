# Simulation Module

# Library Imports
from __future__ import annotations
from typing import Any
import numpy as np  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]
from tqdm import tqdm

# --- Imports from src ---
from src.model import hovorka_equations, compute_optimal_steady_state_from_glucose, ParameterSet
from src.parameters import generate_monte_carlo_patients
from src.input import scenario_with_cached_meals, N_SCENARIOS, clear_meal_cache
from src.sensor import measure_glycemia
from src.export import export_to_formats, ExportConfig
from src.sensitivity import find_icr, find_isf
from src.simulation_config import SimulationConfig
from src.simulation_control import (
    ControllerState,
    apply_guard_iob_isf,
    apply_hypo_rescue_to_derivative,
    count_correction_active_points,
    estimate_iob_from_state,
)
from src.simulation_utils import (
    clip_state_trajectory,
    create_export_directory,
    generate_autocorrelated_noise,
    get_patient_color,
    measure_glycemia_day,
)


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
    now_sim_folder_path = create_export_directory() if any(export_config.to_list()) else None

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
    rejected_quality_hypo = 0
    rejected_quality_hyper = 0
    accepted_total_points = 0
    accepted_guard_active_points = 0
    accepted_rescue_active_points = 0
    accepted_iob_guard_active_points = 0
    accepted_correction_isf_active_points = 0
    accepted_correction_isf_events = 0
    accepted_correction_isf_units = 0.0
    rejection_bounds_mmol = (4.5, 7.2)
    instability_max_glucose_mmol = 17.0
    instability_hyper_pct = 30.0
    quality_max_hypo_pct = 4.0
    quality_max_hyper_pct = 12.0

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
            # TODO: put a range of good glycemias
            x0_initial = compute_optimal_steady_state_from_glucose(
                patient_params, 
                config.initial_target_glucose_mgdl,
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

            # Compute ICR and ISF (sensitivity factors)
            insulin_carbo_ratio_patient = find_icr(params=patient_params, initial_icr=config.init_insulin_carbo_ratio, print_progress=False)
            insulin_sensitivity_patient = find_isf(params=patient_params, initial_isf=config.init_insulin_sensitivity_factor, print_progress=False)

            # Store into patient params
            patient_params["ICR"] = insulin_carbo_ratio_patient
            patient_params["ISF"] = insulin_sensitivity_patient

            # Track state across days
            current_state: np.ndarray = np.array(x0_initial, dtype=np.float64)
            patient_full_trajectory_noisy: list[np.ndarray] = []
            patient_full_trajectory_physio: list[np.ndarray] = []
            patient_total_points = 0
            patient_guard_active_points = 0
            patient_rescue_active_points = 0
            patient_iob_guard_active_points = 0
            patient_correction_isf_active_points = 0
            patient_correction_isf_events = 0
            patient_correction_isf_units = 0.0
            previous_noise_value: float | None = None
            controller_state = ControllerState()
        
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
                    current_min = int(np.floor(t))
                    # Use absolute simulation minute to keep latch timers consistent across days.
                    current_abs_min = day_idx * minutes_per_day + current_min
                    g_est = measure_glycemia(tuple(x_safe.tolist()), patient_params, noise_std=0.0, output_unit='mmol/L')

                    # Basic insulin-on-board (IOB) estimate from subcutaneous depots [U].
                    # S1 and S2 are insulin masses in mU, so divide by 1000 to get units.
                    iob_u = max(0.0, float(x_safe[2]) + float(x_safe[3])) / 1000.0
                    basal_hourly_effective, insulin_carbo_ratio_effective, _guard_latched = apply_guard_iob_isf(
                        current_abs_min=current_abs_min,
                        g_est=g_est,
                        iob_u=iob_u,
                        basal_hourly_patient=basal_hourly_patient,
                        insulin_carbo_ratio_patient=insulin_carbo_ratio_patient,
                        insulin_sensitivity_patient=insulin_sensitivity_patient,
                        config=config,
                        state=controller_state,
                    )

                    result = hovorka_equations(
                        int(t),
                        x_safe.tolist(),
                        patient_params,
                        scenario_with_cached_meals,
                        scenario=scenario,
                        patient_id=sim_patient_id,
                        day=day_idx,
                        basal_hourly=basal_hourly_effective,
                        insulin_carbo_ratio=insulin_carbo_ratio_effective,
                        meal_schedule=None,
                        seed=config.random_seed,
                    )
                    dy = np.asarray(result, dtype=np.float64)
                    apply_hypo_rescue_to_derivative(
                        dy=dy,
                        current_abs_min=current_abs_min,
                        g_est=g_est,
                        patient_params=patient_params,
                        config=config,
                        state=controller_state,
                    )

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
                    state_trajectory = clip_state_trajectory(state_trajectory)
            
                # Update current state for next day (continuity)
                current_state = np.asarray(state_trajectory[:, -1], dtype=np.float64)
            
                # Generate autocorrelated CGM noise for this day
                n_measurements = len(t_eval_day)
                noise_sequence = generate_autocorrelated_noise(
                    n_measurements,
                    config.noise_std,
                    config.noise_autocorr,
                    rng,
                    initial_value=previous_noise_value,
                )
                previous_noise_value = float(noise_sequence[-1]) if noise_sequence.size > 0 else previous_noise_value

                glycemia_day_array = measure_glycemia_day(
                    state_trajectory=state_trajectory,
                    patient_params=patient_params,
                    noise_sequence=noise_sequence,
                    n_measurements=n_measurements,
                )
                glycemia_day_physio = measure_glycemia_day(
                    state_trajectory=state_trajectory,
                    patient_params=patient_params,
                    noise_sequence=np.zeros(n_measurements, dtype=np.float64),
                    n_measurements=n_measurements,
                )
                glycemia_day_physio_mmol = glycemia_day_physio.copy()
            
                # Convert units if needed
                if not config.international_unit:
                    glycemia_day_array = glycemia_day_array * (float(patient_params['MwG']) / 10.0)  # mmol/L -> mg/dL
                    glycemia_day_physio = glycemia_day_physio * (float(patient_params['MwG']) / 10.0)  # mmol/L -> mg/dL
            
                # Store results for this day
                results_tot[sim_patient_id]["days"][day_idx] = glycemia_day_array
                # Day 0 is kept in full (0..1440 = 1441 points).
                # Subsequent days drop their first point (minute 0 = same physical
                # timestamp as minute 1440 of the previous day) to avoid a duplicate
                # in the concatenated trajectory that would create a visible "kink".
                noisy_segment = glycemia_day_array if day_idx == 0 else glycemia_day_array[1:]
                physio_segment = glycemia_day_physio if day_idx == 0 else glycemia_day_physio[1:]
                physio_segment_mmol = glycemia_day_physio_mmol if day_idx == 0 else glycemia_day_physio_mmol[1:]
                iob_day_u = estimate_iob_from_state(state_trajectory)
                iob_segment_u = iob_day_u if day_idx == 0 else iob_day_u[1:]
                patient_full_trajectory_noisy.append(noisy_segment)
                patient_full_trajectory_physio.append(physio_segment)
                patient_total_points += int(physio_segment_mmol.size)
                patient_guard_active_points += int(np.sum(physio_segment_mmol <= config.hypo_guard_mmol))
                patient_rescue_active_points += int(np.sum(physio_segment_mmol < config.hypo_rescue_trigger_mmol))
                if config.enable_iob_bolus_guard:
                    patient_iob_guard_active_points += int(np.sum(iob_segment_u > config.iob_guard_units))

            # Absolute-minute end of this simulated horizon, used to clip correction windows.
            horizon_end_abs_min = config.n_days * minutes_per_day
            patient_correction_isf_active_points += count_correction_active_points(
                windows_abs=controller_state.correction_isf_windows_abs,
                horizon_end_abs_min=horizon_end_abs_min,
            )
            patient_correction_isf_events = controller_state.correction_isf_events
            patient_correction_isf_units = controller_state.correction_isf_units
        
            # Concatenate all days for this patient
            patient_full_trajectory_noisy_concat = np.concatenate(patient_full_trajectory_noisy, dtype=np.float64)  # type: ignore[arg-type]
            patient_full_trajectory_physio_concat = np.concatenate(patient_full_trajectory_physio, dtype=np.float64)  # type: ignore[arg-type]

            hypo_count = int(np.sum(patient_full_trajectory_physio_concat < 3.9))
            hyper_count = int(np.sum(patient_full_trajectory_physio_concat > 10.0))
            total_count = int(patient_full_trajectory_physio_concat.size)
            hypo_pct = (100.0 * hypo_count / total_count) if total_count > 0 else 0.0
            hyper_pct = (100.0 * hyper_count / total_count) if total_count > 0 else 0.0
            max_glucose = float(np.max(patient_full_trajectory_physio_concat)) if total_count > 0 else 0.0
            if hyper_pct > instability_hyper_pct or max_glucose > instability_max_glucose_mmol:
                rejected_patients += 1
                rejected_instability += 1
                del results_tot[sim_patient_id]
                continue
            if hypo_pct > quality_max_hypo_pct:
                rejected_patients += 1
                rejected_quality_hypo += 1
                del results_tot[sim_patient_id]
                continue
            if hyper_pct > quality_max_hyper_pct:
                rejected_patients += 1
                rejected_quality_hyper += 1
                del results_tot[sim_patient_id]
                continue

            accepted_patients += 1
            all_patient_trajectories.append(patient_full_trajectory_physio_concat)
            accepted_total_points += patient_total_points
            accepted_guard_active_points += patient_guard_active_points
            accepted_rescue_active_points += patient_rescue_active_points
            accepted_iob_guard_active_points += patient_iob_guard_active_points
            accepted_correction_isf_active_points += patient_correction_isf_active_points
            accepted_correction_isf_events += patient_correction_isf_events
            accepted_correction_isf_units += patient_correction_isf_units
            pbar.update(1)
        
            # Plot patient trajectory with aesthetically pleasing colors
            time_hours = np.arange(len(patient_full_trajectory_noisy_concat)) / 60.0
            patient_color = get_patient_color(sim_patient_id, max(1, config.n_patients))
            plt.plot(time_hours, patient_full_trajectory_noisy_concat, color=patient_color[:3], alpha=patient_color[3])  # type: ignore[misc]

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
        f"initial_glucose={rejected_initial_glucose}, instability={rejected_instability}, "
        f"quality_hypo={rejected_quality_hypo}, quality_hyper={rejected_quality_hyper}"
    )
    guard_active_pct = (100.0 * accepted_guard_active_points / accepted_total_points) if accepted_total_points > 0 else 0.0
    rescue_active_pct = (100.0 * accepted_rescue_active_points / accepted_total_points) if accepted_total_points > 0 else 0.0
    iob_guard_active_pct = (100.0 * accepted_iob_guard_active_points / accepted_total_points) if accepted_total_points > 0 else 0.0
    correction_isf_active_pct = (100.0 * accepted_correction_isf_active_points / accepted_total_points) if accepted_total_points > 0 else 0.0
    print(
        "Safety controls activity (accepted cohort): "
        f"guard_active={guard_active_pct:.2f}%, rescue_active={rescue_active_pct:.2f}%, "
        f"iob_guard_active={iob_guard_active_pct:.2f}%, correction_isf_active={correction_isf_active_pct:.2f}%"
    )
    avg_correction_isf_events_per_patient = (accepted_correction_isf_events / accepted_patients) if accepted_patients > 0 else 0.0
    avg_correction_isf_units_per_patient = (accepted_correction_isf_units / accepted_patients) if accepted_patients > 0 else 0.0
    print(
        "ISF correction summary (accepted cohort): "
        f"events={accepted_correction_isf_events}, total_units={accepted_correction_isf_units:.2f} U, "
        f"avg_events_per_patient={avg_correction_isf_events_per_patient:.2f}, "
        f"avg_units_per_patient={avg_correction_isf_units_per_patient:.2f} U"
    )

    # Export results if requested
    if now_sim_folder_path and any(export_config.to_list()):
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
                "init_insulin_carbo_ratio_g_U": config.init_insulin_carbo_ratio,
                "init_insulin_sensitivity_factor_mmol_U": config.init_insulin_sensitivity_factor,
                "enable_iob_bolus_guard": config.enable_iob_bolus_guard,
                "iob_guard_units": config.iob_guard_units,
                "iob_full_attenuation_units": config.iob_full_attenuation_units,
                "iob_max_icr_multiplier": config.iob_max_icr_multiplier,
                "enable_correction_isf": config.enable_correction_isf,
                "correction_isf_target_mmol": config.correction_isf_target_mmol,
                "correction_isf_cooldown_min": config.correction_isf_cooldown_min,
                "correction_isf_max_bolus_units": config.correction_isf_max_bolus_units,
                "correction_isf_min_bolus_units": config.correction_isf_min_bolus_units,
                "correction_isf_bolus_duration_min": config.correction_isf_bolus_duration_min,
                "correction_isf_iob_free_units": config.correction_isf_iob_free_units,
                "initial_target_glucose_mgdl": config.initial_target_glucose_mgdl,
                "enable_hypo_guard": config.enable_hypo_guard,
                "hypo_guard_mmol_L": config.hypo_guard_mmol,
                "hypo_guard_retrigger_cooldown_min": config.hypo_guard_retrigger_cooldown_min,
                "suppress_meal_bolus_on_guard": config.suppress_meal_bolus_on_guard,
                "enable_hypo_rescue": config.enable_hypo_rescue,
                "hypo_rescue_trigger_mmol_L": config.hypo_rescue_trigger_mmol,
                "hypo_guard_suspend_min": config.hypo_guard_suspend_min,
                "hypo_rescue_carbs_g": config.hypo_rescue_carbs_g,
                "hypo_rescue_duration_min": config.hypo_rescue_duration_min,
                "hypo_rescue_retrigger_cooldown_min": config.hypo_rescue_retrigger_cooldown_min,
                "solver_method": config.solver_method,
                "solver_max_step": config.solver_max_step,
                "effective_insulin_carbo_ratio_min_g_U": 10.0,
                "effective_insulin_carbo_ratio_max_g_U": 14.0,
                "si3_ratio_scaling_min": 0.85,
                "si3_ratio_scaling_max": 1.15,
                "sampled_patients": sampled_patients,
                "accepted_patients": accepted_patients,
                "rejected_patients": rejected_patients,
                "rejected_initial_glucose": rejected_initial_glucose,
                "rejected_instability": rejected_instability,
                "rejected_quality_hypo": rejected_quality_hypo,
                "rejected_quality_hyper": rejected_quality_hyper,
                "rejection_rate_percent": round(rejection_rate_pct, 2),
                "guard_active_percent_accepted": round(guard_active_pct, 3),
                "rescue_active_percent_accepted": round(rescue_active_pct, 3),
                "iob_guard_active_percent_accepted": round(iob_guard_active_pct, 3),
                "correction_isf_active_percent_accepted": round(correction_isf_active_pct, 3),
                "guard_active_points_accepted": accepted_guard_active_points,
                "rescue_active_points_accepted": accepted_rescue_active_points,
                "iob_guard_active_points_accepted": accepted_iob_guard_active_points,
                "correction_isf_active_points_accepted": accepted_correction_isf_active_points,
                "correction_isf_events_accepted": accepted_correction_isf_events,
                "correction_isf_total_units_accepted": round(accepted_correction_isf_units, 4),
                "total_points_accepted": accepted_total_points,
                "initial_glucose_acceptance_min_mmol_L": rejection_bounds_mmol[0],
                "initial_glucose_acceptance_max_mmol_L": rejection_bounds_mmol[1],
                "instability_max_glucose_mmol_L": instability_max_glucose_mmol,
                "instability_hyper_pct_threshold": instability_hyper_pct,
                "quality_max_hypo_pct_threshold": quality_max_hypo_pct,
                "quality_max_hyper_pct_threshold": quality_max_hyper_pct,
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
