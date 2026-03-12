import numpy as np
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

from src.parameters import get_base_params
from src.model import ParameterSet, hovorka_equations, compute_optimal_steady_state_from_glucose, measure_glycemia
from src.simulation_utils import clip_state_trajectory

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

    sol = solve_ivp(  # type: ignore[unknown-variable-type]
        ode_func,
        (0, duration_minutes),
        np.asarray(initial_state, dtype=np.float64),
        method=solver_method,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        max_step=solver_max_step,
    )

    state_traj = np.asarray(sol.y, dtype=np.float64)  # type: ignore[union-attr]
    state_traj = np.nan_to_num(state_traj, nan=0.0, posinf=1e6, neginf=0.0)
    if clip_states:
        state_traj = clip_state_trajectory(state_traj)

    final_state = state_traj[:, -1]
    final_glycemia = float(measure_glycemia(
        tuple(final_state.tolist()),
        params,
        noise_std=0.0,
        output_unit='mmol/L',
    ))

    return final_state, final_glycemia

def find_insulin_carbo_ratio(
    params: ParameterSet,
    initial_icr: float = 19.3,
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
    initial_icr: initial guess for ICR [g/U] (default 19.3)
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

    best_bolus_mU = cho_mg / initial_icr
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

def find_insulin_sensitivity_factor(
    params: ParameterSet,
    initial_isf: float = 3.1,
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
    initial_isf: initial guess for ISF [mmol/L/U] (default 3.1)
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

    best_bolus_mU = (initial_glucose_mmol - target_glycemia_mmol) / initial_isf * 1000
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


def find_sensitivities(
    p: ParameterSet=get_base_params(),
    cho_grams: float=50.0,
    print_progress: bool=False
) -> tuple[float, float]:
    """
    Find ICR and ISF for a given parameter set.

    Parameters:
    -----------
    p: ParameterSet
        Parameter set for the patient.
    cho_grams: float
        Carbohydrate grams to use for ICR calculation.
    print_progress: bool
        Whether to print progress during calculation.
    
    Returns:
    --------
    icr: float
        Insulin-to-carbohydrate ratio.
    isf: float
        Insulin sensitivity factor.
    """
    # Find ICR
    icr = find_icr(p, cho_grams=cho_grams, print_progress=print_progress)
    # Find ISF
    isf = find_isf(p, print_progress=print_progress)
    
    # Return the dictionaries
    return icr, isf



def find_icr(
    params: ParameterSet,
    initial_icr: float = 19.3,
    cho_grams: float = 50.0,
    target_glycemia_mmol: float = 5.5,
    measurement_time_min: int = 180,
    initial_glucose_mmol: float = 5.5,
    tolerance_mmol: float = 0.3,
    max_iterations: int = 40,
    print_progress: bool = False,
) -> float:
    
    # Find ICR dictionary
    icr_dict = find_insulin_carbo_ratio(
        params=params,
        initial_icr=initial_icr,
        cho_grams=cho_grams,
        target_glycemia_mmol=target_glycemia_mmol,
        measurement_time_min=measurement_time_min,
        initial_glucose_mmol=initial_glucose_mmol,
        tolerance_mmol=tolerance_mmol,
        max_iterations=max_iterations,
        print_progress=print_progress,
    )

    # Return the value
    return icr_dict["icr_g_per_U"]

def find_isf(
    params: ParameterSet,
    initial_isf: float = 3.1,
    initial_glucose_mmol: float = 13.0,
    target_glycemia_mmol: float = 5.5,
    measurement_time_min: int = 180,
    tolerance_mmol: float = 0.3,
    max_iterations: int = 40,
    print_progress: bool = False,
) -> float:
    
    # Find ISF dictionary
    isf_dict = find_insulin_sensitivity_factor(
        params=params,
        initial_isf=initial_isf,
        initial_glucose_mmol=initial_glucose_mmol,
        target_glycemia_mmol=target_glycemia_mmol,
        measurement_time_min=measurement_time_min,
        tolerance_mmol=tolerance_mmol,
        max_iterations=max_iterations,
        print_progress=print_progress,
    )

    # Return the value
    return isf_dict["isf_mmol_per_U"]
