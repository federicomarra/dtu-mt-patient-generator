from __future__ import annotations

import importlib
from typing import Any, Callable, Protocol, cast

import numpy as np

from src.hovorka_exercise import compute_rashid_terms
from src.sensor import measure_glycemia

ParameterSet = dict[str, float]
StateVector = list[float]
StateArray = np.ndarray
InputValues = tuple[float, float] | tuple[float, float, float]
InputFunc = Callable[..., InputValues]


class _NewtonSolver(Protocol):
    def __call__(
        self,
        func: Callable[[float], float],
        x0: float,
        fprime: Callable[[float], float] | None = None,
        args: tuple[Any, ...] = (),
        tol: float = 1.48e-8,
        maxiter: int = 50,
        fprime2: Callable[[float], float] | None = None,
        x1: float | None = None,
        rtol: float = 0.0,
        full_output: bool = False,
        disp: bool = True,
    ) -> float: ...


newton = cast(_NewtonSolver, cast(Any, importlib.import_module("scipy.optimize")).newton)

_HOVORKA_BASE_STATE_COUNT = 10
_HOVORKA_STATE_COUNT = 13


def state_listify(
    Q1: float,
    Q2: float,
    S1: float,
    S2: float,
    I: float,
    x1: float,
    x2: float,
    x3: float,
    D1: float,
    D2: float,
    E1: float = 0.0,
    E2: float = 0.0,
    TE: float = 0.0,
) -> list[float]:
    return [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2, E1, E2, TE]


def state_unlistify(
    x: StateVector,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
    if len(x) == _HOVORKA_BASE_STATE_COUNT:
        q1, q2, s1, s2, i, x1, x2, x3, d1, d2 = x
        return q1, q2, s1, s2, i, x1, x2, x3, d1, d2, 0.0, 0.0, 0.0
    if len(x) != _HOVORKA_STATE_COUNT:
        raise ValueError(f"Expected {_HOVORKA_BASE_STATE_COUNT} or {_HOVORKA_STATE_COUNT} Hovorka states, got {len(x)}")
    q1, q2, s1, s2, i, x1, x2, x3, d1, d2, e1, e2, te = x
    return q1, q2, s1, s2, i, x1, x2, x3, d1, d2, e1, e2, te


def _parse_input_values(values: InputValues) -> tuple[float, float, float]:
    if len(values) == 2:
        u_t, d_t = values
        return float(u_t), float(d_t), 0.0
    u_t, d_t, delta_hr_t = values
    return float(u_t), float(d_t), float(delta_hr_t)


def _get_input_values(
    t: int,
    input_func: InputFunc,
    scenario: int,
    patient_id: int,
    day: int,
    basal_hourly: float,
    insulin_carbo_ratio: float,
    patient_age_years: float,
    meal_schedule: dict[str, float] | None,
    seed: int | None,
    precomputed_inputs: InputValues | None,
) -> tuple[float, float, float]:
    if precomputed_inputs is not None:
        return _parse_input_values(precomputed_inputs)
    if meal_schedule is None:
        return _parse_input_values(
            input_func(
                t,
                patient_id=patient_id,
                day=day,
                basal_hourly=basal_hourly,
                scenario=scenario,
                insulin_carbo_ratio=insulin_carbo_ratio,
                patient_age_years=patient_age_years,
                seed=seed,
            )
        )
    return _parse_input_values(
        input_func(
            t,
            patient_id=patient_id,
            day=day,
            basal_hourly=basal_hourly,
            scenario=scenario,
            insulin_carbo_ratio=insulin_carbo_ratio,
            patient_age_years=patient_age_years,
            meal_schedule=meal_schedule,
            seed=seed,
        )
    )


def hovorka_equations(
    t: int,
    x: StateVector | StateArray,
    params: ParameterSet,
    input_func: InputFunc,
    scenario: int,
    patient_id: int = 0,
    day: int = 0,
    basal_hourly: float = 0.5,
    insulin_carbo_ratio: float = 2.0,
    patient_age_years: float = 35.0,
    meal_schedule: dict[str, float] | None = None,
    seed: int | None = None,
    precomputed_inputs: InputValues | None = None,
) -> StateVector:
    Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2, E1, E2, TE = state_unlistify(list(x))

    EGP0 = float(params["EGP0"])
    F01 = float(params["F01"])
    k12 = float(params["k12"])
    ka1 = float(params["ka1"])
    ka2 = float(params["ka2"])
    ka3 = float(params["ka3"])
    SI1 = float(params["SI1"])
    SI2 = float(params["SI2"])
    SI3 = float(params["SI3"])
    ke = float(params["ke"])
    VI = float(params["VI"])
    VG = float(params["VG"])
    tauI = float(params["tauI"])
    tauG = float(params["tauG"])
    Ag = float(params["Ag"])
    BW = float(params["BW"])

    u_t, d_t, delta_hr_t = _get_input_values(
        t=t,
        input_func=input_func,
        scenario=scenario,
        patient_id=patient_id,
        day=day,
        basal_hourly=basal_hourly,
        insulin_carbo_ratio=insulin_carbo_ratio,
        patient_age_years=patient_age_years,
        meal_schedule=meal_schedule,
        seed=seed,
        precomputed_inputs=precomputed_inputs,
    )
    U = u_t
    D = d_t / float(params["MwG"])

    G = Q1 / (VG * BW) if (VG * BW) > 0.0 else 0.0

    dD1 = (Ag * D) - ((1.0 / tauG) * D1)
    dD2 = (1.0 / tauG) * (D1 - D2)
    UG = (1.0 / tauG) * D2

    dS1 = U - ((1.0 / tauI) * S1)
    dS2 = (1.0 / tauI) * (S1 - S2)
    UI = (1.0 / tauI) * S2
    dI = (UI / (VI * BW)) - (ke * I)

    if G >= 4.5:
        F01c = F01 * BW
    else:
        F01c = max(0.0, F01 * BW * max(0.0, G) / 4.5)

    if G >= 9.0:
        fr = 0.003 * (G - 9.0) * VG * BW
    else:
        fr = 0.0

    kb1 = SI1 * ka1
    kb2 = SI2 * ka2
    kb3 = SI3 * ka3
    dx1 = kb1 * I - ka1 * x1
    dx2 = kb2 * I - ka2 * x2
    dx3 = kb3 * I - ka3 * x3

    dE1 = 0.0
    dE2 = 0.0
    dTE = 0.0

    # Exercise terms from A.16-A.19 and A.23-A.24.
    rashid = compute_rashid_terms(
        E1=E1,
        E2=E2,
        TE=TE,
        delta_hr_t=delta_hr_t,
        x1=x1,
        x2=x2,
        Q1=Q1,
        Q2=Q2,
        params=params,
    )
    dE1 = float(rashid["dE1"])
    dE2 = float(rashid["dE2"])
    dTE = float(rashid["dTE"])
    QE1 = float(rashid["QE1"])
    QE21 = float(rashid["QE21"])
    QE22 = float(rashid["QE22"])

    # Glucose model terms in figure notation.
    R12 = (x1 * Q1) - (k12 * Q2)
    R2 = x2 * Q2
    EGPc = EGP0 * BW * max(0.0, 1.0 - x3)

    dQ1 = UG + EGPc - R12 - F01c - fr - QE21
    dQ2 = R12 - R2 + QE21 - QE22 - QE1
    return [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2, dE1, dE2, dTE]


def compute_fasting_steady_state_from_basal_insulin(u_mu: float, params: ParameterSet) -> StateVector:
    BW = params["BW"]
    tauI = params["tauI"]
    ke = params["ke"]
    VI = params["VI"]
    SI1 = params["SI1"]
    SI2 = params["SI2"]
    SI3 = params["SI3"]
    EGP0 = params["EGP0"]
    F01 = params["F01"]
    k12 = params["k12"]

    Seq = tauI * u_mu
    Ieq = Seq / (ke * tauI * VI * BW)
    x1eq = SI1 * Ieq
    x2eq = SI2 * Ieq
    x3eq = SI3 * Ieq

    F01c = F01 * BW
    EGPc = EGP0 * BW * max(0.0, 1.0 - x3eq)
    A = np.array([[-x1eq, k12], [x1eq, -k12 - x2eq]], dtype=np.float64)
    b = np.array([F01c - EGPc, 0.0], dtype=np.float64)
    try:
        q = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        q, *_ = np.linalg.lstsq(A, b, rcond=None)

    return state_listify(
        Q1=max(0.0, float(q[0])),
        Q2=max(0.0, float(q[1])),
        S1=Seq,
        S2=Seq,
        I=Ieq,
        x1=x1eq,
        x2=x2eq,
        x3=x3eq,
        D1=0.0,
        D2=0.0,
    )


def _convert_target_glucose_to_mmol(
    params: ParameterSet,
    desired_glycemia: float | tuple[float, float],
    international_units: bool,
) -> tuple[float, float]:
    if isinstance(desired_glycemia, tuple):
        target_mmol = float(0.5 * (desired_glycemia[0] + desired_glycemia[1]))
    else:
        target_mmol = float(desired_glycemia)

    if international_units:
        return target_mmol, 0.1
    tolerance = 0.1 * (params["MwG"] / 10.0)
    return target_mmol / (params["MwG"] / 10.0), tolerance


def _evaluate_steady_state_glucose(us: float, params: ParameterSet) -> tuple[float, StateVector]:
    x_state = compute_fasting_steady_state_from_basal_insulin(us, params)
    glucose_mmol = measure_glycemia(tuple(x_state), params)
    return glucose_mmol, x_state


def _estimate_glucose_derivative(us: float, params: ParameterSet) -> float:
    step = max(1e-4, 1e-2 * us)
    lower = max(1e-6, us - step)
    upper = us + step
    glucose_lower, _ = _evaluate_steady_state_glucose(lower, params)
    glucose_upper, _ = _evaluate_steady_state_glucose(upper, params)
    denominator = upper - lower
    if denominator <= 0.0:
        return 0.0
    return (glucose_upper - glucose_lower) / denominator


def _find_newton_bracket(
    params: ParameterSet,
    target_mmol: float,
) -> tuple[float, float, float, float, StateVector, StateVector]:
    pivot = 10.0
    glucose_pivot, state_pivot = _evaluate_steady_state_glucose(pivot, params)

    if glucose_pivot >= target_mmol:
        low = pivot
        glucose_low = glucose_pivot
        state_low = state_pivot
        high = pivot * 2.0
        glucose_high, state_high = _evaluate_steady_state_glucose(high, params)
        while glucose_high > target_mmol and high < 100.0:
            low = high
            glucose_low = glucose_high
            state_low = state_high
            high *= 2.0
            glucose_high, state_high = _evaluate_steady_state_glucose(high, params)
    else:
        high = pivot
        glucose_high = glucose_pivot
        state_high = state_pivot
        low = pivot / 2.0
        glucose_low, state_low = _evaluate_steady_state_glucose(low, params)
        while glucose_low < target_mmol and low > 1e-6:
            high = low
            glucose_high = glucose_low
            state_high = state_low
            low = max(1e-6, low / 2.0)
            glucose_low, state_low = _evaluate_steady_state_glucose(low, params)

    return low, glucose_low, high, glucose_high, state_low, state_high


def _compute_hovorka_steady_state_newton(
    params: ParameterSet,
    desired_glycemia: float | tuple[float, float],
    international_units: bool = True,
    max_iterations: int = 100,
    print_progress: bool = True,
) -> StateVector:
    target_mmol, glucose_tolerance_mmol = _convert_target_glucose_to_mmol(
        params,
        desired_glycemia,
        international_units,
    )

    if print_progress:
        print(f"Computing optimal steady state with Newton's method for glucose: {target_mmol} [mmol/L]")

    (
        min_insulin_amount,
        glucose_low,
        max_insulin_amount,
        glucose_high,
        state_low,
        state_high,
    ) = _find_newton_bracket(params, target_mmol)

    best_state = state_low
    best_error = abs(glucose_low - target_mmol)
    if abs(glucose_high - target_mmol) < best_error:
        best_state = state_high
        best_error = abs(glucose_high - target_mmol)

    if not (glucose_low >= target_mmol >= glucose_high):
        return best_state

    evaluation_count = 0

    def objective(us: float) -> float:
        nonlocal best_state, best_error, evaluation_count
        insulin_amount = float(np.clip(us, min_insulin_amount, max_insulin_amount))
        glucose_mmol, state = _evaluate_steady_state_glucose(insulin_amount, params)
        residual = glucose_mmol - target_mmol
        error = abs(residual)
        evaluation_count += 1

        if error < best_error:
            best_error = error
            best_state = state

        if print_progress:
            print(
                f"Eval {evaluation_count}: G= {glucose_mmol:.4f} [mmol/L], "
                f"I= {insulin_amount:.6f} [mU/min], residual= {residual:.4f}"
            )
        return residual

    def objective_prime(us: float) -> float:
        insulin_amount = float(np.clip(us, min_insulin_amount, max_insulin_amount))
        derivative = _estimate_glucose_derivative(insulin_amount, params)
        if not np.isfinite(derivative):
            return 0.0
        return derivative

    initial_guess = float(
        np.clip(
            min_insulin_amount
            + (target_mmol - glucose_low)
            * (max_insulin_amount - min_insulin_amount)
            / (glucose_high - glucose_low),
            min_insulin_amount,
            max_insulin_amount,
        )
    )

    # SciPy's Newton `tol` is in root-variable units (here insulin mU/min),
    # not glucose units. Keep it small and use glucose_tolerance_mmol for
    # physiological acceptance after solving.
    insulin_step_tol = 1e-4

    try:
        insulin_solution = float(
            newton(
                func=objective,
                x0=initial_guess,
                fprime=objective_prime,
                tol=insulin_step_tol,
                maxiter=max_iterations,
            )
        )
        glucose_solution, state_solution = _evaluate_steady_state_glucose(
            float(np.clip(insulin_solution, min_insulin_amount, max_insulin_amount)),
            params,
        )
        if abs(glucose_solution - target_mmol) <= glucose_tolerance_mmol:
            return state_solution
        if abs(glucose_solution - target_mmol) < best_error:
            return state_solution
    except (RuntimeError, OverflowError, ZeroDivisionError):
        pass
    return best_state


def compute_optimal_steady_state_from_glucose(
    params: ParameterSet,
    desired_glycemia: float | tuple[float, float],
    international_units: bool = True,
    max_iterations: int = 100,
    print_progress: bool = True,
) -> StateVector:
    return _compute_hovorka_steady_state_newton(
        params=params,
        desired_glycemia=desired_glycemia,
        international_units=international_units,
        max_iterations=max_iterations,
        print_progress=print_progress,
    )


def get_glucose_from_state(state: StateVector | StateArray, params: ParameterSet) -> float:
    return measure_glycemia(tuple(float(value) for value in state), params)


def get_glucose_trace_from_trajectory(
    state_trajectory: np.ndarray,
    params: ParameterSet,
    effective: int,
) -> np.ndarray:
    if effective == 0:
        return np.zeros(0, dtype=np.float64)
    vg_bw = float(params["VG"]) * float(params["BW"])
    return np.asarray(state_trajectory[0, :effective], dtype=np.float64) / vg_bw


def estimate_iob_from_state_vector(state: StateVector | StateArray) -> float:
    return max(0.0, float(state[2]) + float(state[3])) / 1000.0


def estimate_iob_from_state_trajectory(state_trajectory: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(state_trajectory[2, :] + state_trajectory[3, :], dtype=np.float64), 0.0) / 1000.0


def estimate_basal_input_from_state(
    state: StateVector | StateArray,
    params: ParameterSet,
) -> float:
    tau_i = float(params["tauI"])
    return float(state[2]) / tau_i if tau_i > 0.0 else 0.0


def get_non_negative_state_indices() -> np.ndarray:
    return np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12], dtype=np.int64)


if __name__ == "__main__":
    from src.parameters import get_base_params

    params = get_base_params()
    compute_optimal_steady_state_from_glucose(
        params,
        100,
        international_units=False,
        max_iterations=100,
        print_progress=True,
    )