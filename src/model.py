from __future__ import annotations

from typing import Callable

import numpy as np

from src.hovorka_exercise import compute_rashid_terms
from src.sensor import measure_glycemia

ParameterSet = dict[str, float]
StateVector = list[float]
StateArray = np.ndarray
InputValues = tuple[float, float] | tuple[float, float, float]
InputFunc = Callable[..., InputValues]

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


def compute_initial_state_from_insulin(u_mu: float, params: ParameterSet) -> StateVector:
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


def _compute_hovorka_steady_state_from_glucose(
    params: ParameterSet,
    desired_glycemia: float | tuple[float, float],
    international_units: bool = True,
    max_iterations: int = 100,
    print_progress: bool = True,
) -> StateVector:
    target_mmol, tolerance = _convert_target_glucose_to_mmol(params, desired_glycemia, international_units)
    min_insulin_amount = 1e-5
    max_insulin_amount = 50.0

    if print_progress:
        print(f"Computing optimal steady state for glucose: {target_mmol} [mmol/L]")

    def eval_glycemia(us: float) -> tuple[float, StateVector]:
        x_state = compute_initial_state_from_insulin(us, params)
        g = measure_glycemia(tuple(x_state), params)
        return g, x_state

    g_low, x_low = eval_glycemia(min_insulin_amount)
    g_high, x_high = eval_glycemia(max_insulin_amount)

    expansions = 0
    while g_high > target_mmol and expansions < 6:
        max_insulin_amount *= 2.0
        g_high, x_high = eval_glycemia(max_insulin_amount)
        expansions += 1

    if not (g_low >= target_mmol >= g_high):
        return x_low if abs(g_low - target_mmol) <= abs(g_high - target_mmol) else x_high

    best_x = x_low
    best_err = abs(g_low - target_mmol)
    for iteration in range(max_iterations):
        mid = 0.5 * (min_insulin_amount + max_insulin_amount)
        g_mid, x_mid = eval_glycemia(mid)
        err = abs(g_mid - target_mmol)

        if print_progress:
            print(f"Iteration {iteration + 1}: G= {g_mid:.2f} [mmol/L], I= {mid:.5f} [mU/min]")

        if err < best_err:
            best_err = err
            best_x = x_mid
        if err < tolerance:
            return x_mid

        if g_mid > target_mmol:
            min_insulin_amount = mid
        else:
            max_insulin_amount = mid
    return best_x


def compute_model_steady_state_from_glucose(
    params: ParameterSet,
    desired_glycemia: float | tuple[float, float],
    international_units: bool = True,
    max_iterations: int = 100,
    print_progress: bool = True,
) -> StateVector:
    return _compute_hovorka_steady_state_from_glucose(
        params=params,
        desired_glycemia=desired_glycemia,
        international_units=international_units,
        max_iterations=max_iterations,
        print_progress=print_progress,
    )


def compute_optimal_steady_state_from_glucose(
    params: ParameterSet,
    desired_glycemia: float | tuple[float, float],
    international_units: bool = True,
    max_iterations: int = 100,
    print_progress: bool = True,
) -> StateVector:
    return compute_model_steady_state_from_glucose(
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