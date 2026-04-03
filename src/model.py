from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.hovorka_exercise import compute_eth_exercise_terms
from src.sensor import measure_glycemia

ParameterSet = dict[str, float]
StateVector = list[float]
StateArray = np.ndarray
InputValues = tuple[float, float] | tuple[float, float, float]
InputFunc = Callable[..., InputValues]
ResidualCallback = Callable[[np.ndarray, ParameterSet, float], np.ndarray]
JacobianCallback = Callable[[np.ndarray, np.ndarray, ParameterSet, float], np.ndarray]
ProjectCallback = Callable[[np.ndarray], np.ndarray]
ObservableCallback = Callable[[np.ndarray, ParameterSet], float]


_HOVORKA_BASE_STATE_COUNT = 10
_HOVORKA_STATE_COUNT = 18  # 10 base + 8 ETH exercise states (Y, Z, rGU, rGP, tPA, PAint, rdepl, th)


@dataclass(frozen=True)
class _SteadyStateCallbacks:
    residual: ResidualCallback
    jacobian: JacobianCallback
    project: ProjectCallback
    observable: ObservableCallback


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
    # ETH exercise states (indices 10-17); default to zero (fasting/rest)
    Y: float = 0.0,      # short-term PA insulin sensitivity accumulator
    Z: float = 0.0,      # long-term post-exercise SI elevation
    rGU: float = 0.0,    # exercise glucose utilization rate [1/min]
    rGP: float = 0.0,    # exercise glucose production rate [1/min]
    tPA: float = 0.0,    # PA tracking state
    PAint: float = 0.0,  # cumulative PA intensity integral [count·min]
    rdepl: float = 0.0,  # glycogen depletion rate [1/min]
    th: float = 0.0,     # high-intensity duration accumulator
) -> list[float]:
    return [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2, Y, Z, rGU, rGP, tPA, PAint, rdepl, th]


def state_unlistify(
    x: StateVector,
) -> tuple[float, float, float, float, float, float, float, float, float, float,
           float, float, float, float, float, float, float, float]:
    if len(x) == _HOVORKA_BASE_STATE_COUNT:
        q1, q2, s1, s2, i, x1, x2, x3, d1, d2 = x
        return q1, q2, s1, s2, i, x1, x2, x3, d1, d2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if len(x) != _HOVORKA_STATE_COUNT:
        raise ValueError(f"Expected {_HOVORKA_BASE_STATE_COUNT} or {_HOVORKA_STATE_COUNT} Hovorka states, got {len(x)}")
    q1, q2, s1, s2, i, x1, x2, x3, d1, d2, Y, Z, rGU, rGP, tPA, PAint, rdepl, th = x
    return q1, q2, s1, s2, i, x1, x2, x3, d1, d2, Y, Z, rGU, rGP, tPA, PAint, rdepl, th


def _parse_input_values(values: InputValues) -> tuple[float, float, float]:
    """Parse (u, d) or (u, d, ac) input tuple. Third element is AC [counts] for the ETH model."""
    if len(values) == 2:
        u_t, d_t = values
        return float(u_t), float(d_t), 0.0
    u_t, d_t, ac_t = values
    return float(u_t), float(d_t), float(ac_t)


def _get_input_values(
    t: int,
    input_func: InputFunc,
    scenario: int,
    patient_id: int,
    day: int,
    basal_hourly: float,
    insulin_carbo_ratio: float,
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
            meal_schedule=meal_schedule,
            seed=seed,
        )
    )


def _dawn_egp_factor(t_min: float) -> float:
    """Circadian EGP0 scaling for the dawn phenomenon.

    Hepatic glucose production rises ~25% at peak (06:00, 360 min) driven by
    cortisol and GH surges, tapering linearly from 04:00 (240 min) to 08:00
    (480 min).  Outside that window the factor is 1.0 (no modification).
    """
    DAWN_START = 240.0   # 04:00
    DAWN_PEAK  = 360.0   # 06:00
    DAWN_END   = 480.0   # 08:00
    DAWN_AMP   = 0.25    # 25% peak EGP elevation
    if t_min <= DAWN_START or t_min >= DAWN_END:
        return 1.0
    if t_min <= DAWN_PEAK:
        frac = (t_min - DAWN_START) / (DAWN_PEAK - DAWN_START)
    else:
        frac = (DAWN_END - t_min) / (DAWN_END - DAWN_PEAK)
    return 1.0 + DAWN_AMP * frac


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
    meal_schedule: dict[str, float] | None = None,
    seed: int | None = None,
    precomputed_inputs: InputValues | None = None,
) -> StateVector:
    Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2, Y, Z, rGU, rGP, tPA, PAint, rdepl, th = state_unlistify(list(x))

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

    u_t, d_t, ac_t = _get_input_values(
        t=t,
        input_func=input_func,
        scenario=scenario,
        patient_id=patient_id,
        day=day,
        basal_hourly=basal_hourly,
        insulin_carbo_ratio=insulin_carbo_ratio,
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

    # ETH Deichmann exercise model — replaces Rashid HR-based states.
    # Input: ac_t [accelerometer counts] instead of delta_hr_t.
    eth = compute_eth_exercise_terms(
        Y=Y, Z=Z, rGU=rGU, rGP=rGP,
        tPA=tPA, PAint=PAint, rdepl=rdepl, th=th,
        ac_t=ac_t,
        x1=x1,
        Q1=Q1,
        params=params,
    )
    dY     = float(eth["dY"])
    dZ     = float(eth["dZ"])
    drGU   = float(eth["drGU"])
    drGP   = float(eth["drGP"])
    dtPA   = float(eth["dtPA"])
    dPAint = float(eth["dPAint"])
    drdepl = float(eth["drdepl"])
    dth    = float(eth["dth"])

    # Hovorka glucose compartment terms
    R12  = (x1 * Q1) - (k12 * Q2)
    R2   = x2 * Q2
    EGPc = EGP0 * BW * max(0.0, 1.0 - x3) * _dawn_egp_factor(float(t))

    # ETH exercise contributions grafted onto Q1:
    #   exercise_uptake — insulin-independent glucose disposal during exercise
    #   exercise_prod   — net EGP boost (reduced by glycogen depletion for prolonged sessions)
    #   exercise_si     — post-exercise enhanced insulin sensitivity (Z amplifies x1)
    dQ1 = UG + EGPc - R12 - F01c - fr \
          - float(eth["exercise_uptake"]) \
          + float(eth["exercise_prod"]) \
          - float(eth["exercise_si"])
    dQ2 = R12 - R2

    return [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2,
            dY, dZ, drGU, drGP, dtPA, dPAint, drdepl, dth]


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

    # ETH exercise states are all zero at fasting (no physical activity)
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
    )  # Y, Z, rGU, rGP, tPA, PAint, rdepl, th default to 0.0


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
    # mg/dL path: convert target to mmol/L; tolerance stays at 0.1 mmol/L (= 1.8 mg/dL).
    # Dividing by MwG/10 would give 0.0055 mmol/L — 18× too tight and physically meaningless.
    return target_mmol / (params["MwG"] / 10.0), 0.1


def _steady_state_input_stub(*_: object, **__: object) -> InputValues:
    return 0.0, 0.0, 0.0


def _project_state_and_insulin(state: np.ndarray, insulin_amount: float) -> tuple[np.ndarray, float]:
    """Enforces box constraints on physiological states and insulin.

    Clips nonnegative states to [0, ∞) and insulin to [1e-6, 200.0] mU/min.
    Ensures all residual and Jacobian evaluations remain in physically valid domains.
    """
    projected_state = np.asarray(state, dtype=np.float64).copy()
    nonnegative_indices = get_non_negative_state_indices()
    valid_indices = nonnegative_indices[nonnegative_indices < projected_state.size]
    projected_state[valid_indices] = np.maximum(projected_state[valid_indices], 0.0)
    projected_insulin = float(np.clip(insulin_amount, 1e-6, 200.0))
    return projected_state, projected_insulin


def _project_hovorka_variables(z: np.ndarray) -> np.ndarray:
    """Adapts _project_state_and_insulin for the combined Newton step vector z = [x, u].

    Unpacks z into state and insulin, projects both, then repacks.
    Serves as the Newton solver's constraint callback after each iteration and step trial.
    """
    projected_z = np.asarray(z, dtype=np.float64).copy()
    projected_state, projected_insulin = _project_state_and_insulin(projected_z[:-1], float(projected_z[-1]))
    projected_z[:-1] = projected_state
    projected_z[-1] = projected_insulin
    return projected_z


def _evaluate_hovorka_steady_state_residual(
    z: np.ndarray,
    params: ParameterSet,
    target_mmol: float,
) -> np.ndarray:
    """Evaluates the residual of the Hovorka model equations and the glucose observable
    for a given combined state and insulin variable vector.
    The residual consists of the derivatives from the Hovorka equations
    (which should be zero at steady state)
    """
    projected_z = _project_hovorka_variables(z)
    state = projected_z[:-1]
    insulin_amount = float(projected_z[-1])

    derivatives = np.asarray(
        hovorka_equations(
            t=0,
            x=state,
            params=params,
            input_func=_steady_state_input_stub,
            scenario=1,
            precomputed_inputs=(insulin_amount, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    glucose_residual = _hovorka_observable(projected_z, params) - target_mmol

    residual = np.empty(state.size + 1, dtype=np.float64)
    residual[:-1] = derivatives
    residual[-1] = glucose_residual
    return residual


def _hovorka_observable(z: np.ndarray, params: ParameterSet) -> float:
    """Calculates the glucose observable for a given combined state and insulin variable vector."""
    projected_z = _project_hovorka_variables(z)
    state = projected_z[:-1]
    return measure_glycemia(tuple(float(v) for v in state), params)


def _estimate_numerical_jacobian(
    residual_callback: ResidualCallback,
    z: np.ndarray,
    base_residual: np.ndarray,
    params: ParameterSet,
    target_mmol: float,
) -> np.ndarray:
    variable_count = z.size
    jacobian = np.zeros((variable_count, variable_count), dtype=np.float64)

    for variable_index in range(variable_count):
        perturbation = max(1e-6, 1e-4 * max(1.0, abs(float(z[variable_index]))))
        z_perturbed = z.copy()
        z_perturbed[variable_index] += perturbation
        residual_perturbed = residual_callback(z_perturbed, params, target_mmol)
        jacobian[:, variable_index] = (residual_perturbed - base_residual) / perturbation

    return jacobian


def _compute_hovorka_analytical_jacobian(
    z: np.ndarray,
    _: np.ndarray,
    params: ParameterSet,
    __: float,
) -> np.ndarray:
    """Analytical Jacobian of the 18-state Hovorka + ETH exercise system.

    This Jacobian is evaluated exclusively at the fasting steady state (no exercise,
    AC=0), where all 8 ETH exercise states (Y…th) are zero. At rest the ETH states
    decouple from the Hovorka core: their ODE rows are diagonal with negative
    self-derivatives, and they contribute no terms to the Q1/Q2 rows.
    The resulting 19×19 matrix therefore has the same top-left 10×10 Hovorka
    block as before, plus an 8×8 diagonal block for the exercise states.
    """
    projected_z = _project_hovorka_variables(z)
    state = projected_z[:-1]

    q1, q2, _s1, _s2, _i, x1, x2, x3, _d1, _d2, \
        _Y, _Z, _rGU, _rGP, _tPA, _PAint, _rdepl, _th = state_unlistify(
            list(float(v) for v in state)
        )

    bw      = float(params["BW"])
    vg      = float(params["VG"])
    vi      = float(params["VI"])
    tau_i   = float(params["tauI"])
    tau_g   = float(params["tauG"])
    ke      = float(params["ke"])
    ka1     = float(params["ka1"])
    ka2     = float(params["ka2"])
    ka3     = float(params["ka3"])
    si1     = float(params["SI1"])
    si2     = float(params["SI2"])
    si3     = float(params["SI3"])
    k12     = float(params["k12"])
    egp0    = float(params["EGP0"])
    f01     = float(params["F01"])
    tau_AC  = max(1.0, float(params["eth_tau_AC"]))
    tau_Z   = max(1.0, float(params["eth_tau_Z"]))
    q2_eth  = max(0.0, float(params["eth_q2"]))   # rGU decay rate
    q4l_eth = max(0.0, float(params["eth_q4l"]))  # rGP aerobic decay (dominant at rest)
    q5_eth  = max(0.0, float(params["eth_q5"]))   # th decay rate (EXPERIMENTAL)
    q6_eth  = max(0.0, float(params["eth_q6"]))   # glycogen depletion decay rate

    vg_bw     = vg * bw
    inv_tau_i = 1.0 / tau_i
    inv_tau_g = 1.0 / tau_g
    inv_vi_bw = 1.0 / (vi * bw)

    glucose = q1 / vg_bw if vg_bw > 0.0 else 0.0

    d_f01c_d_q1 = 0.0 if glucose >= 4.5 else f01 / (4.5 * vg)
    d_fr_d_q1   = 0.003 if glucose >= 9.0 else 0.0
    d_egpc_d_x3 = -egp0 * bw if x3 < 1.0 else 0.0

    # 19×19: 18 state rows + 1 glucose residual row, columns = 18 states + u
    jacobian = np.zeros((19, 19), dtype=np.float64)

    idx_q1, idx_q2           = 0, 1
    idx_s1, idx_s2           = 2, 3
    idx_i                    = 4
    idx_x1, idx_x2, idx_x3  = 5, 6, 7
    idx_d1, idx_d2           = 8, 9
    # ETH exercise state indices
    idx_Y, idx_Z             = 10, 11
    idx_rGU, idx_rGP         = 12, 13
    idx_tPA, idx_PAint       = 14, 15
    idx_rdepl, idx_th        = 16, 17
    idx_u                    = 18

    # --- Hovorka core block (unchanged from base model) ---

    # dQ1/d* row — at fasting ETH states are 0 so no exercise terms appear
    jacobian[0, idx_q1] = -x1 - d_f01c_d_q1 - d_fr_d_q1
    jacobian[0, idx_q2] = k12
    jacobian[0, idx_x1] = -q1
    jacobian[0, idx_x3] = d_egpc_d_x3
    jacobian[0, idx_d2] = inv_tau_g

    # dQ2/d* row
    jacobian[1, idx_q1] = x1
    jacobian[1, idx_q2] = -k12 - x2
    jacobian[1, idx_x1] = q1
    jacobian[1, idx_x2] = -q2

    # dS1, dS2, dI rows
    jacobian[2, idx_s1] = -inv_tau_i
    jacobian[2, idx_u]  = 1.0
    jacobian[3, idx_s1] = inv_tau_i
    jacobian[3, idx_s2] = -inv_tau_i
    jacobian[4, idx_s2] = inv_tau_i * inv_vi_bw
    jacobian[4, idx_i]  = -ke

    # dx1, dx2, dx3 rows
    jacobian[5, idx_i]  = si1 * ka1
    jacobian[5, idx_x1] = -ka1
    jacobian[6, idx_i]  = si2 * ka2
    jacobian[6, idx_x2] = -ka2
    jacobian[7, idx_i]  = si3 * ka3
    jacobian[7, idx_x3] = -ka3

    # dD1, dD2 rows
    jacobian[8, idx_d1]  = -inv_tau_g
    jacobian[9, idx_d1]  = inv_tau_g
    jacobian[9, idx_d2]  = -inv_tau_g

    # --- ETH exercise states block (diagonal at fasting: AC=0, all states=0) ---
    # Each row i has d(d_state_i)/d(state_i) = -decay_rate_i.
    # Cross-derivatives involving fY, fAC, fHI are all zero when Y=0 (fY=0) and AC=0.

    jacobian[10, idx_Y]     = -1.0 / tau_AC          # dY: decay to AC (= 0 at rest)
    jacobian[11, idx_Z]     = -1.0 / tau_Z            # dZ: decay term dominates (fY=0)
    jacobian[12, idx_rGU]   = -q2_eth                 # drGU: decay
    jacobian[13, idx_rGP]   = -q4l_eth                # drGP: aerobic decay (fp=0 at rest)
    jacobian[14, idx_tPA]   = -1.0                    # dtPA: decays to fAC=0 at rest
    jacobian[15, idx_PAint] = -1.0                    # dPAint: decays to fAC*AC=0 at rest
    jacobian[16, idx_rdepl] = -q6_eth                 # drdepl: decay (ft=0 when tPA=0)
    jacobian[17, idx_th]    = -q5_eth if q5_eth > 0.0 else -1e-6  # dth: decay (fHI=0 at rest)

    # Glucose residual row: d(G - target)/d* = d(Q1/(VG*BW))/dQ1
    jacobian[18, idx_q1] = 1.0 / vg_bw
    return jacobian


def _build_hovorka_steady_state_callbacks() -> _SteadyStateCallbacks:
    return _SteadyStateCallbacks(
        residual=_evaluate_hovorka_steady_state_residual,
        jacobian=_compute_hovorka_analytical_jacobian,
        project=_project_hovorka_variables,
        observable=_hovorka_observable,
    )


def _solve_multivariate_steady_state_newton(
    callbacks: _SteadyStateCallbacks,
    params: ParameterSet,
    target_mmol: float,
    warm_start_z: np.ndarray,
    glucose_tolerance_mmol: float,
    max_iterations: int,
    print_progress: bool,
    label: str,
) -> np.ndarray:
    z = callbacks.project(np.asarray(warm_start_z, dtype=np.float64))
    best_z = z.copy()
    best_score = float("inf")
    state_residual_tolerance = 1e-6
    eval_count = 0

    if print_progress:
        print(f"Computing optimal steady state with multivariate Newton ({label}) for glucose: {target_mmol} [mmol/L]")

    for iteration in range(1, max_iterations + 1):
        residual = callbacks.residual(z, params, target_mmol)
        eval_count += 1
        z = callbacks.project(z)

        state_residual_inf = float(np.linalg.norm(residual[:-1], ord=np.inf))
        glucose_residual_abs = abs(float(residual[-1]))
        score = max(state_residual_inf, glucose_residual_abs)

        if score < best_score:
            best_score = score
            best_z = z.copy()

        if print_progress:
            print(
                f"Iter {iteration:02d} (eval {eval_count}): "
                f"G= {callbacks.observable(z, params):.4f} [mmol/L], "
                f"I= {float(z[-1]):.6f} [mU/min], "
                f"||F_state||_inf= {state_residual_inf:.3e}, "
                f"|F_glucose|= {glucose_residual_abs:.3e}"
            )

        if glucose_residual_abs <= glucose_tolerance_mmol and state_residual_inf <= state_residual_tolerance:
            return z

        jacobian = callbacks.jacobian(z, residual, params, target_mmol)
        # Rare safeguard for unstable analytical derivatives near non-smooth branch points.
        if not np.all(np.isfinite(jacobian)):
            jacobian = _estimate_numerical_jacobian(callbacks.residual, z, residual, params, target_mmol)

        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)

        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
            z_trial = callbacks.project(z + damping * step)
            trial_residual = callbacks.residual(z_trial, params, target_mmol)
            eval_count += 1

            trial_state_residual_inf = float(np.linalg.norm(trial_residual[:-1], ord=np.inf))
            trial_glucose_residual_abs = abs(float(trial_residual[-1]))
            trial_score = max(trial_state_residual_inf, trial_glucose_residual_abs)

            if trial_score < best_score:
                best_score = trial_score
                best_z = z_trial.copy()

            if trial_score <= score:
                z = z_trial
                accepted = True
                break

        if not accepted:
            break

    return best_z


def _compute_hovorka_steady_state_multivariate_newton(
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

    warm_start_insulin = 10.0
    warm_start_state = compute_fasting_steady_state_from_basal_insulin(warm_start_insulin, params)
    warm_start_state[0] = max(0.0, target_mmol * float(params["VG"]) * float(params["BW"]))

    warm_start_z = np.asarray([*warm_start_state, warm_start_insulin], dtype=np.float64)
    callbacks = _build_hovorka_steady_state_callbacks()
    solved_z = _solve_multivariate_steady_state_newton(
        callbacks=callbacks,
        params=params,
        target_mmol=target_mmol,
        warm_start_z=warm_start_z,
        glucose_tolerance_mmol=glucose_tolerance_mmol,
        max_iterations=max_iterations,
        print_progress=print_progress,
        label="Hovorka analytical Jacobian",
    )
    return [float(v) for v in solved_z[:-1]]


def compute_optimal_steady_state_from_glucose(
    params: ParameterSet,
    desired_glycemia: float | tuple[float, float],
    international_units: bool = True,
    max_iterations: int = 100,
    print_progress: bool = True,
) -> StateVector:
    return _compute_hovorka_steady_state_multivariate_newton(
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


def estimate_basal_input_from_state(
    state: StateVector | StateArray,
    params: ParameterSet,
) -> float:
    tau_i = float(params["tauI"])
    return float(state[2]) / tau_i if tau_i > 0.0 else 0.0


def get_non_negative_state_indices() -> np.ndarray:
    # Indices 0-9: standard Hovorka states (Q1,Q2,S1,S2,I,x1,x2,x3,D1,D2)
    # Indices 10-17: ETH exercise states (Y,Z,rGU,rGP,tPA,PAint,rdepl,th) — all non-negative
    return np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64)


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