from __future__ import annotations

from typing import Callable

import numpy as np

from src.sensor import measure_glycemia

# Type aliases
ParameterSet = dict[str, float]
StateVector = list[float]
InputFunc = Callable[..., tuple[float, float]]

# Helper functions
def state_listify(Q1: float, Q2: float, S1: float, S2: float, I: float, x1: float, x2: float, x3: float, D1: float, D2: float) -> list[float]:
    """Helper function to pack state variables into a numpy array for ODE solvers."""
    x = {
        'Q1': Q1, 'Q2': Q2,
        'S1': S1, 'S2': S2,
        'I': I,
        'x1': x1, 'x2': x2, 'x3': x3,
        'D1': D1, 'D2': D2
    }
    return [x["Q1"], x["Q2"], x["S1"], x["S2"], x["I"], x["x1"], x["x2"], x["x3"], x["D1"], x["D2"]]

def state_unlistify(x: StateVector) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Helper function to unpack state variables from a list."""
    Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2 = x
    return Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2

# Model equations
def hovorka_equations(
    t: int,
    x: StateVector,
    params: ParameterSet,
    input_func: InputFunc,
    scenario: int,
    patient_id: int = 0,
    day: int = 0,
    basal_hourly: float = 0.5,
    insulin_sensitivity: float = 2.0,
    meal_schedule: dict[str, float] | None = None,
    seed: int | None = None,
) -> StateVector:
    """
    Standard Hovorka Model ODEs with named intermediate variables for clarity.
    
    States x: [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2]
      Q1, Q2: glucose mass [mmol] in accessible and remote compartments
      S1, S2: insulin mass [mU] in absorption depots
      I: plasma insulin [mU/L]
      x1, x2, x3: insulin action on transport, disposal, EGP
      D1, D2: meal carbs [mmol] in stomach and intestine
    
    Returns: [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2]
    """
    Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2 = x

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

    if meal_schedule is None:
        u_t, d_t = input_func(
            t,
            patient_id=patient_id,
            day=day,
            basal_hourly=basal_hourly,
            scenario=scenario,
            insulin_sensitivity=insulin_sensitivity,
            seed=seed,
        )
    else:
        u_t, d_t = input_func(
            t,
            patient_id=patient_id,
            day=day,
            basal_hourly=basal_hourly,
            scenario=scenario,
            insulin_sensitivity=insulin_sensitivity,
            meal_schedule=meal_schedule,
            seed=seed,
        )
    U = float(u_t)
    # Convert carbs from g/min to mmol/min
    # d_t is provided in mg/min from input.py, convert to mmol/min.
    D = float(d_t) / float(params["MwG"])
    # Calculate current glucose concentration in blood (mmol/L)
    G = Q1 / (VG * BW) if (VG * BW) > 0 else 0.0


    # --- 1. CHO ABSORPTION ---
    # Note: If multiple meals occur simultaneously in input_func, only the last D value
    # is used (overwrites previous); carbs can be lost. See input.py for meal scheduling.
    # eq:2.8a: drift of the glucose concentration in the stomach
    # [mmol/min] = [1 * mmol/min] - [1/min * mmol]
    dD1 = (Ag * D) - ((1.0 / tauG) * D1)

    # eq:2.8b: drift of the glucose concentration in the intestine
    # [mmol/min] = [1/min * mmol]
    dD2 = (1.0 / tauG) * (D1 - D2)

    # eq:2.9: glucose absorption rate in the blood
    # [mmol/min] = [1/min * mmol]
    UG = (1.0 / tauG) * D2


    # --- 2. INSULIN ABSORPTION ---
    # eq:2.11a: drift of the insulin concentration in the first adipose tissue
    # [mU/min] = [mU/min] - [1/min * mU]
    dS1 = U - ((1.0 / tauI) * S1)

    # eq:2.11b: drift of the insulin concentration in the second adipose tissue
    # [mU/min] = [1/min * mU]
    dS2 = (1.0 / tauI) * (S1 - S2)

    # eq:2.12: insulin absorption rate in the blood
    # [mU/min] = [1/min * mU]
    UI = (1.0 / tauI) * S2

    # eq:2.6: drift of the insulin concentration in the blood
    # [mU/L/min] = [mU/min * kg/L * 1/kg] - [1/min] * [mU/L]
    dI = (UI / (VI * BW)) - (ke * I)


    # --- 3. GLUCOSE KINETICS ---
    # eq:2.4: glucose absorbed by the central neural system
    # [mmol/min] = [mmol/kg/min] * [kg]
    # Note: max(0.0, G) guards against negative glucose from sensor noise,
    # preventing F01c from becoming negative (which would increase EGP unphysically)
    if G >= 4.5:
        F01c = F01 * BW
    else:
        F01c = max(0.0, F01 * BW * max(0.0, G) / 4.5)

    # eq:2.5: excretion rate of glucose in the kidneys
    # [mmol/min] = [mmol/L] * [L/kg] * [kg]
    # Note: Simplified kidney threshold model. Real threshold/gradient more complex,
    # but this approximation is acceptable for physiological range (40-600 mg/dL)
    if G >= 9.0:
        fr = 0.003 * (G - 9.0) * VG * BW
    else:
        fr = 0.0

    # eq:2.13: insulin sensitivity factors
    kb1 = SI1 * ka1  # [min^-2/(mU/L)] = [1/min/(mU/L)] * [1/min]
    kb2 = SI2 * ka2  # [min^-2/(mU/L)] = [1/min/(mU/L)] * [1/min]
    kb3 = SI3 * ka3  # [min^-1/(mU/L)] = [L/mU] * [1/min]

    # eq:2.7a: drift of the insulin action on glucose transport
    # [1/min^2] = [min^-2 * L/mU] * [mU/L] - [1/min] * [1/min]
    dx1 = kb1 * I - ka1 * x1

    # eq:2.7b: drift of the insulin action on glucose disposal
    # [1/min^2] = [min^-2/(mU/L)] * [mU/L] - [1/min] * [1/min]
    dx2 = kb2 * I - ka2 * x2

    # eq:2.7c: drift of the insulin action on endogenous glucose production (liver)
    # [1/min] = [1/min * L/mU] * [mU/L] - [1/min] * [1]
    dx3 = kb3 * I - ka3 * x3

    # Transfer rates between glucose compartments
    # Q1 -> Q2 transfer is insulin-dependent (x1 * Q1)
    # Q2 -> Q1 transfer is basal diffusion (k12 * Q2)
    R12 = x1 * Q1
    R21 = k12 * Q2
    # Endogenous glucose production
    EGPc = EGP0 * BW * max(0.0, 1.0 - x3)

    # eq:2.1: drift of glucose mass in the accessible compartment
    # [mmol/min] = [mmol/min] - [mmol/min] - [mmol/min] - [1/min * mmol] + [1/min * mmol] + [mmol/min/kg * kg]
    # UG: gut absorption, F01c: CNS uptake, fr: renal excretion,
    # R12: Q1→Q2 transfer, R21: Q2→Q1 transfer, EGPc: liver production
    dQ1 = UG - F01c - fr - R12 + R21 + EGPc

    # eq:2.2: drift of glucose mass in the non-accessible compartment
    # [mmol/min] = [1/min * mmol] - [[1/min + 1/min] * mmol]
    dQ2 = R12 - (k12 + x2) * Q2

    # Note: No state bounds validation - ODE can produce negative masses (Q1<0, etc.)
    # if parameters are pathological. Consider adding clipping for production use.
    return [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2]


# Initial state functions
def compute_initial_state_from_insulin(
    u_mu: float,
    params: ParameterSet
    ) -> StateVector:
    """
    Compute initial state from insulin.
    
    Parameters:
    -----------
    u_mu: basal insulin rate [mU/min]
    params: parameter set
    
    Returns:
    --------
    StateVector: initial state
    """
    BW = params['BW']      # [kg] body weight
    tauI = params['tauI']  # [min] maximum insulin absorption time
    ke = params['ke']      # [1/min] insulin elimination rate from plasma
    VI = params['VI']      # [L/kg] insulin distribution volume
    SI1 = params['SI1']    # [1/min*(mU/L)] insulin sensitivity of distribution/transport
    SI2 = params['SI2']    # [1/min*(mU/L)] insulin sensitivity of disposal
    SI3 = params['SI3']    # [L/mU]         insulin sensitivity of EGP
    EGP0 = params['EGP0']  # [mmol/kg/min] endogenous glucose production extrapolated to zero insulin concentration
    F01 = params['F01']    # [mmol/kg/min] non–insulin-dependent glucose flux (per kg)
    k12 = params['k12']    # [1/min] transfer rate between non-accessible and accessible glucose compartments

    Seq = tauI * u_mu                  # [mU] = [min] * [mU/min]
    Ieq = Seq / (ke * tauI * VI * BW)  # [mU/L] = [mU/min] * [1/min] * [kg/L] * [1/kg] * [min]
    x1eq = SI1 * Ieq                   # [1/min] = [1/min] * [L/mU] * [mU/L]
    x2eq = SI2 * Ieq                   # [1/min] = [1/min] * [L/mU] * [mU/L]
    x3eq = SI3 * Ieq                   # [1] = [L/mU] * [mU/L]

    # Fasting steady-state approximation (UG=0, fr≈0):
    # -x1*Q1 + k12*Q2 = F01c - EGPc
    # x1*Q1 - (k12 + x2)*Q2 = 0
    F01c = F01 * BW
    EGPc = EGP0 * BW * max(0.0, 1.0 - x3eq)
    b1 = F01c - EGPc  # [mmol/min]
    a11 = -x1eq                          # [1/min]
    a12 = k12                            # [1/min]
    # b2 = 0                             # [mmol/min]
    a21 = x1eq                           # [1/min]
    a22 = - k12 - x2eq                   # [1/min]

    # Solve robustly to avoid division-by-zero in singular/near-singular cases.
    # Closed-form formulas can fail when insulin approaches zero.
    A = np.array([[a11, a12], [a21, a22]], dtype=np.float64)
    b = np.array([b1, 0.0], dtype=np.float64)
    try:
        q = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        q, *_ = np.linalg.lstsq(A, b, rcond=None)

    Q1eq = max(0.0, float(q[0]))
    Q2eq = max(0.0, float(q[1]))

    return state_listify(
        Q1=Q1eq,   # [mmol] glucose in blood
        Q2=Q2eq,   # [mmol] glucose in muscles
        S1=Seq,    # [mU] insulin absorption in first adipose tissue
        S2=Seq,    # [mU] insulin absorption in second adipose tissue
        I=Ieq,     # [mU/L] insulin in plasma,
        x1=x1eq,   # [1/min] insulin action on glucose transport
        x2=x2eq,   # [1/min] insulin action on glucose disposal
        x3=x3eq,   # [1] insulin action on endogenous glucose production (liver)
        D1=0,      # [mmol] glucose absorption in the stomach
        D2=0,      # [mmol] glucose absorption in the intestine
    )

# Steady state functions
def compute_optimal_steady_state_from_glucose(
    params: ParameterSet,
    desired_glycemia: float | tuple[float, float],
    international_units: bool = True,
    max_iterations: int = 100,
    print_progress: bool = True
    ) -> StateVector:
    """
    Compute steady-state via bounded bisection on basal insulin rate [mU/min].

    Uses physiological basal bounds and evaluates glycemia from the analytical
    fasting equilibrium returned by compute_initial_state_from_insulin.
    """
    if international_units:
        tolerance = 0.1  # mmol/L
    else:
        # Input desired_glycemia is in mg/dL in this branch.
        # Convert both target and tolerance to mmol/L for internal comparisons.
        tolerance = 1.0 / (params['MwG'] / 10.0)  # 1 mg/dL in mmol/L
        desired_glycemia = desired_glycemia / (params['MwG'] / 10.0)

    min_insulin_amount = 1e-5    # [mU/min]
    max_insulin_amount = 50.0    # [mU/min]

    if print_progress:
        print(f"Computing optimal steady state for glucose: {desired_glycemia} [mmol/L]")

    def eval_glycemia(us: float) -> tuple[float, StateVector]:
        x = compute_initial_state_from_insulin(us, params)
        g = measure_glycemia(tuple(x), params)
        return g, x

    g_low, x_low = eval_glycemia(min_insulin_amount)
    g_high, x_high = eval_glycemia(max_insulin_amount)

    # Expand upper bound if needed to try bracketing the target.
    expansions = 0
    while g_high > desired_glycemia and expansions < 6:
        max_insulin_amount *= 2.0
        g_high, x_high = eval_glycemia(max_insulin_amount)
        expansions += 1

    # If still not bracketed, return the closer endpoint.
    if not (g_low >= desired_glycemia >= g_high):
        return x_low if abs(g_low - desired_glycemia) <= abs(g_high - desired_glycemia) else x_high

    best_x = x_low
    best_err = abs(g_low - desired_glycemia)

    for i in range(max_iterations):
        mid = 0.5 * (min_insulin_amount + max_insulin_amount)
        g_mid, x_mid = eval_glycemia(mid)
        err = abs(g_mid - desired_glycemia)

        if print_progress:
            print(f"Iteration {i+1}: G= {g_mid:.2f} [mmol/L], I= {mid:.5f} [mU/min]")

        if err < best_err:
            best_err = err
            best_x = x_mid

        if err < tolerance:
            return x_mid

        if g_mid > desired_glycemia:
            min_insulin_amount = mid
        else:
            max_insulin_amount = mid

    return best_x

if __name__ == "__main__":
    print("Get steady state testing")
    #from parameters import generate_monte_carlo_patients
    #p = generate_monte_carlo_patients(1, standard_patient=True)[0]
    from parameters import get_base_params
    p = get_base_params()
    compute_optimal_steady_state_from_glucose(100, p, international_units=False, max_iterations=100000, print_progress=True)