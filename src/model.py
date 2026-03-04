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
    D = float(d_t) * (1000.0 / float(params["MwG"]))
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
    R12 = k12 * Q2
    R21 = x1 * Q1
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
def compute_initial_state_from_insulin(us: float, params: ParameterSet) -> StateVector:
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

    Seq = tauI * us                    # [mU] = [min] * [mU/min]
    Ieq = Seq / (ke * tauI * VI * BW)  # [mU/L] = [mU/min] * [1/min] * [kg/L] * [1/kg] * [min]
    x1eq = SI1 * Ieq                   # [1/min] = [1/min] * [L/mU] * [mU/L]
    x2eq = SI2 * Ieq                   # [1/min] = [1/min] * [L/mU] * [mU/L]
    x3eq = SI3 * Ieq                   # [1] = [L/mU] * [mU/L]

    # -x1*Q1 + k12*Q2 = F01 + EGP0(x3-1) * BW
    # x1*Q1 - (k12 + x2)*Q2 = 0
    b1 = (EGP0 * BW * (x3eq - 1)) + F01  # [mmol/min] = [mmol/kg/min] * [kg] * [1]
    a11 = -x1eq                          # [1/min]
    a12 = k12                            # [1/min]
    # b2 = 0                             # [mmol/min]
    a21 = x1eq                           # [1/min]
    a22 = - k12 - x2eq                   # [1/min]

    Q1eq = b1 / (a11 - (a12 * a21 / a22))  # [mmol] = [mmol/min] / [1/min]

    Q2eq = - a21 * Q1eq / a22              # [mmol] = [1/min] * [mmol] / [1/min]

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
def compute_optimal_steady_state_from_glucose(desired_glycemia: float, params: ParameterSet, international_units: bool = True, max_iterations: int = 100, print_progress: bool = True) -> StateVector:
    """
    Compute steady-state using crude fixed-step search.
    
    Note: This non-adaptive algorithm with fixed 100 mU steps is inefficient but works.
    Bisection would converge faster, but this is acceptable for initialization.
    Steps can overshoot target; not critical for initial state computation.
    """
    if international_units:
        tolerance = 0.1
    else:
        tolerance = 1
        desired_glycemia = desired_glycemia / (params['MwG'] / 10)

    # Initial guess for insulin
    insulin_amount = 1500        # [mU]
    insulin_offset = 100         # [mU] - fixed step size; bisection would be more efficient

    if print_progress:
        print(f"Computing optimal steady state for glucose: {desired_glycemia} [mmol/L]")

    x0 = compute_initial_state_from_insulin(insulin_amount, params)

    # Cycle until good glycemia is achieved
    for i in range(max_iterations):

        x0 = compute_initial_state_from_insulin(insulin_amount, params)
        G = measure_glycemia(tuple(x0), params)

        if print_progress: print(f"Iteration {i+1}: G= {G:.2f} [mmol/L], I= {insulin_amount:.2f} [mU/L]")

        if np.abs(G - desired_glycemia) < tolerance:
            return x0
        if G > desired_glycemia:
            insulin_amount += insulin_offset
        else:
            insulin_amount -= insulin_offset
        #insulin_offset *= 0.95
        #if insulin_amount <= 0:
        #    break

    return x0


# ============================================================================
# LEGACY: Gemini Reference Implementation
# ============================================================================
# The following is a legacy reference implementation. It is NOT used by the main pipeline.
# Kept for historical reference only.

def hovorka_equations_gemini(
    t: int, 
    x: StateVector, 
    params: ParameterSet, 
    input_func: InputFunc, 
    scenario: int = 1
) -> StateVector:
    """
    LEGACY: Gemini reference implementation of Hovorka Model ODEs.
    This function is kept for historical reference and is not used in the main pipeline.
    
    WARNING: input_func signature mismatch - this function calls input_func(t)
    with only one argument, but modern input.py functions (scenario_inputs, 
    scenario_with_cached_meals) require multiple arguments. This will fail
    if called directly with current input functions.
    
    States x: [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2]
    """
    # Unpack states
    Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2 = x

    # Unpack parameters
    # MwG = params['MwG']     # Molecular weight of glucose (180.16 mg/mmol) - not used in current implementation
    EGP0 = params['EGP0']   # Endogenous glucose production at zero insulin [mmol/kg/min]
    F01 = params['F01']     # Non-insulin dependent glucose flux [mmol/kg/min]
    k12 = params['k12']     # Transfer rate between Q1 and Q2 [1/min]
    ka1 = params['ka1']     # Deactivation rate for x1 [1/min]
    ka2 = params['ka2']     # Deactivation rate for x2 [1/min]
    ka3 = params['ka3']     # Deactivation rate for x3 [1/min]
    #SI1 = params['SI1']     # Sensitivity for x1 (transport) [L/min/mU]
    #SI2 = params['SI2']     # Sensitivity for x2 (disposal) [L/min/mU]
    #SI3 = params['SI3']     # Sensitivity for x3 (EGP) [L/mU]
    kb1 = params['kb1']     #TODO: remove??
    kb2 = params['kb2']
    kb3 = params['kb3']
    ke = params['ke']       # Insulin elimination rate [1/min]
    VI = params['VI']       # Insulin distribution volume [L/kg]
    VG = params['VG']       # Glucose distribution volume [L/kg]
    tauI = params['tauI'] # Insulin absorption time constant [min]
    tauG = params['tauG'] # Meal absorption time constant [min]
    Ag = params['Ag']       # Carbohydrate bioavailability (fraction)
    # BW = params['BW']       # Body weight [kg] - not used in current implementation

    # Get inputs (Insulin u(t) and Carbs d(t)) at current time t
    u_t, d_t = input_func(t)
    # Note: U and D conversion factors computed but not used in current implementation
    # _U = u_t * 1000      # Convert insulin from U/min to mU/min
    # _D = d_t * (1000 / MwG)  # Convert carbs from g/min to mg/min (MwG = 180.16 mg/mmol)
    # _G = Q1 / (VG * BW) # Current glucose concentration (computed for reference)

    # --- Glucose Subsystem ---
    # Non-insulin dependent glucose flux
    if Q1 >= F01:
        F01_c = F01
    else:
        F01_c = Q1  # Simplification for low glucose

    # Renal excretion (approximated)
    FR = 0 if Q1 / VG < 9 else 0.003 * (Q1 / VG - 9) * VG

    # Gut absorption rate
    UG = D2 / tauG

    # Endogenous Glucose Production
    egp_rate = EGP0 * (1 - x3)
    if egp_rate < 0: 
        egp_rate = 0

    # Glucose Kinetics
    # 0 -> Q1 (Accessible compartment)
    dQ1 = - (F01_c + FR + x1 * Q1 + k12 * Q1) + k12 * Q2 + UG + egp_rate
    # Q1 -> Q2 (Non-accessible compartment)
    dQ2 = x1 * Q1 - (k12 + x2) * Q2

    # --- Insulin Subsystem ---
    # S1, S2: Subcutaneous insulin absorption
    dS1 = u_t - S1 / tauI
    dS2 = S1 / tauI - S2 / tauI

    # Plasma Insulin (I)
    U_I = S2 / tauI  # Absorption rate
    dI = U_I / VI - ke * I

    # --- Insulin Action Subsystem ---
    # x1: Effect on distribution/transport
    dx1 = -ka1 * x1 + kb1 * I
    # x2: Effect on disposal
    dx2 = -ka2 * x2 + kb2 * I
    # x3: Effect on EGP
    dx3 = -ka3 * x3 + kb3 * I

    # --- Gut Absorption Subsystem ---
    # D1, D2: Carbohydrate digestion
    dD1 = Ag * d_t - D1 / tauG
    dD2 = D1 / tauG - D2 / tauG

    return [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2]

if __name__ == "__main__":
    print("Get steady state testing")
    #from parameters import generate_monte_carlo_patients
    #p = generate_monte_carlo_patients(1, standard_patient=True)[0]
    from parameters import get_base_params
    p = get_base_params()
    compute_optimal_steady_state_from_glucose(100, p, international_units=False, max_iterations=100000, print_progress=True)