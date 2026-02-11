# Library Imports
from typing import Any

import numpy as np

def hovorka_equations(t, x, params, input_func, scenario):
    """
    Standard Hovorka Model ODEs.
    States x: [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2]
    """
    # Unpack states
    Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2 = x

    # Unpack parameters
    MwG = params['MwG']     # Molecular weight of glucose (180.16 mg/mmol)
    EGP0 = params['EGP0']   # Endogenous glucose production at zero insulin [mmol/kg/min]
    F01 = params['F01']     # Non-insulin dependent glucose flux [mmol/kg/min]
    k12 = params['k12']     # Transfer rate between Q1 and Q2 [1/min]
    ka1 = params['ka1']     # Deactivation rate for x1 [1/min]
    ka2 = params['ka2']     # Deactivation rate for x2 [1/min]
    ka3 = params['ka3']     # Deactivation rate for x3 [1/min]
    SI1 = params['SI1']     # Sensitivity for x1 (transport) [L/min/mU]
    SI2 = params['SI2']     # Sensitivity for x2 (disposal) [L/min/mU]
    SI3 = params['SI3']     # Sensitivity for x3 (EGP) [L/mU]
    ke = params['ke']       # Insulin elimination rate [1/min]
    VI = params['VI']       # Insulin distribution volume [L/kg]
    VG = params['VG']       # Glucose distribution volume [L/kg]
    tauI = params['tauI'] # Insulin absorption time constant [min]
    tauG = params['tauG'] # Meal absorption time constant [min]
    Ag = params['Ag']       # Carbohydrate bioavailability (fraction)
    BW = params['BW']       # Body weight [kg]

    # Get inputs (Insulin u(t) and Carbs d(t)) at current time t
    u_t, d_t = input_func(t, scenario)
    # Convert insulin from U/min to mU/min
    U = u_t
    # Convert carbs from g/min to mg/min (MwG = 180.16 mg/mmol)
    D = d_t * (1000 / MwG)
    # Calculate current glucose concentration in blood (mmol/L)
    G = Q1 / (VG * BW)


    # --- 1. CHO ABSORPTION ---
    # eq:2.8a: drift of the glucose concentration in the stomach
    # [mmol/min] = [1 * mmol/min] - [1/min * mmol]
    dD1 = (Ag * D) - ((1 / tauG) * D1)

    # eq:2.8b: drift of the glucose concentration in the intestine
    # [mmol/min] = [1/min * mmol]
    dD2 = (1 / tauG) * (D1 - D2)

    # eq:2.9: glucose absorption rate in the blood
    # [mmol/min] = [1/min * mmol]
    UG = (1 / tauG) * D2


    # --- 2. INSULIN ABSORPTION ---
    # eq:2.11a: drift of the insulin concentration in the first adipose tissue
    # [mU/min] = [mU/min] - [1/min * mU]
    dS1 = U - ((1 / tauI) * S1)

    # eq:2.11b: drift of the insulin concentration in the second adipose tissue
    # [mU/min] = [1/min * mU]
    dS2 = (1 / tauI) * (S1 - S2)

    # eq:2.12: insulin absorption rate in the blood
    # [mU/min] = [1/min * mU]
    UI = (1 / tauI) * S2

    # eq:2.6: drift of the insulin concentration in the blood
    # [mU/L/min] = [mU/min * kg/L * 1/kg] - [1/min] * [mU/L]
    dI = (UI / (VI * BW)) - (ke * I)


    # --- 3. GLUCOSE KINETICS ---
    # eq:2.4: glucose absorbed by the central neural system
    # [mmol/min] = [mmol/kg/min] * [kg]
    if G >= 4.5: F01c = F01 * BW
    else:        F01c = F01 * BW * G / 4.5

    # eq:2.5: excretion rate of glucose in the kidneys
    # [mmol/min] = [mmol/L] * [L/kg] * [kg]
    if G >= 9: FR = 0.003 * (G - 9) * VG * BW
    else:      FR = 0

    # eq:2.1: drift of glucose mass in the accessible compartment
    # [mmol/min] = [mmol/min] - [mmol/min] - [mmol/min] - [1/min * mmol] + [1/min * mmol] + [mmol/min/kg * kg]
    dQ1 = UG - F01c - FR - (x1 * Q1) + (k12 * Q2) + (EGP0 * BW * (1 - x3))

    # eq:2.2: drift of glucose mass in the non-accessible compartment
    # [mmol/min] = [1/min * mmol] - [[1/min + 1/min] * mmol]
    dQ2 = x1 * Q1 - (k12 + x2) * Q2


    # --- 4. INSULIN ACTION ---
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


    return [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dD1, dD2]












def hovorka_equations_gemini(t, x, params, input_func, scenario=1):
    """
    Standard Hovorka Model ODEs.
    States x: [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2]
    """
    # Unpack states
    Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2 = x

    # Unpack parameters
    MwG = params['MwG']     # Molecular weight of glucose (180.16 mg/mmol)
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
    BW = params['BW']       # Body weight [kg]

    # Get inputs (Insulin u(t) and Carbs d(t)) at current time t
    u_t, d_t = input_func(t)
    U = u_t * 1000      # Convert insulin from U/min to mU/min
    D = d_t * (1000 / MwG)  # Convert carbs from g/min to mg/min (MwG = 180.16 mg/mmol)
    G = Q1 / (VG * BW)

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
    EGP = EGP0 * (1 - x3)
    if EGP < 0: EGP = 0

    # Glucose Kinetics
    # 0 -> Q1 (Accessible compartment)
    dQ1 = - (F01_c + FR + x1 * Q1 + k12 * Q1) + k12 * Q2 + UG + EGP
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