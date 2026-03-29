from __future__ import annotations

from typing import TypedDict


class ETHExerciseTerms(TypedDict):
    """Output of the ETH Deichmann exercise model per ODE timestep.

    State derivatives drive the 8 new exercise states forward in time.
    Q1 interaction terms are grafted onto the Hovorka glucose equation.
    """

    # --- State derivatives ---
    dY: float       # d/dt of short-term PA insulin sensitivity accumulator [count·min/min]
    dZ: float       # d/dt of long-term post-exercise SI elevation [count·min/min]
    drGU: float     # d/dt of exercise glucose utilization rate [1/min²]
    drGP: float     # d/dt of exercise glucose production rate [1/min²]
    dtPA: float     # d/dt of PA tracking state [-/min]
    dPAint: float   # d/dt of cumulative PA intensity integral [count/min]
    drdepl: float   # d/dt of glycogen depletion rate [1/min²]
    dth: float      # d/dt of high-intensity duration accumulator [-/min]

    # --- Hovorka Q1 interaction terms ---
    exercise_uptake: float  # rGU * Q1: insulin-independent glucose disposal [mmol/min]
    exercise_prod: float    # (rGP - rdepl) * Q1: net exercise EGP [mmol/min]
    exercise_si: float      # Z * x1 * Q1: post-exercise enhanced insulin sensitivity [mmol/min]


def compute_eth_exercise_terms(
    Y: float,
    Z: float,
    rGU: float,
    rGP: float,
    tPA: float,
    PAint: float,
    rdepl: float,
    th: float,
    ac_t: float,
    x1: float,
    Q1: float,
    params: dict[str, float],
) -> ETHExerciseTerms:
    """Compute ETH Deichmann PA exercise model ODEs and Hovorka Q1 interaction terms.

    Based on: Deichmann et al., PLOS Comput Biol 2023.
    Validated on 5 T1D children from Basel University Children's Hospital
    (data_patients/ in gitlab.com/csb.ethz/t1d-exercise-model).

    State vector (8 new states replacing Rashid E1/E2/TE):
      Y      — short-term PA insulin sensitivity, driven by AC with τ=tau_AC (~5 min)
      Z      — long-term post-exercise SI elevation, decays with τ=tau_Z (~600 min = 10h)
      rGU    — exercise glucose utilization rate [1/min]
      rGP    — exercise glucose production rate [1/min]
      tPA    — PA tracking state (1 = active, 0 = rest)
      PAint  — cumulative PA intensity integral [count·min], drives glycogen depletion
      rdepl  — glycogen depletion rate [1/min], limits EGP as exercise prolongs
      th     — high-intensity duration accumulator, drives aerobic→anaerobic transition

    Input:
      ac_t  — accelerometer counts at current minute (0 = rest, ~1000 = moderate, ~5600+ = high)
      x1    — Hovorka insulin action state for glucose disposal [1/min]
      Q1    — Hovorka glucose compartment 1 [mmol]

    Parameter prefix: "eth_*" in the patient ParameterSet.

    NOTE: Anaerobic parameters (eth_q3h, eth_q4h, eth_q5, eth_q6) are marked
    EXPERIMENTAL — they are not validated on T1D patients and use population-level
    values from params_standard.csv as a physiologically plausible starting point.
    """
    # --- Extract parameters ---
    tau_AC = max(1.0, float(params["eth_tau_AC"]))   # [min] AC → Y time constant (≈5 min)
    b      = max(0.0, float(params["eth_b"]))         # [1/(count·min)] Z drive coefficient
    tau_Z  = max(1.0, float(params["eth_tau_Z"]))     # [min] long-term SI decay (~600 min)
    # EXPERIMENTAL: multi-day Z cap (not in original Deichmann et al. 2023 paper).
    # Prevents unbounded cross-day accumulation by saturating the Z drive as Z → Z_max.
    Z_max  = max(1e-3, float(params.get("eth_Z_max", float("inf"))))
    q1     = max(0.0, float(params["eth_q1"]))        # [1/(count·min²)] rGU drive
    q2     = max(0.0, float(params["eth_q2"]))        # [1/min] rGU decay
    q3l    = max(0.0, float(params["eth_q3l"]))       # rGP drive — aerobic (T1D-validated)
    q4l    = max(0.0, float(params["eth_q4l"]))       # rGP decay — aerobic (T1D-validated)
    q3h    = max(0.0, float(params["eth_q3h"]))       # rGP drive — anaerobic (EXPERIMENTAL)
    q4h    = max(0.0, float(params["eth_q4h"]))       # rGP decay — anaerobic (EXPERIMENTAL)
    q5     = max(0.0, float(params["eth_q5"]))        # [1/min] th decay rate (EXPERIMENTAL)
    q6     = max(0.0, float(params["eth_q6"]))        # [1/min] glycogen depletion rate
    adepl  = float(params["eth_adepl"])               # [min/count] depletion threshold slope
    bdepl  = max(1.0, float(params["eth_bdepl"]))     # [count·min] depletion threshold intercept
    aY     = max(1.0, float(params["eth_aY"]))        # [count·min] fY half-saturation
    aAC    = max(1.0, float(params["eth_aAC"]))       # [count] fAC half-saturation
    ah     = max(1.0, float(params["eth_ah"]))        # [count] high-intensity threshold
    n1     = max(1.0, float(params["eth_n1"]))        # [-] fY Hill coefficient
    n2     = max(1.0, float(params["eth_n2"]))        # [-] fAC/fHI Hill coefficient
    tp     = max(1e-6, float(params["eth_tp"]))       # [min] fp half-saturation

    # Guard all states to non-negative (projection is also applied by clip_state_trajectory)
    AC       = max(0.0, ac_t)
    Y_s      = max(0.0, Y)
    Z_s      = max(0.0, Z)
    rGU_s    = max(0.0, rGU)
    rGP_s    = max(0.0, rGP)
    tPA_s    = max(0.0, tPA)
    PAint_s  = max(0.0, PAint)
    rdepl_s  = max(0.0, rdepl)
    th_s     = max(0.0, th)
    Q1_s     = max(0.0, Q1)
    x1_s     = max(0.0, x1)

    # --- Transfer functions (sigmoid-shaped) ---
    # fY: PA detection — rises as Y accumulates, saturates at high Y
    y_ratio  = Y_s / aY
    fY_num   = y_ratio ** n1
    fY       = fY_num / (1.0 + fY_num) if fY_num < 1e15 else 1.0

    # fAC: activity intensity — switches on when AC exceeds aAC (~1000 counts)
    ac_ratio = AC / aAC
    fAC_num  = ac_ratio ** n2
    fAC      = fAC_num / (1.0 + fAC_num) if fAC_num < 1e15 else 1.0

    # fHI: high-intensity switch — activates near anaerobic threshold (ah ~5600 counts)
    ahi_ratio = AC / ah
    fHI_num   = ahi_ratio ** n2
    fHI       = fHI_num / (1.0 + fHI_num) if fHI_num < 1e15 else 1.0

    # fp: anaerobic fraction — proportion of activity that is high-intensity
    th_ratio = th_s / tp
    fp_num   = th_ratio ** n2
    fp       = fp_num / (1.0 + fp_num) if fp_num < 1e15 else 1.0

    # Intensity-weighted rGP parameters: blend aerobic ↔ anaerobic based on fp
    q3 = (1.0 - fp) * q3l + fp * q3h
    q4 = (1.0 - fp) * q4l + fp * q4h

    # ft: glycogen depletion fraction — rises as cumulative exercise approaches t_depl
    # t_depl is the estimated time-to-depletion given current average intensity PAint/tPA
    if tPA_s > 1e-6 and PAint_s > 1e-6:
        avg_intensity = PAint_s / tPA_s
        t_depl = max(1e-3, -adepl * avg_intensity + bdepl)
        tpa_depl_ratio = tPA_s / t_depl
        ft_num = tpa_depl_ratio ** n1
        ft = ft_num / (1.0 + ft_num) if ft_num < 1e15 else 1.0
    else:
        ft = 0.0

    # rm: maximum EGP rate (current rGP serves as the ceiling for glycogen replenishment)
    rm = rGP_s

    # --- State derivatives ---

    # Y: short-term PA insulin sensitivity accumulator (τ = tau_AC ≈ 5 min)
    dY = (-1.0 / tau_AC) * Y_s + (1.0 / tau_AC) * AC

    # Z: long-term post-exercise SI elevation — driven by fY*Y during exercise,
    #    decays with τ = tau_Z ≈ 600 min (~10h) after exercise stops (fY → 0).
    # EXPERIMENTAL: saturation factor (1 - Z/Z_max) caps multi-day accumulation.
    # When Z << Z_max (single session) the drive is unchanged; as Z → Z_max it shuts off.
    Z_sat_factor = max(0.0, 1.0 - Z_s / Z_max)
    dZ = b * fY * Y_s * Z_sat_factor - (1.0 - fY) / tau_Z * Z_s

    # rGU: exercise glucose utilization rate (rises with activity, decays after)
    drGU = q1 * fY * Y_s - q2 * rGU_s

    # rGP: exercise glucose production rate (q3/q4 blend aerobic↔anaerobic via fp)
    drGP = q3 * fY * Y_s - q4 * rGP_s

    # tPA: PA tracking state — sigmoid step up when active, step down at rest
    dtPA = fAC - (1.0 - fAC) * tPA_s

    # PAint: cumulative PA intensity integral for glycogen depletion threshold
    dPAint = fAC * AC - (1.0 - fAC) * PAint_s

    # rdepl: glycogen depletion rate — rises as ft approaches 1 (glycogen nearly exhausted)
    #        q6 = 0 for T1D-validated aerobic; non-zero for prolonged/anaerobic (EXPERIMENTAL)
    drdepl = q6 * (ft * rm - rdepl_s)

    # th: high-intensity duration accumulator — drives fp (anaerobic fraction)
    #     q5 = 0 in T1D aerobic-only data; non-zero in params_standard (EXPERIMENTAL)
    dth = fHI - (1.0 - fHI) * q5 * th_s

    # --- Q1 interaction terms (grafted onto Hovorka dQ1) ---

    # exercise_uptake: insulin-independent glucose uptake during exercise [mmol/min]
    # Equivalent to Rashid's QE21 term but driven by rGU instead of E2²·x1
    exercise_uptake = rGU_s * Q1_s

    # exercise_prod: net exercise EGP contribution [mmol/min]
    # rGP increases endogenous production during exercise; rdepl progressively
    # limits it as glycogen depletes (prolonged exercise → late hypoglycemia)
    exercise_prod = max(0.0, rGP_s - rdepl_s) * Q1_s

    # exercise_si: post-exercise insulin sensitivity boost [mmol/min]
    # Z amplifies the x1 insulin action on Q1 for up to ~10h after exercise
    exercise_si = Z_s * x1_s * Q1_s

    return {
        "dY": dY,
        "dZ": dZ,
        "drGU": drGU,
        "drGP": drGP,
        "dtPA": dtPA,
        "dPAint": dPAint,
        "drdepl": drdepl,
        "dth": dth,
        "exercise_uptake": exercise_uptake,
        "exercise_prod": exercise_prod,
        "exercise_si": exercise_si,
    }
