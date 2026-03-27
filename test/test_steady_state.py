"""
Comprehensive fasting steady-state verification test.

Three levels of verification for each target glucose:
  1. Glucose accuracy   — returned state glucose matches target within tight tolerance
  2. Fixed-point check  — all 18 ODE derivatives are ~0 at the returned state
  3. Integration stability — forward ODE integration with basal-only for 60 min stays flat

Also tests the mg/dL input path (what the main simulation uses) in addition to mmol/L.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import (
    compute_optimal_steady_state_from_glucose,
    hovorka_equations,
    get_glucose_from_state,
)
from src.parameters import get_base_params, generate_monte_carlo_patients

# ── Tolerances ────────────────────────────────────────────────────────────────
GLUCOSE_TOLERANCE_MMOL   = 0.05   # max acceptable |G_actual - G_target| [mmol/L]
DERIVATIVE_TOLERANCE     = 1e-4   # max acceptable |dX/dt| at steady state
DRIFT_TOLERANCE_MMOL     = 0.10   # max acceptable glucose drift over 60 min integration

TARGETS_MMOL   = [4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
TARGETS_MGDL   = [81.0, 90.0, 100.0, 108.0, 126.0, 144.0]   # same values, different units
N_RANDOM_PATIENTS = 5


def _basal_from_state(state: list[float], params: dict) -> float:
    """Derive calibrated basal [mU/min] from S1 at steady state."""
    tau_i = float(params["tauI"])
    return float(state[2]) / tau_i if tau_i > 0 else 0.5 * 1000.0 / 60.0


def _run_level1_glucose_accuracy(state, target_mmol, params, label):
    """Level 1: returned glucose matches target within GLUCOSE_TOLERANCE_MMOL."""
    glucose = get_glucose_from_state(state, params)
    err = abs(glucose - target_mmol)
    assert err <= GLUCOSE_TOLERANCE_MMOL, (
        f"[{label}] Level 1 FAILED: target={target_mmol:.3f}, got={glucose:.4f}, "
        f"err={err:.4f} mmol/L (tol={GLUCOSE_TOLERANCE_MMOL})"
    )
    return glucose


def _run_level2_fixed_point(state, params, label):
    """Level 2: all ODE derivatives are ~0 at the returned state."""
    basal = _basal_from_state(state, params)

    def basal_only(*_args, **_kwargs):
        return basal, 0.0, 0.0

    derivs = np.array(
        hovorka_equations(
            t=0,
            x=state,
            params=params,
            input_func=basal_only,
            scenario=1,
            precomputed_inputs=(basal, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    max_deriv = float(np.max(np.abs(derivs)))
    assert max_deriv <= DERIVATIVE_TOLERANCE, (
        f"[{label}] Level 2 FAILED: max |dX/dt|={max_deriv:.3e} (tol={DERIVATIVE_TOLERANCE}). "
        f"Per-state: {np.round(derivs, 8)}"
    )
    return max_deriv


def _run_level3_integration_stability(state, params, label):
    """Level 3: forward ODE integration with basal-only for 60 min stays flat."""
    basal = _basal_from_state(state, params)
    vg_bw = float(params["VG"]) * float(params["BW"])
    g0 = float(state[0]) / vg_bw

    def ode_func(t: float, x: np.ndarray) -> np.ndarray:
        x_safe = np.nan_to_num(np.asarray(x, dtype=np.float64), copy=True, nan=0.0)
        result = hovorka_equations(
            int(t), x_safe, params,
            input_func=lambda *a, **k: (basal, 0.0, 0.0),
            scenario=1,
            precomputed_inputs=(basal, 0.0, 0.0),
        )
        return np.asarray(result, dtype=np.float64)

    sol = solve_ivp(
        ode_func,
        (0, 60),
        np.array(state, dtype=np.float64),
        method="RK45",
        t_eval=np.arange(0, 61),
        rtol=1e-8,
        atol=1e-10,
        max_step=1.0,
    )

    glucose_trace = sol.y[0, :] / vg_bw
    drift = float(np.max(np.abs(glucose_trace - g0)))
    assert drift <= DRIFT_TOLERANCE_MMOL, (
        f"[{label}] Level 3 FAILED: glucose drifted {drift:.4f} mmol/L over 60 min "
        f"(tol={DRIFT_TOLERANCE_MMOL}). Start={g0:.4f}, End={glucose_trace[-1]:.4f}"
    )
    return drift


def run_all_tests():
    params = get_base_params()
    passed = 0
    failed = 0

    print("=" * 70)
    print("STEADY STATE TEST — base patient, mmol/L inputs")
    print("=" * 70)
    for target in TARGETS_MMOL:
        label = f"base  {target:.1f} mmol/L"
        try:
            state = compute_optimal_steady_state_from_glucose(
                params, target, international_units=True, max_iterations=100, print_progress=False,
            )
            g = _run_level1_glucose_accuracy(state, target, params, label)
            d = _run_level2_fixed_point(state, params, label)
            drift = _run_level3_integration_stability(state, params, label)
            print(f"  PASS  {label}: G={g:.4f}, max|dX/dt|={d:.2e}, drift={drift:.4f} mmol/L")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failed += 1

    print()
    print("=" * 70)
    print("STEADY STATE TEST — base patient, mg/dL inputs (simulation path)")
    print("=" * 70)
    MwG = float(params["MwG"])
    for target_mgdl in TARGETS_MGDL:
        target_mmol = target_mgdl / (MwG / 10.0)
        label = f"base  {target_mgdl:.0f} mg/dL ({target_mmol:.3f} mmol/L)"
        try:
            state = compute_optimal_steady_state_from_glucose(
                params, target_mgdl, international_units=False, max_iterations=100, print_progress=False,
            )
            g = _run_level1_glucose_accuracy(state, target_mmol, params, label)
            d = _run_level2_fixed_point(state, params, label)
            drift = _run_level3_integration_stability(state, params, label)
            print(f"  PASS  {label}: G={g:.4f}, max|dX/dt|={d:.2e}, drift={drift:.4f} mmol/L")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"STEADY STATE TEST — {N_RANDOM_PATIENTS} random Monte Carlo patients, 100 mg/dL target")
    print("=" * 70)
    patients = generate_monte_carlo_patients(N_RANDOM_PATIENTS, standard_patient=False, seed=42)
    for i, p in enumerate(patients):
        MwG_p = float(p["MwG"])
        target_mmol = 100.0 / (MwG_p / 10.0)
        label = f"mc[{i}] 100 mg/dL ({target_mmol:.3f} mmol/L)"
        try:
            state = compute_optimal_steady_state_from_glucose(
                p, 100.0, international_units=False, max_iterations=100, print_progress=False,
            )
            g = _run_level1_glucose_accuracy(state, target_mmol, p, label)
            d = _run_level2_fixed_point(state, p, label)
            drift = _run_level3_integration_stability(state, p, label)
            print(f"  PASS  {label}: G={g:.4f}, max|dX/dt|={d:.2e}, drift={drift:.4f} mmol/L")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
