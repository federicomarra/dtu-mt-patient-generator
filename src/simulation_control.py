from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # type: ignore[import-untyped]

from src.model import ParameterSet
from src.simulation_config import SimulationConfig


def _empty_correction_windows() -> list[tuple[int, int]]:
    return []


@dataclass
class ControllerState:
    """Mutable per-patient controller state carried across all simulated days."""

    guard_suspend_until_min: int = -1
    rescue_active_until_min: int = -1
    guard_next_trigger_min: int = 0
    rescue_next_trigger_min: int = 0

    rescue_l2_active_until_min: int = -1   # L2 (<3.0 mmol/L) independent cooldown
    rescue_l2_next_trigger_min: int = 0

    correction_isf_active_until_min: int = -1
    correction_isf_next_trigger_min: int = 0
    correction_isf_rate_mU_min: float = 0.0
    correction_isf_windows_abs: list[tuple[int, int]] = field(default_factory=_empty_correction_windows)
    correction_isf_events: int = 0
    correction_isf_units: float = 0.0


def apply_guard_iob_isf(
    current_abs_min: int,
    g_est: float,
    iob_u: float,
    basal_hourly_patient: float,
    insulin_carbo_ratio_patient: float,
    insulin_sensitivity_patient: float,
    config: SimulationConfig,
    state: ControllerState,
) -> tuple[float, float, bool]:
    """Return effective basal/ICR at current minute after all insulin-side control policies."""
    basal_hourly_effective = basal_hourly_patient
    insulin_carbo_ratio_effective = insulin_carbo_ratio_patient

    guard_latched = False
    if config.enable_hypo_guard:
        if (
            g_est <= config.hypo_guard_mmol
            and current_abs_min > state.guard_suspend_until_min
            and current_abs_min >= state.guard_next_trigger_min
        ):
            state.guard_suspend_until_min = current_abs_min + max(1, config.hypo_guard_suspend_min) - 1
            state.guard_next_trigger_min = (
                state.guard_suspend_until_min + max(0, config.hypo_guard_retrigger_cooldown_min)
            )
        if current_abs_min <= state.guard_suspend_until_min:
            guard_latched = True
            basal_hourly_effective = 0.0
            if config.suppress_meal_bolus_on_guard:
                insulin_carbo_ratio_effective = 1e6

    if config.enable_iob_bolus_guard:
        iob_guard = max(0.0, config.iob_guard_units)
        iob_full = max(iob_guard + 1e-6, config.iob_full_attenuation_units)
        if iob_u > iob_guard:
            frac = (iob_u - iob_guard) / (iob_full - iob_guard)
            frac = min(1.0, max(0.0, frac))
            icr_mult_max = max(1.0, config.iob_max_icr_multiplier)
            icr_mult = 1.0 + frac * (icr_mult_max - 1.0)
            insulin_carbo_ratio_effective = insulin_carbo_ratio_effective * icr_mult

    if config.enable_correction_isf:
        check_interval_min = max(1, int(config.correction_isf_check_interval_min))
        is_check_minute = (current_abs_min % check_interval_min) == 0
        if current_abs_min > state.correction_isf_active_until_min:
            state.correction_isf_rate_mU_min = 0.0

        if (
            (not guard_latched)
            and is_check_minute
            and g_est > config.correction_isf_target_mmol
            and current_abs_min > state.correction_isf_active_until_min
            and current_abs_min >= state.correction_isf_next_trigger_min
        ):
            isf_eff = max(1e-6, insulin_sensitivity_patient)
            raw_correction_u = (g_est - config.correction_isf_target_mmol) / isf_eff
            iob_credit_u = max(0.0, iob_u - max(0.0, config.correction_isf_iob_free_units))
            net_correction_u = max(0.0, raw_correction_u - iob_credit_u)
            dose_u = min(max(0.0, config.correction_isf_max_bolus_units), net_correction_u)

            if dose_u >= max(0.0, config.correction_isf_min_bolus_units):
                corr_duration = max(1, config.correction_isf_bolus_duration_min)
                state.correction_isf_rate_mU_min = (dose_u * 1000.0) / corr_duration
                corr_start = current_abs_min
                corr_end = current_abs_min + corr_duration - 1
                state.correction_isf_active_until_min = corr_end
                state.correction_isf_next_trigger_min = corr_end + max(0, config.correction_isf_cooldown_min)
                state.correction_isf_windows_abs.append((corr_start, corr_end))
                state.correction_isf_events += 1
                state.correction_isf_units += float(dose_u)

        if current_abs_min <= state.correction_isf_active_until_min and not guard_latched:
            basal_hourly_effective += state.correction_isf_rate_mU_min * 60.0 / 1000.0

    return basal_hourly_effective, insulin_carbo_ratio_effective, guard_latched


def apply_hypo_rescue_to_derivative(
    dy: np.ndarray,
    current_abs_min: int,
    g_est: float,
    patient_params: ParameterSet,
    config: SimulationConfig,
    state: ControllerState,
) -> None:
    """Mutate ODE derivative with gut-delivered rescue carbs when rescue is active.

    Two-tier rescue (ADA/Battelino 2019):
      L1 (≤ hypo_rescue_trigger_mmol, default 3.9): 15 g fast carbs, 45-min cooldown.
          Fires simultaneously with the hypo guard basal suspend.
      L2 (≤ hypo_rescue_l2_trigger_mmol, default 3.0): 30 g carbs, 60-min cooldown.
          Independent cooldown — L2 can fire even while L1 cooldown is active, because
          at <3.0 mmol/L waiting is not acceptable.
    Both tiers can be active simultaneously; their carb rates are summed.
    """
    ag  = float(patient_params["Ag"])
    mwg = float(patient_params["MwG"])

    # --- L1 rescue ---
    if (
        config.enable_hypo_rescue
        and g_est <= config.hypo_rescue_trigger_mmol
        and current_abs_min > state.rescue_active_until_min
        and current_abs_min >= state.rescue_next_trigger_min
    ):
        state.rescue_active_until_min = current_abs_min + max(1, config.hypo_rescue_duration_min) - 1
        state.rescue_next_trigger_min = state.rescue_active_until_min + max(
            0, config.hypo_rescue_retrigger_cooldown_min
        )

    if current_abs_min <= state.rescue_active_until_min:
        l1_rate_mg_min = max(0.0, config.hypo_rescue_carbs_g) * 1000.0 / max(1, config.hypo_rescue_duration_min)
        dy[8] += ag * (l1_rate_mg_min / mwg)

    # --- L2 rescue (independent cooldown) ---
    if (
        config.enable_hypo_rescue_l2
        and g_est <= config.hypo_rescue_l2_trigger_mmol
        and current_abs_min > state.rescue_l2_active_until_min
        and current_abs_min >= state.rescue_l2_next_trigger_min
    ):
        state.rescue_l2_active_until_min = current_abs_min + max(1, config.hypo_rescue_l2_duration_min) - 1
        state.rescue_l2_next_trigger_min = state.rescue_l2_active_until_min + max(
            0, config.hypo_rescue_l2_retrigger_cooldown_min
        )

    if current_abs_min <= state.rescue_l2_active_until_min:
        l2_rate_mg_min = max(0.0, config.hypo_rescue_l2_carbs_g) * 1000.0 / max(1, config.hypo_rescue_l2_duration_min)
        dy[8] += ag * (l2_rate_mg_min / mwg)


def count_correction_active_points(
    windows_abs: list[tuple[int, int]],
    horizon_end_abs_min: int,
) -> int:
    """Count total minutes with active ISF correction windows within a horizon."""
    points = 0
    for corr_start, corr_end in windows_abs:
        start = max(0, corr_start)
        end = min(horizon_end_abs_min, corr_end)
        if end >= start:
            points += end - start + 1
    return points


def estimate_iob_from_state(state_trajectory: np.ndarray) -> np.ndarray:
    """Compute minute-wise IOB estimate [U] from S1+S2 depot masses [mU]."""
    return np.maximum(np.asarray(state_trajectory[2, :] + state_trajectory[3, :], dtype=np.float64), 0.0) / 1000.0
