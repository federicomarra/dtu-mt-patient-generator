from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

    n_patients: int = 100
    n_days: int = 7
    international_unit: bool = True
    noise_std: float = 0.10
    noise_autocorr: float = 0.7   # AR(1) φ coefficient for the lagged CGM noise model
    cgm_lag_alpha: float = 0.25   # first-order CGM lag blend factor (0 < α ≤ 1); 0.25 ≈ 4-min physiological lag
    random_scenarios: bool = False
    fixed_scenario: int = 1
    clip_states: bool = True
    enable_plots: bool = True
    random_seed: Optional[int] = None
    basal_hourly: float = 0.5
    use_calibrated_basal: bool = True
    initial_target_glucose_mgdl: float = 100.0
    initial_glucose_acceptance_min_mmol: float = 4.5
    initial_glucose_acceptance_max_mmol: float = 7.2
    instability_max_glucose_mmol: float = 30.53  # 550 mg/dL / 18.016
    instability_hyper_pct_threshold: float = 60.0
    # Rejection thresholds applied on a worst-day basis (see simulation.py rejection logic).
    # Exercise days (scenarios 2, 7, 8, 9) use the _exercise variants because physiological
    # hypo during/after vigorous exercise is expected and the rescue system handles it.
    #
    # NOTE (thesis): When scenario 2 (active afternoon) is fixed for all simulated days,
    # acceptance rates are significantly lower (~15-20%) than for mixed-scenario runs (~45%).
    # This is expected and physiologically justified: running aerobic exercise every day for
    # 3+ consecutive days with burn-in causes cumulative post-exercise insulin sensitivity
    # elevation (ETH Z-state) that persistently lowers fasting glucose. Most virtual patients
    # cannot sustain acceptable glycaemic control under this chronic load within the 15%
    # per-day hypo threshold. This finding reflects a genuine limitation of fixed-exercise
    # cohort generation and should be reported as such in validation experiments.
    quality_max_hypo_pct_threshold: float = 10.0
    quality_max_hypo_pct_exercise_threshold: float = 15.0
    # Bonus added to the hypo threshold per exercise day found in the 2-day lookback window
    # (scenarios 2,7,8,9). The Z state (post-exercise insulin sensitivity, tau_Z≈600 min)
    # pre-loads at ~40% for 1 day prior and compounds further for 2 consecutive exercise days.
    # Each exercise day in the [d-2, d-1] window contributes this bonus independently:
    #   non-ex→exercise→any:      base + 2%  (d-1 spillover)
    #   exercise→non-ex→any:      base + 2%  (d-2 spillover; Z still ~9% after 2 days)
    #   exercise→exercise→non-ex: 10% + 2% + 2% = 14%
    #   exercise→exercise→exercise: 15% + 2% + 2% = 19%
    quality_max_hypo_pct_spillover_bonus: float = 2.0
    quality_max_hyper_pct_threshold: float = 75.0
    # Hard floor: reject if glucose drops below this at any point regardless of hypo%.
    # Set to 36 mg/dL (= 2.0 mmol/L) — slightly tighter than the previous 1.78 to
    # compensate for the looser safety stack (guard 4.0, rescue 3.5, ISF target 10.0),
    # keeping CGM sensor readings (which can undershoot true glucose by ~0.7 mmol/L
    # due to AR(1) noise) above ~1.3 mmol/L in the worst case.
    quality_min_glucose_mmol: float = 2.0  # 36 mg/dL / 18.016

    # Burn-in days run the same scenario before recording starts, letting the ETH exercise
    # states (Y, Z post-exercise insulin sensitivity) reach a cyclic steady state so that
    # Day 1 of the recorded horizon does not look artificially "clean" vs later days.
    n_warmup_days: int = 3

    enable_hypo_guard: bool = True
    hypo_guard_mmol: float = 4.0
    hypo_guard_suspend_min: int = 20
    hypo_guard_retrigger_cooldown_min: int = 20
    suppress_meal_bolus_on_guard: bool = False

    enable_hypo_rescue: bool = True
    hypo_rescue_trigger_mmol: float = 3.5
    hypo_rescue_carbs_g: float = 15.0
    hypo_rescue_duration_min: int = 15
    hypo_rescue_retrigger_cooldown_min: int = 45

    solver_method: str = "RK45"
    solver_max_step: float = 1.0
    derivative_clip: float = 1e5
    std_patient: bool = False

    init_insulin_carbo_ratio: float = 11.8
    init_insulin_sensitivity_factor: float = 2.8

    enable_iob_bolus_guard: bool = True
    iob_guard_units: float = 4.0
    iob_full_attenuation_units: float = 8.0
    iob_max_icr_multiplier: float = 1.6

    enable_correction_isf: bool = True
    correction_isf_target_mmol: float = 14.0
    correction_isf_check_interval_min: int = 5
    correction_isf_cooldown_min: int = 90
    correction_isf_max_bolus_units: float = 2.0
    correction_isf_min_bolus_units: float = 0.05
    correction_isf_bolus_duration_min: int = 5
    correction_isf_iob_free_units: float = 0.5
