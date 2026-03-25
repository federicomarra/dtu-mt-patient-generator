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
    noise_autocorr: float = 0.7
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
    instability_max_glucose_mmol: float = 17.0
    instability_hyper_pct_threshold: float = 30.0
    # Rejection thresholds applied on a worst-day basis (see simulation.py rejection logic).
    # Exercise days (scenarios 2, 7, 8, 9) use the _exercise variants because physiological
    # hypo during/after vigorous exercise is expected and the rescue system handles it.
    #
    # NOTE (thesis): When scenario 2 (active afternoon) is fixed for all simulated days,
    # acceptance rates are significantly lower (~15-20%) than for mixed-scenario runs (~45%).
    # This is expected and physiologically justified: running aerobic exercise every day for
    # 3+ consecutive days with burn-in causes cumulative post-exercise insulin sensitivity
    # elevation (ETH Z-state) that persistently lowers fasting glucose. Most virtual patients
    # cannot sustain acceptable glycaemic control under this chronic load within the 8%
    # per-day hypo threshold. This finding reflects a genuine limitation of fixed-exercise
    # cohort generation and should be reported as such in validation experiments.
    quality_max_hypo_pct_threshold: float = 4.0
    quality_max_hypo_pct_exercise_threshold: float = 8.0
    quality_max_hyper_pct_threshold: float = 12.0
    # Hard floor: reject if glucose drops below this at any point regardless of hypo%.
    # Prevents deeply hypoglycaemic spikes in accepted patients (e.g. 2.3 mmol/L).
    quality_min_glucose_mmol: float = 3.0

    # Burn-in days run the same scenario before recording starts, letting the ETH exercise
    # states (Y, Z post-exercise insulin sensitivity) reach a cyclic steady state so that
    # Day 1 of the recorded horizon does not look artificially "clean" vs later days.
    n_warmup_days: int = 3

    enable_hypo_guard: bool = True
    hypo_guard_mmol: float = 4.2
    hypo_guard_suspend_min: int = 20
    hypo_guard_retrigger_cooldown_min: int = 20
    suppress_meal_bolus_on_guard: bool = False

    enable_hypo_rescue: bool = True
    hypo_rescue_trigger_mmol: float = 3.9
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
    correction_isf_target_mmol: float = 6.5
    correction_isf_cooldown_min: int = 60
    correction_isf_max_bolus_units: float = 2.0
    correction_isf_min_bolus_units: float = 0.05
    correction_isf_bolus_duration_min: int = 5
    correction_isf_iob_free_units: float = 0.5
