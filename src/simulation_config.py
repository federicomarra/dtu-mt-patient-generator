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
    clip_states: bool = True
    random_seed: Optional[int] = None
    basal_hourly: float = 0.5
    use_calibrated_basal: bool = True
    initial_target_glucose_mgdl: float = 100.0

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
    iob_guard_units: float = 3.0
    iob_full_attenuation_units: float = 6.0
    iob_max_icr_multiplier: float = 2.0

    enable_correction_isf: bool = True
    correction_isf_target_mmol: float = 6.5
    correction_isf_cooldown_min: int = 60
    correction_isf_max_bolus_units: float = 2.0
    correction_isf_min_bolus_units: float = 0.05
    correction_isf_bolus_duration_min: int = 5
    correction_isf_iob_free_units: float = 0.5
