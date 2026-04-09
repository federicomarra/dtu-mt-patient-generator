from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

    n_patients: int = 100
    n_days: int = 7
    international_unit: bool = True
    noise_std: float = 0.33  # real Dexcom/Libre MARD 8-10% → ±0.56-0.70 mmol/L at 7 mmol/L mean; AR(1) stationary std = noise_std
    noise_autocorr: float = 0.7   # AR(1) φ coefficient for the lagged CGM noise model
    cgm_lag_alpha: float = 0.25   # first-order CGM lag blend factor (0 < α ≤ 1); 0.25 ≈ 4-min physiological lag
    random_scenarios: bool = False
    fixed_scenario: int = 1  # base scenario for all patients when random_scenarios=False: 1=normal, 2=active aerobic, 3=sedentary
    clip_states: bool = True
    enable_plots: bool = True
    random_seed: Optional[int] = None
    basal_hourly: float = 0.5
    use_calibrated_basal: bool = True
    initial_target_glucose_mgdl: float = 126.0  # ~7.0 mmol/L: upper end of ADA pre-meal target (80–130 mg/dL); representative of real-world T1D moderate control rather than near-euglycaemic lab conditions
    initial_glucose_acceptance_min_mmol: float = 4.5
    initial_glucose_acceptance_max_mmol: float = 7.2
    instability_max_glucose_mmol: float = 33.3   # 600 mg/dL / 18.016 — raised from 550 to reduce
    # rejection from cortisol/dawn-driven single-day peaks while remaining within
    # physiologically possible T1D range (DKA onset >600 mg/dL)
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
    # Two-tier hypo rejection for non-exercise days (sc1,3,4,5,6):
    #   Tier 1 — per-day hard cap: reject immediately if any non-exercise day exceeds
    #     quality_max_hypo_pct_threshold (15%). A single terrible day at 15% TBR (216 min in
    #     hypo) sits at the outer edge of what literature documents for T1D adults without exercise
    #     (Battelino 2019 consensus: >10% = attention threshold, >25% = high risk). Values above
    #     15% on a non-exercise day indicate a fundamental dosing or sensitivity problem.
    #   Tier 2 — chronic pattern check: count non-exercise days that exceed
    #     quality_max_hypo_pct_soft_threshold (10%). Reject if more than
    #     quality_max_hypo_bad_nonex_days (2) such days occur across the simulation. One or two
    #     rough non-exercise days is physiologically realistic; three or more signals chronic
    #     instability. The 10% soft threshold matches Battelino 2019's "attention needed" level.
    # Exercise days (sc2,7,8,9) use quality_max_hypo_pct_exercise_threshold (17%) plus the
    # spillover bonuses and do NOT count toward the non-exercise chronic-pattern check.
    # Ordering coherence: exercise max (17–21% with spillover) > non-exercise hard cap (15%) ✓
    quality_max_hypo_pct_threshold: float = 15.0          # non-exercise per-day hard cap
    quality_max_hypo_pct_soft_threshold: float = 10.0     # counting threshold for chronic-pattern check
    quality_max_hypo_bad_nonex_days: int = 2              # max allowed non-exercise days above soft threshold
    quality_max_hypo_pct_exercise_threshold: float = 17.0
    # Bonus added to the hypo threshold per exercise day found in the 2-day lookback window
    # (scenarios 2,7,8,9). The Z state (post-exercise insulin sensitivity, tau_Z≈600 min)
    # pre-loads at ~40% for 1 day prior and compounds further for 2 consecutive exercise days.
    # Each exercise day in the [d-2, d-1] window contributes this bonus independently:
    #   non-ex→exercise→any:      base + 2%  (d-1 spillover)
    #   exercise→non-ex→any:      base + 2%  (d-2 spillover; Z still ~9% after 2 days)
    #   exercise→exercise→non-ex: 17% + 2% + 2% = 21%
    #   exercise→exercise→exercise: 17% + 2% + 2% = 21%
    quality_max_hypo_pct_spillover_bonus: float = 2.0
    quality_max_hyper_pct_threshold: float = 70.0
    # Hard floor: reject if glucose drops below this at any point regardless of hypo%.
    # Set to 36 mg/dL (= 2.0 mmol/L). With the tighter safety stack (guard 3.9, L1 rescue
    # 3.9, L2 rescue 3.0) floor breaches should be rare; the floor guards against the tail
    # of cases where L2 still cannot arrest the fall.
    quality_min_glucose_mmol: float = 2.0  # 36 mg/dL / 18.016

    # Burn-in days run the same scenario before recording starts, letting the ETH exercise
    # states (Y, Z post-exercise insulin sensitivity) reach a cyclic steady state so that
    # Day 1 of the recorded horizon does not look artificially "clean" vs later days.
    n_warmup_days: int = 3

    enable_hypo_guard: bool = True
    hypo_guard_mmol: float = 3.9           # ADA/Battelino 2019 Level 1 alert threshold
    hypo_guard_suspend_min: int = 20
    hypo_guard_retrigger_cooldown_min: int = 20
    suppress_meal_bolus_on_guard: bool = False

    # Two-tier hypo rescue (ADA/Battelino 2019):
    #   L1 (3.0–3.9 mmol/L): treat with 15 g fast carbs, suspend basal (guard already fires simultaneously)
    #   L2 (<3.0 mmol/L):    treat with 30 g carbs; independent cooldown so L2 can fire
    #                         even if L1 recently fired — at 3.0 you do not wait.
    enable_hypo_rescue: bool = True
    hypo_rescue_trigger_mmol: float = 3.9  # L1: fires at same threshold as hypo guard
    hypo_rescue_carbs_g: float = 15.0
    hypo_rescue_duration_min: int = 15
    hypo_rescue_retrigger_cooldown_min: int = 45

    enable_hypo_rescue_l2: bool = True
    hypo_rescue_l2_trigger_mmol: float = 3.0  # L2: serious hypoglycaemia floor
    hypo_rescue_l2_carbs_g: float = 30.0
    hypo_rescue_l2_duration_min: int = 15
    hypo_rescue_l2_retrigger_cooldown_min: int = 60

    solver_method: str = "RK45"
    solver_max_step: float = 1.0
    derivative_clip: float = 1e5
    std_patient: bool = False

    init_insulin_carbo_ratio: float = 11.8
    init_insulin_sensitivity_factor: float = 2.8

    # Target glucose for ICR/ISF bisection calibration [mmol/L].
    # At 5.5 mmol/L (euglycaemia), ICR/ISF are tuned to return glucose to near-normal after
    # every meal/correction — producing overly well-controlled virtual patients (TIR ~89%,
    # hyper ~7%). Raising to 6.5 mmol/L calibrates ICR to deliver less insulin per gram of
    # carbs (stopping at a higher post-meal plateau) and ISF to correct less aggressively,
    # producing sustained post-meal excursions above 10 mmol/L consistent with HbA1c ~7.5–8%
    # and real-world T1D moderate control (T1D Exchange 2016 median HbA1c 8.4%).
    calibration_target_glycemia_mmol: float = 6.5

    enable_iob_bolus_guard: bool = True
    iob_guard_units: float = 4.0
    iob_full_attenuation_units: float = 8.0
    iob_max_icr_multiplier: float = 1.6

    # Lower clamp for CGM readings. Real sensors report "LOW" below ~2.2 mmol/L;
    # AR(1) noise accumulation can otherwise push readings below 1.3 mmol/L even
    # when true glucose is at the 2.0 mmol/L rejection floor.
    cgm_min_glucose_mmol: float = 1.5

    enable_correction_isf: bool = True
    correction_isf_target_mmol: float = 10.5  # ~189 mg/dL: fires when glucose first exceeds 10.5 mmol/L, capping the duration of post-meal excursions and preventing the 10–12.5 mmol/L "dead zone" that caused chronic quality_hyper rejections at long horizons (14+ days). IOB guard prevents double-dosing.
    correction_isf_check_interval_min: int = 5
    correction_isf_cooldown_min: int = 90
    correction_isf_max_bolus_units: float = 2.0
    correction_isf_min_bolus_units: float = 0.05
    correction_isf_bolus_duration_min: int = 5
    correction_isf_iob_free_units: float = 0.5
