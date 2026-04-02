from __future__ import annotations

import numpy as np
from typing import TypedDict, Optional, Tuple

N_SCENARIOS: int = 9

# Sampling weights for scenarios 1-9.
# Scenarios 1-3 (normal, active, sedentary) — common day-to-day patterns.
# Scenarios 4-6 (restaurant meal, missed bolus, late bolus) — meal anomalies, rarer.
# Scenarios 7-9 (prolonged aerobic, anaerobic, exercise + missed bolus) — rare exercise anomalies.
#
# Target: ~3:1 normal-to-anomaly ratio. Scenarios 7-9 are intentionally rarer
# than 4-6 to reflect real-world prevalence of prolonged/anaerobic exercise days.
#
# Scenario 2 (active) is deliberately lower than 1 and 3 (18% vs 25/27%) because:
#   - Each exercise day causes Z-state spillover into the next day (~10h tau_Z), effectively
#     making two consecutive days exercise-affected; reducing frequency limits this coupling.
#   - Real-world adherence to structured afternoon exercise in T1D adults is ~3 sessions/week
#     (~43%), but scenario 2 represents a specific afternoon workout, not incidental activity.
#   - Acceptance rate is sensitive to scenario 2 frequency due to the cross-day hypo spillover.
#
# ML NOTE — scenario 4 (restaurant meal) combines a larger carb load with systematic
# bolus underestimation and extended absorption, applied to either lunch or dinner (50/50).
# This gives a strong post-prandial glucose excursion and is a clean ML target.
# Scenario 8 (anaerobic) uses EXPERIMENTAL parameters — see hovorka_exercise.py.
_SCENARIO_WEIGHTS_RAW: list[float] = [0.25, 0.18, 0.20, 0.08, 0.08, 0.10, 0.04, 0.04, 0.03]
SCENARIO_WEIGHTS: list[float] = [
    w / sum(_SCENARIO_WEIGHTS_RAW) for w in _SCENARIO_WEIGHTS_RAW
]


# ============================================================================
# Time Conversion Helper
# ============================================================================

def time_to_minutes(h: int, m: int) -> int:
    """Convert hours and minutes to total minutes."""
    return h * 60 + m


# ============================================================================
# Meal Parameters (Constants)
# Tuned for realistic 80kg active T1D patient with frequent meals and activity
# ============================================================================

# Breakfast: larger, leisurely eating (15-20 min) at consistent time
BREAKFAST_CARBS_MIN: int = 45
BREAKFAST_CARBS_MAX: int = 70
BREAKFAST_TIME_MIN: int = time_to_minutes(6, 30)
BREAKFAST_TIME_MAX: int = time_to_minutes(8, 0)
BREAKFAST_DURATION_MIN: int = 15
BREAKFAST_DURATION_MAX: int = 20

# Morning snack: light, quick (8-12 min) mid-morning snack for activity buffer
MORNING_SNACK_CARBS_MIN: int = 12
MORNING_SNACK_CARBS_MAX: int = 20
MORNING_SNACK_TIME_MIN: int = time_to_minutes(10, 0)
MORNING_SNACK_TIME_MAX: int = time_to_minutes(11, 0)
MORNING_SNACK_DURATION_MIN: int = 8
MORNING_SNACK_DURATION_MAX: int = 12

# Lunch: moderate (50-75g) with realistic 20-25 min eating window
LUNCH_CARBS_MIN: int = 50
LUNCH_CARBS_MAX: int = 75                       # increase to 90 or even higher just to see some variability
LUNCH_TIME_MIN: int = time_to_minutes(12, 0)
LUNCH_TIME_MAX: int = time_to_minutes(13, 0)
LUNCH_DURATION_MIN: int = 20
LUNCH_DURATION_MAX: int = 25

# Afternoon snack: light, mid-afternoon (15:00-16:30) for activity afternoon
AFTERNOON_SNACK_CARBS_MIN: int = 12
AFTERNOON_SNACK_CARBS_MAX: int = 25
AFTERNOON_SNACK_TIME_MIN: int = time_to_minutes(15, 0)
AFTERNOON_SNACK_TIME_MAX: int = time_to_minutes(16, 30)
AFTERNOON_SNACK_DURATION_MIN: int = 5
AFTERNOON_SNACK_DURATION_MAX: int = 10

# Realistic snack adherence variability: some days one snack is skipped,
# and more rarely both snacks are skipped.
SNACK_SKIP_ONE_PROB: float = 0.20
SNACK_SKIP_BOTH_PROB: float = 0.07

# Dinner: moderate with realistic 20-25 min and earlier timing for active schedule
DINNER_CARBS_MIN: int = 50
DINNER_CARBS_MAX: int = 75
DINNER_TIME_MIN: int = time_to_minutes(18, 30)
DINNER_TIME_MAX: int = time_to_minutes(20, 0)
DINNER_DURATION_MIN: int = 20
DINNER_DURATION_MAX: int = 25

# Periodic larger main meal (lunch or dinner, not both): once every 1-2 weeks.
HIGH_MAIN_MEAL_CARBS_MIN: int = 70
HIGH_MAIN_MEAL_CARBS_MAX: int = 120
HIGH_MAIN_MEAL_GAP_DAYS_MIN: int = 7
HIGH_MAIN_MEAL_GAP_DAYS_MAX: int = 14

# Bolus timing: 15 min pre-meal for fast-paced active schedule
PREANNOUNCED_BOLUS_TIME: int = 15
BOLUS_DURATION: int = 3  # minutes over which the bolus dose is spread

# ============================================================================
# Meal Schedule Type
# ============================================================================

class MealSpec(TypedDict):
    """Specification of a single meal (time window, carbs, duration)."""
    time: int           # [min] when meal starts (minutes from 00:00)
    duration: int       # [min] how long meal lasts
    carbs: int          # [g] total carbs in meal


class MealSchedule(TypedDict):
    """Complete daily meal schedule with 5 regularly-scheduled meals."""
    breakfast: MealSpec
    morning_snack: MealSpec
    lunch: MealSpec
    afternoon_snack: MealSpec
    dinner: MealSpec
    breakfast_bolus_carbs_g: float
    morning_snack_bolus_carbs_g: float
    lunch_bolus_carbs_g: float
    afternoon_snack_bolus_carbs_g: float
    dinner_bolus_carbs_g: float
    missed_meal_id: Optional[int]  # For missed bolus scenario (1-5 or None)
    late_bolus_ids: set[int]          # For late bolus scenario: set of meal IDs (1-5) with late bolus
    late_bolus_delay_min: int


class ExerciseSpec(TypedDict):
    """Specification of one exercise block represented as accelerometer counts (AC)."""
    start: int
    duration: int
    ac_counts: float  # [counts] representative accelerometer intensity for this session


class ExerciseSchedule(TypedDict):
    """Daily exercise schedule with one optional main session and optional anaerobic bursts."""
    enabled: bool
    session: ExerciseSpec
    burst_period_min: int
    burst_on_min: int
    burst_multiplier: float


_PATIENT_SEED_FACTOR: int = 10_000
_SCENARIO_SEED_FACTOR: int = 1_000_000
_DAY_SEED_FACTOR: int = 97
_HIGH_CARB_STREAM_OFFSET: int = 31
_BOLUS_ESTIMATION_STREAM_OFFSET: int = 53
_SMALL_TIME_JITTER_MIN: int = -5
_SMALL_TIME_JITTER_MAX: int = 6
_SMALL_CARB_JITTER_PCT: float = 0.06
_IRREGULAR_DAY_PROBABILITY: float = 0.15
_IRREGULAR_TIME_JITTER_MIN: int = -20
_IRREGULAR_TIME_JITTER_MAX: int = 21

# Bolus uses estimated carbs; gut absorption uses actual meal carbs.
_PATIENT_BOLUS_BIAS_MIN: float = 0.90
_PATIENT_BOLUS_BIAS_MAX: float = 1.05
_MEAL_EST_NOISE_MIN: float = 0.90
_MEAL_EST_NOISE_MAX: float = 1.10
_LUNCH_DINNER_UNDEREST_MIN: float = 0.90
_LUNCH_DINNER_UNDEREST_MAX: float = 0.98
_SNACK_EST_MIN: float = 0.96
_SNACK_EST_MAX: float = 1.04
# Scenario 4 (restaurant meal): carb load multiplier and systematic bolus underestimation.
# Guests routinely underestimate restaurant portions; these ranges are applied on top of the
# normal meal-estimation noise so the net estimate is well below actual intake.
_RESTAURANT_CARB_FACTOR_MIN: float = 1.8
_RESTAURANT_CARB_FACTOR_MAX: float = 2.1
_RESTAURANT_UNDEREST_MIN: float = 0.70
_RESTAURANT_UNDEREST_MAX: float = 0.85
_LATE_BOLUS_DELAY_MIN: int = 20
_LATE_BOLUS_DELAY_MAX: int = 60

_EXERCISE_START_MIN: int = time_to_minutes(16, 30)
_EXERCISE_START_MAX: int = time_to_minutes(20, 30)
_EXERCISE_DURATION_MIN: int = 30
_EXERCISE_DURATION_MAX: int = 75
# Scenario 7 (prolonged aerobic) minimum is set to 80 min so it never overlaps with
# scenario 2 (max 75 min). This ensures a clean duration separation for ML anomaly
# detection: any scenario 7 session is always longer than any scenario 2 session,
# producing a consistently more pronounced Z/rGU accumulation in the glucose trace.
_EXERCISE_DURATION_PROLONGED_MIN: int = 80   # scenario 7: prolonged aerobic (always > scenario 2 max)
_EXERCISE_DURATION_PROLONGED_MAX: int = 90
_EXERCISE_DURATION_ANAEROBIC_MIN: int = 30   # scenario 8: anaerobic/resistance
_EXERCISE_DURATION_ANAEROBIC_MAX: int = 60
_EXERCISE_OVERLAP_PROBABILITY: float = 0.15
_EXERCISE_MIN_GAP_AFTER_SNACK_MIN: int = 20
_EXERCISE_MIN_GAP_BEFORE_DINNER_MIN: int = 30

_SCENARIO_NORMAL: int = 1
_SCENARIO_ACTIVE_AFTERNOON: int = 2
_SCENARIO_SEDENTARY: int = 3
_SCENARIO_RESTAURANT_MEAL: int = 4
_SCENARIO_MISSED_BOLUS: int = 5
_SCENARIO_LATE_BOLUS: int = 6
_SCENARIO_PROLONGED_AEROBIC: int = 7
_SCENARIO_ANAEROBIC: int = 8
_SCENARIO_EXERCISE_MISSED_BOLUS: int = 9

# Scenario guide:
# 1: Normal day — light incidental AC, no planned workout
# 2: Active day — moderate aerobic session (30-75 min, 1200-2000 AC)
# 3: Sedentary day — minimal AC baseline, no workout
# 4: Restaurant meal — larger carbs (1.8-2.1×), extended duration (2×), under-bolused; lunch or dinner
# 5: Missed bolus — no insulin for one meal (meal anomaly)
# 6: Late bolus — bolus at meal time instead of pre-meal (meal anomaly)
# 7: Prolonged aerobic — long session (60-90 min, 1500-2500 AC); triggers glycogen depletion
# 8: Anaerobic/resistance — high-intensity session (30-60 min, 6000-9000 AC); EXPERIMENTAL
# 9: Exercise + missed bolus — moderate session combined with missed bolus (compound anomaly)

# Accelerometer count (AC) ranges for the ETH exercise model.
# Reference: aAC=1000 → fAC=0.5 (moderate onset); ah=5600 → fHI=0.5 (anaerobic onset).
_AC_INCIDENTAL_NORMAL: float = 300.0    # light baseline (morning walk, lunch, dinner)
_AC_INCIDENTAL_SEDENTARY: float = 100.0 # minimal baseline
_AC_AEROBIC_MIN: float = 1200.0         # moderate aerobic lower bound
_AC_AEROBIC_MAX: float = 2000.0         # moderate aerobic upper bound
_AC_PROLONGED_MIN: float = 1500.0       # prolonged aerobic lower bound
_AC_PROLONGED_MAX: float = 2500.0       # prolonged aerobic upper bound
_AC_ANAEROBIC_MIN: float = 6000.0       # anaerobic/resistance lower bound (> ah=5600)
_AC_ANAEROBIC_MAX: float = 9000.0       # anaerobic/resistance upper bound
_AC_COMPOUND_MIN: float = 1000.0        # compound scenario (exercise + bolus error)
_AC_COMPOUND_MAX: float = 1800.0        # compound scenario upper bound


# ============================================================================
# Meal Schedule Generation
# ============================================================================

def generate_meal_schedule(
    scenario: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> MealSchedule:
    """
    Generate a deterministic daily meal schedule.
    
    Parameters:
    scenario: which scenario (1-6) determines meal variations
    rng: optional numpy random generator (uses global if None)
    
    Returns:
    MealSchedule dict with all meal timings and carbs for the day.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Base meals for all scenarios - 5 regular meals for active patient
    breakfast: MealSpec = {
        "time": int(rng.integers(BREAKFAST_TIME_MIN, BREAKFAST_TIME_MAX + 1)),
        "duration": int(rng.integers(BREAKFAST_DURATION_MIN, BREAKFAST_DURATION_MAX + 1)),
        "carbs": int(rng.integers(BREAKFAST_CARBS_MIN, BREAKFAST_CARBS_MAX + 1)),
    }
    
    morning_snack: MealSpec = {
        "time": int(rng.integers(MORNING_SNACK_TIME_MIN, MORNING_SNACK_TIME_MAX + 1)),
        "duration": int(rng.integers(MORNING_SNACK_DURATION_MIN, MORNING_SNACK_DURATION_MAX + 1)),
        "carbs": int(rng.integers(MORNING_SNACK_CARBS_MIN, MORNING_SNACK_CARBS_MAX + 1)),
    }
    
    lunch: MealSpec = {
        "time": int(rng.integers(LUNCH_TIME_MIN, LUNCH_TIME_MAX + 1)),
        "duration": int(rng.integers(LUNCH_DURATION_MIN, LUNCH_DURATION_MAX + 1)),
        "carbs": int(rng.integers(LUNCH_CARBS_MIN, LUNCH_CARBS_MAX + 1)),
    }
    
    afternoon_snack: MealSpec = {
        "time": int(rng.integers(AFTERNOON_SNACK_TIME_MIN, AFTERNOON_SNACK_TIME_MAX + 1)),
        "duration": int(rng.integers(AFTERNOON_SNACK_DURATION_MIN, AFTERNOON_SNACK_DURATION_MAX + 1)),
        "carbs": int(rng.integers(AFTERNOON_SNACK_CARBS_MIN, AFTERNOON_SNACK_CARBS_MAX + 1)),
    }
    
    dinner: MealSpec = {
        "time": int(rng.integers(DINNER_TIME_MIN, DINNER_TIME_MAX + 1)),
        "duration": int(rng.integers(DINNER_DURATION_MIN, DINNER_DURATION_MAX + 1)),
        "carbs": int(rng.integers(DINNER_CARBS_MIN, DINNER_CARBS_MAX + 1)),
    }
    
    # Scenario-specific modifications
    missed_meal_id: Optional[int] = None
    late_bolus_ids: set[int] = set()
    lunch_bolus_carbs_g_override: Optional[float] = None
    dinner_bolus_carbs_g_override: Optional[float] = None

    if scenario == 4:
        # Restaurant meal: pick lunch or dinner (50/50), larger carbs, extended duration, under-bolused.
        restaurant_target = "lunch" if bool(rng.integers(0, 2)) else "dinner"
        carb_factor = float(rng.uniform(_RESTAURANT_CARB_FACTOR_MIN, _RESTAURANT_CARB_FACTOR_MAX))
        underest = float(rng.uniform(_RESTAURANT_UNDEREST_MIN, _RESTAURANT_UNDEREST_MAX))
        if restaurant_target == "lunch":
            lunch["carbs"] = int(round(lunch["carbs"] * carb_factor))
            lunch["duration"] = lunch["duration"] * 2
            lunch_bolus_carbs_g_override = float(lunch["carbs"]) * underest
        else:
            dinner["carbs"] = int(round(dinner["carbs"] * carb_factor))
            dinner["duration"] = dinner["duration"] * 2
            dinner_bolus_carbs_g_override = float(dinner["carbs"]) * underest
    elif scenario == 5:
        # Missed bolus: one of the five meal boluses is skipped (1-5)
        missed_meal_id = int(rng.integers(1, 6))
    elif scenario == 6:
        # Late bolus: 1-2 randomly chosen meals (uniform, no meal-type bias) get a late bolus
        k = int(rng.integers(1, 3))  # 1 or 2 meals affected
        late_bolus_ids = set(int(x) for x in rng.choice(np.arange(1, 6), size=k, replace=False))
    elif scenario == _SCENARIO_EXERCISE_MISSED_BOLUS:
        # Compound anomaly: exercise session + missed bolus for one meal.
        missed_meal_id = int(rng.integers(1, 6))

    return {
        "breakfast": breakfast,
        "morning_snack": morning_snack,
        "lunch": lunch,
        "afternoon_snack": afternoon_snack,
        "dinner": dinner,
        "breakfast_bolus_carbs_g": float(breakfast["carbs"]),
        "morning_snack_bolus_carbs_g": float(morning_snack["carbs"]),
        "lunch_bolus_carbs_g": lunch_bolus_carbs_g_override if lunch_bolus_carbs_g_override is not None else float(lunch["carbs"]),
        "afternoon_snack_bolus_carbs_g": float(afternoon_snack["carbs"]),
        "dinner_bolus_carbs_g": dinner_bolus_carbs_g_override if dinner_bolus_carbs_g_override is not None else float(dinner["carbs"]),
        "missed_meal_id": missed_meal_id,
        "late_bolus_ids": late_bolus_ids,
        "late_bolus_delay_min": 0,
    }


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _generate_exercise_schedule(
    *,
    rng: np.random.Generator,
    scenario: int,
    meal_schedule: Optional[MealSchedule] = None,
) -> ExerciseSchedule:
    """Create a daily exercise pattern from scenario intent.

    Scenario mapping:
    - 1 (normal): no planned exercise
    - 2 (active): moderate aerobic session (30-75 min, 1200-2000 AC)
    - 3 (sedentary): no planned exercise
    - 4-6 (meal perturbations): no planned exercise
    - 7 (prolonged aerobic): long session (60-90 min, 1500-2500 AC)
    - 8 (anaerobic): high-intensity session (30-60 min, 6000-9000 AC); EXPERIMENTAL
    - 9 (exercise + missed bolus): moderate session (30-75 min, 1000-1800 AC)
    """
    _exercise_scenarios = {
        _SCENARIO_ACTIVE_AFTERNOON,
        _SCENARIO_PROLONGED_AEROBIC,
        _SCENARIO_ANAEROBIC,
        _SCENARIO_EXERCISE_MISSED_BOLUS,
    }
    enabled = scenario in _exercise_scenarios

    # Duration and AC intensity range depend on scenario type.
    if scenario == _SCENARIO_PROLONGED_AEROBIC:
        dur_min, dur_max = _EXERCISE_DURATION_PROLONGED_MIN, _EXERCISE_DURATION_PROLONGED_MAX
        ac_min, ac_max = _AC_PROLONGED_MIN, _AC_PROLONGED_MAX
    elif scenario == _SCENARIO_ANAEROBIC:
        dur_min, dur_max = _EXERCISE_DURATION_ANAEROBIC_MIN, _EXERCISE_DURATION_ANAEROBIC_MAX
        ac_min, ac_max = _AC_ANAEROBIC_MIN, _AC_ANAEROBIC_MAX
    elif scenario == _SCENARIO_EXERCISE_MISSED_BOLUS:
        dur_min, dur_max = _EXERCISE_DURATION_MIN, _EXERCISE_DURATION_MAX
        ac_min, ac_max = _AC_COMPOUND_MIN, _AC_COMPOUND_MAX
    else:
        dur_min, dur_max = _EXERCISE_DURATION_MIN, _EXERCISE_DURATION_MAX
        ac_min, ac_max = _AC_AEROBIC_MIN, _AC_AEROBIC_MAX

    duration = int(rng.integers(dur_min, dur_max + 1))
    ac_counts = float(rng.uniform(ac_min, ac_max))

    start = int(rng.integers(_EXERCISE_START_MIN, _EXERCISE_START_MAX + 1))
    if enabled and meal_schedule is not None:
        allow_overlap = bool(rng.random() < _EXERCISE_OVERLAP_PROBABILITY)

        snack_skipped = meal_schedule["afternoon_snack"]["carbs"] == 0
        snack_start = int(meal_schedule["afternoon_snack"]["time"])
        snack_end = snack_start + int(meal_schedule["afternoon_snack"]["duration"])
        dinner_start = int(meal_schedule["dinner"]["time"])

        if not allow_overlap:
            # Main dataset mode: keep exercise separated from meals.
            # If the afternoon snack was skipped there is nothing to clear — use the
            # exercise window start directly as the lower bound.
            if snack_skipped:
                clean_start_min = _EXERCISE_START_MIN
            else:
                clean_start_min = max(_EXERCISE_START_MIN, snack_end + _EXERCISE_MIN_GAP_AFTER_SNACK_MIN)
            clean_start_max = min(_EXERCISE_START_MAX, dinner_start - duration - _EXERCISE_MIN_GAP_BEFORE_DINNER_MIN)
            if clean_start_min <= clean_start_max:
                start = int(rng.integers(clean_start_min, clean_start_max + 1))
        else:
            # Controlled outlier mode: intentionally allow overlap with snack or dinner.
            overlap_windows: list[tuple[int, int]] = []

            # Only add snack overlap window when the snack actually exists.
            if not snack_skipped:
                snack_overlap_min = max(_EXERCISE_START_MIN, snack_start - duration + 1)
                snack_overlap_max = min(_EXERCISE_START_MAX, snack_end - 1)
                if snack_overlap_min <= snack_overlap_max:
                    overlap_windows.append((snack_overlap_min, snack_overlap_max))

            dinner_overlap_min = max(_EXERCISE_START_MIN, dinner_start - duration + 1)
            dinner_overlap_max = min(_EXERCISE_START_MAX, dinner_start - 1)
            if dinner_overlap_min <= dinner_overlap_max:
                overlap_windows.append((dinner_overlap_min, dinner_overlap_max))

            if overlap_windows:
                window_idx = int(rng.integers(0, len(overlap_windows)))
                win_min, win_max = overlap_windows[window_idx]
                start = int(rng.integers(win_min, win_max + 1))

    # Burst pattern: anaerobic scenario uses periodic high-intensity bursts.
    # Burst AC = ac_counts * burst_multiplier during burst_on minutes out of every burst_period.
    burst_period = 1
    burst_on = 1
    burst_multiplier = 1.0
    if scenario == _SCENARIO_ANAEROBIC:
        # Resistance intervals: 2 min on / 3 min rest, AC spikes ~50% above session mean.
        burst_multiplier = 1.5
        burst_period = 5
        burst_on = 2

    return {
        "enabled": enabled,
        "session": {
            "start": start,
            "duration": duration,
            "ac_counts": ac_counts,
        },
        "burst_period_min": burst_period,
        "burst_on_min": burst_on,
        "burst_multiplier": burst_multiplier,
    }


def _baseline_ac_for_scenario(time: int, scenario: int) -> float:
    """Return incidental movement baseline as accelerometer counts.

    Synchronized with meal windows (light walking to kitchen/cafeteria):
    - breakfast window: 06:30-08:30
    - lunch window: 12:00-13:30
    - dinner window: 18:30-20:00
    Planned exercise sessions (scenarios 2, 7, 8, 9) add on top of this baseline.
    Reference: aAC=1000 → fAC=0.5; values here are well below exercise onset.
    """
    minute = int(max(0, min(1439, time)))

    if minute < time_to_minutes(6, 0) or minute >= time_to_minutes(22, 30):
        return 0.0  # Night/sleep

    if scenario in {4, 5, 6}:
        # Meal-perturbation scenarios are exercise-neutral.
        return 0.0

    if scenario == _SCENARIO_SEDENTARY:
        if time_to_minutes(6, 30) <= minute < time_to_minutes(8, 30):
            return 150.0  # light breakfast movement
        if time_to_minutes(12, 0) <= minute < time_to_minutes(13, 30):
            return 200.0  # light lunch movement
        if time_to_minutes(18, 30) <= minute < time_to_minutes(20, 0):
            return 150.0  # light dinner movement
        return 75.0  # minimal daytime baseline

    # Scenarios 1, 2, 3=handled above, 7, 8, 9 share this incidental baseline.
    if time_to_minutes(6, 30) <= minute < time_to_minutes(8, 30):
        return 400.0  # morning routine + walking
    if time_to_minutes(12, 0) <= minute < time_to_minutes(13, 30):
        return 500.0  # lunch walk
    if time_to_minutes(18, 30) <= minute < time_to_minutes(20, 0):
        return 450.0  # dinner + post-dinner walk
    if time_to_minutes(20, 0) <= minute < time_to_minutes(22, 30):
        return 200.0  # light evening activity
    return _AC_INCIDENTAL_NORMAL  # general daytime light movement


def _exercise_ac_at_minute(time: int, schedule: ExerciseSchedule) -> float:
    """Return exercise-session accelerometer counts at a given minute."""
    if not schedule["enabled"]:
        return 0.0

    session = schedule["session"]
    start = session["start"]
    end = start + session["duration"]
    if not (start <= time < end):
        return 0.0

    ac = float(session["ac_counts"])
    period = max(1, int(schedule["burst_period_min"]))
    on = max(1, min(period, int(schedule["burst_on_min"])))
    local_t = time - start
    in_burst = (local_t % period) < on
    if in_burst:
        ac *= float(schedule["burst_multiplier"])
    return max(0.0, ac)


def _create_seed(
    seed: Optional[int],
    patient_id: int,
    scenario: int,
    day: Optional[int] = None,
) -> Optional[int]:
    """
    Create a seed for the random number generator.
    """
    if seed is None:
        return None

    value = (
        int(seed)
        + int(patient_id) * _PATIENT_SEED_FACTOR
        + int(scenario) * _SCENARIO_SEED_FACTOR
    )
    if day is not None:
        value += int(day) * _DAY_SEED_FACTOR
    return value


def _seeded_rng(
    seed: Optional[int],
    patient_id: int,
    scenario: int,
    day: Optional[int] = None,
) -> np.random.Generator:
    """
    Generate seed for the random number generator.
    """
    composed_seed = _create_seed(seed, patient_id, scenario, day)
    if composed_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(composed_seed)


def _jitter_meal(
    meal: MealSpec,
    time_min: int,
    time_max: int,
    carbs_min: int,
    carbs_max: int,
    duration_min: int,
    duration_max: int,
    rng: np.random.Generator,
    jitter_min: int,
    jitter_max: int,
    carb_jitter_pct: float,
) -> MealSpec:
    """Add realistic day-to-day jitter to meal timing, carbs, and duration."""
    time_jitter = int(rng.integers(jitter_min, jitter_max))
    new_time = _clamp_int(meal["time"] + time_jitter, time_min, time_max)

    carb_factor = 1.0 + float(rng.uniform(-carb_jitter_pct, carb_jitter_pct))
    new_carbs = _clamp_int(
        int(round(meal["carbs"] * carb_factor)),
        carbs_min,
        carbs_max,
    )
    
    # Add small random duration variation (±1-2 min)
    duration_jitter = int(rng.integers(-2, 3))
    new_duration = _clamp_int(meal["duration"] + duration_jitter, duration_min, duration_max)

    return {
        "time": new_time,
        "duration": new_duration,
        "carbs": new_carbs,
    }


def _build_daily_schedule_from_baseline(
    baseline: MealSchedule,
    scenario: int,
    rng: np.random.Generator,
    high_carb_main_meal: Optional[str] = None,
    patient_bolus_bias: float = 1.0,
) -> MealSchedule:
    """Build patient-specific daily schedule with realistic variation from baseline."""
    breakfast = _jitter_meal(
        baseline["breakfast"],
        BREAKFAST_TIME_MIN,
        BREAKFAST_TIME_MAX,
        BREAKFAST_CARBS_MIN,
        BREAKFAST_CARBS_MAX,
        BREAKFAST_DURATION_MIN,
        BREAKFAST_DURATION_MAX,
        rng,
        _SMALL_TIME_JITTER_MIN,
        _SMALL_TIME_JITTER_MAX,
        _SMALL_CARB_JITTER_PCT,
    )
    morning_snack = _jitter_meal(
        baseline["morning_snack"],
        MORNING_SNACK_TIME_MIN,
        MORNING_SNACK_TIME_MAX,
        MORNING_SNACK_CARBS_MIN,
        MORNING_SNACK_CARBS_MAX,
        MORNING_SNACK_DURATION_MIN,
        MORNING_SNACK_DURATION_MAX,
        rng,
        _SMALL_TIME_JITTER_MIN,
        _SMALL_TIME_JITTER_MAX,
        _SMALL_CARB_JITTER_PCT,
    )
    lunch = _jitter_meal(
        baseline["lunch"],
        LUNCH_TIME_MIN,
        LUNCH_TIME_MAX,
        LUNCH_CARBS_MIN,
        LUNCH_CARBS_MAX,
        LUNCH_DURATION_MIN,
        LUNCH_DURATION_MAX,
        rng,
        _SMALL_TIME_JITTER_MIN,
        _SMALL_TIME_JITTER_MAX,
        _SMALL_CARB_JITTER_PCT,
    )
    afternoon_snack = _jitter_meal(
        baseline["afternoon_snack"],
        AFTERNOON_SNACK_TIME_MIN,
        AFTERNOON_SNACK_TIME_MAX,
        AFTERNOON_SNACK_CARBS_MIN,
        AFTERNOON_SNACK_CARBS_MAX,
        AFTERNOON_SNACK_DURATION_MIN,
        AFTERNOON_SNACK_DURATION_MAX,
        rng,
        _SMALL_TIME_JITTER_MIN,
        _SMALL_TIME_JITTER_MAX,
        _SMALL_CARB_JITTER_PCT,
    )
    dinner = _jitter_meal(
        baseline["dinner"],
        DINNER_TIME_MIN,
        DINNER_TIME_MAX,
        DINNER_CARBS_MIN,
        DINNER_CARBS_MAX,
        DINNER_DURATION_MIN,
        DINNER_DURATION_MAX,
        rng,
        _SMALL_TIME_JITTER_MIN,
        _SMALL_TIME_JITTER_MAX,
        _SMALL_CARB_JITTER_PCT,
    )

    irregular_day = bool(rng.random() < _IRREGULAR_DAY_PROBABILITY)
    if irregular_day:
        breakfast = _jitter_meal(
            breakfast,
            BREAKFAST_TIME_MIN,
            BREAKFAST_TIME_MAX,
            BREAKFAST_CARBS_MIN,
            BREAKFAST_CARBS_MAX,
            BREAKFAST_DURATION_MIN,
            BREAKFAST_DURATION_MAX,
            rng,
            _IRREGULAR_TIME_JITTER_MIN,
            _IRREGULAR_TIME_JITTER_MAX,
            _SMALL_CARB_JITTER_PCT,
        )
        morning_snack = _jitter_meal(
            morning_snack,
            MORNING_SNACK_TIME_MIN,
            MORNING_SNACK_TIME_MAX,
            MORNING_SNACK_CARBS_MIN,
            MORNING_SNACK_CARBS_MAX,
            MORNING_SNACK_DURATION_MIN,
            MORNING_SNACK_DURATION_MAX,
            rng,
            _IRREGULAR_TIME_JITTER_MIN,
            _IRREGULAR_TIME_JITTER_MAX,
            _SMALL_CARB_JITTER_PCT,
        )
        lunch = _jitter_meal(
            lunch,
            LUNCH_TIME_MIN,
            LUNCH_TIME_MAX,
            LUNCH_CARBS_MIN,
            LUNCH_CARBS_MAX,
            LUNCH_DURATION_MIN,
            LUNCH_DURATION_MAX,
            rng,
            _IRREGULAR_TIME_JITTER_MIN,
            _IRREGULAR_TIME_JITTER_MAX,
            _SMALL_CARB_JITTER_PCT,
        )
        afternoon_snack = _jitter_meal(
            afternoon_snack,
            AFTERNOON_SNACK_TIME_MIN,
            AFTERNOON_SNACK_TIME_MAX,
            AFTERNOON_SNACK_CARBS_MIN,
            AFTERNOON_SNACK_CARBS_MAX,
            AFTERNOON_SNACK_DURATION_MIN,
            AFTERNOON_SNACK_DURATION_MAX,
            rng,
            _IRREGULAR_TIME_JITTER_MIN,
            _IRREGULAR_TIME_JITTER_MAX,
            _SMALL_CARB_JITTER_PCT,
        )
        dinner = _jitter_meal(
            dinner,
            DINNER_TIME_MIN,
            DINNER_TIME_MAX,
            DINNER_CARBS_MIN,
            DINNER_CARBS_MAX,
            DINNER_DURATION_MIN,
            DINNER_DURATION_MAX,
            rng,
            _IRREGULAR_TIME_JITTER_MIN,
            _IRREGULAR_TIME_JITTER_MAX,
            _SMALL_CARB_JITTER_PCT,
        )

    missed_meal_id: Optional[int] = None
    late_bolus_ids: set[int] = set()
    restaurant_meal_target: Optional[str] = None

    # Apply scenario perturbations regardless of irregular_day so that ML labels
    # remain consistent: a scenario-5 day always has a missed bolus, etc.
    # irregular_day only affects timing jitter amplitude, not the anomaly label.
    if scenario == 4:
        # Restaurant meal: pick lunch or dinner (50/50), larger carbs + extended duration.
        restaurant_meal_target = "lunch" if bool(rng.integers(0, 2)) else "dinner"
        carb_factor = float(rng.uniform(_RESTAURANT_CARB_FACTOR_MIN, _RESTAURANT_CARB_FACTOR_MAX))
        if restaurant_meal_target == "lunch":
            lunch["carbs"] = int(round(lunch["carbs"] * carb_factor))
            lunch["duration"] = lunch["duration"] * 2
        else:
            dinner["carbs"] = int(round(dinner["carbs"] * carb_factor))
            dinner["duration"] = dinner["duration"] * 2
    elif scenario == 5:
        missed_meal_id = int(rng.integers(1, 6))  # 1-5 for the 5 meals
    elif scenario == 6:
        # Late bolus: 1-2 randomly chosen meals (uniform, no meal-type bias) get a late bolus
        k = int(rng.integers(1, 3))  # 1 or 2 meals affected
        late_bolus_ids = set(int(x) for x in rng.choice(np.arange(1, 6), size=k, replace=False))
    elif scenario == _SCENARIO_EXERCISE_MISSED_BOLUS:
        # Compound anomaly: exercise session + missed bolus for one meal.
        missed_meal_id = int(rng.integers(1, 6))

    # Once every 1-2 weeks, increase either lunch or dinner carbs (not both).
    if high_carb_main_meal == "lunch":
        lunch["carbs"] = int(rng.integers(HIGH_MAIN_MEAL_CARBS_MIN, HIGH_MAIN_MEAL_CARBS_MAX + 1))
    elif high_carb_main_meal == "dinner":
        dinner["carbs"] = int(rng.integers(HIGH_MAIN_MEAL_CARBS_MIN, HIGH_MAIN_MEAL_CARBS_MAX + 1))

    # Some days snacks are skipped entirely (no carbs => no snack bolus).
    skip_roll = float(rng.random())
    if skip_roll < SNACK_SKIP_BOTH_PROB:
        morning_snack["carbs"] = 0
        afternoon_snack["carbs"] = 0
    elif skip_roll < (SNACK_SKIP_BOTH_PROB + SNACK_SKIP_ONE_PROB):
        if bool(rng.integers(0, 2)):
            morning_snack["carbs"] = 0
        else:
            afternoon_snack["carbs"] = 0

    # Meal-level carb estimation for bolus: lunch/dinner are more often
    # underestimated than snacks.
    breakfast_est_factor = patient_bolus_bias * float(rng.uniform(_MEAL_EST_NOISE_MIN, _MEAL_EST_NOISE_MAX))
    morning_snack_est_factor = patient_bolus_bias * float(rng.uniform(_SNACK_EST_MIN, _SNACK_EST_MAX))
    if scenario == _SCENARIO_RESTAURANT_MEAL and restaurant_meal_target == "lunch":
        # Restaurant lunch: strong systematic underestimation on top of meal noise.
        lunch_est_factor = (
            patient_bolus_bias
            * float(rng.uniform(_MEAL_EST_NOISE_MIN, _MEAL_EST_NOISE_MAX))
            * float(rng.uniform(_RESTAURANT_UNDEREST_MIN, _RESTAURANT_UNDEREST_MAX))
        )
    else:
        lunch_est_factor = (
            patient_bolus_bias
            * float(rng.uniform(_MEAL_EST_NOISE_MIN, _MEAL_EST_NOISE_MAX))
            * float(rng.uniform(_LUNCH_DINNER_UNDEREST_MIN, _LUNCH_DINNER_UNDEREST_MAX))
        )
    afternoon_snack_est_factor = patient_bolus_bias * float(rng.uniform(_SNACK_EST_MIN, _SNACK_EST_MAX))
    if scenario == _SCENARIO_RESTAURANT_MEAL and restaurant_meal_target == "dinner":
        # Restaurant dinner: strong systematic underestimation on top of meal noise.
        dinner_est_factor = (
            patient_bolus_bias
            * float(rng.uniform(_MEAL_EST_NOISE_MIN, _MEAL_EST_NOISE_MAX))
            * float(rng.uniform(_RESTAURANT_UNDEREST_MIN, _RESTAURANT_UNDEREST_MAX))
        )
    else:
        dinner_est_factor = (
            patient_bolus_bias
            * float(rng.uniform(_MEAL_EST_NOISE_MIN, _MEAL_EST_NOISE_MAX))
            * float(rng.uniform(_LUNCH_DINNER_UNDEREST_MIN, _LUNCH_DINNER_UNDEREST_MAX))
        )

    late_bolus_delay_min = 0
    if scenario == _SCENARIO_LATE_BOLUS:
        late_bolus_delay_min = int(rng.integers(_LATE_BOLUS_DELAY_MIN, _LATE_BOLUS_DELAY_MAX + 1))

    return {
        "breakfast": breakfast,
        "morning_snack": morning_snack,
        "lunch": lunch,
        "afternoon_snack": afternoon_snack,
        "dinner": dinner,
        "breakfast_bolus_carbs_g": float(breakfast["carbs"]) * breakfast_est_factor,
        "morning_snack_bolus_carbs_g": float(morning_snack["carbs"]) * morning_snack_est_factor,
        "lunch_bolus_carbs_g": float(lunch["carbs"]) * lunch_est_factor,
        "afternoon_snack_bolus_carbs_g": float(afternoon_snack["carbs"]) * afternoon_snack_est_factor,
        "dinner_bolus_carbs_g": float(dinner["carbs"]) * dinner_est_factor,
        "missed_meal_id": missed_meal_id,
        "late_bolus_ids": late_bolus_ids,
        "late_bolus_delay_min": late_bolus_delay_min,
    }


# ============================================================================
# Meal Computation Helper
# ============================================================================

def _apply_meal(
    meal: MealSpec,
    current_time: int,
    insulin_carbo_ratio: float,
    bolus_time: int,
    bolus_carbs_grams: Optional[float] = None,
    missed: bool = False,
    late_bolus: bool = False,
    late_bolus_delay_min: int = 0,
) -> Tuple[float, float]:
    """
    Compute carb intake and insulin bolus for a single meal at current_time.
    
    Parameters:
    meal: MealSpec with time, duration, carbs
    current_time: current minute in day
    insulin_carbo_ratio: carbs per insulin unit [g/unit]
                        (e.g., 2 = 1 unit covers 2g carbs)
    bolus_time: pre-bolus time offset (typically -15 min)
    missed: if True, skip insulin bolus for this meal
    late_bolus: if True, deliver bolus at meal time instead of pre-bolus
    
    Returns:
    (u_increment, d_increment): insulin [mU/min], carbs [mg/min]
    """
    u_inc: float = 0.0
    d_inc: float = 0.0
    
    meal_time = meal["time"]
    duration = meal["duration"]
    carbs_grams = float(meal["carbs"])
    bolus_carbs_g = carbs_grams if bolus_carbs_grams is None else max(0.0, float(bolus_carbs_grams))
    
    # Carb intake during meal consumption
    if meal_time <= current_time < meal_time + duration:
        d_inc = carbs_grams * 1000 / duration  # mg/min
    
    # Insulin bolus
    if not missed:
        bolus_amount = bolus_carbs_g / insulin_carbo_ratio  # [units]
        
        # Spread the bolus evenly over BOLUS_DURATION minutes so the total
        # delivered dose stays equal to bolus_amount [U] while avoiding the
        # sharp S1 spike that a 1-minute dump produces (which can falsely
        # trigger the IOB guard and suppress subsequent meal boluses).
        bolus_rate = bolus_amount * 1000 / BOLUS_DURATION  # mU/min
        if late_bolus:
            # Late-bolus anomaly with variable delay after meal start.
            bolus_start = meal_time + max(0, late_bolus_delay_min)
            if bolus_start <= current_time < bolus_start + BOLUS_DURATION:
                u_inc = bolus_rate
        else:
            # Pre-bolus (15 min before meal)
            bolus_start = meal_time - bolus_time
            if bolus_start <= current_time < bolus_start + BOLUS_DURATION:
                u_inc = bolus_rate
    
    return u_inc, d_inc


# ============================================================================
# Main Scenario Function
# ============================================================================

def scenario_inputs(
    time: int,
    basal_hourly: float = 0.5,
    scenario: int = 1,
    insulin_carbo_ratio: float = 2.0,
    meal_schedule: Optional[MealSchedule] = None,
    exercise_schedule: Optional[ExerciseSchedule] = None,
) -> Tuple[float, float, float]:
    """
    Compute insulin delivery (u), carbohydrate intake (d), and accelerometer counts at a given time.

    Parameters:
    time: current minute (0-1439 within a day)
    basal_hourly: basal insulin rate [U/hr]
    scenario: which scenario to simulate (1-9)
    insulin_carbo_ratio: insulin-to-carb ratio [g/unit]
    meal_schedule: optional pre-computed MealSchedule (for determinism)
    exercise_schedule: optional pre-computed ExerciseSchedule (for determinism)
    Returns:
    (u, d, activity): insulin [mU/min], carbs [mg/min], activity as accelerometer counts [AC]
    """
    # Convert basal from U/hr to mU/min
    basal = basal_hourly * 1000 / 60  # ~8.33 mU/min for 0.5 U/hr
    
    # Generate or use provided meal schedule
    if meal_schedule is None:
        meal_schedule = generate_meal_schedule(scenario=scenario)

    u: float = basal
    d: float = 0.0

    if exercise_schedule is None:
        exercise_schedule = _generate_exercise_schedule(
            rng=np.random.default_rng(),
            scenario=scenario,
            meal_schedule=meal_schedule,
        )
    
    bolus_lead_min = PREANNOUNCED_BOLUS_TIME

    # Apply meals
    breakfast_u, breakfast_d = _apply_meal(
        meal_schedule["breakfast"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        bolus_carbs_grams=meal_schedule["breakfast_bolus_carbs_g"],
        missed=(meal_schedule["missed_meal_id"] == 1),
        late_bolus=(1 in meal_schedule["late_bolus_ids"]),
        late_bolus_delay_min=meal_schedule["late_bolus_delay_min"],
    )
    u += breakfast_u
    d += breakfast_d

    morning_snack_u, morning_snack_d = _apply_meal(
        meal_schedule["morning_snack"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        bolus_carbs_grams=meal_schedule["morning_snack_bolus_carbs_g"],
        missed=(meal_schedule["missed_meal_id"] == 2),
        late_bolus=(2 in meal_schedule["late_bolus_ids"]),
        late_bolus_delay_min=meal_schedule["late_bolus_delay_min"],
    )
    u += morning_snack_u
    d += morning_snack_d
    
    lunch_u, lunch_d = _apply_meal(
        meal_schedule["lunch"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        bolus_carbs_grams=meal_schedule["lunch_bolus_carbs_g"],
        missed=(meal_schedule["missed_meal_id"] == 3),
        late_bolus=(3 in meal_schedule["late_bolus_ids"]),
        late_bolus_delay_min=meal_schedule["late_bolus_delay_min"],
    )
    u += lunch_u
    d += lunch_d

    afternoon_snack_u, afternoon_snack_d = _apply_meal(
        meal_schedule["afternoon_snack"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        bolus_carbs_grams=meal_schedule["afternoon_snack_bolus_carbs_g"],
        missed=(meal_schedule["missed_meal_id"] == 4),
        late_bolus=(4 in meal_schedule["late_bolus_ids"]),
        late_bolus_delay_min=meal_schedule["late_bolus_delay_min"],
    )
    u += afternoon_snack_u
    d += afternoon_snack_d
    
    dinner_u, dinner_d = _apply_meal(
        meal_schedule["dinner"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        bolus_carbs_grams=meal_schedule["dinner_bolus_carbs_g"],
        missed=(meal_schedule["missed_meal_id"] == 5),
        late_bolus=(5 in meal_schedule["late_bolus_ids"]),
        late_bolus_delay_min=meal_schedule["late_bolus_delay_min"],
    )
    u += dinner_u
    d += dinner_d

    baseline_ac = _baseline_ac_for_scenario(time=time, scenario=scenario)
    session_ac = _exercise_ac_at_minute(time=time, schedule=exercise_schedule)
    activity = baseline_ac + session_ac

    return u, d, activity


# ============================================================================
# Caching Wrapper for Determinism
# ============================================================================

# WARNING — process-global caches. Safe for mp.Pool (each worker gets its own
# memory copy) and for sequential run_simulation calls (clear_meal_cache() is
# called at the top of every run). NOT safe for thread-based parallelism
# (ThreadPoolExecutor) or direct calls to scenario_with_cached_meals outside
# of run_simulation — in those cases caches bleed across runs and produce
# non-reproducible results even with a fixed seed. If thread-based parallelism
# is ever needed, move these into a per-simulation context object.
_patient_baseline_cache: dict[tuple[int, int], MealSchedule] = {}
_meal_cache: dict[tuple[int, int, int], MealSchedule] = {}
_exercise_cache: dict[tuple[int, int, int], ExerciseSchedule] = {}
_high_carb_next_day_cache: dict[tuple[int, int], int] = {}
_patient_bolus_bias_cache: dict[tuple[int, int], float] = {}


def scenario_with_cached_meals(
    time: int,
    patient_id: int,
    day: int,
    basal_hourly: float = 0.5,
    scenario: int = 1,
    insulin_carbo_ratio: float = 2.0,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Wrapper around scenario_inputs that caches meal schedules per (patient_id, day, scenario).

    The third return value is accelerometer counts (AC) representing exercise intensity.
    """
    patient_key = (patient_id, scenario)
    cache_key = (patient_id, day, scenario)
    
    if cache_key not in _meal_cache:
        if patient_key not in _patient_baseline_cache:
            patient_rng = _seeded_rng(
                seed=seed,
                patient_id=patient_id,
                scenario=scenario,
                day=None,
            )
            _patient_baseline_cache[patient_key] = generate_meal_schedule(
                scenario=1,
                rng=patient_rng,
            )

        if patient_key not in _high_carb_next_day_cache:
            # Use a dedicated deterministic RNG stream so weekly high-carb events
            # are reproducible without coupling to regular meal jitter draws.
            phase_rng = _seeded_rng(
                seed=seed,
                patient_id=patient_id,
                scenario=scenario + _HIGH_CARB_STREAM_OFFSET,
                day=None,
            )
            _high_carb_next_day_cache[patient_key] = int(
                phase_rng.integers(HIGH_MAIN_MEAL_GAP_DAYS_MIN, HIGH_MAIN_MEAL_GAP_DAYS_MAX + 1)
            )

        if patient_key not in _patient_bolus_bias_cache:
            bias_rng = _seeded_rng(
                seed=seed,
                patient_id=patient_id,
                scenario=scenario + _BOLUS_ESTIMATION_STREAM_OFFSET,
                day=None,
            )
            _patient_bolus_bias_cache[patient_key] = float(
                bias_rng.uniform(_PATIENT_BOLUS_BIAS_MIN, _PATIENT_BOLUS_BIAS_MAX)
            )

        day_rng = _seeded_rng(
            seed=seed,
            patient_id=patient_id,
            scenario=scenario,
            day=day,
        )

        high_carb_main_meal: Optional[str] = None
        next_high_day = _high_carb_next_day_cache[patient_key]
        if day >= next_high_day:
            high_carb_main_meal = "lunch" if bool(day_rng.integers(0, 2)) else "dinner"
            gap_days = int(day_rng.integers(HIGH_MAIN_MEAL_GAP_DAYS_MIN, HIGH_MAIN_MEAL_GAP_DAYS_MAX + 1))
            _high_carb_next_day_cache[patient_key] = day + gap_days

        _meal_cache[cache_key] = _build_daily_schedule_from_baseline(
            baseline=_patient_baseline_cache[patient_key],
            scenario=scenario,
            rng=day_rng,
            high_carb_main_meal=high_carb_main_meal,
            patient_bolus_bias=_patient_bolus_bias_cache[patient_key],
        )

    if cache_key not in _exercise_cache:
        ex_rng = _seeded_rng(
            seed=seed,
            patient_id=patient_id,
            scenario=scenario + 17,
            day=day,
        )
        _exercise_cache[cache_key] = _generate_exercise_schedule(
            rng=ex_rng,
            scenario=scenario,
            meal_schedule=_meal_cache[cache_key],
        )
    
    return scenario_inputs(
        time,
        basal_hourly=basal_hourly,
        scenario=scenario,
        insulin_carbo_ratio=insulin_carbo_ratio,
        meal_schedule=_meal_cache[cache_key],
        exercise_schedule=_exercise_cache[cache_key],
    )


def get_cached_meal_schedule(
    patient_id: int,
    day: int,
    scenario: int,
) -> Optional[MealSchedule]:
    """Return the cached MealSchedule for a given (patient_id, day, scenario).

    Returns None if scenario_with_cached_meals has not yet been called for this
    combination (i.e. the cache entry doesn't exist yet).
    """
    return _meal_cache.get((patient_id, day, scenario))


def clear_meal_cache() -> None:
    """Clear the meal cache (useful for tests or starting a new cohort)."""
    global _meal_cache
    global _patient_baseline_cache
    global _exercise_cache
    global _high_carb_next_day_cache
    global _patient_bolus_bias_cache
    _meal_cache = {}
    _patient_baseline_cache = {}
    _exercise_cache = {}
    _high_carb_next_day_cache = {}
    _patient_bolus_bias_cache = {}
