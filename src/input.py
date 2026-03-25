from __future__ import annotations

import numpy as np
from typing import TypedDict, Optional, Tuple

N_SCENARIOS: int = 6  # Number of different input scenarios (e.g., different meal times, insulin doses)

# Sampling weights for scenarios 1-6.
# Scenarios 1-3 (normal, active, sedentary) are common day-to-day patterns.
# Scenarios 4-6 (long lunch, missed bolus, late bolus) are deliberate anomalies
# and should be rarer so the dataset is realistic and anomaly-class imbalance
# matches real-world prevalence.
_SCENARIO_WEIGHTS_RAW: list[float] = [0.25, 0.25, 0.25, 0.083, 0.083, 0.083]
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
LUNCH_CARBS_MAX: int = 75
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

# Dinner: moderate with realistic 20-25 min and earlier timing for active schedule
DINNER_CARBS_MIN: int = 50
DINNER_CARBS_MAX: int = 75
DINNER_TIME_MIN: int = time_to_minutes(18, 30)
DINNER_TIME_MAX: int = time_to_minutes(20, 0)
DINNER_DURATION_MIN: int = 20
DINNER_DURATION_MAX: int = 25

# Bolus timing: 15 min pre-meal for fast-paced active schedule
PREANNOUNCED_BOLUS_TIME: int = 15
BOLUS_DURATION: int = 1

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
    missed_meal_id: Optional[int]  # For missed bolus scenario (1-5 or None)
    late_bolus_id: Optional[int]   # For late bolus scenario (1-5 or None)


class ExerciseSpec(TypedDict):
    """Specification of one exercise block represented as a fraction of HR reserve."""
    start: int
    duration: int
    hr_reserve_fraction: float


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
_SMALL_TIME_JITTER_MIN: int = -5
_SMALL_TIME_JITTER_MAX: int = 6
_SMALL_CARB_JITTER_PCT: float = 0.06
_IRREGULAR_DAY_PROBABILITY: float = 0.15
_IRREGULAR_TIME_JITTER_MIN: int = -20
_IRREGULAR_TIME_JITTER_MAX: int = 21

_EXERCISE_START_MIN: int = time_to_minutes(16, 30)
_EXERCISE_START_MAX: int = time_to_minutes(20, 30)
_EXERCISE_DURATION_MIN: int = 30
_EXERCISE_DURATION_MAX: int = 75
_EXERCISE_INTENSITY_MIN: int = 35
_EXERCISE_INTENSITY_MAX: int = 90
_ANAEROBIC_THRESHOLD_PCT: int = 75
_EXERCISE_OVERLAP_PROBABILITY: float = 0.15
_EXERCISE_MIN_GAP_AFTER_SNACK_MIN: int = 20
_EXERCISE_MIN_GAP_BEFORE_DINNER_MIN: int = 30

_SCENARIO_NORMAL: int = 1
_SCENARIO_ACTIVE_AFTERNOON: int = 2
_SCENARIO_SEDENTARY: int = 3

# Scenario guide for input behavior:
# 1: Normal day (regular meals, light incidental movement, no planned workout)
# 2: Active day (regular meals + planned afternoon workout)
# 3: Sedentary day (regular meals, lower incidental movement, no planned workout)
# 4: Long lunch disturbance (meal-only perturbation)
# 5: Missed bolus disturbance (meal-only perturbation)
# 6: Late bolus disturbance (meal-only perturbation)

# Hovorka / Rashid HR constants for converting scenario intensity into ΔHR.
_HR_REST_BPM: float = 60.0
_HR_MAX_FLOOR_BPM: float = _HR_REST_BPM + 1.0
_HR_RESERVE_FRACTION_MAX_SAFE: float = 1.0


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
    late_bolus_id: Optional[int] = None
    
    if scenario == 4:
        # Long lunch: extend lunch duration by 2×
        lunch["duration"] = lunch["duration"] * 2
    elif scenario == 5:
        # Missed bolus: one of the five meal boluses is skipped (1-5)
        missed_meal_id = int(rng.integers(1, 6))
    elif scenario == 6:
        # Late bolus: one meal gets bolus at meal time instead of 15 min before
        late_bolus_id = int(rng.integers(1, 6))
    
    return {
        "breakfast": breakfast,
        "morning_snack": morning_snack,
        "lunch": lunch,
        "afternoon_snack": afternoon_snack,
        "dinner": dinner,
        "missed_meal_id": missed_meal_id,
        "late_bolus_id": late_bolus_id,
    }


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _intensity_percent_to_hr_reserve_fraction(intensity_pct: float) -> float:
    """Map scenario intensity percentage to a 0..1 fraction of HR reserve."""
    return float(max(0.0, min(100.0, intensity_pct))) / 100.0


def _hr_max_from_age(age_years: float) -> float:
    """Use the classic Fox-Haskell formula requested by the user."""
    return max(_HR_MAX_FLOOR_BPM, 220.0 - float(age_years))


def _hr_reserve_from_age(age_years: float, hr0: float = _HR_REST_BPM) -> float:
    return max(1.0, _hr_max_from_age(age_years) - hr0)


def _hr_reserve_fraction_to_delta_hr(hr_reserve_fraction: float, age_years: float, hr0: float = _HR_REST_BPM) -> float:
    clipped_fraction = max(0.0, min(_HR_RESERVE_FRACTION_MAX_SAFE, float(hr_reserve_fraction)))
    return clipped_fraction * _hr_reserve_from_age(age_years, hr0)


def _generate_exercise_schedule(
    *,
    rng: np.random.Generator,
    scenario: int,
    meal_schedule: Optional[MealSchedule] = None,
) -> ExerciseSchedule:
    """Create a daily exercise pattern from scenario intent.

    Scenario mapping:
    - 1 (normal): no planned exercise
    - 2 (active): one afternoon exercise session (17:00-19:30 neighborhood)
    - 3 (sedentary): no planned exercise
    - 4-6 (meal perturbations): no planned exercise
    """
    enabled = scenario == _SCENARIO_ACTIVE_AFTERNOON

    duration = int(rng.integers(_EXERCISE_DURATION_MIN, _EXERCISE_DURATION_MAX + 1))
    intensity_pct = int(rng.integers(_EXERCISE_INTENSITY_MIN, _EXERCISE_INTENSITY_MAX + 1))
    hr_reserve_fraction = _intensity_percent_to_hr_reserve_fraction(float(intensity_pct))

    start = int(rng.integers(_EXERCISE_START_MIN, _EXERCISE_START_MAX + 1))
    if enabled and meal_schedule is not None:
        allow_overlap = bool(rng.random() < _EXERCISE_OVERLAP_PROBABILITY)

        snack_start = int(meal_schedule["afternoon_snack"]["time"])
        snack_end = snack_start + int(meal_schedule["afternoon_snack"]["duration"])
        dinner_start = int(meal_schedule["dinner"]["time"])

        if not allow_overlap:
            # Main dataset mode: keep exercise separated from afternoon snack and dinner.
            clean_start_min = max(_EXERCISE_START_MIN, snack_end + _EXERCISE_MIN_GAP_AFTER_SNACK_MIN)
            clean_start_max = min(_EXERCISE_START_MAX, dinner_start - duration - _EXERCISE_MIN_GAP_BEFORE_DINNER_MIN)
            if clean_start_min <= clean_start_max:
                start = int(rng.integers(clean_start_min, clean_start_max + 1))
        else:
            # Controlled outlier mode: intentionally allow overlap with snack or dinner.
            overlap_windows: list[tuple[int, int]] = []

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

    # For high-intensity sessions, model intermittent anaerobic bouts as periodic spikes.
    burst_period = 1
    burst_on = 1
    burst_multiplier = 1.0
    if enabled and intensity_pct >= _ANAEROBIC_THRESHOLD_PCT:
        burst_multiplier = 1.25
        burst_period = 3
        burst_on = 1
    elif enabled and intensity_pct >= 65:
        burst_multiplier = 1.12

    return {
        "enabled": enabled,
        "session": {
            "start": start,
            "duration": duration,
            "hr_reserve_fraction": hr_reserve_fraction,
        },
        "burst_period_min": burst_period,
        "burst_on_min": burst_on,
        "burst_multiplier": burst_multiplier,
    }


def _baseline_hr_reserve_fraction_for_scenario(time: int, scenario: int) -> float:
    """Return incidental movement baseline as a fraction of HR reserve.

    This baseline is intentionally modest and synchronized with meal windows:
    - breakfast window: 06:30-08:30
    - lunch window: 12:00-13:30
    - dinner window: 18:30-20:00
    Scenario 2 adds a separate planned workout on top of this baseline.
    """
    minute = int(max(0, min(1439, time)))

    if minute < time_to_minutes(6, 0) or minute >= time_to_minutes(22, 30):
        return 0.0  # Night/sleep: at resting HR

    if scenario in {4, 5, 6}:
        # Meal-perturbation scenarios are intentionally exercise-neutral.
        return 0.0

    if scenario == _SCENARIO_SEDENTARY:
        if time_to_minutes(6, 30) <= minute < time_to_minutes(8, 30):
            return 0.02
        if time_to_minutes(12, 0) <= minute < time_to_minutes(13, 30):
            return 0.03
        if time_to_minutes(18, 30) <= minute < time_to_minutes(20, 0):
            return 0.02
        return 0.015

    # Scenario 1 and 2 share this light incidental baseline.
    if time_to_minutes(6, 30) <= minute < time_to_minutes(8, 30):
        return 0.05
    if time_to_minutes(12, 0) <= minute < time_to_minutes(13, 30):
        return 0.07
    if time_to_minutes(18, 30) <= minute < time_to_minutes(20, 0):
        return 0.06
    if time_to_minutes(19, 30) <= minute < time_to_minutes(22, 30):
        return 0.03
    return 0.025


def _exercise_hr_reserve_fraction_at_minute(time: int, schedule: ExerciseSchedule) -> float:
    """Return exercise-session effort as a fraction of HR reserve."""
    if not schedule["enabled"]:
        return 0.0

    session = schedule["session"]
    start = session["start"]
    end = start + session["duration"]
    if not (start <= time < end):
        return 0.0

    hr_reserve_fraction = float(session["hr_reserve_fraction"])
    period = max(1, int(schedule["burst_period_min"]))
    on = max(1, min(period, int(schedule["burst_on_min"])))
    local_t = time - start
    in_burst = (local_t % period) < on
    if in_burst:
        hr_reserve_fraction *= float(schedule["burst_multiplier"])
    return max(0.0, min(_HR_RESERVE_FRACTION_MAX_SAFE, hr_reserve_fraction))


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
    late_bolus_id: Optional[int] = None

    # Apply scenario perturbations regardless of irregular_day so that ML labels
    # remain consistent: a scenario-5 day always has a missed bolus, etc.
    # irregular_day only affects timing jitter amplitude, not the anomaly label.
    if scenario == 4:
        lunch["duration"] = lunch["duration"] * 2
    elif scenario == 5:
        missed_meal_id = int(rng.integers(1, 6))  # 1-5 for the 5 meals
    elif scenario == 6:
        late_bolus_id = int(rng.integers(1, 6))  # 1-5 for the 5 meals

    return {
        "breakfast": breakfast,
        "morning_snack": morning_snack,
        "lunch": lunch,
        "afternoon_snack": afternoon_snack,
        "dinner": dinner,
        "missed_meal_id": missed_meal_id,
        "late_bolus_id": late_bolus_id,
    }


# ============================================================================
# Meal Computation Helper
# ============================================================================

def _apply_meal(
    meal: MealSpec,
    current_time: int,
    insulin_carbo_ratio: float,
    bolus_time: int,
    missed: bool = False,
    late_bolus: bool = False,
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
    
    # Carb intake during meal consumption
    if meal_time <= current_time < meal_time + duration:
        d_inc = carbs_grams * 1000 / duration  # mg/min
    
    # Insulin bolus
    if not missed:
        bolus_amount = carbs_grams / insulin_carbo_ratio  # [units]
        
        if late_bolus:
            # Bolus at meal time
            if meal_time <= current_time < meal_time + BOLUS_DURATION:
                u_inc = bolus_amount * 1000  # mU/min
        else:
            # Pre-bolus (15 min before meal)
            bolus_start = meal_time - bolus_time
            if bolus_start <= current_time < bolus_start + BOLUS_DURATION:
                u_inc = bolus_amount * 1000  # mU/min
    
    return u_inc, d_inc


# ============================================================================
# Main Scenario Function
# ============================================================================

def scenario_inputs(
    time: int,
    basal_hourly: float = 0.5,
    scenario: int = 1,
    insulin_carbo_ratio: float = 2.0,
    patient_age_years: float = 35.0,
    meal_schedule: Optional[MealSchedule] = None,
    exercise_schedule: Optional[ExerciseSchedule] = None,
) -> Tuple[float, float, float]:
    """
    Compute insulin delivery (u), carbohydrate intake (d), and ΔHR activity at a given time.

    Parameters:
    time: current minute (0-1439 within a day)
    basal_hourly: basal insulin rate [U/hr]
    scenario: which scenario to simulate (1-6)
    insulin_carbo_ratio: insulin-to-carb ratio [g/unit]
    patient_age_years: patient age for the 220-age HRmax conversion
    meal_schedule: optional pre-computed MealSchedule (for determinism)
    exercise_schedule: optional pre-computed ExerciseSchedule (for determinism)
    Returns:
    (u, d, activity): insulin [mU/min], carbs [mg/min], activity as ΔHR bpm above rest
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
        missed=(meal_schedule["missed_meal_id"] == 1),
        late_bolus=(meal_schedule["late_bolus_id"] == 1),
    )
    u += breakfast_u
    d += breakfast_d

    morning_snack_u, morning_snack_d = _apply_meal(
        meal_schedule["morning_snack"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        missed=(meal_schedule["missed_meal_id"] == 2),
        late_bolus=(meal_schedule["late_bolus_id"] == 2),
    )
    u += morning_snack_u
    d += morning_snack_d
    
    lunch_u, lunch_d = _apply_meal(
        meal_schedule["lunch"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        missed=(meal_schedule["missed_meal_id"] == 3),
        late_bolus=(meal_schedule["late_bolus_id"] == 3),
    )
    u += lunch_u
    d += lunch_d

    afternoon_snack_u, afternoon_snack_d = _apply_meal(
        meal_schedule["afternoon_snack"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        missed=(meal_schedule["missed_meal_id"] == 4),
        late_bolus=(meal_schedule["late_bolus_id"] == 4),
    )
    u += afternoon_snack_u
    d += afternoon_snack_d
    
    dinner_u, dinner_d = _apply_meal(
        meal_schedule["dinner"],
        time,
        insulin_carbo_ratio,
        bolus_lead_min,
        missed=(meal_schedule["missed_meal_id"] == 5),
        late_bolus=(meal_schedule["late_bolus_id"] == 5),
    )
    u += dinner_u
    d += dinner_d

    baseline_hrr = _baseline_hr_reserve_fraction_for_scenario(time=time, scenario=scenario)
    session_hrr = _exercise_hr_reserve_fraction_at_minute(time=time, schedule=exercise_schedule)
    activity = _hr_reserve_fraction_to_delta_hr(
        hr_reserve_fraction=min(_HR_RESERVE_FRACTION_MAX_SAFE, baseline_hrr + session_hrr),
        age_years=patient_age_years,
    )

    return u, d, activity


# ============================================================================
# Caching Wrapper for Determinism
# ============================================================================

_patient_baseline_cache: dict[tuple[int, int], MealSchedule] = {}
_meal_cache: dict[tuple[int, int, int], MealSchedule] = {}
_exercise_cache: dict[tuple[int, int, int], ExerciseSchedule] = {}


def scenario_with_cached_meals(
    time: int,
    patient_id: int,
    day: int,
    basal_hourly: float = 0.5,
    scenario: int = 1,
    insulin_carbo_ratio: float = 2.0,
    patient_age_years: float = 35.0,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Wrapper around scenario_inputs that caches meal schedules per (patient_id, day, scenario).

    The third return value is ΔHR in bpm above resting heart rate.
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

        day_rng = _seeded_rng(
            seed=seed,
            patient_id=patient_id,
            scenario=scenario,
            day=day,
        )
        _meal_cache[cache_key] = _build_daily_schedule_from_baseline(
            baseline=_patient_baseline_cache[patient_key],
            scenario=scenario,
            rng=day_rng,
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
        patient_age_years=patient_age_years,
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
    _meal_cache = {}
    _patient_baseline_cache = {}
    _exercise_cache = {}
