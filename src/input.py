from __future__ import annotations

import numpy as np
from typing import TypedDict, Optional, Tuple

N_SCENARIOS: int = 6  # Number of different input scenarios (e.g., different meal times, insulin doses)


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


_PATIENT_SEED_FACTOR: int = 10_000
_SCENARIO_SEED_FACTOR: int = 1_000_000
_DAY_SEED_FACTOR: int = 97
_SMALL_TIME_JITTER_MIN: int = -5
_SMALL_TIME_JITTER_MAX: int = 6
_SMALL_CARB_JITTER_PCT: float = 0.06
_IRREGULAR_DAY_PROBABILITY: float = 0.15
_IRREGULAR_TIME_JITTER_MIN: int = -20
_IRREGULAR_TIME_JITTER_MAX: int = 21


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

    if not irregular_day:
        if scenario == 4:
            lunch["duration"] = lunch["duration"] * 2
        if scenario == 5:
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
    meal_schedule: Optional[MealSchedule] = None,
) -> Tuple[float, float]:
    """
    Compute insulin delivery (u) and carbohydrate intake (d) at a given time.
    
    Parameters:
    time: current minute (0-1439 within a day)
    basal_hourly: basal insulin rate [U/hr]
    scenario: which scenario to simulate (1-6)
    insulin_carbo_ratio: insulin-to-carb ratio [g/unit]
    meal_schedule: optional pre-computed MealSchedule (for determinism)
                   If None, generates a new one (non-deterministic per call)
    
    Returns:
    (u, d): insulin [mU/min], carbohydrate intake [mg/min]
    
    Note:
    For deterministic simulations, pre-compute meal_schedule once per day
    and pass it to all 1440 minute calls. See scenario_with_cached_meals().
    """
    # Convert basal from U/hr to mU/min
    basal = basal_hourly * 1000 / 60  # ~8.33 mU/min for 0.5 U/hr
    
    # Generate or use provided meal schedule
    if meal_schedule is None:
        meal_schedule = generate_meal_schedule(scenario=scenario)

    u: float = basal
    d: float = 0.0
    
    # Apply each meal - 5 regularly scheduled meals for active patient
    breakfast_u, breakfast_d = _apply_meal(
        meal_schedule["breakfast"],
        time,
        insulin_carbo_ratio,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 1),
        late_bolus=(meal_schedule["late_bolus_id"] == 1),
    )
    u += breakfast_u
    d += breakfast_d
    
    morning_snack_u, morning_snack_d = _apply_meal(
        meal_schedule["morning_snack"],
        time,
        insulin_carbo_ratio,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 2),
        late_bolus=(meal_schedule["late_bolus_id"] == 2),
    )
    u += morning_snack_u
    d += morning_snack_d
    
    lunch_u, lunch_d = _apply_meal(
        meal_schedule["lunch"],
        time,
        insulin_carbo_ratio,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 3),
        late_bolus=(meal_schedule["late_bolus_id"] == 3),
    )
    u += lunch_u
    d += lunch_d
    
    afternoon_snack_u, afternoon_snack_d = _apply_meal(
        meal_schedule["afternoon_snack"],
        time,
        insulin_carbo_ratio,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 4),
        late_bolus=(meal_schedule["late_bolus_id"] == 4),
    )
    u += afternoon_snack_u
    d += afternoon_snack_d
    
    dinner_u, dinner_d = _apply_meal(
        meal_schedule["dinner"],
        time,
        insulin_carbo_ratio,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 5),
        late_bolus=(meal_schedule["late_bolus_id"] == 5),
    )
    u += dinner_u
    d += dinner_d
    
    return u, d


# ============================================================================
# Caching Wrapper for Determinism
# ============================================================================

_patient_baseline_cache: dict[tuple[int, int], MealSchedule] = {}
_meal_cache: dict[tuple[int, int, int], MealSchedule] = {}


def scenario_with_cached_meals(
    time: int,
    patient_id: int,
    day: int,
    basal_hourly: float = 0.5,
    scenario: int = 1,
    insulin_carbo_ratio: float = 2.0,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Wrapper around scenario_inputs that caches meal schedules per (patient_id, day, scenario).
    
    Ensures deterministic behavior with realistic structure:
    - same patient has a stable baseline routine,
    - each day has small jitter,
    - some days have larger deviations (irregular days).
    
    Parameters:
    time: current minute (0-1439)
    patient_id: unique patient identifier
    day: day number (for caching key)
    basal_hourly: basal insulin rate [U/hr]
    scenario: scenario number (1-6)
    insulin_carbo_ratio: carb-to-insulin ratio
    seed: optional random seed (for reproducibility across runs)
    
    Returns:
    (u, d): insulin [mU/min], carbohydrate [mg/min]
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
    
    return scenario_inputs(
        time,
        basal_hourly=basal_hourly,
        scenario=scenario,
        insulin_carbo_ratio=insulin_carbo_ratio,
        meal_schedule=_meal_cache[cache_key],
    )


def clear_meal_cache() -> None:
    """Clear the meal cache (useful for tests or starting a new cohort)."""
    global _meal_cache
    global _patient_baseline_cache
    _meal_cache = {}
    _patient_baseline_cache = {}
