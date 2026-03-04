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
# ============================================================================

BREAKFAST_CARBS_MIN: int = 30
BREAKFAST_CARBS_MAX: int = 60
BREAKFAST_TIME_MIN: int = time_to_minutes(7, 0)
BREAKFAST_TIME_MAX: int = time_to_minutes(9, 0)
BREAKFAST_DURATION_MIN: int = 10
BREAKFAST_DURATION_MAX: int = 15

LUNCH_CARBS_MIN: int = 50
LUNCH_CARBS_MAX: int = 100
LUNCH_TIME_MIN: int = time_to_minutes(12, 0)
LUNCH_TIME_MAX: int = time_to_minutes(14, 0)
LUNCH_DURATION_MIN: int = 25
LUNCH_DURATION_MAX: int = 30

DINNER_CARBS_MIN: int = 50
DINNER_CARBS_MAX: int = 80
DINNER_TIME_MIN: int = time_to_minutes(18, 0)
DINNER_TIME_MAX: int = time_to_minutes(21, 0)
DINNER_DURATION_MIN: int = 10
DINNER_DURATION_MAX: int = 15

SNACK_CARBS_MIN: int = 10
SNACK_CARBS_MAX: int = 30
SNACK_TIME_MIN: int = time_to_minutes(10, 0)
SNACK_TIME_MAX: int = time_to_minutes(11, 0)
SNACK_DURATION_MIN: int = 5
SNACK_DURATION_MAX: int = 10

PREANNOUNCED_BOLUS_TIME: int = 15
BOLUS_DURATION: int = 1

# ============================================================================
# Meal Schedule Type
# ============================================================================

class MealSpec(TypedDict):
    """Specification of a single meal (time window, carbs, duration)."""
    time: int           # [min] when meal starts
    duration: int       # [min] how long meal lasts
    carbs: int          # [g] total carbs in meal


class MealSchedule(TypedDict):
    """Complete daily meal schedule."""
    breakfast: MealSpec
    snack: MealSpec
    lunch: MealSpec
    dinner: MealSpec
    missed_meal_id: Optional[int]  # For missed bolus scenario (1-4 or None)
    late_bolus_id: Optional[int]   # For late bolus scenario (1-4 or None)


_PATIENT_SEED_FACTOR: int = 10_000
_SCENARIO_SEED_FACTOR: int = 1_000_000
_DAY_SEED_FACTOR: int = 97
_SMALL_TIME_JITTER_MIN: int = -5
_SMALL_TIME_JITTER_MAX: int = 6
_SMALL_CARB_JITTER_PCT: float = 0.06
_IRREGULAR_DAY_PROBABILITY: float = 0.30
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
    
    # Base meals for all scenarios
    breakfast: MealSpec = {
        "time": int(rng.integers(BREAKFAST_TIME_MIN, BREAKFAST_TIME_MAX + 1)),
        "duration": int(rng.integers(BREAKFAST_DURATION_MIN, BREAKFAST_DURATION_MAX + 1)),
        "carbs": int(rng.integers(BREAKFAST_CARBS_MIN, BREAKFAST_CARBS_MAX + 1)),
    }
    
    snack: MealSpec = {
        "time": int(rng.integers(SNACK_TIME_MIN, SNACK_TIME_MAX + 1)),
        "duration": int(rng.integers(SNACK_DURATION_MIN, SNACK_DURATION_MAX + 1)),
        "carbs": int(rng.integers(SNACK_CARBS_MIN, SNACK_CARBS_MAX + 1)),
    }
    
    lunch: MealSpec = {
        "time": int(rng.integers(LUNCH_TIME_MIN, LUNCH_TIME_MAX + 1)),
        "duration": int(rng.integers(LUNCH_DURATION_MIN, LUNCH_DURATION_MAX + 1)),
        "carbs": int(rng.integers(LUNCH_CARBS_MIN, LUNCH_CARBS_MAX + 1)),
    }
    
    dinner: MealSpec = {
        "time": int(rng.integers(DINNER_TIME_MIN, DINNER_TIME_MAX + 1)),
        "duration": int(rng.integers(DINNER_DURATION_MIN, DINNER_DURATION_MAX + 1)),
        "carbs": int(rng.integers(DINNER_CARBS_MIN, DINNER_CARBS_MAX + 1)),
    }
    
    # Scenario-specific modifications
    missed_meal_id: Optional[int] = None
    late_bolus_id: Optional[int] = None
    
    if scenario == 3:
        # Long lunch: extend lunch duration by 2×
        lunch["duration"] = lunch["duration"] * 2
    elif scenario == 4:
        # Missed bolus: one of the meal boluses is skipped (1-4)
        missed_meal_id = int(rng.integers(1, 5))
    elif scenario == 5:
        # Late bolus: one meal gets bolus at meal time instead of 15 min before
        late_bolus_id = int(rng.integers(1, 5))
    
    return {
        "breakfast": breakfast,
        "snack": snack,
        "lunch": lunch,
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
    rng: np.random.Generator,
    jitter_min: int,
    jitter_max: int,
    carb_jitter_pct: float,
) -> MealSpec:
    time_jitter = int(rng.integers(jitter_min, jitter_max))
    new_time = _clamp_int(meal["time"] + time_jitter, time_min, time_max)

    carb_factor = 1.0 + float(rng.uniform(-carb_jitter_pct, carb_jitter_pct))
    new_carbs = _clamp_int(
        int(round(meal["carbs"] * carb_factor)),
        carbs_min,
        carbs_max,
    )

    return {
        "time": new_time,
        "duration": meal["duration"],
        "carbs": new_carbs,
    }


def _build_daily_schedule_from_baseline(
    baseline: MealSchedule,
    scenario: int,
    rng: np.random.Generator,
) -> MealSchedule:
    breakfast = _jitter_meal(
        baseline["breakfast"],
        BREAKFAST_TIME_MIN,
        BREAKFAST_TIME_MAX,
        BREAKFAST_CARBS_MIN,
        BREAKFAST_CARBS_MAX,
        rng,
        _SMALL_TIME_JITTER_MIN,
        _SMALL_TIME_JITTER_MAX,
        _SMALL_CARB_JITTER_PCT,
    )
    snack = _jitter_meal(
        baseline["snack"],
        SNACK_TIME_MIN,
        SNACK_TIME_MAX,
        SNACK_CARBS_MIN,
        SNACK_CARBS_MAX,
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
            rng,
            _IRREGULAR_TIME_JITTER_MIN,
            _IRREGULAR_TIME_JITTER_MAX,
            _SMALL_CARB_JITTER_PCT,
        )
        snack = _jitter_meal(
            snack,
            SNACK_TIME_MIN,
            SNACK_TIME_MAX,
            SNACK_CARBS_MIN,
            SNACK_CARBS_MAX,
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
            rng,
            _IRREGULAR_TIME_JITTER_MIN,
            _IRREGULAR_TIME_JITTER_MAX,
            _SMALL_CARB_JITTER_PCT,
        )

    missed_meal_id: Optional[int] = None
    late_bolus_id: Optional[int] = None

    if scenario == 3:
        lunch["duration"] = lunch["duration"] * 2

    if scenario == 4 and irregular_day:
        missed_meal_id = int(rng.integers(1, 5))
    elif scenario == 5 and irregular_day:
        late_bolus_id = int(rng.integers(1, 5))

    return {
        "breakfast": breakfast,
        "snack": snack,
        "lunch": lunch,
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
    insulin_sensitivity: float,
    bolus_time: int,
    missed: bool = False,
    late_bolus: bool = False,
) -> Tuple[float, float]:
    """
    Compute carb intake and insulin bolus for a single meal at current_time.
    
    Parameters:
    meal: MealSpec with time, duration, carbs
    current_time: current minute in day
    insulin_sensitivity: carbs per insulin unit [g/unit]
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
    carbs_grams = meal["carbs"]
    
    # Carb intake during meal consumption
    if meal_time <= current_time < meal_time + duration:
        d_inc = carbs_grams * 1000 / duration  # mg/min
    
    # Insulin bolus
    if not missed:
        bolus_amount = carbs_grams / insulin_sensitivity  # [units]
        
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
    insulin_sensitivity: float = 2.0,
    meal_schedule: Optional[MealSchedule] = None,
) -> Tuple[float, float]:
    """
    Compute insulin delivery (u) and carbohydrate intake (d) at a given time.
    
    Parameters:
    time: current minute (0-1439 within a day)
    basal_hourly: basal insulin rate [U/hr]
    scenario: which scenario to simulate (1-6)
    insulin_sensitivity: insulin-to-carb ratio [g/unit]
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
    
    # Apply each meal
    breakfast_u, breakfast_d = _apply_meal(
        meal_schedule["breakfast"],
        time,
        insulin_sensitivity,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 1),
        late_bolus=(meal_schedule["late_bolus_id"] == 1),
    )
    u += breakfast_u
    d += breakfast_d
    
    snack_u, snack_d = _apply_meal(
        meal_schedule["snack"],
        time,
        insulin_sensitivity,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 2),
        late_bolus=(meal_schedule["late_bolus_id"] == 2),
    )
    u += snack_u
    d += snack_d
    
    lunch_u, lunch_d = _apply_meal(
        meal_schedule["lunch"],
        time,
        insulin_sensitivity,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 3),
        late_bolus=(meal_schedule["late_bolus_id"] == 3),
    )
    u += lunch_u
    d += lunch_d
    
    dinner_u, dinner_d = _apply_meal(
        meal_schedule["dinner"],
        time,
        insulin_sensitivity,
        PREANNOUNCED_BOLUS_TIME,
        missed=(meal_schedule["missed_meal_id"] == 4),
        late_bolus=(meal_schedule["late_bolus_id"] == 4),
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
    insulin_sensitivity: float = 2.0,
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
    insulin_sensitivity: carb-to-insulin ratio
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
        insulin_sensitivity=insulin_sensitivity,
        meal_schedule=_meal_cache[cache_key],
    )


def clear_meal_cache() -> None:
    """Clear the meal cache (useful for tests or starting a new cohort)."""
    global _meal_cache
    global _patient_baseline_cache
    _meal_cache = {}
    _patient_baseline_cache = {}


# ============================================================================
# LEGACY: Gemini Reference Implementation (Commented)
# ============================================================================

# The following is a legacy reference implementation. It is NOT used by the main pipeline.
# Kept for historical reference only.
#
# def scenario_inputs_gemini(t: int, scenario: int = 1) -> Tuple[float, float]:
#     """
#     Legacy Gemini reference: hardcoded single meal, no variability.
#
#     Defines what happens at time t.
#     Time is in minutes. 0 to 1440 (24 hours).
#     """
#     # Basal Insulin (constant background)
#     u_val = 8.33  # 0.5 U/hr = 8.33 mU/min
#
#     # Meal at t=120 min (2 hours in)
#     # 50g carbs eaten over 15 mins
#     d_val = 0.0
#     if 120 <= t <= 135:
#         d_val = 50000 / 15  # mg/min (Total 50g)
#
#     # Meal Bolus Insulin at t=120
#     # 5 Units bolus delivered over 1 min
#     if 120 <= t <= 121:
#         u_val += 5000  # 5 U = 5000 mU
#
#     return u_val, d_val
