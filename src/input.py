from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Time Conversion Helper
# ============================================================================

def time_to_minutes(h: int, m: int) -> int:
    """Convert hours and minutes to total minutes from midnight."""
    return h * 60 + m


# ============================================================================
# Patient-Level Scenario Probabilities
# ============================================================================

# Exercise tendency distribution — sampled ONCE per patient at profile creation.
# Maps onto the old SC1/SC2/SC3 archetypes but drives a continuous daily probability
# rather than a deterministic 100%-exercise-every-day for active patients.
# Indices: 0=normal, 1=active, 2=sedentary  (preserved as base_scenario 1/2/3 for export)
EXERCISE_TENDENCY_WEIGHTS: list[float] = [0.38, 0.27, 0.35]  # normal / active / sedentary
EXERCISE_TENDENCY_LABELS:  list[str]   = ['normal', 'active', 'sedentary']

# Per-tendency daily exercise probability ranges [min, max].
# Active patients exercise most days but not every day — more realistic than the old SC2
# guarantee of 100% daily exercise.
EXERCISE_DAILY_PROB: dict[str, tuple[float, float]] = {
    'sedentary': (0.03, 0.10),
    'normal':    (0.20, 0.45),
    'active':    (0.55, 0.80),   # was 1.0 for SC2 — now stochastic
}

# Per-tendency exercise type probability distribution.
EXERCISE_TYPE_PROBS: dict[str, dict[str, float]] = {
    'sedentary': {'aerobic': 0.80, 'prolonged': 0.10, 'anaerobic': 0.10},
    'normal':    {'aerobic': 0.70, 'prolonged': 0.15, 'anaerobic': 0.15},
    'active':    {'aerobic': 0.65, 'prolonged': 0.25, 'anaerobic': 0.10},
}

# Per-tendency intensity bias range; active patients push harder so their AC ranges
# are scaled up from the base constants.
EXERCISE_INTENSITY_BIAS: dict[str, tuple[float, float]] = {
    'sedentary': (0.70, 0.90),
    'normal':    (0.85, 1.10),
    'active':    (1.00, 1.25),
}

# Baseline meal count: equal thirds across the cohort.
MEAL_COUNT_PROBS: list[float] = [1/3, 1/3, 1/3]  # 3, 4, or 5 meals

# Probability that a patient's daily meal count deviates by ±1 from their baseline.
MEAL_COUNT_DEVIATION_PROB: float = 0.20


# ============================================================================
# Per-Day Overlay Probabilities
# ============================================================================
# Each overlay is sampled independently (Bernoulli) each day.
# Marginals match the target cohort-level rates:
#   sc4: 9% of all days → large meal (restaurant/event dining)
#   sc5: 6% of all days → missed bolus
#   sc6: 6% of all days → late bolus (primary event)
# Exercise overlays (sc7/sc8) are now driven by EXERCISE_TYPE_PROBS above,
# not per-day Bernoulli draws; P_PROLONGED_AEROBIC_GIVEN_SC1 / P_ANAEROBIC_GIVEN_SC1
# are no longer used.

P_LARGE_MEAL:   float = 0.09  # large-meal event (restaurant, party, event dining, etc.)
P_MISSED_BOLUS: float = 0.09  # raised from 0.06: primary lever for physiological CV (each missed bolus adds a clean 2–4h excursion to 12–16 mmol/L, increasing population std without causing chronic hyperglycemia)
P_LATE_BOLUS:   float = 0.06
# Conditional probability of a 2nd late-bolus event given one was already sampled.
# Inflates the sc6 daily marginal slightly above 0.06 (to ~0.069); documented in thesis.
P_LATE_BOLUS_SECOND: float = 0.15


# ============================================================================
# Meal Slot Definitions
# ============================================================================
# Slot convention: 1=breakfast, 2=morning snack, 3=lunch, 4=afternoon snack, 5=dinner

_MAIN_SLOTS:  frozenset[int] = frozenset({1, 3, 5})
_SNACK_SLOTS: frozenset[int] = frozenset({2, 4})


# ============================================================================
# Meal Times (minute-of-day) by Slot and Meal Count
# ============================================================================
# Each dict covers exactly the slots that are active for that meal-count.
# 3-meal patients skip snack slots entirely (exercise session may fill that gap).
# 4-meal patients define all 5 slot anchor times so whichever snack slot is
# sampled for the day (slot 2 or 4) can be looked up.
# 5-meal patients use slightly earlier slot-4/slot-5 times to preserve room for
# a prolonged aerobic session (75–120 min) between afternoon snack and dinner.
# Worst-case with max main jitter: slot-4 end+45 = 1005, dinner−45 = 1215 → 210 min free.

_MEAL_TIMES_3MEALS: dict[int, int] = {
    1: time_to_minutes(8, 0),    # breakfast       08:00
    # slot 2: no morning snack (exercise session may occur here)
    3: time_to_minutes(13, 0),   # lunch           13:00
    # slot 4: no afternoon snack (exercise session may occur here)
    5: time_to_minutes(20, 0),   # dinner          20:00
}

_MEAL_TIMES_4MEALS: dict[int, int] = {
    1: time_to_minutes(7, 15),   # breakfast       07:15
    2: time_to_minutes(10, 30),  # morning snack   10:30 (or exercise slot if not active)
    3: time_to_minutes(13, 0),   # lunch           13:00
    4: time_to_minutes(16, 0),   # afternoon snack 16:00 (or exercise slot if not active)
    5: time_to_minutes(21, 0),   # dinner          21:00
}

_MEAL_TIMES_5MEALS: dict[int, int] = {
    1: time_to_minutes(7, 15),   # breakfast       07:15
    2: time_to_minutes(10, 30),  # morning snack   10:30
    3: time_to_minutes(13, 0),   # lunch           13:00
    4: time_to_minutes(16, 0),   # afternoon snack 16:00
    5: time_to_minutes(21, 0),   # dinner          21:00
}

_MEAL_TIME_MAIN_JITTER_MAX: int  = 45  # uniform ±45 min (main meals: breakfast/lunch/dinner)
_MEAL_TIME_SNACKS_JITTER_MAX: int = 30  # uniform ±30 min (snacks: morning/afternoon)

# Meal eating-duration range (min, max) in minutes by slot
_MEAL_DURATION_RANGE: dict[int, tuple[int, int]] = {
    1: (15, 20),  # breakfast
    2: (8, 12),   # morning snack
    3: (20, 25),  # lunch
    4: (5, 10),   # afternoon snack
    5: (20, 25),  # dinner
}


# ============================================================================
# Carb Amounts by Meal Count and Slot
# ============================================================================
# 3-meal days have larger mains (no snacks to distribute energy).
# 5-meal days have smaller mains + two snacks.

_CARBS_BY_COUNT: dict[int, dict[int, int]] = {
    3: {1: 75, 3: 90, 5: 90},
    4: {1: 70, 2: 20, 3: 80, 4: 20, 5: 80},
    5: {1: 65, 2: 20, 3: 75, 4: 20, 5: 75},
}

_SMALL_CARB_JITTER_PCT: float = 0.10  # ±10% day-to-day carb variability


# ============================================================================
# Bolus Estimation Constants
# ============================================================================

BOLUS_DURATION: int = 3  # minutes over which a bolus is spread (prevents sharp S1 spike)

_PATIENT_BOLUS_BIAS_MIN: float = 0.78
_PATIENT_BOLUS_BIAS_MAX: float = 0.95

_MEAL_EST_NOISE_MIN: float = 0.80
_MEAL_EST_NOISE_MAX: float = 1.20
_LUNCH_DINNER_UNDEREST_MIN: float = 0.82
_LUNCH_DINNER_UNDEREST_MAX: float = 0.93
_SNACK_EST_MIN: float = 0.96
_SNACK_EST_MAX: float = 1.04

_LARGE_MEAL_CARB_FACTOR_MIN: float = 1.8    # actual carb load inflation for large-meal event
_LARGE_MEAL_CARB_FACTOR_MAX: float = 2.1
_LARGE_MEAL_UNDEREST_MIN: float = 0.70      # systematic bolus underestimation (patient misjudges portion)
_LARGE_MEAL_UNDEREST_MAX: float = 0.85

_LATE_BOLUS_DELAY_MIN: int = 30   # minutes after meal start
_LATE_BOLUS_DELAY_MAX: int = 90

# Per-meal bolus lead-time distribution.
# Positive = pre-meal, 0 = at meal onset, negative = post-meal bolus delivery.
# Empirical T1D breakdown: ~60% pre-meal, ~25% at onset, ~15% post-meal.
_BOLUS_LEAD_PRE_MIN: int  = 5
_BOLUS_LEAD_PRE_MAX: int  = 20
_BOLUS_LEAD_AT_MIN: int   = 0
_BOLUS_LEAD_AT_MAX: int   = 4
_BOLUS_LEAD_POST_MIN: int = -20   # negative → bolus N min after meal starts
_BOLUS_LEAD_POST_MAX: int = -1
_BOLUS_LEAD_PRE_PROB: float  = 0.60
_BOLUS_LEAD_AT_PROB: float   = 0.25
# post-meal probability is implicitly 1 − 0.60 − 0.25 = 0.15


# ============================================================================
# Exercise Constants
# ============================================================================

EARLIEST_EXERCISE_TIME: int  = time_to_minutes(8, 0)  # 08:00 = 480 min
MIN_GAP_BEFORE_MEAL_MIN: int = 20   # no exercise within 20 min before a meal
MIN_GAP_AFTER_MEAL_MIN: int  = 20   # no exercise within 20 min after meal ends

# Duration windows (min, max) in minutes by exercise type
_EXERCISE_DURATION: dict[str, tuple[int, int]] = {
    'aerobic':   (30, 60),    # sc2 baseline aerobic
    'anaerobic': (20, 50),    # sc8 overlay
    'prolonged': (75, 120),   # sc7 overlay; always longer than any sc2 session for clean ML separation
}

# Accelerometer count (AC) ranges by exercise type.
# Reference: aAC=1000 → fAC=0.5 (moderate aerobic onset); ah=5600 → fHI=0.5 (anaerobic onset).
_AC_AEROBIC_MIN: float   = 1200.0
_AC_AEROBIC_MAX: float   = 2000.0
_AC_PROLONGED_MIN: float = 1500.0
_AC_PROLONGED_MAX: float = 2500.0
_AC_ANAEROBIC_MIN: float = 6000.0
_AC_ANAEROBIC_MAX: float = 9000.0

_AC_INCIDENTAL_NORMAL: float    = 300.0   # sc1/sc2 incidental movement baseline
_AC_INCIDENTAL_SEDENTARY: float = 100.0   # sc3 incidental movement baseline


# ============================================================================
# ML Label Window Constants
# ============================================================================
# All windows start at LABEL_WINDOW_BOLUS_START after the meal event to account
# for the gastric-emptying + CGM interstitial lag before a CGM-visible signal.
# Window endpoints chosen from Hovorka-model average glucose trajectories:
#   Normal bolus:  post-meal excursion resolves ~2 h after meal
#   Missed bolus:  sustained high, still elevated at 3 h
#   Late bolus:    elevated peak + delayed correction, visible until ~4 h
#   Restaurant:    high-fat delayed absorption, visible 5 h post-meal
#
# Exercise labels cover session + recovery period where CGM shows anomalous
# glucose change (hypoglycaemia during/after aerobic; hyper spike from anaerobic).

LABEL_WINDOW_BOLUS_START: int = 15     # min after meal: earliest CGM-visible signal

LABEL_WINDOW_NORMAL_END: int  = 120    # normal bolus post-meal window
LABEL_WINDOW_MISSED_END: int  = 180    # missed bolus: sustained high
LABEL_WINDOW_LATE_END: int    = 240    # late bolus: includes delayed correction
LABEL_WINDOW_LARGE_END: int   = 300    # large meal: prolonged fat-delayed absorption (restaurant/event dining)

LABEL_WINDOW_AEROBIC_POST: int   = 60   # aerobic recovery (sc2 + sc7 both use this end cap)
LABEL_WINDOW_PROLONGED_POST: int = 120  # prolonged aerobic: extended post-exercise sensitivity
LABEL_WINDOW_ANAEROBIC_POST: int = 60   # anaerobic: acute hyper spike resolves ~60 min


# ============================================================================
# RNG Seeding Factors
# ============================================================================

_PATIENT_SEED_FACTOR: int    = 10_000
_DAY_SEED_FACTOR: int        = 97
_PROFILE_STREAM_OFFSET: int  = 0    # patient profile uses base stream
_DAY_PLAN_STREAM_OFFSET: int = 13   # meal/anomaly draws use this offset
_EXERCISE_STREAM_OFFSET: int = 17   # exercise placement uses a separate stream


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PatientProfile:
    """Fixed per-patient characteristics sampled once at cohort creation."""
    baseline_meal_count: int           # 3, 4, or 5
    baseline_snack_choice: Optional[int]  # 2 (morning) or 4 (afternoon) if count=4; None otherwise
    base_scenario: int                 # 1 (normal activity), 2 (active aerobic), 3 (sedentary) — kept for export compat
    bolus_bias: float                  # persistent carb-estimation multiplier [0.78–0.95]
    # Continuous exercise traits replace the old deterministic SC2 guarantee
    exercise_tendency: str             # 'sedentary' | 'normal' | 'active'
    exercise_daily_prob: float         # P(exercise today) ∈ [0.03, 0.80]; sampled once per patient
    exercise_intensity_bias: float     # multiplier on AC range [0.70, 1.25]


@dataclass
class MealEvent:
    """One meal for a single simulated day, with full anomaly annotation."""
    slot: int           # 1–5 (1=breakfast 2=morning-snack 3=lunch 4=afternoon-snack 5=dinner)
    time_min: int       # minute-of-day the meal starts
    duration: int       # minutes the meal lasts
    carbs: int          # actual carbohydrates consumed [g]
    bolus_carbs: float  # estimated carbs used for bolus dose [g]; 0 if missed
    bolus_status: str   # 'normal' | 'missed' | 'late'
    meal_size: str      # 'normal' | 'large' (large-meal event: restaurant, party, etc.)
    bolus_lead_min: int     # minutes before meal start for normal delivery (positive = pre-meal)
    late_bolus_delay_min: int  # minutes after meal start for late delivery; 0 if not late


@dataclass
class ExerciseEvent:
    """One exercise session for a single simulated day."""
    start_min: int          # minute-of-day the session starts
    duration_min: int       # session duration [min]
    exercise_type: str      # 'aerobic' | 'anaerobic' | 'prolonged'
    is_anomaly_overlay: bool   # False = sc2 baseline activity; True = sc7/sc8 overlay
    ac_counts: float           # representative accelerometer counts for session intensity
    burst_period_min: int      # burst interval in minutes (1 = no burst pattern)
    burst_on_min: int          # active portion of each burst period
    burst_multiplier: float    # AC scaling during burst-on phase (1.0 = no burst)


@dataclass
class DayPlan:
    """Complete daily simulation specification: activity base, meals, and overlays."""
    base_scenario: int            # 1, 2, or 3
    meal_count: int               # number of meals today
    meals: list[MealEvent]
    exercise: Optional[ExerciseEvent]
    # Overlay metadata for export and verification
    had_large_meal: bool  # True if a large-meal event (restaurant/party) occurred today
    had_missed_bolus: bool
    n_late_boluses: int
    exercise_overlay: Optional[int]  # 7 (prolonged aerobic), 8 (anaerobic), or None
    # Note: aerobic exercise (normal or sedentary-anomaly) always maps to None here —
    # distinguish via exercise.exercise_type + exercise.is_anomaly_overlay if needed.

    @property
    def is_exercise_day(self) -> bool:
        """True if any exercise session occurs today (any tendency, any type)."""
        return self.exercise is not None


# ============================================================================
# Internal helper: _MealAnomalyAssignment
# ============================================================================

@dataclass
class _MealAnomalyAssignment:
    large_meal_slot: Optional[int] = None
    missed_bolus_slot: Optional[int] = None
    late_bolus_slots: list[int] = field(default_factory=lambda: [])


# ============================================================================
# Utility Functions
# ============================================================================

def _seeded_rng(
    seed: Optional[int],
    patient_id: int,
    stream_offset: int = 0,
    day: Optional[int] = None,
) -> np.random.Generator:
    """Create a reproducible per-(patient, stream, day) RNG from a global seed."""
    if seed is None:
        return np.random.default_rng()
    value = int(seed) + int(patient_id) * _PATIENT_SEED_FACTOR + stream_offset
    if day is not None:
        d = int(day)
        if d < 0:
            # Warmup days use negative indices (-n_warmup_days … -1).  Map them to
            # large unique positive offsets (1_000_000+d) so the combined seed stays
            # non-negative even when seed=0 and patient_id=0.  The mapped range
            # (~999_997–999_999) is far above any realistic recorded day count.
            d = 1_000_000 + d
        value += d * _DAY_SEED_FACTOR
    return np.random.default_rng(value)


def _sample_bolus_lead(rng: np.random.Generator) -> int:
    """Draw per-meal bolus lead time from a realistic T1D timing distribution.

    Returns minutes before meal start (positive = pre-meal, negative = post-meal).
    Uses exactly 2 RNG draws (category + value).
    """
    r = float(rng.random())
    if r < _BOLUS_LEAD_PRE_PROB:
        return int(rng.integers(_BOLUS_LEAD_PRE_MIN, _BOLUS_LEAD_PRE_MAX + 1))
    elif r < _BOLUS_LEAD_PRE_PROB + _BOLUS_LEAD_AT_PROB:
        return int(rng.integers(_BOLUS_LEAD_AT_MIN, _BOLUS_LEAD_AT_MAX + 1))
    else:
        return int(rng.integers(_BOLUS_LEAD_POST_MIN, _BOLUS_LEAD_POST_MAX + 1))


def _compute_free_windows(
    day_start: int,
    day_end: int,
    forbidden: list[tuple[int, int]],
    min_duration: int,
) -> list[tuple[int, int]]:
    """Return (start, end) intervals in [day_start, day_end] avoiding forbidden zones.

    Forbidden zones are merged, then the gaps are collected.  Only windows with
    length >= min_duration are returned.
    """
    forbidden_sorted = sorted(forbidden)
    merged: list[tuple[int, int]] = []
    for lo, hi in forbidden_sorted:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    free: list[tuple[int, int]] = []
    cursor = day_start
    for lo, hi in merged:
        if lo > cursor:
            free.append((cursor, min(lo, day_end)))
        cursor = max(cursor, hi)
        if cursor >= day_end:
            break
    if cursor < day_end:
        free.append((cursor, day_end))

    return [(a, b) for a, b in free if (b - a) >= min_duration]


# ============================================================================
# Patient Profile Generation
# ============================================================================

def _generate_patient_profile(
    patient_id: int,
    seed: Optional[int],
    base_scenario_override: Optional[int] = None,
) -> PatientProfile:
    """Sample per-patient fixed characteristics.

    base_scenario_override: when config.random_scenarios=False, forces all
        patients to this base scenario (1–3), mapped to the corresponding
        exercise tendency for backwards compatibility.
    """
    rng = _seeded_rng(seed, patient_id, _PROFILE_STREAM_OFFSET)

    # Baseline meal count (equal probability for 3, 4, 5)
    baseline_meal_count = int(rng.choice([3, 4, 5], p=MEAL_COUNT_PROBS))

    # For 4-meal patients: habitual snack slot (morning=2 or afternoon=4)
    if baseline_meal_count == 4:
        baseline_snack_choice = int(rng.choice([2, 4]))
    else:
        baseline_snack_choice = None

    # Exercise tendency — replaces hard SC1/SC2/SC3 for exercise logic.
    # base_scenario is preserved (1=normal, 2=active, 3=sedentary) for export compat.
    if base_scenario_override is not None:
        base_scenario = max(1, min(3, int(base_scenario_override)))
        tendency_map = {1: 'normal', 2: 'active', 3: 'sedentary'}
        exercise_tendency = tendency_map[base_scenario]
    else:
        tendency_idx = int(rng.choice(3, p=EXERCISE_TENDENCY_WEIGHTS))
        exercise_tendency = EXERCISE_TENDENCY_LABELS[tendency_idx]
        base_scenario = tendency_idx + 1  # 0→1 (normal), 1→2 (active), 2→3 (sedentary)

    # Per-patient daily exercise probability drawn from the tendency's range
    p_lo, p_hi = EXERCISE_DAILY_PROB[exercise_tendency]
    exercise_daily_prob = float(rng.uniform(p_lo, p_hi))

    # Per-patient intensity bias drawn from the tendency's range
    i_lo, i_hi = EXERCISE_INTENSITY_BIAS[exercise_tendency]
    exercise_intensity_bias = float(rng.uniform(i_lo, i_hi))

    # Stable per-patient bolus estimation bias [0.78–0.95]
    bolus_bias = float(rng.uniform(_PATIENT_BOLUS_BIAS_MIN, _PATIENT_BOLUS_BIAS_MAX))

    return PatientProfile(
        baseline_meal_count=baseline_meal_count,
        baseline_snack_choice=baseline_snack_choice,
        base_scenario=base_scenario,
        bolus_bias=bolus_bias,
        exercise_tendency=exercise_tendency,
        exercise_daily_prob=exercise_daily_prob,
        exercise_intensity_bias=exercise_intensity_bias,
    )


# ============================================================================
# Day Plan Generation Helpers
# ============================================================================

def _sample_daily_meal_count(rng: np.random.Generator, profile: PatientProfile) -> int:
    """Sample today's meal count; ±1 deviation from baseline with MEAL_COUNT_DEVIATION_PROB.

    Boundary rules: baseline=3 can only go up to 4; baseline=5 can only go down to 4.
    """
    if float(rng.random()) >= MEAL_COUNT_DEVIATION_PROB:
        return profile.baseline_meal_count
    if profile.baseline_meal_count == 3:
        return 4
    if profile.baseline_meal_count == 5:
        return 4
    # baseline == 4: equal chance of 3 or 5
    return 3 if float(rng.random()) < 0.5 else 5


def _sample_active_slots(
    rng: np.random.Generator,
    profile: PatientProfile,
    meal_count: int,
) -> frozenset[int]:
    """Return the active meal slots for today given today's meal count."""
    if meal_count == 3:
        return frozenset({1, 3, 5})
    if meal_count == 5:
        return frozenset({1, 2, 3, 4, 5})
    # meal_count == 4: pick which snack to include
    if (profile.baseline_meal_count == 4
            and profile.baseline_snack_choice is not None
            and float(rng.random()) < 0.8):
        # 80% of days: use the patient's habitual snack slot
        snack = profile.baseline_snack_choice
    else:
        snack = int(rng.choice([2, 4]))
    return frozenset({1, snack, 3, 5})


def _sample_meal_anomalies(
    rng: np.random.Generator,
    active_slots: frozenset[int],
) -> _MealAnomalyAssignment:
    """Sample meal-level anomaly overlays for one day.

    Constraints enforced:
      - sc4 (large meal): at most 1 per day; biased toward lunch/dinner (slots 3/5)
      - sc5 (missed bolus): at most 1 per day; excluded if 2 late boluses exist
      - sc6 (late bolus): at most 2 per day; sc5 and sc6 cannot share a meal
    """
    assignment = _MealAnomalyAssignment()
    slots = sorted(active_slots)

    # sc4: large-meal event (restaurant/party) — prefer main meal slots 3 (lunch) or 5 (dinner)
    if float(rng.random()) < P_LARGE_MEAL:
        main_candidates = [s for s in slots if s in (3, 5)]
        if not main_candidates:
            main_candidates = [s for s in slots if s in _MAIN_SLOTS]
        if main_candidates:
            assignment.large_meal_slot = int(rng.choice(main_candidates))

    # sc5: tentatively sample missed bolus
    missed_sampled = float(rng.random()) < P_MISSED_BOLUS

    # sc6: late bolus — 0, 1, or 2 per day
    n_late = 0
    if float(rng.random()) < P_LATE_BOLUS:
        n_late = 1
        if float(rng.random()) < P_LATE_BOLUS_SECOND:
            n_late = 2

    # Constraint: 2 late boluses → no missed bolus this day
    if n_late == 2:
        missed_sampled = False

    # Assign late-bolus slots (without replacement from all active slots)
    if n_late > 0:
        late_pool = list(slots)
        for _ in range(min(n_late, len(late_pool))):
            chosen = int(rng.choice(late_pool))
            assignment.late_bolus_slots.append(chosen)
            late_pool.remove(chosen)

    # Assign missed-bolus slot (cannot coincide with any late-bolus slot)
    if missed_sampled:
        candidates = [s for s in slots if s not in assignment.late_bolus_slots]
        if candidates:
            assignment.missed_bolus_slot = int(rng.choice(candidates))

    return assignment


def _sample_exercise_for_day(
    rng: np.random.Generator,
    profile: PatientProfile,
) -> tuple[Optional[str], bool]:
    """Sample exercise type and anomaly flag for one day.

    Returns (exercise_type, is_anomaly_overlay) or (None, False).

    All tendencies go through the same probabilistic path — there is no
    guaranteed daily exercise even for 'active' patients (replaces SC2 hardcode).

    Anomaly flag rules:
      - 'prolonged' and 'anaerobic' are always anomalies (out-of-routine sessions)
      - 'aerobic' is normal for active/normal patients; anomaly for sedentary
        (unexpected structured exercise for an otherwise sedentary patient)
    """
    # First: does exercise happen today at all?
    if float(rng.random()) >= profile.exercise_daily_prob:
        return None, False

    # Sample exercise type from tendency-specific distribution
    type_probs = EXERCISE_TYPE_PROBS[profile.exercise_tendency]
    ex_types   = list(type_probs.keys())
    ex_weights = [type_probs[t] for t in ex_types]
    ex_type    = str(rng.choice(ex_types, p=ex_weights))

    if ex_type in ('prolonged', 'anaerobic'):
        is_anomaly = True
    elif profile.exercise_tendency == 'sedentary':
        is_anomaly = True   # unexpected structured exercise for sedentary patient
    else:
        is_anomaly = False

    return ex_type, is_anomaly


def _build_meal_events(
    rng: np.random.Generator,
    active_slots: frozenset[int],
    meal_count: int,
    anomalies: _MealAnomalyAssignment,
    patient_bolus_bias: float,
) -> list[MealEvent]:
    """Build MealEvent list with times, carbs, estimation noise, and anomaly flags.

    Each slot always consumes exactly 10 RNG draws regardless of anomaly type,
    ensuring a consistent downstream RNG state.
    """
    events: list[MealEvent] = []
    if meal_count == 3:
        meal_times = _MEAL_TIMES_3MEALS
    elif meal_count == 4:
        meal_times = _MEAL_TIMES_4MEALS
    else:
        meal_times = _MEAL_TIMES_5MEALS

    for slot in sorted(active_slots):
        # ── Fixed-draw block (10 draws per slot) ────────────────────────────
        # Jitter range depends on whether this is a main meal or snack slot.
        jitter_max     = _MEAL_TIME_MAIN_JITTER_MAX if slot in _MAIN_SLOTS else _MEAL_TIME_SNACKS_JITTER_MAX
        time_jitter    = int(rng.integers(-jitter_max, jitter_max + 1))
        dur_lo, dur_hi = _MEAL_DURATION_RANGE[slot]
        duration_raw   = int(rng.integers(dur_lo, dur_hi + 1))
        carb_jitter    = 1.0 + float(rng.uniform(-_SMALL_CARB_JITTER_PCT, _SMALL_CARB_JITTER_PCT))
        large_meal_factor  = float(rng.uniform(_LARGE_MEAL_CARB_FACTOR_MIN, _LARGE_MEAL_CARB_FACTOR_MAX))
        est_noise      = float(rng.uniform(_MEAL_EST_NOISE_MIN, _MEAL_EST_NOISE_MAX))
        snack_noise    = float(rng.uniform(_SNACK_EST_MIN, _SNACK_EST_MAX))
        main_underest  = float(rng.uniform(_LUNCH_DINNER_UNDEREST_MIN, _LUNCH_DINNER_UNDEREST_MAX))
        large_meal_underest = float(rng.uniform(_LARGE_MEAL_UNDEREST_MIN, _LARGE_MEAL_UNDEREST_MAX))
        bolus_lead     = _sample_bolus_lead(rng)   # 2 draws
        late_delay_raw = int(rng.integers(_LATE_BOLUS_DELAY_MIN, _LATE_BOLUS_DELAY_MAX + 1))
        # ────────────────────────────────────────────────────────────────────

        # Actual carbs with jitter
        base_carbs   = _CARBS_BY_COUNT[meal_count][slot]
        actual_carbs = max(1, int(round(base_carbs * carb_jitter)))

        # Large-meal event (restaurant/party): inflates carbs and eating duration.
        meal_size = 'normal'
        duration  = duration_raw
        if slot == anomalies.large_meal_slot:
            actual_carbs = int(round(actual_carbs * large_meal_factor))
            duration     = duration_raw * 2
            meal_size    = 'large'

        meal_time = meal_times[slot] + time_jitter

        # Bolus status
        if slot == anomalies.missed_bolus_slot:
            bolus_status = 'missed'
        elif slot in anomalies.late_bolus_slots:
            bolus_status = 'late'
        else:
            bolus_status = 'normal'

        # Bolus carb estimation (0 if missed)
        if bolus_status == 'missed':
            bolus_carbs = 0.0
        elif slot in _SNACK_SLOTS:
            bolus_carbs = float(actual_carbs) * patient_bolus_bias * snack_noise
        elif slot == anomalies.large_meal_slot:
            # Large-meal event: patient systematically underestimates the portion size
            bolus_carbs = float(actual_carbs) * patient_bolus_bias * est_noise * large_meal_underest
        else:
            # Main meal: moderate underestimation typical of T1D adults
            bolus_carbs = float(actual_carbs) * patient_bolus_bias * est_noise * main_underest

        # Late bolus delay (0 for non-late meals)
        late_delay = late_delay_raw if bolus_status == 'late' else 0

        events.append(MealEvent(
            slot=slot,
            time_min=meal_time,
            duration=duration,
            carbs=actual_carbs,
            bolus_carbs=bolus_carbs,
            bolus_status=bolus_status,
            meal_size=meal_size,
            bolus_lead_min=bolus_lead,
            late_bolus_delay_min=late_delay,
        ))

    return events


def _build_exercise_event(
    rng: np.random.Generator,
    profile: PatientProfile,
    meals: list[MealEvent],
    ex_type: Optional[str],
    is_anomaly: bool,
) -> Optional[ExerciseEvent]:
    """Build an exercise session respecting meal-gap and dinner-cutoff constraints.

    ex_type / is_anomaly come from _sample_exercise_for_day; None → no session.

    Placement: free windows in [EARLIEST_EXERCISE_TIME, dinner−20] avoiding
    meal ±20 min buffers.  Falls back to minimum duration; returns None if no
    room is available (rare, occurs <1% of days).
    """
    if ex_type is None:
        return None

    dur_lo, dur_hi = _EXERCISE_DURATION[ex_type]
    duration = int(rng.integers(dur_lo, dur_hi + 1))

    # Apply patient-specific intensity bias to AC ranges
    bias = profile.exercise_intensity_bias
    if ex_type == 'aerobic':
        ac_counts = float(rng.uniform(_AC_AEROBIC_MIN * bias, _AC_AEROBIC_MAX * bias))
        burst_period, burst_on, burst_mult = 1, 1, 1.0
    elif ex_type == 'prolonged':
        ac_counts = float(rng.uniform(_AC_PROLONGED_MIN * bias, _AC_PROLONGED_MAX * bias))
        burst_period, burst_on, burst_mult = 1, 1, 1.0
    else:  # anaerobic: periodic high-intensity intervals, 2 min on / 3 min rest
        ac_counts = float(rng.uniform(_AC_ANAEROBIC_MIN * bias, _AC_ANAEROBIC_MAX * bias))
        burst_period, burst_on, burst_mult = 5, 2, 1.5

    # Dinner time establishes the exercise cutoff (no exercise in the pre-dinner window)
    dinner_mins = [m.time_min for m in meals if m.slot == 5]
    dinner_time = dinner_mins[0] if dinner_mins else time_to_minutes(21, 0)

    day_start = EARLIEST_EXERCISE_TIME
    day_end   = dinner_time - MIN_GAP_BEFORE_MEAL_MIN

    # Forbidden intervals: [meal_start − gap, meal_end + gap] for every meal
    forbidden = [
        (m.time_min - MIN_GAP_BEFORE_MEAL_MIN,
         m.time_min + m.duration + MIN_GAP_AFTER_MEAL_MIN)
        for m in meals
    ]

    # Find valid placement windows; fall back to minimum duration if needed
    free = _compute_free_windows(day_start, day_end, forbidden, duration)
    if not free and duration > dur_lo:
        duration = dur_lo
        free = _compute_free_windows(day_start, day_end, forbidden, duration)
    if not free:
        return None   # genuinely no room — skip exercise today

    # Sample start time weighted by window length (longer windows more likely)
    lengths = np.maximum(np.array([b - a - duration + 1 for a, b in free], dtype=float), 1.0)
    probs   = lengths / lengths.sum()
    chosen  = int(rng.choice(len(free), p=probs))
    a, b    = free[chosen]
    start   = int(rng.integers(a, max(a + 1, b - duration + 1)))

    return ExerciseEvent(
        start_min=start,
        duration_min=duration,
        exercise_type=ex_type,
        is_anomaly_overlay=is_anomaly,
        ac_counts=ac_counts,
        burst_period_min=burst_period,
        burst_on_min=burst_on,
        burst_multiplier=burst_mult,
    )


def _generate_day_plan(
    patient_id: int,
    day: int,
    seed: Optional[int],
    profile: PatientProfile,
) -> DayPlan:
    """Generate the complete daily schedule for one patient-day.

    Two separate RNG streams maintain reproducibility:
      - meal_rng (_DAY_PLAN_STREAM_OFFSET): meal count, slot selection, anomaly
        sampling, exercise type draw, and meal event building.
      - ex_rng (_EXERCISE_STREAM_OFFSET): exercise placement and AC sampling.
    """
    meal_rng = _seeded_rng(seed, patient_id, _DAY_PLAN_STREAM_OFFSET, day)
    ex_rng   = _seeded_rng(seed, patient_id, _EXERCISE_STREAM_OFFSET, day)

    # 1. Today's meal count (±1 deviation from baseline)
    meal_count = _sample_daily_meal_count(meal_rng, profile)

    # 2. Active meal slots for today
    active_slots = _sample_active_slots(meal_rng, profile, meal_count)

    # 3. Meal anomaly overlays (restaurant, missed bolus, late bolus)
    anomalies = _sample_meal_anomalies(meal_rng, active_slots)

    # 4. Unified exercise sampling — all tendencies now go through the same
    #    probabilistic path; no more hard SC2 guarantee of daily exercise.
    ex_type, is_anomaly = _sample_exercise_for_day(meal_rng, profile)

    # 5. Build meal events with carbs, timing, bolus estimation, and anomaly flags
    meals = _build_meal_events(meal_rng, active_slots, meal_count, anomalies, profile.bolus_bias)

    # 6. Build exercise event (uses dedicated ex_rng to decouple from meal draws)
    exercise = _build_exercise_event(ex_rng, profile, meals, ex_type, is_anomaly)

    # Map ex_type back to the legacy overlay integer for export compatibility
    exercise_overlay: Optional[int] = None
    if is_anomaly and ex_type == 'prolonged':
        exercise_overlay = 7
    elif is_anomaly and ex_type == 'anaerobic':
        exercise_overlay = 8

    return DayPlan(
        base_scenario=profile.base_scenario,
        meal_count=meal_count,
        meals=meals,
        exercise=exercise,
        had_large_meal=anomalies.large_meal_slot is not None,
        had_missed_bolus=anomalies.missed_bolus_slot is not None,
        n_late_boluses=len(anomalies.late_bolus_slots),
        exercise_overlay=exercise_overlay,
    )


# ============================================================================
# Per-Minute Computation from DayPlan
# ============================================================================

def _meal_inputs_at_minute(
    meal: MealEvent,
    current_time: int,
    insulin_carbo_ratio: float,
) -> tuple[float, float]:
    """Return (u_increment [mU/min], d_increment [mg/min]) for this meal at current_time."""
    u_inc = 0.0
    d_inc = 0.0

    # Carbohydrate delivery during the eating window
    if meal.time_min <= current_time < meal.time_min + meal.duration:
        d_inc = float(meal.carbs) * 1000.0 / float(meal.duration)  # mg/min

    if meal.bolus_status == 'missed':
        return u_inc, d_inc  # no insulin for this meal

    # Bolus: spread over BOLUS_DURATION minutes to avoid sharp S1 depot spike
    bolus_units = meal.bolus_carbs / insulin_carbo_ratio  # [U]
    bolus_rate  = bolus_units * 1000.0 / BOLUS_DURATION    # mU/min

    if meal.bolus_status == 'late':
        bolus_start = meal.time_min + meal.late_bolus_delay_min
    else:
        # Normal delivery: bolus_lead_min positive = pre-meal, negative = post-meal
        bolus_start = meal.time_min - meal.bolus_lead_min

    if bolus_start <= current_time < bolus_start + BOLUS_DURATION:
        u_inc = bolus_rate

    return u_inc, d_inc


def _exercise_ac_at_minute(exercise: ExerciseEvent, current_time: int) -> float:
    """Return accelerometer counts for the exercise session at current_time."""
    if not (exercise.start_min <= current_time < exercise.start_min + exercise.duration_min):
        return 0.0
    ac      = exercise.ac_counts
    local_t = current_time - exercise.start_min
    period  = max(1, exercise.burst_period_min)
    on      = max(1, min(period, exercise.burst_on_min))
    if (local_t % period) < on:
        ac *= exercise.burst_multiplier
    return max(0.0, ac)


def _baseline_ac_at_minute(current_time: int, base_scenario: int) -> float:
    """Return incidental movement AC for non-exercise minutes.

    Sedentary patients (sc3) have substantially lower baseline.
    Meal-anomaly overlay days follow their base scenario AC (sc1 or sc2).
    """
    minute = max(0, min(1439, current_time))

    if minute < time_to_minutes(6, 0) or minute >= time_to_minutes(22, 30):
        return 0.0   # sleep

    if base_scenario == 3:
        if time_to_minutes(6, 30) <= minute < time_to_minutes(8, 30):
            return 150.0
        if time_to_minutes(12, 0) <= minute < time_to_minutes(13, 30):
            return 200.0
        if time_to_minutes(18, 30) <= minute < time_to_minutes(20, 30):
            return 150.0
        return _AC_INCIDENTAL_SEDENTARY

    # sc1 and sc2: same incidental baseline; exercise session adds on top for sc2
    if time_to_minutes(6, 30) <= minute < time_to_minutes(8, 30):
        return 400.0   # morning routine + commute
    if time_to_minutes(12, 0) <= minute < time_to_minutes(13, 30):
        return 500.0   # lunch walk
    if time_to_minutes(18, 30) <= minute < time_to_minutes(20, 30):
        return 450.0   # dinner + post-dinner walk
    if time_to_minutes(20, 30) <= minute < time_to_minutes(22, 30):
        return 200.0   # light evening activity
    return _AC_INCIDENTAL_NORMAL


def _day_plan_inputs_at_minute(
    t: int,
    day_plan: DayPlan,
    basal_hourly: float,
    insulin_carbo_ratio: float,
) -> tuple[float, float, float]:
    """Compute (u [mU/min], d [mg/min], activity [AC]) at minute t from a DayPlan."""
    u = basal_hourly * 1000.0 / 60.0   # basal delivery rate
    d = 0.0

    for meal in day_plan.meals:
        mu, md = _meal_inputs_at_minute(meal, t, insulin_carbo_ratio)
        u += mu
        d += md

    baseline_ac = _baseline_ac_at_minute(t, day_plan.base_scenario)
    session_ac  = 0.0
    if day_plan.exercise is not None:
        session_ac = _exercise_ac_at_minute(day_plan.exercise, t)

    return u, d, baseline_ac + session_ac


# ============================================================================
# ML Label Computation
# ============================================================================

def compute_day_labels(
    day_plan: DayPlan,
    n_minutes: int = 1441,
) -> tuple[list[Optional[str]], list[Optional[str]], list[str]]:
    """Compute per-minute anomaly labels for ML ground truth.

    Returns three lists of length n_minutes:
      bolus_status_arr:  'normal' | 'missed' | 'late' | None
      meal_size_arr:     'normal' | 'large'            | None
      exercise_type_arr: 'aerobic' | 'anaerobic' | 'prolonged' | 'none'

    Windowed labeling: the label window starts LABEL_WINDOW_BOLUS_START minutes
    after the meal to account for gastric emptying + CGM interstitial lag.
    Window width depends on anomaly type — see LABEL_WINDOW_* constants.

    Precedence when windows overlap:
      bolus_status:  missed > late > normal
      meal_size:     large > normal
      exercise_type: prolonged > anaerobic > aerobic > 'none'
    """
    bolus_status_arr:  list[Optional[str]] = [None]   * n_minutes
    meal_size_arr:     list[Optional[str]] = [None]   * n_minutes
    exercise_type_arr: list[str]            = ['none'] * n_minutes

    _BOLUS_PRIORITY  = {'missed': 3, 'late': 2, 'normal': 1}
    _MEAL_PRIORITY   = {'large': 2, 'normal': 1}
    _EX_PRIORITY     = {'prolonged': 3, 'anaerobic': 2, 'aerobic': 1, 'none': 0}

    for meal in day_plan.meals:
        t_window_start = meal.time_min + LABEL_WINDOW_BOLUS_START

        if meal.bolus_status == 'missed':
            bs_end = meal.time_min + LABEL_WINDOW_MISSED_END
        elif meal.bolus_status == 'late':
            bs_end = meal.time_min + LABEL_WINDOW_LATE_END
        else:
            bs_end = meal.time_min + LABEL_WINDOW_NORMAL_END

        ms_end = (meal.time_min + LABEL_WINDOW_LARGE_END
                  if meal.meal_size == 'large'
                  else meal.time_min + LABEL_WINDOW_NORMAL_END)

        new_bs_pri = _BOLUS_PRIORITY[meal.bolus_status]
        new_ms_pri = _MEAL_PRIORITY[meal.meal_size]

        for t in range(max(0, t_window_start), min(n_minutes, bs_end)):
            curr = bolus_status_arr[t]
            if curr is None or _BOLUS_PRIORITY.get(curr, 0) < new_bs_pri:
                bolus_status_arr[t] = meal.bolus_status

        for t in range(max(0, t_window_start), min(n_minutes, ms_end)):
            curr = meal_size_arr[t]
            if curr is None or _MEAL_PRIORITY.get(curr, 0) < new_ms_pri:
                meal_size_arr[t] = meal.meal_size

    if day_plan.exercise is not None:
        ex = day_plan.exercise
        if ex.exercise_type == 'prolonged':
            post = LABEL_WINDOW_PROLONGED_POST
        elif ex.exercise_type == 'anaerobic':
            post = LABEL_WINDOW_ANAEROBIC_POST
        else:
            post = LABEL_WINDOW_AEROBIC_POST

        ex_end = ex.start_min + ex.duration_min + post
        ex_pri = _EX_PRIORITY[ex.exercise_type]

        for t in range(max(0, ex.start_min), min(n_minutes, ex_end)):
            if _EX_PRIORITY.get(exercise_type_arr[t], 0) < ex_pri:
                exercise_type_arr[t] = ex.exercise_type

    return bolus_status_arr, meal_size_arr, exercise_type_arr


# ============================================================================
# Caching Infrastructure
# ============================================================================
# WARNING: process-global caches.  Safe for mp.Pool (each worker gets its own
# memory copy).  NOT safe for ThreadPoolExecutor.  Call clear_meal_cache() at
# the top of every run_simulation call.

_patient_profile_cache: dict[int, PatientProfile] = {}
_day_plan_cache: dict[tuple[int, int], DayPlan]   = {}


def _get_or_create_profile(
    patient_id: int,
    seed: Optional[int],
    base_scenario_override: Optional[int] = None,
) -> PatientProfile:
    """Return cached patient profile, generating it on first access."""
    if patient_id not in _patient_profile_cache:
        _patient_profile_cache[patient_id] = _generate_patient_profile(
            patient_id, seed, base_scenario_override
        )
    return _patient_profile_cache[patient_id]


def _get_or_create_day_plan(
    patient_id: int,
    day: int,
    seed: Optional[int],
    base_scenario_override: Optional[int] = None,
) -> DayPlan:
    """Return cached day plan, generating it on first access."""
    key = (patient_id, day)
    if key not in _day_plan_cache:
        profile = _get_or_create_profile(patient_id, seed, base_scenario_override)
        _day_plan_cache[key] = _generate_day_plan(patient_id, day, seed, profile)
    return _day_plan_cache[key]


# ============================================================================
# Public API
# ============================================================================

def get_cached_day_plan(patient_id: int, day: int) -> Optional[DayPlan]:
    """Return the cached DayPlan for (patient_id, day), or None if not yet generated.

    Will be populated after scenario_with_cached_meals is first called for minute 0
    of that day during ODE integration.
    """
    return _day_plan_cache.get((patient_id, day))


def scenario_with_cached_meals(
    time: int,
    patient_id: int,
    day: int,
    basal_hourly: float = 0.5,
    scenario: Optional[int] = None,
    insulin_carbo_ratio: float = 2.0,
    seed: Optional[int] = None,
    meal_schedule: Optional[object] = None,  # ignored; kept for model.py fallback-path compat
) -> tuple[float, float, float]:
    """Per-minute (u [mU/min], d [mg/min], activity [AC]) from the cached DayPlan.

    scenario: if 1–3, used as the base-scenario override for this patient
        (i.e. config.random_scenarios=False with fixed_scenario=N).  Values > 3
        or None mean "use the patient's randomly drawn base scenario."
    """
    del meal_schedule  # unused; model.py passes this for its legacy fallback path
    base_sc_override = max(1, min(3, int(scenario))) if scenario is not None else None
    day_plan = _get_or_create_day_plan(patient_id, day, seed, base_sc_override)
    return _day_plan_inputs_at_minute(time, day_plan, basal_hourly, insulin_carbo_ratio)


def clear_meal_cache() -> None:
    """Clear all per-run caches.  Must be called at the start of each run_simulation."""
    global _patient_profile_cache, _day_plan_cache
    _patient_profile_cache = {}
    _day_plan_cache = {}
