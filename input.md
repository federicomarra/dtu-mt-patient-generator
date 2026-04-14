# Input Logic: Daily Routine Generation

This document explains how `src/input.py` constructs the daily insulin, carbohydrate, and
activity inputs fed into the Hovorka glucose ODE model. It covers the base scenario
architecture, overlay system, meal structure, bolus estimation, exercise scheduling, and
reproducibility. The goal is to give a clear picture of *what a simulated patient's day
looks like* and *why it was designed that way*.

---

## 1. The Big Picture

Each simulated day produces a per-minute time series of three inputs:

| Signal | Units | Description |
| -------- | ------- | ------------- |
| `u` | mU/min | Insulin delivery (basal + meal boluses) |
| `d` | mg/min | Carbohydrate absorption entering the gut |
| `activity` | accelerometer counts (AC) | Physical activity level from wearable |

All three are computed minute-by-minute by `scenario_inputs()`, which dispatches to a
pre-generated `DayPlan` for that patient × day combination. Plans are generated once and
cached so consecutive calls for the same minute are deterministic and fast.

---

## 2. Architecture: Base Scenarios + Overlays

The simulation does **not** use a fixed library of mutually exclusive day types.
Instead, each patient-day is constructed from two independent layers:

1. **Base scenario** — sampled once per patient, determines activity type across all days.
2. **Overlays** — sampled independently each day via Bernoulli draws; any combination can co-occur.

### 2.1 Base Scenarios

The base scenario is a per-patient constant, determining the activity profile of every day:

| ID | Name | Core Feature |
| ---- | ------ | ------------- |
| SC1 | Normal activity | Mixed-activity patient; moderate daily-exercise probability |
| SC2 | Active aerobic | High daily-exercise probability; predominantly aerobic |
| SC3 | Sedentary | Low daily-exercise probability; minimal incidental movement |

**Sampling weights** (drawn once per patient at cohort generation):

| Base scenario | Weight | Effective % |
| --------------- | -------- | ------------- |
| SC1 normal | 0.38 | 38% |
| SC2 active aerobic | 0.27 | 27% |
| SC3 sedentary | 0.35 | 35% |

The base scenario maps to a continuous `exercise_daily_prob` range, sampled once per patient:

| Tendency | `exercise_daily_prob` range |
| --------- | --------------------------- |
| SC3 sedentary | 0.03–0.10 |
| SC1 normal | 0.20–0.45 |
| SC2 active | 0.55–0.80 |

Each day, whether exercise occurs is a Bernoulli draw using the patient's `exercise_daily_prob`. SC2 patients exercise on most days but not necessarily every day. SC3 patients rarely exercise. Exercise type is drawn from per-tendency type probabilities (`EXERCISE_TYPE_PROBS`); prolonged aerobic and anaerobic sessions can occur for any tendency, not just SC1.

### 2.2 Per-Day Overlays

Each overlay is sampled independently every day by a Bernoulli draw. Multiple overlays
can co-occur on the same day (e.g., large meal + missed bolus, or prolonged aerobic +
late bolus).

| Overlay | Probability | Behaviour |
| --------- | ------------ | ----------- |
| Large meal | 9% | One main meal (lunch or dinner) scaled ×1.8–2.1, strong underestimation |
| Missed bolus | 9% | One meal receives no insulin |
| Late bolus (1st event) | 6% | 1 meal bolused 30–90 min after meal start |
| Late bolus (2nd event) | 15% conditional | A second meal is also late (given 1st was drawn) |

Exercise sessions (aerobic, prolonged aerobic, anaerobic) are not Bernoulli-overlay driven.
They are scheduled from a per-tendency type distribution (`EXERCISE_TYPE_PROBS`) on days where
the patient's daily exercise draw fires. Exercise and meal-perturbation overlays are independent
and can co-occur. At most one exercise session per day (no stacking).

**Exercise type probabilities by tendency:**

| Tendency | aerobic | prolonged | anaerobic |
| --------- | ------- | --------- | --------- |
| SC3 sedentary | 80% | 10% | 10% |
| SC1 normal | 70% | 15% | 15% |
| SC2 active | 65% | 25% | 10% |

**Missed bolus and late bolus constraints:**

- A single meal cannot be both missed and late — the missed bolus slot is always drawn from the set of slots not already assigned as late.
- If 2 late boluses fire on the same day, the missed bolus overlay is suppressed entirely for that day (a day with 2 late meals + 1 missed is considered too anomalous to be realistic).
- 1 late bolus + 1 missed bolus on *different* meals is allowed (they are independent Bernoulli draws).

**Effective marginal rates** (cohort-level, averaged across patients):

- Large meal: ~9% of all days
- Missed bolus: ~9% of all days
- Late bolus: ~6% + tail from 2nd event ≈ ~6.9% of all days
- Exercise (any type): depends on patient tendency and `exercise_daily_prob`; roughly
  0.38 × 0.33 + 0.27 × 0.68 + 0.35 × 0.07 ≈ ~35% of all days across a random-scenario cohort
- Of exercise days: ~67–70% aerobic, ~12–25% prolonged, ~10–15% anaerobic (tendency-weighted)

---

## 3. Meal Structure

### 3.1 Meal Count

Each patient has a fixed **baseline meal count** of 3, 4, or 5 meals, sampled at cohort
creation from equal thirds:

| Meal count | Probability | Snack structure |
| ------------ | ------------ | ----------------- |
| 3 meals | 1/3 | Breakfast, lunch, dinner only |
| 4 meals | 1/3 | Three mains + one snack (morning or afternoon, fixed per patient) |
| 5 meals | 1/3 | Three mains + morning snack + afternoon snack |

On 20% of days, the meal count deviates by ±1 from the patient's baseline (minimum 3,
maximum 5), reflecting natural week-to-week variation.

### 3.2 Meal Times and Carbs

Anchor times differ by meal count to avoid overlap between snacks and potential exercise windows:

**3-meal patients:**

| Meal | Anchor time |
| ------ | ------------- |
| Breakfast | 08:00 |
| Lunch | 13:00 |
| Dinner | 20:00 |

**4 and 5-meal patients:**

| Meal | Anchor time |
| ------ | ------------- |
| Breakfast | 07:15 |
| Morning snack (slots 2) | 10:30 |
| Lunch | 13:00 |
| Afternoon snack (slot 4) | 16:00 |
| Dinner | 21:00 |

**Carbohydrate targets by meal count:**

| Meal count | Breakfast | Snack | Lunch | Snack | Dinner |
| ------------ | ----------- | ------- | ------- | ------- | -------- |
| 3 meals | 75 g | — | 90 g | — | 90 g |
| 4 meals | 70 g | 20 g | 80 g | 20 g | 80 g |
| 5 meals | 65 g | 20 g | 75 g | 20 g | 75 g |

A ±10% day-to-day carb jitter is applied to each meal independently.

### 3.3 Timing Jitter

Each day a random timing shift is applied to each meal independently:

| Meal type | Jitter range |
| ----------- | ------------- |
| Main meals (breakfast, lunch, dinner) | ±45 min |
| Snacks | ±30 min |

This reflects realistic schedule variation across days without changing the meal count or
bolus allocation.

---

## 4. Bolus Estimation Model

The most important realism feature: patients estimate meal carbs and calculate their
bolus from that estimate, not the true absorbed carbs.

### 4.1 Two-Layer Estimation Error

**Layer 1 — Patient-level persistent bias** (computed once per patient):

| Parameter | Range | Meaning |
| ----------- | ------- | --------- |
| `bolus_bias` | 0.78–0.95 | Multiplier on all estimated carbs |

The range is intentionally skewed low (systematically below 1.0): real patients tend to
undercount carbs more than overcount. A value of 0.85 means this patient underestimates
every meal by 15%.

**Layer 2 — Meal-level random noise** (sampled independently per meal per day):

| Meal type | Noise range | Systematic underestimation |
| ----------- | ------------ | --------------------------- |
| Breakfast | ×0.80–1.20 | — |
| Morning snack | ×0.96–1.04 | — |
| Lunch (normal day) | ×0.80–1.20 | ×0.82–0.93 |
| Afternoon snack | ×0.96–1.04 | — |
| Dinner (normal day) | ×0.80–1.20 | ×0.82–0.93 |
| Lunch/dinner (large-meal overlay) | ×0.80–1.20 | ×0.70–0.85 (stronger) |

Snacks have tight noise because small portions are easier to judge. Main meals have an
additional systematic underestimation factor (grounded in dietary assessment literature:
larger plates are harder to gauge).

**Combined example:** Patient with `bolus_bias = 0.85` having a normal lunch of 75 g:
`75 × 0.85 × ~1.00 (noise) × 0.87 (underest) ≈ 55 g` → bolus covers ~55 g while gut absorbs 75 g.

### 4.2 Bolus Lead-Time Distribution

Normal boluses are not always delivered exactly 15 min pre-meal. The lead time is drawn
from a three-way distribution matching empirical T1D behaviour:

| Category | Probability | Lead time range | Interpretation |
| ---------- | ------------ | ----------------- | ---------------- |
| Pre-meal | 60% | 5–20 min before | Planned ahead |
| At onset | 25% | 0–4 min before | Remembered at table |
| Post-meal | 15% | 1–20 min after | Forgot, corrected |

Each meal on each day independently draws from this distribution.

---

## 5. Overlay Details

### 5.1 Large-Meal Overlay

On days with a large-meal event (restaurant, party, event dining), one main meal (lunch
or dinner, uniform 50/50) is replaced:

```text
carb_factor ~ Uniform([1.8, 2.1])
underest_factor ~ Uniform([0.70, 0.85])
actual_carbs = base_carbs × carb_factor        → gut absorption
bolus_carbs  = actual_carbs × bolus_bias × meal_noise × underest_factor
duration     = original_duration × 2           → slower restaurant eating
```

The combination of 1.8–2.1× carb load, strong underestimation, and doubled eating
duration produces a sustained hyperglycaemic excursion (peak ~2–3 h post-meal).

### 5.2 Missed Bolus Overlay

One meal is chosen uniformly from the patient's active meals that day. That meal
receives zero insulin; all other meals are bolused normally.

This is the strongest hyperglycaemia-producing overlay: a clean 2–4 h excursion
reaching 12–16 mmol/L with no attenuation.

### 5.3 Late Bolus Overlay

One meal (up to two if the second Bernoulli fires) has its bolus delivered after the
meal starts:

```text
delay ~ Uniform([30, 90] min)   # sampled independently per affected meal
```

At 30–90 min post-meal delay, insulin peaks ~90–150 min post-meal while glucose already
peaked ~45–60 min post-meal — a distinguishable spike-then-correction CGM signature.

---

## 6. Exercise Scheduling

### 6.1 Which Days Have Exercise

Each day, whether a patient exercises is determined by a Bernoulli draw against their
`exercise_daily_prob` (sampled once per patient from their tendency range). When exercise
fires, the type is drawn from the per-tendency `EXERCISE_TYPE_PROBS` distribution:

| Patient tendency | `exercise_daily_prob` | Exercise type draw |
| ---------------- | --------------------- | -------------------|
| SC3 sedentary | 0.03–0.10 | 80% aerobic / 10% prolonged / 10% anaerobic |
| SC1 normal | 0.20–0.45 | 70% aerobic / 15% prolonged / 15% anaerobic |
| SC2 active | 0.55–0.80 | 65% aerobic / 25% prolonged / 10% anaerobic |

All exercise sessions are scheduled with `is_anomaly_overlay=False` for aerobic (SC2-like
background activity) and `is_anomaly_overlay=True` for prolonged and anaerobic.

### 6.2 Exercise Parameters

| Type | Duration | AC range |
| ------ | ---------- | ---------- |
| Aerobic | 30–60 min | 1200–2000 |
| Prolonged aerobic | 75–120 min | 1500–2500 |
| Anaerobic (EXPERIMENTAL) | 20–50 min | 6000–9000 |

Reference thresholds: `aAC=1000` → `fAC=0.5` (aerobic onset); `ah=5600` → `fHI=0.5`
(anaerobic onset). The anaerobic overlay's AC range (6000–9000) keeps it firmly in the
anaerobic regime. The prolonged-aerobic minimum (75 min) is strictly above the SC2 maximum
(60 min), ensuring clean ML separation by duration.

### 6.3 Session Timing

Exercise sessions are scheduled on any day starting no earlier than **08:00**. The
scheduler enforces:

- At least 20 min after the preceding meal ends
- At least 20 min before the next meal starts

If no valid window exists in the day, the exercise session is dropped (the fallback
prevents infinite retries and is documented in simulation logs).

### 6.4 Anaerobic Burst Pattern

Resistance and HIIT training uses a periodic burst pattern:

```text
burst_period = 5 min
burst_on     = 2 min    (active interval)
burst_multiplier = 1.5 × session_ac   during burst_on minutes
```

This alternating 2-min on / 3-min rest cycle matches a typical resistance training
interval structure and distinguishes the anaerobic AC signal from steady aerobic.

### 6.5 Incidental Movement (Baseline AC)

All patients have background AC from incidental daily movement, independent of planned
sessions:

| Patient type | Incidental AC baseline |
| ------------- | ---------------------- |
| SC1 / SC2 | 300 counts |
| SC3 (sedentary) | 100 counts |

On days with a meal-perturbation overlay only (missed bolus, late bolus) and no exercise,
this baseline is present but low, keeping the glucose signal uncontaminated by
exercise-related insulin sensitisation.

---

## 7. Determinism and Reproducibility

The entire generation pipeline is fully deterministic given a global seed, patient ID,
and day index. The seed composition:

```text
stream_seed = global_seed + patient_id × 10_000 + stream_offset
daily_seed  = stream_seed + day × 97
```

The `× 97` on day prevents accidental seed aliasing across days. Three independent
RNG streams are used:

| Stream | Offset | Content |
| -------- | -------- | --------- |
| Patient profile | +0 | Base scenario, meal count, bolus bias |
| Day plan | +13 | Overlays, meal anomalies, timing, carbs |
| Exercise placement | +17 | Session duration, start time, AC counts |

Stream isolation ensures, for example, that a patient whose large-meal overlay fires
on day 7 does not also have systematically different timing on that day due to shared
RNG state.

---

## 8. Exported Metadata Columns

Each day's `DayPlan` exports ground-truth labels alongside the glucose/insulin signals:

| Column | Type | Values |
| -------- | ------ | -------- |
| `base_scenario` | int | 1, 2, 3 |
| `had_large_meal` | bool | True/False |
| `had_missed_bolus` | bool | True/False |
| `n_late_boluses` | int | 0, 1, 2 |
| `exercise_overlay` | int or None | 7 (prolonged), 8 (anaerobic), None |
| `bolus_status` | str per minute | 'normal', 'missed', 'late', 'none' |
| `meal_size` | str per minute | 'normal', 'large', 'none' |
| `exercise_type` | str per minute | 'aerobic', 'anaerobic', 'prolonged', 'none' |

---

## 9. End-to-End Flow Summary

For each simulated day for a given patient:

```text
1. Load PatientProfile (cached per patient):
   a. exercise_tendency ~ Categorical([0.38, 0.27, 0.35]) → normal/active/sedentary
      base_scenario exported as 1/2/3 for backward compat
   b. baseline_meal_count ~ Categorical([1/3, 1/3, 1/3]) → 3/4/5 meals
   c. bolus_bias ~ Uniform([0.78, 0.95])
   d. exercise_daily_prob ~ Uniform(tendency range)  # P(exercise today)
   e. exercise_intensity_bias ~ Uniform(tendency range)  # AC scale factor

2. Generate DayPlan for this day:
   a. Sample meal count (20% chance ±1 deviation from baseline)
   b. Sample meal overlays independently (Bernoulli):
      - large_meal (P=0.09), missed_bolus (P=0.09)
      - late_bolus_1 (P=0.06), late_bolus_2 conditional (P=0.15)
   c. Sample exercise: Bernoulli(exercise_daily_prob); if fires, draw type from
      EXERCISE_TYPE_PROBS[tendency] → aerobic / prolonged / anaerobic
   d. Place meals at anchor times + jitter
   e. Apply anomaly modifiers (large-meal carb factor, missed/late bolus assignment)
   f. Compute per-meal bolus_carbs = actual × bolus_bias × noise × underest
   g. Schedule exercise session in a valid time window (≥20 min from any meal)

3. At each minute t:
   a. For active meals: compute u_inc (bolus insulin) and d_inc (carb absorption)
   b. Add incidental AC + session AC if exercise is active
   c. Return (u, d, activity) to ODE integrator
```
