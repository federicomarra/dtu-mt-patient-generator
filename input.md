# Input Logic: Daily Routine Generation

This document explains how `src/input.py` constructs the daily insulin, carbohydrate, and
activity inputs fed into the Hovorka glucose ODE model. It covers the nine simulated day
types (scenarios), how meals are generated, how bolus estimation uncertainty is modelled,
and how exercise is scheduled. The goal is to give a clear picture of *what a simulated
patient's day looks like* and *why it was designed that way*.

---

## 1. The Big Picture

Each simulated day produces a per-minute time series of three inputs:

| Signal | Units | Description |
|--------|-------|-------------|
| `u` | mU/min | Insulin delivery (basal + meal boluses) |
| `d` | mg/min | Carbohydrate absorption entering the gut |
| `activity` | accelerometer counts (AC) | Physical activity level from wearable |

All three are computed minute-by-minute by `scenario_inputs()`, which dispatches to a
pre-generated `MealSchedule` and `ExerciseSchedule` for that patient × scenario × day
combination. Schedules are generated once and cached so consecutive calls for the same
minute are deterministic and fast.

---

## 2. Scenarios

There are **9 distinct day types**. Each patient has the same 9 scenarios available; the
multi-day library generator samples them daily according to the weights below.

### 2.1 Scenario Definitions

| ID | Name | Core Feature |
|----|------|-------------|
| 1 | Normal | Typical active day, no anomaly |
| 2 | Active (aerobic) | Afternoon moderate aerobic session |
| 3 | Sedentary | Desk-bound day, minimal movement |
| 4 | Restaurant meal | Large meal, extended eating, under-bolused |
| 5 | Missed bolus | One meal receives no insulin at all |
| 6 | Late bolus | 1–2 meals: bolus given 20–60 min *after* meal starts instead of 15 min before |
| 7 | Prolonged aerobic | Long exercise session (≥80 min); glycogen depletion dominates |
| 8 | Anaerobic/resistance | High-intensity intervals (EXPERIMENTAL parameters) |
| 9 | Exercise + missed bolus | Compound: moderate aerobic session *and* one missed bolus |

Scenarios 1–3 (sc1, sc2, sc3) cover normal daily variation. Scenarios 4–6 (sc4, sc5, sc6) are meal-behaviour anomalies.
Scenarios 7–9 (sc7, sc8, sc9) are exercise-driven anomalies.

### 2.2 Sampling Weights

When the library generator draws a scenario for a given patient-day it samples from this
discrete distribution (normalized to sum to 1.0):

| Scenario | Raw weight | Approx. % | Rationale |
|----------|-----------|-----------|-----------|
| 1 normal | 0.25 | 25% | Most common day type |
| 2 active | 0.18 | 18% | ~3 sessions/week typical T1D adherence |
| 3 sedentary | 0.20 | 20% | Common desk/study day |
| 4 restaurant | 0.08 | 8% | Occasional anomaly, ~1 per 2 weeks |
| 5 missed bolus | 0.08 | 8% | Discrete forgetting event |
| 6 late bolus | 0.10 | 10% | More common than full omission |
| 7 prolonged aerobic | 0.04 | 4% | Rare long session |
| 8 anaerobic | 0.04 | 4% | Rare resistance training |
| 9 exercise+missed | 0.03 | 3% | Compound anomaly |

**Design note on scenario 2 (18%):** Each aerobic exercise session accumulates a
cross-day physiological state (the Z-state, τ≈10 h) that spills into the following
day. Keeping sc2 at 18% instead of 25% limits consecutive-day coupling effects
and matches realistic gym attendance patterns.

---

## 3. Meal Structure

### 3.1 Five Meals Per Day

Every day, regardless of scenario, contains five scheduled eating events:

| Meal | Time window | Carbs range | Eating duration |
|------|------------|-------------|----------------|
| Breakfast | 06:30–08:00 | 45–70 g | 15–20 min |
| Morning snack | 10:00–11:00 | 12–20 g | 8–12 min |
| Lunch | 12:00–13:00 | 50–75 g | 20–25 min |
| Afternoon snack | 15:00–16:30 | 12–25 g | 5–10 min |
| Dinner | 18:30–20:00 | 50–75 g | 20–25 min |

All values are drawn from **uniform integer distributions** within the stated ranges.
Times are in minutes from midnight (e.g. 06:30 = minute 390).

### 3.2 Snack Skipping

Real patients don't eat every snack every day. On each day, a single independent random
draw determines whether snacks are skipped:

| Outcome | Probability |
|---------|------------|
| Both snacks skipped | 7% |
| Exactly one snack skipped (morning or afternoon, 50/50) | 20% |
| Both snacks eaten | 73% |

When a snack is skipped, its carbs are set to 0 and no bolus is given for it.

### 3.3 Periodic High-Carb Meal

Once every **7–14 days** (sampled uniformly) a randomly selected main meal (either
lunch or dinner, not both) is replaced by a larger portion to capture natural weekly
variability — e.g., a celebration dinner or a heavier Friday lunch:

| Parameter | Range |
|-----------|-------|
| Carb amount | 70–120 g |
| Recurrence gap | 7–14 days |

This is separate from the restaurant meal anomaly (sc4) and applies on any scenario day.

### 3.4 Day-to-Day Timing Jitter

On most days a small independent random timing shift is applied to each meal to reflect
realistic schedule variation:

| Day type (probability) | Time jitter | Carb jitter |
|-----------------------|-------------|-------------|
| Regular day (85%) | ±5 min | ±6% |
| Irregular day (15%) | ±20 min | ±6% |

Irregular days simulate atypical schedules (travel, late meetings, etc.). The anomaly label
(scenario ID) is **not** affected — a scenario-5 day always has a missed bolus regardless
of timing irregularity.

---

## 4. Bolus Estimation Model

This is the most important realism feature for non-exercise scenarios. In real T1D
management, patients estimate meal carbs and calculate their bolus from that estimate.
The *actual* carbs absorbed by the gut are the true meal carbs; the bolus is based on
what the patient *thinks* they ate. These two numbers are almost never the same.

### 4.1 How Bolus is Computed

```
actual_carbs → d_inc (gut absorption, mg/min)
estimated_carbs → bolus_amount = estimated_carbs / insulin_carbo_ratio
```

Estimated carbs are always separate from actual carbs. The simulation applies
the estimation factors only to the bolus calculation, not to gut absorption.

### 4.2 Two-Layer Estimation Error

**Layer 1 — Patient-level persistent bias** (computed once per patient per scenario):

Each patient has a stable over/under-estimation tendency that reflects their personal
habits, shaped at simulation start and reused across all days:

| Parameter | Range | Meaning |
|-----------|-------|---------|
| `patient_bolus_bias` | 0.90–1.05 | Multiplier on all estimated carbs |

A value of 0.92 means this patient systematically underestimates by 8% across all meals,
every day. A value of 1.03 means slight overestimation. The asymmetric range (0.90–1.05,
skewed low) reflects that patients tend to undercount more than overcount.

**Layer 2 — Meal-level random noise** (sampled independently for each meal each day):

On top of the patient bias, each individual meal estimate varies further:

| Meal type | Noise range | Extra systematic factor |
|-----------|------------|------------------------|
| Breakfast | ×0.90–1.10 | — |
| Morning snack | ×0.96–1.04 | — |
| Lunch (normal day) | ×0.90–1.10 | ×0.90–0.98 (underestimation bias) |
| Afternoon snack | ×0.96–1.04 | — |
| Dinner (normal day) | ×0.90–1.10 | ×0.90–0.98 (underestimation bias) |
| Lunch (sc4 restaurant) | ×0.90–1.10 | ×0.70–0.85 (strong underestimation) |
| Dinner (sc4 restaurant) | ×0.90–1.10 | ×0.70–0.85 (strong underestimation) |

Snacks have tighter, near-symmetric noise because small portions are easier to judge.
Main meals have an additional systematic underestimation factor (0.90–0.98) because
larger plates are harder to gauge — this is grounded in dietary assessment literature.

**Combined example:** A patient with `patient_bolus_bias = 0.93` having a normal lunch
of 65 g would have an estimated bolus based on approximately:
`65 × 0.93 × ~1.00 (noise) × 0.94 (underest) ≈ 57 g`
— resulting in a bolus covering ~57 g worth of carbs while the gut absorbs 65 g.

### 4.3 RNG Stream Isolation

To prevent the bolus-estimation noise from affecting the meal-timing draws (or vice versa),
the RNG seed used for bolus estimation is shifted by a fixed offset (`_BOLUS_ESTIMATION_STREAM_OFFSET = 53`)
relative to the base patient×scenario seed. Similarly, the periodic high-carb meal uses
offset `_HIGH_CARB_STREAM_OFFSET = 31`. This ensures each behavioral feature draws from a
reproducible but independent random stream.

---

## 5. Bolus Delivery

### 5.1 Normal (Pre-meal) Bolus

The standard bolus is delivered **15 minutes before** the meal starts, spread evenly over
3 minutes to avoid a sharp instantaneous insulin spike. This mimics the pre-announcement
behaviour of a patient who remembers to bolus before eating:

```
bolus_start = meal_time − 15 min
bolus delivered uniformly over [bolus_start, bolus_start + 3 min]
```

### 5.2 Missed Bolus (Scenario 5)

On a sc5 day, **one meal** is chosen uniformly at random from the five meals. That meal
receives no insulin whatsoever. Every other meal is bolused normally.

```
missed_meal_id ~ Uniform({1, 2, 3, 4, 5})
```

This is the strongest hyper-producing anomaly (~10% hyper time).

### 5.3 Late Bolus (Scenario 6)

On a sc6 day, **1 or 2 meals** are selected uniformly at random (without bias toward any
particular meal type):

```
k ~ Uniform({1, 2})
affected_meals ~ Choose(k meals from {1,2,3,4,5}, without replacement)
```

For each affected meal, the bolus is shifted to **after the meal starts**, with a random
delay:

```
bolus_start = meal_start + delay
delay ~ Uniform([20, 60] min)
```

The delay (20–60 min) is resampled independently for each affected meal. The remaining
meals are bolused normally (15 min pre-meal).

**Design rationale:** A 20–60 min post-meal bolus delay means insulin peaks ~80–100 min
post-meal, while glucose peaks ~45–60 min post-meal. The mismatch produces a measurable
glucose spike followed by a delayed correction — a distinguishable CGM signature.

---

## 6. Scenario 4: Restaurant Meal

This is the most structurally complex meal anomaly. It combines three simultaneous effects 
that all occur together at real restaurant dining:

1. **Larger portion** — restaurant servings are typically 30–60% bigger than home meals
2. **Slower eating** — a social meal takes twice as long
3. **Carb underestimation** — restaurant carb content is notoriously hard to judge

**Implementation:**

On each sc4 day, either lunch or dinner is randomly selected (50/50) as the restaurant meal:

```
target ~ Uniform({lunch, dinner})
carb_factor ~ Uniform([1.8, 2.1])
underest_factor ~ Uniform([0.70, 0.85])

actual_carbs = original_carbs × carb_factor        → gut absorption
bolus_carbs  = actual_carbs × patient_bolus_bias
                            × meal_noise (×0.90–1.10)
                            × underest_factor        → bolus calculation
duration = original_duration × 2
```

**Numerical example:** A dinner of 60 g becomes ~126 g actual (60 × 2.1). The patient sees
a large plate and estimates ~92 g. After the normal noise chain:
`126 × 0.95 × 0.77 ≈ 92 g estimated` → bolus covers ~92 g while gut absorbs 126 g.
Net under-bolus: ~27%. Together with the slower absorption (spread over 2× duration), this 
produces a sustained hyperglycaemic excursion starting ~30–60 min into the meal.

**Why 1.8–2.1×:** The baseline range is 50–75 g. At 1.8× minimum the result (92 g) is over the 
top of the normal range; combined with the underestimation factor, the net anomaly signal is 
consistent across all random draws.

---

## 7. Exercise Scheduling

### 7.1 Which Scenarios Have Exercise

Only scenarios 2, 7, 8, and 9 include a planned exercise session. Scenarios 1, 3, 4, 5,
and 6 have zero planned exercise (accelerometer baseline only).

### 7.2 Exercise Parameters by Scenario

| Scenario | Duration | AC counts | Type |
|----------|----------|-----------|------|
| 2 active aerobic | 30–75 min | 1200–2000 | Moderate continuous |
| 7 prolonged aerobic | 80–90 min | 1500–2500 | Long continuous |
| 8 anaerobic | 30–60 min | 6000–9000 | High-intensity intervals |
| 9 exercise+missed bolus | 30–75 min | 1000–1800 | Moderate continuous |

**Important ML design property:** Scenario 7 has a guaranteed minimum duration of 80 min,
which is strictly above scenario 2's maximum of 75 min. This means any sc7 session always
produces greater glycogen depletion than any sc2 session — the scenarios are cleanly
separable by duration alone.

Reference thresholds for the ETH exercise model: `aAC = 1000` counts → `fAC = 0.5`
(aerobic onset); `ah = 5600` counts → `fHI = 0.5` (anaerobic onset). Scenario 8's
AC range (6000–9000) keeps it firmly in the anaerobic regime.

### 7.3 Session Timing

All sessions start in the **afternoon/evening window**: between 16:30 and 20:30.

On most days (85%) the session is **temporally separated** from meals:
- At least 20 min after the afternoon snack ends (gap is waived when the afternoon snack was skipped that day)
- At least 30 min before dinner starts

On 15% of days, a **controlled overlap** is allowed — the session deliberately overlaps
with either the snack or dinner window. If the snack was skipped, only dinner-overlap is
considered. This reflects real-world behaviour and prevents the ML model from over-relying
on clean temporal separation as the only exercise signal.

### 7.4 Anaerobic Burst Pattern (Scenario 8)

Resistance and HIIT training involves periodic intensity bursts. In sc8, a burst pattern
is applied to the exercise session:

```
burst_period = 5 min
burst_on     = 2 min   (active interval)
burst_multiplier = 1.5 × session ac_counts during burst_on minutes
```

This produces an alternating pattern of high-AC bursts (2 min) and rest periods (3 min)
throughout the session, matching a typical resistance training interval structure.

### 7.5 Incidental Movement (Baseline AC)

Even on non-exercise days, patients move incidentally. A background AC baseline is
added at meal-associated times to simulate walking to/from a cafeteria, kitchen, etc.:

| Time window | Normal / active / compound scenario | Sedentary scenario |
|-------------|------------------------------------|--------------------|
| 06:30–08:30 | 400 AC | 150 AC |
| 12:00–13:30 | 500 AC | 200 AC |
| 18:30–20:00 | 450 AC | 150 AC |
| 20:00–22:30 | 200 AC | 75 AC |
| Other daytime | 300 AC | 75 AC |
| Night (00:00–06:00) | 0 | 0 |
| Scenarios 4, 5, 6 | 0 (no movement baseline) | — |

Sedentary days have approximately half the incidental movement of active days.
Meal-anomaly scenarios (4–6) have zero AC baseline to keep their glucose signal
uncontaminated by exercise-related insulin sensitisation.

---

## 8. Determinism and Reproducibility

The entire generation pipeline is fully deterministic given a patient ID, scenario, and
day index. Seeds are composed as:

```
base_seed = patient_id × 10_000_000 + scenario × 1_000_000 + day × 97
```

The `× 97` on day ensures seeds don't accidentally alias across days. Each behavioral
feature uses an offset on top of the base seed to draw from an independent random stream:

| Feature | RNG offset |
|---------|-----------|
| Meal timing and carbs | 0 (base) |
| High-carb meal recurrence | +31 |
| Bolus estimation factors | +53 |

These offsets prevent, for example, a patient whose high-carb cycle happens to be on day 7
from also having systematically different bolus estimation on day 7 due to shared RNG state.

---

## 9. End-to-End Flow Summary

For each simulated minute `t` on a given patient×scenario×day:

```
1. Look up cached MealSchedule (or generate it deterministically)
   a. Sample 5 base meals from uniform distributions
   b. Apply daily timing jitter (±5 min regular, ±20 min irregular day)
   c. Apply snack-skipping (7% both, 20% one)
   d. Apply periodic high-carb event if due (once per 7–14 days)
   e. Apply scenario perturbation:
       sc4 → pick lunch or dinner; scale carbs ×1.8–2.1, duration ×2
       sc5 → pick 1 meal to miss bolus
       sc6 → pick 1–2 meals; sample late-bolus delay (20–60 min each)
   f. Compute per-meal bolus carbs = actual_carbs × patient_bias × noise × underest

2. Look up cached ExerciseSchedule (or generate it deterministically)
   a. sc2/7/8/9: sample duration and AC from scenario-specific range
   b. Sample start time (16:30–20:30), enforce meal-separation on 85% of days

3. At minute t:
   a. For each of 5 meals: compute u_inc (bolus insulin) and d_inc (carb absorption)
   b. Compute baseline AC for scenario + session AC if exercise is active
   c. Sum all contributions → (u, d, activity)
```

This three-signal time series is the input to the Hovorka ODE integrator for that minute.
