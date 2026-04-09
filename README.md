# Type 1 Diabetes Monte Carlo Simulator

A virtual-patient simulator for Type 1 Diabetes based on the Hovorka model, extended with the ETH Deichmann accelerometer-driven exercise model (Deichmann et al., PLOS Comput Biol 2023).

The project is designed for:

- synthetic cohort generation
- control/safety algorithm validation
- anomaly-detection dataset creation
- reproducible simulation runs with configurable realism

## Highlights

- Hovorka glucose-insulin ODE simulation with minute-level inputs
- ETH Deichmann exercise extension: 8 AC-driven states (`Y`, `Z`, `rGU`, `rGP`, `tPA`, `PAint`, `rdepl`, `th`) with glycogen depletion and post-exercise insulin sensitivity decay
- Monte Carlo patient generation from physiological distributions
- Base scenario (SC1/SC2/SC3) + independent per-day overlay architecture: large meal, missed bolus, late bolus, prolonged aerobic, anaerobic — any combination can co-occur
- Ground-truth ML labels exported per minute: `base_scenario`, `had_large_meal`, `had_missed_bolus`, `n_late_boluses`, `exercise_overlay`, `bolus_status`, `meal_size`, `exercise_type`
- Dawn phenomenon (GH-driven EGP elevation 03:00–07:00) and cortisol morning insulin resistance (06:00–10:00) with per-patient `dawn_amp`
- Safety/control stack:
  - hypo guard (basal suspension logic)
  - hypo rescue carbohydrates
  - IOB-aware meal bolus attenuation
  - ISF correction bolus channel
- Configurable rejection pipeline for accepted patient quality
- CSV/Parquet export plus metadata sidecar
- Built-in analysis script for latest run

## Model Overview

### Core dynamics

The simulator integrates an 18-state model:

- 10 classic Hovorka states (glucose, insulin compartments, insulin action, gut absorption)
- 8 ETH Deichmann exercise states (`Y`, `Z`, `rGU`, `rGP`, `tPA`, `PAint`, `rdepl`, `th`) coupled into glucose dynamics via three Q1 interaction terms: exercise uptake, exercise EGP, and post-exercise insulin sensitivity boost

Implementation:

- `src/model.py`
- `src/hovorka_exercise.py`

### Steady-state initialization

Each patient/day simulation starts from a computed fasting steady state obtained by solving for basal insulin that matches the target glucose.

Current method:

- In-house bounded Newton-Raphson solver with explicit bracketing and damping safeguards
- target passed from config (`initial_target_glucose_mgdl`)

Important defaults:

- target: `100 mg/dL` (configurable)
- acceptance window: `4.5 - 7.2 mmol/L` (configurable)

## Input/Scenario System

Implemented in `src/input.py`.

### Patient profile (sampled once per patient)

- **Base scenario**: SC1 (normal, 38%), SC2 (active aerobic every day, 27%), SC3 (sedentary, 35%)
- **Meal count**: 3, 4, or 5 meals (equal thirds); deviates ±1 on 20% of days
- **Bolus bias**: 0.78–0.95 (persistent per-patient carb underestimation)

### Per-day overlays (independent Bernoulli each day)

| Overlay | Rate | Effect |
| --------- | ------ | -------- |
| Large meal | 9% | One main meal ×1.8–2.1 carbs, ×2 duration, 0.70–0.85 underest |
| Missed bolus | 9% | One meal gets no insulin |
| Late bolus | ~7% | 1–2 meals bolused 30–90 min post-meal |
| Prolonged aerobic | ~4.8% (SC1 only) | 75–120 min aerobic session |
| Anaerobic | ~4.8% (SC1 only) | 20–50 min HIIT/resistance, 6000–9000 AC (EXPERIMENTAL) |

SC2 patients always have a standard aerobic session (30–60 min, 1200–2000 AC) every day.
Exercise overlays are mutually exclusive (XOR) and only apply to SC1 days.

### Bolus timing distribution

Normal boluses are delivered: 60% pre-meal (5–20 min before), 25% at onset (0–4 min),
15% post-meal (1–20 min after) — matching empirical T1D behaviour.

## Safety and Control Stack

Implemented primarily in `src/simulation_control.py`, applied in the simulation loop.

- Hypo guard: modifies/suspends insulin delivery under low-glucose risk
- Hypo rescue: injects rescue carbs via gut dynamics
- IOB bolus guard: attenuates meal bolus aggressiveness based on IOB
- Correction ISF: bounded correction bolus with cooldown and IOB handling

Run summary includes activation percentages/events for these controls.

## Rejection Pipeline

Candidates are sampled from `generate_monte_carlo_patients(...)` and filtered.

Stages:

1. Initial-state rejection
   - initial glucose must be in `[initial_glucose_acceptance_min_mmol, initial_glucose_acceptance_max_mmol]`
2. Instability rejection
   - `max glucose > instability_max_glucose_mmol` (default 33.3 mmol/L / 600 mg/dL) — **fail-fast**: checked per day, aborts the loop immediately if any day exceeds the hard cap
   - `hyper% > instability_hyper_pct_threshold` (default 60%) — evaluated over the **full concatenated trajectory** after all days complete (cumulative average; cannot be checked per-day)
3. Quality rejection — evaluated **per day** with **fail-fast**: the loop aborts on the first failing day
   - exercise days (SC2 base or exercise overlay): hypo% ≤ `quality_max_hypo_pct_exercise_threshold` (default 17%)
   - non-exercise days: hypo% ≤ `quality_max_hypo_pct_threshold` (default 15%) — hard cap
   - non-exercise days soft cap: hypo% ≤ 10%; if >2 days exceed this, patient rejected (chronic overinsulinisation check)
   - +`quality_max_hypo_pct_spillover_bonus` (2%) per exercise day in the **2-day lookback** window [d-2, d-1] — accounts for Z-state (τ_Z ≈ 600 min) compounding across consecutive exercise days
   - all days: hyper% ≤ `quality_max_hyper_pct_threshold` (default 70%)
   - hard floor: any minute with glucose < `quality_min_glucose_mmol` (default 2.0 mmol/L / 36 mg/dL) rejects the patient

All thresholds are config-driven from `SimulationConfig`.

## Configuration

Main configuration dataclass: `src/simulation_config.py`

Key groups:

- Cohort/runtime:
  - `n_patients`, `n_days`, `random_seed`
  - `random_scenarios`, `fixed_scenario`
- Signal/noise/solver:
  - `noise_std`, `noise_autocorr`
  - `solver_method`, `solver_max_step`, `derivative_clip`
- Initialization and filtering:
  - `initial_target_glucose_mgdl`
  - `initial_glucose_acceptance_min_mmol`, `initial_glucose_acceptance_max_mmol`
  - `instability_max_glucose_mmol`, `instability_hyper_pct_threshold`
  - `quality_max_hypo_pct_threshold`, `quality_max_hypo_pct_exercise_threshold`, `quality_max_hypo_pct_spillover_bonus`
  - `quality_max_hyper_pct_threshold`, `quality_min_glucose_mmol`
  - `n_warmup_days` (burn-in days before recording; lets ETH Z-state reach cyclic steady state)
- Basal and calibration:
  - `basal_hourly`, `use_calibrated_basal`
  - `init_insulin_carbo_ratio`, `init_insulin_sensitivity_factor`
- Safety/control toggles and thresholds:
  - hypo guard/rescue settings
  - IOB guard settings
  - correction ISF settings

## Exports

Exporter: `src/export.py`

Per run, output is written to:

- `monte_carlo_results/YYYYMMDD/HHMMSS/`

Files:

- `results_<Np>p_<Nd>d.parquet` (optional)
- `results_<Np>p_<Nd>d.csv` (optional)
- `config_<Np>p_<Nd>d.txt` (written with CSV export)
- `simulation_plot.png` (if plotting enabled + export folder exists)
- `inputs_plot.png` (if plotting enabled + export folder exists)

Data columns include:

- `patient_id`, `patient_age_years`
- `day`, `minute`, `absolute_minute`, `time`
- `blood_glucose`, `insulin_mU_min`, `cho_mg_min`
- **Ground-truth ML labels**: `base_scenario`, `had_large_meal`, `had_missed_bolus`, `n_late_boluses`, `exercise_overlay` (per-day); `bolus_status`, `meal_size`, `exercise_type` (per-minute)

## Analysis Tooling

`analyze_simulation.py`:

- auto-finds latest CSV/Parquet if path omitted
- reports glucose stats, CV%, P5/P95
- per-patient and per-day breakdowns
- input channel summaries
- age distribution when available

`analyze_large.py`:

- designed for large cohorts (100K+ patients)
- population-level glucose distribution, TIR/hypo/hyper rates
- per-scenario breakdown

`fix_parquet_boundaries.py`:

- removes the duplicate row at each day boundary (day exports include minute 1440 which is also minute 0 of the next day)
- should be run before analysis on multi-day exports

`main.py` can run optional post-analysis automatically on latest results.

## Installation

### Requirements

- Python 3.12+
- Dependencies in `requirements.txt`

Install:

```bash
git clone https://github.com/federicomarra/dtu-mt-patient-generator
cd dtu-mt-patient-generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run main simulation:

```bash
python main.py
```

Analyze latest export:

```bash
python analyze_simulation.py
```

Run smoke simulation test:

```bash
python test/test_simulation.py --patients 10 --days 3 --random-scenarios
```

Run steady-state Newton check:

```bash
python test/test_steady_state.py
```

## Current Project Structure

```text
dtu-mt-patient-generator/
├── main.py
├── analyze_simulation.py
├── library_generator.py
├── requirements.txt
├── README.md
├── src/
│   ├── export.py
│   ├── hovorka_exercise.py
│   ├── input.py
│   ├── library_generation.py
│   ├── model.py
│   ├── parameters.py
│   ├── sensitivity.py
│   ├── sensor.py
│   ├── simulation.py
│   ├── simulation_config.py
│   ├── simulation_control.py
│   └── simulation_utils.py
└── test/
    ├── test_library_parallel.py
    ├── test_steady_state.py
    ├── test_sensitivity.py
    └── test_simulation.py
```

## Notes

- Plot rendering is backend-aware (headless/Agg runs skip interactive `show()`).
- Age is sampled as integer years (stored as float in parameter container).
- The anaerobic exercise overlay uses EXPERIMENTAL ETH parameters (not validated on T1D patients).
- `fix_parquet_boundaries.py` should always be run before analysis on multi-day exports.

## References

- Hovorka, R. et al. (2004). Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes.
- Deichmann, J. et al. (2023). A physiological model of the effect of physical activity on the glucose-insulin system for people with type 1 diabetes. PLOS Computational Biology.
- Boiroux, D. (2012). Model Predictive Control for Type 1 Diabetes (PhD thesis, DTU).
- Dalla Man, C., Rizza, R. A., Cobelli, C. (2007). Meal simulation model of glucose-insulin system.
- Wilinska, M. E. et al. (2010). Simulation environment for closed-loop insulin delivery evaluation.

## License

MIT License

Copyright (c) 2026 DTU

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
