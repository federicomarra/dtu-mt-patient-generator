# Type 1 Diabetes Monte Carlo Simulator

A virtual-patient simulator for Type 1 Diabetes based on the Hovorka model, extended with an exercise-driven Rashid-Hovorka component.

The project is designed for:

- synthetic cohort generation
- control/safety algorithm validation
- anomaly-detection dataset creation
- reproducible simulation runs with configurable realism

## Highlights

- Hovorka glucose-insulin ODE simulation with minute-level inputs
- Exercise extension with HR-driven states (`E1`, `E2`, `TE`)
- Monte Carlo patient generation from physiological distributions
- Deterministic daily meal schedules with per-day jitter and anomalies
- Weighted scenario sampling (common days vs rarer anomaly days)
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

The simulator integrates a 13-state model:

- 10 classic Hovorka states (glucose, insulin compartments, insulin action, gut absorption)
- 3 exercise states (`E1`, `E2`, `TE`) coupled into glucose dynamics via `QE1`, `QE21`, `QE22`

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

### Daily meal structure

Five meals per day:

- breakfast
- morning snack
- lunch
- afternoon snack
- dinner

with configurable timing/carb windows and deterministic cache behavior per patient/day/scenario.

### Scenarios

`N_SCENARIOS = 6`

- `1`: normal day
- `2`: active day (planned afternoon exercise)
- `3`: sedentary day
- `4`: long lunch disturbance
- `5`: missed bolus disturbance
- `6`: late bolus disturbance

### Weighted scenario sampling

When `random_scenarios=True`, scenario draws are weighted with `SCENARIO_WEIGHTS`:

- scenarios `1-3`: common
- scenarios `4-6`: intentionally rarer for anomaly realism

### Exercise-meal overlap policy

For scenario 2:

- majority of days enforce clean temporal separation from snack/dinner windows
- minority of days allow controlled overlap as realistic outliers

This supports cleaner labels for downstream ML while preserving rare overlap behavior.

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
   - initial glucose must be inside:
     - `initial_glucose_acceptance_min_mmol`
     - `initial_glucose_acceptance_max_mmol`
2. Instability rejection
   - max glucose and hyper % must pass instability thresholds
3. Quality rejection
   - hypo % and hyper % must pass quality thresholds

All thresholds are now config-driven from `SimulationConfig`.

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
  - `initial_glucose_acceptance_min_mmol`
  - `initial_glucose_acceptance_max_mmol`
  - `instability_max_glucose_mmol`
  - `instability_hyper_pct_threshold`
  - `quality_max_hypo_pct_threshold`
  - `quality_max_hyper_pct_threshold`
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

- `patient_id`
- `patient_age_years`
- `day`, `minute`, `absolute_minute`, `time`
- `blood_glucose`
- `insulin_mU_min`
- `cho_mg_min`

## Analysis Tooling

`analyze_simulation.py`:

- auto-finds latest CSV/Parquet if path omitted
- reports glucose stats, CV%, P5/P95
- per-patient and per-day breakdowns
- input channel summaries
- age distribution when available

`main.py` can run optional post-analysis automatically on latest results.

## Installation

### Requirements

- Python 3.9+
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
python test/test_model_steady_state.py
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
    ├── test_model_steady_state.py
    ├── test_sensitivity.py
    └── test_simulation.py
```

## Notes

- Plot rendering is backend-aware (headless/Agg runs skip interactive `show()`).
- Scenario perturbations (4/5/6) are applied consistently for label fidelity.
- Age is sampled as integer years (stored as float in parameter container).

## References

- Hovorka, R. et al. (2004). Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes.
- Rashid, M. et al. Exercise model extension for the Hovorka glucose-insulin framework (HR-driven exercise states and glucose-utilisation coupling).
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
