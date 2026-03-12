# Type 1 Diabetes Monte Carlo Simulator

A Monte Carlo virtual-patient simulator for Type 1 Diabetes based on the Hovorka physiological model. The project focuses on realistic meal-driven glucose dynamics, safety control behaviors, and reproducible cohort generation for algorithm testing.

## What This Simulator Does

- Simulates glucose/insulin dynamics with the Hovorka ODE model.
- Generates virtual patient cohorts from published parameter distributions.
- Builds patient-specific meal schedules with deterministic per-seed jitter.
- Estimates per-patient ICR and ISF before running day-by-day simulations.
- Applies safety and dosing control layers:
  - Hypo guard (temporary basal suspension)
  - Hypo rescue (fixed rescue carbs via gut input)
  - IOB-aware meal bolus attenuation
  - ISF-based correction channel with cooldown and IOB subtraction
- Exports full trajectories and run metadata to CSV/Parquet.

## Installation

Requirements:

- Python 3.9+
- Dependencies in `requirements.txt`

Setup:

```bash
git clone https://github.com/federicomarra/dtu-mt-patient-generator
cd dtu-mt-patient-generator
pip install -r requirements.txt
```

## Quick Start

Run simulation:

```bash
python main.py
```

Analyze latest output:

```bash
python analyze_simulation.py
```

## Current Project Structure

```text
dtu-mt-patient-generator/
├── main.py
├── analyze_simulation.py
├── requirements.txt
├── README.md
├── src/
│   ├── model.py
│   ├── parameters.py
│   ├── input.py
│   ├── sensitivity.py
│   ├── sensor.py
│   ├── export.py
│   ├── simulation.py
│   ├── simulation_config.py
│   ├── simulation_control.py
│   └── simulation_utils.py
└── test/
    └── test_sensitivity.py
```

### Simulation modules (new modular layout)

- `src/simulation.py`: orchestration loop, rejection pipeline, plotting, export metadata.
- `src/simulation_config.py`: `SimulationConfig` dataclass with all tuning knobs.
- `src/simulation_control.py`: guard/rescue/IOB/ISF correction control policies and controller state.
- `src/simulation_utils.py`: reusable utilities (noise generation, glycemia measurement, clipping, export folder creation, plotting colors).

## Control and Safety Pipeline

For each patient and each minute:

1. Meal insulin and carbs are generated from `scenario_with_cached_meals`.
2. Guard/IOB/ISF logic adjusts effective insulin delivery:
   - Guard can suspend basal for a fixed window.
   - IOB guard can attenuate meal bolus by increasing effective ICR.
   - ISF correction can add correction insulin above target with cooldown and IOB subtraction.
3. Rescue logic adds fixed rescue carbs through gut absorption when below trigger.
4. ODE derivatives are clipped to improve numerical robustness.

## Rejection Pipeline

Patients are filtered in stages:

1. Initial-state rejection:
   - Initial glucose must be inside configured bounds.
2. Instability rejection:
   - Reject if max glucose too high or hyperglycemia percentage above threshold.
3. Quality rejection:
   - Reject if per-patient hypo% or hyper% exceeds quality thresholds.

The run summary prints rejection counts by reason and safety/control activity percentages.

## Key Configuration Fields

Configured through `SimulationConfig` in `src/simulation_config.py`.

### Core simulation

- `n_patients`, `n_days`, `random_seed`
- `noise_std`, `noise_autocorr`
- `solver_method`, `solver_max_step`, `derivative_clip`

### Basal and patient-specific dosing

- `basal_hourly`, `use_calibrated_basal`
- `init_insulin_carbo_ratio`, `init_insulin_sensitivity_factor`

### Hypoglycemia safety

- `enable_hypo_guard`
- `hypo_guard_mmol`
- `hypo_guard_suspend_min`
- `hypo_guard_retrigger_cooldown_min`
- `suppress_meal_bolus_on_guard`

- `enable_hypo_rescue`
- `hypo_rescue_trigger_mmol`
- `hypo_rescue_carbs_g`
- `hypo_rescue_duration_min`
- `hypo_rescue_retrigger_cooldown_min`

### IOB-aware meal attenuation

- `enable_iob_bolus_guard`
- `iob_guard_units`
- `iob_full_attenuation_units`
- `iob_max_icr_multiplier`

### ISF correction channel

- `enable_correction_isf`
- `correction_isf_target_mmol`
- `correction_isf_cooldown_min`
- `correction_isf_max_bolus_units`
- `correction_isf_min_bolus_units`
- `correction_isf_bolus_duration_min`
- `correction_isf_iob_free_units`

## Outputs

Per run, the simulator writes:

- `results_<Np>p_<Nd>d.csv`
- `results_<Np>p_<Nd>d.parquet`
- `config_<Np>p_<Nd>d.txt`
- `simulation_plot.png`

inside a timestamped folder under `monte_carlo_results/YYYYMMDD/HHMMSS/`.

Metadata includes:

- all run configuration fields
- rejection counters by reason
- safety activity percentages
- ISF correction usage metrics (events, active %, total units)

## Debugging Workflow (Recommended)

1. Run simulation with fixed seed.
2. Check console summary:
   - rejection reason split
   - safety activity rates
   - ISF correction summary
3. Use `analyze_simulation.py` to inspect:
   - cohort TIR/hypo/hyper
   - per-patient outliers
   - day-by-day control drift
4. If outliers persist, adjust:
   - quality rejection thresholds first
   - then control gains/cooldowns

## References

- Hovorka, R. et al. (2004). Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes.
- Boiroux, D. (2012). Model Predictive Control for Type 1 Diabetes, PhD thesis, DTU.
- Wilinska, M. E. et al. (2010). Simulation environment to evaluate closed-loop insulin delivery systems in type 1 diabetes.

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
