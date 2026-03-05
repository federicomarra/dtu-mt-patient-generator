# Type 1 Diabetes Monte Carlo Simulator

A high-fidelity Monte Carlo simulation framework for Type 1 Diabetes glucose dynamics using the Hovorka physiological model. Generate realistic virtual patient cohorts with diverse physiological parameters for testing control algorithms, meal timing strategies, and safety protocols.

## Features

- **Physiological Model**: Full 11-state Hovorka ODE model with insulin and glucose dynamics
- **Monte Carlo Sampling**: Generate diverse patient cohorts with physiologically plausible parameter distributions (aligned with published research)
- **Realistic Meal Scheduling**: 5-meal daily schedule (breakfast, snacks, lunch, dinner) with temporal jitter and realistic eating durations
- **Safety Controls**:
  - Hypo-guard: Automatic basal suspension at configurable glucose threshold
  - Hypo-rescue: Proportional glucose injection below critical threshold
- **Patient-Specific Dosing**: Insulin-to-carb ratios scaled by individual insulin sensitivity (SI3)
- **Rejection Sampling**: Two-stage filtering eliminates physiologically unstable parameter combinations
- **CGM Noise Simulation**: Autocorrelated AR(1) noise model for realistic sensor error
- **Reproducibility**: Fully seeded random number generation for deterministic results
- **Clean Output**: Progress tracking, rejection statistics, and exportable results (CSV/Parquet)

## Installation

### Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- pandas
- tqdm

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd dtu-mt-patient-generator

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run a 10-patient, 5-day Monte Carlo simulation:

```bash
python main.py
```

Output:

- **CSV file**: `monte_carlo_results/YYYYMMDD_HHMMSS/results_10p_5d.csv`
- **Config metadata**: `monte_carlo_results/YYYYMMDD_HHMMSS/config_10p_5d.txt`
- **Trajectory plot**: `monte_carlo_results/YYYYMMDD_HHMMSS/plot_10p_5d.png`

Analyze results:

```bash
python analyze_simulation.py monte_carlo_results/YYYYMMDD_HHMMSS/results_10p_5d.csv
```

## Configuration

Edit simulation parameters in `main.py`:

```python
@dataclass
class SimulationConfig:
    n_patients: int = 10                       # Number of patients in cohort
    n_days: int = 5                            # Simulation duration (days)
    random_seed: Optional[int] = 42            # For reproducibility
    
    # CGM Noise
    noise_std: float = 0.10                    # Sensor noise (mmol/L)
    noise_autocorr: float = 0.7                # AR(1) autocorrelation
    
    # Insulin Dosing
    insulin_sensitivity: float = 12.0          # Baseline I:C ratio (g/U)
    basal_hourly: float = 0.5                  # Basal rate (U/hr)
    use_calibrated_basal: bool = True          # Derive from steady state
    
    # Safety Thresholds
    hypo_guard_mmol: float = 5.0               # Suspend basal below (mmol/L)
    hypo_rescue_trigger_mmol: float = 4.5      # Emergency glucose below
    hypo_rescue_gain_per_min: float = 25.0     # Rescue scaling
    
    # Initial Conditions
    initial_target_glucose_mgdl: float = 100.0 # Steady-state target
```

### Rejection Thresholds

Configured in `main.py` simulation loop:

```python
rejection_bounds_mmol = (5.0, 6.5)            # Initial glucose acceptance range
instability_max_glucose_mmol = 20.0           # Max allowed glucose spike
instability_hyper_pct = 30.0                  # Max % time in hyperglycemia
```

## Project Structure

```txt
dtu-mt-patient-generator/
├── main.py                    # Simulation driver + config
├── analyze_simulation.py      # Result analysis tool
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── src/
    ├── model.py               # Hovorka ODE equations + steady-state solver
    ├── parameters.py          # Monte Carlo patient parameter generation
    ├── input.py               # Meal schedule generation + caching
    ├── sensor.py              # CGM measurement with noise
    └── export.py              # CSV/Parquet export + metadata logging
```

## How It Works

### 1. Patient Parameter Sampling

Patients are sampled from published physiological distributions (Boiroux et al., Table 2.1):

| Parameter | Distribution | Description |
| ----------- | -------------- | ------------- |
| `SI1` | N(51.2e-4, 32.09e-4²) | Insulin sensitivity (transport) |
| `SI2` | N(8.2e-4, 7.84e-4²) | Insulin sensitivity (disposal) |
| `SI3` | N(520e-4, 306.2e-4²) | Insulin sensitivity (EGP suppression) |
| `VG` | exp(VG) ~ N(1.16, 0.23²) | Glucose distribution volume |
| `EGP0` | N(0.0161, 0.0039²) | Endogenous glucose production |
| `BW` | U(65, 95) kg | Body weight |

**Plausibility Filtering**: Parameters outside physiological bounds are rejected during sampling.

### 2. Steady-State Initialization

Each patient is initialized at a stable fasting state (target: 100 mg/dL) using Newton-Raphson optimization to solve for the fixed point of the Hovorka ODE system.

**Rejection Stage 1**: Patients with initial glucose outside [5.0, 6.5] mmol/L are rejected (indicates pathological parameter combinations).

### 3. Meal Schedule Generation

5-meal daily schedule cached per patient per day:

| Meal | Time Window | Carbs (g) | Duration (min) |
| ------ | ------------- | ----------- | ---------------- |
| Breakfast | 6:30-8:00am | 45-70 | 15-20 |
| Morning Snack | 10:00-11:00am | 12-20 | 8-12 |
| Lunch | 12:00-1:00pm | 50-75 | 20-25 |
| Afternoon Snack | 3:00-4:30pm | 12-25 | 5-10 |
| Dinner | 6:30-8:00pm | 50-75 | 20-25 |

Meals are deterministically jittered (±30 min) and bolus insulin is delivered 10 minutes pre-meal.

### 4. ODE Simulation

Each day is simulated using SciPy's `solve_ivp` (RK45 method) with:

- **Time resolution**: 1 minute
- **State vector**: 10 compartments (glucose, insulin, gut absorption, etc.)
- **Adaptive stepping**: `max_step=1.0` min to capture meal/bolus discontinuities

**Patient-Specific Scaling**:

- Insulin sensitivity: `12.0 g/U × SI3_ratio` (clamped to [10.0, 14.0])
- SI3 ratio: Limited to ±15% of reference (0.85-1.15)
- Basal rate: Derived from steady-state insulin level

### 5. Post-Simulation Rejection

**Rejection Stage 2**: After 5-day simulation, reject if:

- Hyperglycemia percentage > 30%, OR
- Maximum glucose > 20.0 mmol/L

This catches rare parameter combinations that pass initial checks but produce unstable trajectories.

### 6. CGM Noise Addition

Realistic sensor noise is added via AR(1) autocorrelated process:

$$\varepsilon[t] = \rho \cdot \varepsilon[t-1] + \sqrt{1-\rho^2} \cdot \eta[t]$$

where $\eta[t] \sim N(0, \sigma^2)$, $\rho = 0.7$, $\sigma = 0.1$ mmol/L.

## Output Format

### CSV Export

Columns:

- `patient_id`: Integer patient identifier (0-indexed)
- `day`: Day number (0-indexed)
- `minute`: Minute within day (0-1440)
- `time`: Formatted time string (HH:MM)
- `blood_glucose`: CGM reading (mmol/L or mg/dL)

### Metadata File

Text file containing:

- All simulation parameters
- Rejection statistics (count by reason, rejection rate %)
- Safety thresholds
- Timestamp and random seed

## Analysis

Use `analyze_simulation.py` to compute:

- **Glycemic metrics**: Mean, min, max, standard deviation
- **Time-in-range**: % time in hypo (<3.9), target (3.9-10.0), hyper (>10.0)
- **Per-patient breakdown**: Individual glucose statistics
- **Per-day TIR**: Tracking control quality over time
- **Extreme values**: Top 5 hypo/hyper events with timestamps

Example output:

```txt
=== Glycemic Control ===
Hypoglycemia (<3.9):       0/72050 (  0.0%)
In Range (3.9-10.0):    69575/72050 ( 96.6%)
Hyperglycemia (>10.0):  2475/72050 (  3.4%)
```

## References

**Hovorka Model**:

- Hovorka, R., et al. (2004). "Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes." *Physiological Measurement*, 25(4), 905.

**Parameter Distributions**:

- Boiroux, D. (2012). "Model Predictive Control for Type 1 Diabetes," PhD Thesis, Technical University of Denmark (DTU), Table 2.1.

**Meal Scheduling**:

- Wilinska, M. E., et al. (2010). "Simulation environment to evaluate closed-loop insulin delivery systems in type 1 diabetes." *Journal of Diabetes Science and Technology*, 4(1), 132-144.

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
