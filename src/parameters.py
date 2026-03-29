from __future__ import annotations

from typing import Optional

import numpy as np

ParameterSet = dict[str, float]

_TAUI_RATE_MIN: float = 1e-3
_MIN_POSITIVE: float = 1e-9
_VG_EXP_NORMAL_MEAN: float = 1.16
_VG_EXP_NORMAL_STD: float = 0.23
_VG_EXP_MIN_FOR_POSITIVE_VG: float = 1.01
_MAX_RESAMPLE_ATTEMPTS: int = 50
_AGE_YEARS_MIN: float = 18.0
_AGE_YEARS_MAX: float = 65.0
_AGE_YEARS_MEAN: float = 35.0
_AGE_YEARS_STD: float = 12.0


def _get_hovorka_base_params() -> ParameterSet:
    """Return standard Hovorka parameters for a reference patient."""
    return {
        "MwG": 180.16,  # [mg/mmol] molecular weight of glucose
        "EGP0": 0.0161,  # [mmol/kg/min] endogenous glucose production
        "F01": 0.0097,  # [mmol/kg/min] non-insulin dependent flux
        "k12": 0.0649,  # [1/min] transfer rate
        "ka1": 0.0055,  # [1/min] deactivation rate for transport
        "ka2": 0.0683,  # [1/min] deactivation rate for disposal
        "ka3": 0.0304,  # [1/min] deactivation rate for EGP
        "SI1": 32.0e-4,  # [L/min/mU] sensitivity transport  (scaled ~38% down from Hovorka 2004 to match broader T1D population; targets ICR ~12 g/U per Dalla Man et al. 2007)
        "SI2": 5.1e-4,  # [L/min/mU] sensitivity disposal
        "SI3": 325.0e-4,  # [L/mU] sensitivity EGP
        "kb1": 0.0055 * 32.0e-4,  # [min^-2/(mU/L)] sensitivity transport
        "kb2": 0.0683 * 5.1e-4,  # [min^-2/(mU/L)] sensitivity disposal
        "kb3": 0.0304 * 325.0e-4,  # [min^-1/(mU/L)] sensitivity EGP
        "ke": 0.138,  # [1/min] elimination rate
        "VI": 0.12,  # [L/kg] insulin distribution volume
        "VG": 0.1484,  # [L/kg] glucose distribution volume
        "tauI": 55.871,  # [min] absorption time constant
        "tauG": 39.908,  # [min] glucose absorption time
        "Ag": 0.7943,  # [1] glucose absorption rate
        "BW": 80.0,  # [kg] body weight
        "age_years": 35.0,  # [years] adult T1D reference cohort center
        # -----------------------------------------------------------------------
        # ETH Deichmann exercise parameters (Deichmann et al., PLOS CB 2023).
        # AC-driven states: Y, Z, rGU, rGP, tPA, PAint, rdepl, th.
        # Source: gitlab.com/csb.ethz/t1d-exercise-model
        #
        # Aerobic parameters — nominal values from params_T1D-V1 / params_standard
        # (Deichmann et al. 2023, gitlab.com/csb.ethz/t1d-exercise-model).
        # NOTE: all 5 individual patient files share identical exercise parameters;
        # inter-patient variability is captured by the V1 posterior (261 samples).
        "eth_tau_AC":  5.0,       # [min] AC → Y time constant (fixed across all patients)
        "eth_tau_Z":   600.0,     # [min] post-exercise SI decay (~10h, fixed)
        "eth_b":       3.64e-6,   # [1/(count·min)] Z drive; V1/standard nominal
        "eth_q1":      6.46e-7,   # [1/(count·min²)] rGU drive; V1/standard nominal
        "eth_q2":      0.0617,    # [1/min] rGU decay; V1/standard nominal
        "eth_q3l":     4.46e-7,   # rGP aerobic drive; V1/standard nominal
        "eth_q4l":     0.0705,    # [1/min] rGP aerobic decay; V1/standard nominal
        "eth_adepl":   0.0108,    # [min/count] glycogen depletion slope (fixed)
        "eth_bdepl":   180.6,     # [count·min] glycogen depletion intercept (fixed)
        "eth_aY":      1500.0,    # [count·min] fY half-saturation (fixed)
        "eth_aAC":     1000.0,    # [count] fAC half-saturation (fixed)
        "eth_ah":      5600.0,    # [count] high-intensity threshold (fixed)
        "eth_n1":      20.0,      # [-] fY Hill coefficient (fixed)
        "eth_n2":      100.0,     # [-] fAC/fHI Hill coefficient (fixed)
        "eth_tp":      2.0,       # [min] fp half-saturation (fixed)
        "eth_alpha":   0.27,      # scaling constant (fixed across patients)
        # Anaerobic parameters — EXPERIMENTAL (not validated on T1D patients).
        # Values from params_standard.csv; partially supported by params_T1D-V3.
        "eth_q3h":     1.17e-6,   # rGP anaerobic drive (EXPERIMENTAL)
        "eth_q4h":     0.0705,    # [1/min] rGP anaerobic decay (EXPERIMENTAL)
        "eth_q5":      0.03,      # [1/min] th decay rate (EXPERIMENTAL)
        "eth_q6":      0.0,       # [1/min] glycogen depletion rate (0 = aerobic-only)
        # Multi-day accumulation cap — EXPERIMENTAL (original paper validated on single sessions only).
        # Z naturally accumulates when exercise days repeat: each session raises Z and tau_Z=600 min
        # means only ~56% decays overnight, leading to unbounded multi-day buildup.
        # eth_Z_max caps the drive term via (1 - Z/Z_max), bounding Z to a single-session ceiling.
        #
        # Calibration rationale (Z_max=0.2):
        #   Scenario 2 (aerobic, 45 min, AC=1500): ΔZ ≈ 0.12 per session → naturally below cap,
        #   grows freely and realistically. Multi-day spillover compounds to ≤0.15 at most.
        #   Scenarios 7/8 (prolonged/anaerobic, 80-90 min, AC≥1500): ΔZ ≈ 0.65 uncapped → capped
        #   at 0.2. Post-exercise effective insulin sensitization 20% (was 40% at Z_max=0.4).
        #   This halves the exercise_si=Z*x1*Q1 overnight drain for intense exercise days while
        #   preserving the anomaly glucose signature needed for ML labeling.
        "eth_Z_max":   0.2,       # [count·min] soft ceiling on Z accumulation (EXPERIMENTAL)
    }


def get_base_params() -> ParameterSet:
    """Return standard parameters for the supported Hovorka model."""
    return _get_hovorka_base_params()


def _sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    lower: float = _MIN_POSITIVE,
    upper: float = float("inf"),
    max_attempts: int = _MAX_RESAMPLE_ATTEMPTS,
) -> float:
    """Sample from Normal(mean, std) truncated to (lower, upper) by rejection."""
    for _ in range(max_attempts):
        sample = float(rng.normal(mean, std))
        if np.isfinite(sample) and sample > lower and sample < upper:
            return sample
    return float(np.clip(mean, lower, upper))


def _is_plausible_patient(p: ParameterSet) -> bool:
    """Loose physiological plausibility checks for sampled parameter sets."""
    positive_keys = (
        "EGP0",
        "F01",
        "k12",
        "ka1",
        "ka2",
        "ka3",
        "SI1",
        "SI2",
        "SI3",
        "kb1",
        "kb2",
        "kb3",
        "ke",
        "VI",
        "VG",
        "tauI",
        "tauG",
        "Ag",
        "BW",
    )
    if any(p[k] <= 0.0 or not np.isfinite(p[k]) for k in positive_keys):
        return False

    if not (40.0 <= p["BW"] <= 180.0):
        return False
    if not (_AGE_YEARS_MIN <= p["age_years"] <= _AGE_YEARS_MAX):
        return False
    if not (0.03 <= p["VI"] <= 0.5):
        return False
    # Tightened VG lower bound to reduce unstable low-distribution outliers.
    if not (0.10 <= p["VG"] <= 0.62):
        return False
    if not (5.0 <= p["tauI"] <= 400.0):
        return False
    if not (5.0 <= p["tauG"] <= 240.0):
        return False

    # Keep insulin sensitivities within physiological ranges based on published
    # Hovorka distributions (Boiroux-Cap2 thesis, Table 2.1)
    # SI1 ~ N(32e-4, 20e-4²), allow ±3σ → [0, 92e-4]; clamp at 1e-4
    if not (1.0e-4 <= p["SI1"] <= 1.0e-2):
        return False
    # SI2 ~ N(5.1e-4, 4.9e-4²), allow ±3σ → [0, 19.8e-4]; clamp at 1e-5
    if not (1e-5 <= p["SI2"] <= 2.0e-3):
        return False
    # SI3 ~ N(325e-4, 191e-4²), allow ±3σ → [0, 898e-4]; clamp at 1e-3
    if not (1e-3 <= p["SI3"] <= 9.0e-2):
        return False

    # Endogenous glucose production: N(0.0161, 0.0039²)
    # Allow ±3σ: 0.0161 ± 0.0117 = -0.0 to 0.028, clamp at 0.001
    if not (0.001 <= p["EGP0"] <= 0.030):
        return False

    # F01 (non-insulin dependent glucose): N(0.0097, 0.0022²)
    # Allow ±3σ: 0.0097 ± 0.0066 = 0.003 to 0.016
    if not (0.001 <= p["F01"] <= 0.020):
        return False

    return True


def _sample_single_patient(rng: np.random.Generator, base: ParameterSet) -> ParameterSet:
    """Sample one patient parameter set with guarded distributions."""
    p = base.copy()

    # NOTE: kb1/kb2/kb3 are NOT sampled here — the ODE always recomputes them
    # as kb = SI × ka at runtime (model.py). Sampling them here would have no effect.

    # Glucose parameters (truncated normal, no abs-folding)
    p["EGP0"] = _sample_truncated_normal(rng, 0.0161, 0.0039)
    p["F01"] = _sample_truncated_normal(rng, 0.0097, 0.0022)
    p["k12"] = _sample_truncated_normal(rng, 0.0649, 0.0282)

    # Activation/deactivation rates (ka1, ka2, ka3)
    # The Boiroux thesis reports CVs of ~100%, 74%, and 77% respectively, but those
    # values were back-calculated from a 7-patient cohort. Because kb = SI × ka is what
    # the ODE actually uses, sampling both SI and ka with high independent variance
    # compounds the effective spread of kb roughly as (CV_SI² + CV_ka²)^0.5.
    # Holding ka near their nominal values and concentrating variability in SI (which
    # is the clinically meaningful parameter) better matches published ICR/ISF distributions
    # (Dalla Man et al. 2007). CV reduced to ~10% here.
    p["ka1"] = _sample_truncated_normal(rng, 0.0055, 0.0006)   # was std=0.0056 (CV≈102%)
    p["ka2"] = _sample_truncated_normal(rng, 0.0683, 0.0068)   # was std=0.0507 (CV≈74%)
    p["ka3"] = _sample_truncated_normal(rng, 0.0304, 0.0030)   # was std=0.0235 (CV≈77%)

    # Insulin sensitivities (cannot be negative)
    # Means scaled ~38% down from Hovorka 2004 (7-patient cohort) to represent a broader
    # T1D adult population targeting ICR ~12 g/U (Dalla Man et al. 2007, IEEE TBME).
    # Stds scaled proportionally to preserve published CVs (Boiroux thesis Table 2.1).
    p["SI1"] = _sample_truncated_normal(rng, 32.0e-4, 20.0e-4, lower=1e-6)
    p["SI2"] = _sample_truncated_normal(rng, 5.1e-4, 4.9e-4, lower=1e-6)
    p["SI3"] = _sample_truncated_normal(rng, 325.0e-4, 191.0e-4, lower=1e-6)

    # Elimination and volumes
    p["ke"] = _sample_truncated_normal(rng, 0.14, 0.035)
    p["VI"] = _sample_truncated_normal(rng, 0.12, 0.012)

    # VG derivation per spec: exp(VG) ~ N(1.16, 0.23^2)
    # Use truncated normal (>1.01) to avoid non-physical VG <= 0 and avoid clipping spikes.
    vg_exp = _sample_truncated_normal(
        rng,
        _VG_EXP_NORMAL_MEAN,
        _VG_EXP_NORMAL_STD,
        lower=_VG_EXP_MIN_FOR_POSITIVE_VG,
    )
    p["VG"] = float(np.log(vg_exp))

    # tauI derivation: 1/tauI ~ N(0.018, 0.0045^2), guarded away from ~0
    tau_i_rate = _sample_truncated_normal(rng, 0.018, 0.0045, lower=_TAUI_RATE_MIN)
    p["tauI"] = 1.0 / tau_i_rate

    # tauG derivation: ln(tauG) ~ N(3.689, 0.25^2)
    p["tauG"] = float(np.exp(rng.normal(3.689, 0.25)))

    # Carbohydrate bioavailability: physiological range 0.7–0.9 for T1D adults
    # (upper bound was incorrectly 1.2 = 120% absorption, which is physically impossible)
    p["Ag"] = float(rng.uniform(0.7, 0.9))
    # Adult T1D cohort assumption: broad outpatient population, not pediatric.
    # Keep age integer at generation time, but store as float for consistency
    # with the model parameter container type.
    sampled_age = int(np.rint(rng.normal(_AGE_YEARS_MEAN, _AGE_YEARS_STD)))
    clamped_age = max(int(_AGE_YEARS_MIN), min(int(_AGE_YEARS_MAX), sampled_age))
    p["age_years"] = float(clamped_age)
    # Body weight: constrained male cohort range for this simulation setup.
    p["BW"] = float(rng.uniform(65.0, 95.0))

    # ETH exercise parameter sampling — distributions from params_T1D-V1_pred.csv
    # (261-sample V1 posterior, Deichmann et al. 2023). The 5 individual patient
    # files share identical exercise parameters and do not represent inter-patient
    # variability — the posterior is the correct source for Monte Carlo sampling.
    # Fixed structural parameters (tau_AC, tau_Z, adepl, bdepl, aY, aAC, ah, n1, n2, tp, alpha)
    # are not sampled — they are consistent across all T1D patient files.
    # eth_b: V1 nominal 3.64e-6; std reflects V1/V2/V3 spread (1.55–3.64e-6).
    p["eth_b"]   = _sample_truncated_normal(rng, 3.64e-6, 1.05e-6, lower=1e-9)  # Z drive
    # eth_q1/q2: posterior mean/std; observed range [4.4e-7, 1.2e-6] / [0.032, 0.134].
    p["eth_q1"]  = _sample_truncated_normal(rng, 7.33e-7, 1.60e-7, lower=1e-9, upper=1.3e-6)  # rGU drive
    p["eth_q2"]  = _sample_truncated_normal(rng, 0.0707,  0.0219,  lower=0.032)               # rGU decay
    p["eth_q3l"] = _sample_truncated_normal(rng, 5.79e-7, 1.83e-7, lower=1e-10)  # rGP aerobic drive
    p["eth_q4l"] = _sample_truncated_normal(rng, 0.0993,  0.0378,  lower=1e-4)   # rGP aerobic decay
    # Anaerobic params are fixed (EXPERIMENTAL) — not sampled per patient.

    return p


def generate_monte_carlo_patients(
    n: int = 10,
    standard_patient: bool = False,
    seed: Optional[int] = None,
) -> list[ParameterSet]:
    """
    Generate N patient parameter sets.

    - `standard_patient=True` returns base parameters duplicated N times.
    - Otherwise, sampled parameters are resampled until plausibility checks pass.
    """
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)
    base = get_base_params()

    patients: list[ParameterSet] = []

    for _ in range(n):
        if standard_patient:
            patients.append(base.copy())
            continue

        sampled = base.copy()
        for _ in range(_MAX_RESAMPLE_ATTEMPTS):
            sampled = _sample_single_patient(rng, base)
            if _is_plausible_patient(sampled):
                break
        else:
            sampled = base.copy()

        patients.append(sampled)

    return patients
