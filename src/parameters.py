from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

ParameterSet = Dict[str, float]

_CV_KB: float = 0.30
_TAUI_RATE_MIN: float = 1e-3
_MIN_POSITIVE: float = 1e-9
_VG_EXP_NORMAL_MEAN: float = 1.16
_VG_EXP_NORMAL_STD: float = 0.23
_VG_EXP_MIN_FOR_POSITIVE_VG: float = 1.01
_MAX_RESAMPLE_ATTEMPTS: int = 50


def get_base_params() -> ParameterSet:
    """Return standard Hovorka parameters for a reference patient."""
    return {
        "MwG": 180.16,  # [mg/mmol] molecular weight of glucose
        "EGP0": 0.0161,  # [mmol/kg/min] endogenous glucose production
        "F01": 0.0097,  # [mmol/kg/min] non-insulin dependent flux
        "k12": 0.0649,  # [1/min] transfer rate
        "ka1": 0.0055,  # [1/min] deactivation rate for transport
        "ka2": 0.0683,  # [1/min] deactivation rate for disposal
        "ka3": 0.0304,  # [1/min] deactivation rate for EGP
        "SI1": 51.2,  # [L/min/mU] sensitivity transport
        "SI2": 8.2,  # [L/min/mU] sensitivity disposal
        "SI3": 520.0,  # [L/mU] sensitivity EGP
        "kb1": 0.0055 * 51.2,  # [min^-2/(mU/L)] sensitivity transport
        "kb2": 0.0683 * 8.2,  # [min^-2/(mU/L)] sensitivity disposal
        "kb3": 0.0304 * 520.0,  # [min^-1/(mU/L)] sensitivity EGP
        "ke": 0.138,  # [1/min] elimination rate
        "VI": 0.12,  # [L/kg] insulin distribution volume
        "VG": 0.1484,  # [L/kg] glucose distribution volume
        "tauI": 55.871,  # [min] absorption time constant
        "tauG": 39.908,  # [min] glucose absorption time
        "Ag": 0.7943,  # [1] glucose absorption rate
        "BW": 80.0,  # [kg] body weight
    }


def _sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    lower: float = _MIN_POSITIVE,
    max_attempts: int = _MAX_RESAMPLE_ATTEMPTS,
) -> float:
    """Sample from Normal(mean, std) truncated at lower bound (no folding by abs)."""
    for _ in range(max_attempts):
        sample = float(rng.normal(mean, std))
        if np.isfinite(sample) and sample > lower:
            return sample
    return max(lower, float(mean))


def _sample_lognormal_with_cv(
    rng: np.random.Generator,
    mean: float,
    cv: float,
) -> float:
    """
    Sample log-normal using desired arithmetic mean and coefficient of variation.

    If X ~ LogNormal(mu, sigma^2):
      CV^2 = exp(sigma^2) - 1
      mu = ln(mean) - sigma^2 / 2
    """
    sigma_sq = float(np.log(1.0 + cv * cv))
    sigma = float(np.sqrt(sigma_sq))
    mu = float(np.log(mean) - 0.5 * sigma_sq)
    return float(rng.lognormal(mean=mu, sigma=sigma))


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
    if not (0.03 <= p["VI"] <= 0.5):
        return False
    if not (0.02 <= p["VG"] <= 0.8):
        return False
    if not (5.0 <= p["tauI"] <= 400.0):
        return False
    if not (5.0 <= p["tauG"] <= 240.0):
        return False

    return True


def _sample_single_patient(rng: np.random.Generator, base: ParameterSet) -> ParameterSet:
    """Sample one patient parameter set with guarded distributions."""
    p = base.copy()

    # Insulin sensitivity factors: CV = 30% (implemented as true coefficient of variation)
    p["kb1"] = _sample_lognormal_with_cv(rng, base["kb1"], _CV_KB)
    p["kb2"] = _sample_lognormal_with_cv(rng, base["kb2"], _CV_KB)
    p["kb3"] = _sample_lognormal_with_cv(rng, base["kb3"], _CV_KB)

    # Glucose parameters (truncated normal, no abs-folding)
    p["EGP0"] = _sample_truncated_normal(rng, 0.0161, 0.0039)
    p["F01"] = _sample_truncated_normal(rng, 0.0097, 0.0022)
    p["k12"] = _sample_truncated_normal(rng, 0.0649, 0.0282)

    # Activation rates (ka)
    p["ka1"] = _sample_truncated_normal(rng, 0.0055, 0.0056)
    p["ka2"] = _sample_truncated_normal(rng, 0.0683, 0.0507)
    p["ka3"] = _sample_truncated_normal(rng, 0.0304, 0.0235)

    # Insulin sensitivities (cannot be negative)
    p["SI1"] = _sample_truncated_normal(rng, 51.2, 32.09, lower=0.1)
    p["SI2"] = _sample_truncated_normal(rng, 8.2, 7.84, lower=0.1)
    p["SI3"] = _sample_truncated_normal(rng, 520.0, 306.2, lower=0.1)

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

    # Carbohydrate bioavailability and body weight
    p["Ag"] = float(rng.uniform(0.7, 1.2))
    p["BW"] = float(rng.uniform(65.0, 95.0))

    return p


def generate_monte_carlo_patients(
    n: int = 10,
    standard_patient: bool = False,
    seed: Optional[int] = None,
) -> List[ParameterSet]:
    """
    Generate N patient parameter sets.

    - `standard_patient=True` returns base parameters duplicated N times.
    - Otherwise, sampled parameters are resampled until plausibility checks pass.
    """
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)
    base = get_base_params()
    patients: List[ParameterSet] = []

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


# ============================================================================
# LEGACY / UNUSED (Gemini reference helpers)
# ============================================================================
# Kept for historical comparison only. Main pipeline should use get_base_params
# and generate_monte_carlo_patients above.

def get_base_params_gemini(bw: float = 70.0) -> ParameterSet:
    """Legacy Gemini baseline parameters (unused by current main pipeline)."""
    return {
        "MwG": 180.16,
        "BW": float(bw),
        "EGP0": 16.9,
        "F01": 9.7,
        "k12": 0.066,
        "VG": 0.16 * float(bw),
        "kb1": 14.6e-4 * 60,
        "kb2": 8.2e-4 * 60,
        "kb3": 52e-5 * 60,
        "ka1": 0.006 * 60,
        "ka2": 0.06 * 60,
        "ka3": 0.03 * 60,
        "ke": 0.138,
        "VI": 0.12 * float(bw),
        "tauI": 55.0,
        "tauG": 40.0,
        "Ag": 0.8,
    }


def generate_monte_carlo_patients_gemini(
    n: int = 10,
    seed: Optional[int] = None,
) -> List[ParameterSet]:
    """Legacy Gemini patient generator (unused by current main pipeline)."""
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)
    patients: List[ParameterSet] = []
    base = get_base_params_gemini()

    for _ in range(n):
        p = base.copy()
        p["kb1"] = _sample_lognormal_with_cv(rng, base["kb1"], _CV_KB)
        p["kb2"] = _sample_lognormal_with_cv(rng, base["kb2"], _CV_KB)
        p["kb3"] = _sample_lognormal_with_cv(rng, base["kb3"], _CV_KB)
        p["tauG"] = _sample_truncated_normal(rng, base["tauG"], 5.0)
        patients.append(p)

    return patients