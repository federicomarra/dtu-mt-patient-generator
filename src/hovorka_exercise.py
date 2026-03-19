from __future__ import annotations

from typing import TypedDict


class RashidTerms(TypedDict):
    dE1: float
    dE2: float
    dTE: float
    QE1: float
    QE21: float
    QE22: float



def compute_rashid_terms(
    E1: float,
    E2: float,
    TE: float,
    delta_hr_t: float,
    x1: float,
    x2: float,
    Q1: float,
    Q2: float,
    params: dict[str, float],
) -> RashidTerms:
    """Compute Exercise Model (A.16-A.19) and exercise glucose terms (A.23-A.24)."""
    tau_HR = max(1.0, float(params["rashid_tau_hr"]))
    tau_ex = max(1.0, float(params["rashid_tau_ex"]))
    tau_in = max(1.0, float(params["rashid_tau_in"]))
    c1 = float(params["rashid_c1"])
    c2 = float(params["rashid_c2"])
    alpha = max(1e-6, float(params["rashid_alpha"]))
    beta = max(0.0, float(params["rashid_beta"]))
    HR0 = max(1e-6, float(params["rashid_hr0"]))
    n = max(1.0, float(params["rashid_n"]))

    # A.16 with HR(t) = HR0 + delta_hr_t from the scenario input.
    HR_t = HR0 + max(0.0, delta_hr_t)
    dE1 = (HR_t - HR0 - E1) / tau_HR

    # A.19: fE1(E1) = (E1/(alpha*HR0))^n / (1 + (E1/(alpha*HR0))^n)
    e1_scale = max(1e-6, alpha * HR0)
    e1_ratio = max(0.0, E1) / e1_scale
    f_e1_num = e1_ratio**n
    f_e1 = f_e1_num / (1.0 + f_e1_num)

    # A.17 with explicit resting offset c2.
    dTE = (c1 * f_e1 + c2 - TE) / tau_ex
    TE_safe = max(1e-6, TE)

    # A.18 (TE is clamped only to avoid numeric division by zero).
    dE2 = -((f_e1 / tau_in) + (1.0 / TE_safe)) * E2 + (f_e1 * TE_safe) / max(1e-6, c1 + c2)

    # A.23-A.24 exercise terms in glucose dynamics.
    e2_pos = max(0.0, E2)
    QE1 = beta * max(0.0, E1) / HR0
    QE21 = alpha * (e2_pos**2) * max(0.0, x1) * max(0.0, Q1)
    QE22 = alpha * (e2_pos**2) * max(0.0, x2) * max(0.0, Q2)

    return {
        "dE1": dE1,
        "dE2": dE2,
        "dTE": dTE,
        "QE1": QE1,
        "QE21": QE21,
        "QE22": QE22,
    }
