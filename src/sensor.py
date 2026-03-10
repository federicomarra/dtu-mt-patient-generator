from __future__ import annotations

from typing import Mapping, MutableMapping, Optional, SupportsFloat

import numpy as np

SensorMode = str
SensorState = MutableMapping[str, float]


def _to_float(value: SupportsFloat, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def _get_param(params: Mapping[str, SupportsFloat], key: str) -> float:
    if key not in params:
        raise ValueError(f"Missing required parameter: {key}")
    return _to_float(params[key], key)


def _convert_units(glucose_mmol_l: float, params: Mapping[str, SupportsFloat], output_unit: str) -> float:
    if output_unit == "mmol/L":
        return glucose_mmol_l
    if output_unit == "mg/dL":
        mwg = _get_param(params, "MwG")  # [mg/mmol]
        return glucose_mmol_l * (mwg / 10.0)
    raise ValueError(f"Unsupported output_unit: {output_unit!r}. Use 'mmol/L' or 'mg/dL'.")


def measure_glycemia(
    x: np.ndarray | list[SupportsFloat] | tuple[SupportsFloat, ...],
    params: Mapping[str, SupportsFloat],
    noise_std: float = 0.0,
    *,
    mode: SensorMode = "gaussian",
    bias: float = 0.0,
    phi: float = 0.85,
    lag_alpha: float = 0.25,
    sensor_state: Optional[SensorState] = None,
    rng: Optional[np.random.Generator] = None,
    output_unit: str = "mmol/L",
    min_glucose: float = 0.0,
) -> float:
    """
    Measure blood glucose from model state with configurable sensor behavior.

    Parameters:
    - x: model state vector (x[0] = Q1 [mmol])
    - params: model parameters containing at least VG, BW (and MwG if mg/dL output)
    - noise_std: standard deviation of sensor noise in output units
    - mode:
        - "none": no noise
        - "gaussian": white Gaussian noise
        - "bias_gaussian": constant bias + white Gaussian noise
        - "lagged": first-order lag + AR(1)-like correlated error
    - bias: additive bias in output units (used by "bias_gaussian" and "lagged")
    - phi: AR(1) coefficient for correlated error in "lagged" mode (0 <= phi < 1)
    - lag_alpha: lag blend factor in "lagged" mode (0 < lag_alpha <= 1)
    - sensor_state: mutable dict preserving error/lag states across calls
    - rng: optional Generator for reproducible trajectories (recommended)
    - output_unit: "mmol/L" or "mg/dL"
    - min_glucose: lower clamp for physiologically impossible negative readings
    """
    if len(x) == 0:
        raise ValueError("x must contain at least one state value (Q1 at index 0)")

    q1 = _to_float(x[0], "x[0]")  # [mmol]
    vg = _get_param(params, "VG")  # [L/kg]
    bw = _get_param(params, "BW")  # [kg]

    if vg <= 0.0:
        raise ValueError(f"VG must be > 0, got {vg}")
    if bw <= 0.0:
        raise ValueError(f"BW must be > 0, got {bw}")

    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}")
    if min_glucose < 0:
        raise ValueError(f"min_glucose must be >= 0, got {min_glucose}")

    if mode not in {"none", "gaussian", "bias_gaussian", "lagged"}:
        raise ValueError(
            f"Unsupported mode: {mode!r}. Use 'none', 'gaussian', 'bias_gaussian', or 'lagged'."
        )

    if not (0.0 <= phi < 1.0):
        raise ValueError(f"phi must be in [0, 1), got {phi}")
    if not (0.0 < lag_alpha <= 1.0):
        raise ValueError(f"lag_alpha must be in (0, 1], got {lag_alpha}")

    local_rng = rng if rng is not None else np.random.default_rng()

    true_glucose_mmol_l = q1 / (vg * bw)
    true_glucose = _convert_units(true_glucose_mmol_l, params, output_unit)

    if mode == "none" or noise_std == 0.0:
        measured = true_glucose + (bias if mode in {"bias_gaussian", "lagged"} else 0.0)
        return max(min_glucose, measured)

    if mode == "gaussian":
        measured = true_glucose + float(local_rng.normal(0.0, noise_std))
        return max(min_glucose, measured)

    if mode == "bias_gaussian":
        measured = true_glucose + bias + float(local_rng.normal(0.0, noise_std))
        return max(min_glucose, measured)

    if sensor_state is None:
        sensor_state = {}

    prev_display = float(sensor_state.get("display", true_glucose))
    prev_error = float(sensor_state.get("error", 0.0))

    lagged_true = prev_display + lag_alpha * (true_glucose - prev_display)
    innovation_std: float = float(noise_std * np.sqrt(max(0.0, 1.0 - phi * phi)))
    innovation: float = float(local_rng.normal(loc=0.0, scale=innovation_std))
    correlated_error = (phi * prev_error) + innovation

    measured = lagged_true + bias + correlated_error
    measured = max(min_glucose, measured)

    sensor_state["display"] = measured
    sensor_state["error"] = correlated_error
    return measured
