import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import compute_optimal_steady_state_from_glucose, get_glucose_from_state
from src.parameters import get_base_params


def _assert_close(actual: float, expected: float, tolerance: float) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Steady-state glucose mismatch: expected {expected:.4f}, got {actual:.4f}, tolerance {tolerance:.4f}"
        )


if __name__ == "__main__":
    params = get_base_params()
    tolerance = 1.0

    for target in (6.0,):
        state = compute_optimal_steady_state_from_glucose(
            params=params,
            desired_glycemia=target,
            international_units=True,
            max_iterations=30,
            print_progress=False,
        )
        glucose = get_glucose_from_state(state, params)
        _assert_close(glucose, target, tolerance)
        print(f"target={target:.2f} mmol/L -> steady_state={glucose:.4f} mmol/L")

    print("Steady-state Newton solver checks passed.")