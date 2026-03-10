import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parameters import get_base_params
from src.simulation import find_icr, find_isf
from src.model import ParameterSet

def find_sensitivities(p: ParameterSet=get_base_params(), cho_grams: float=50.0, print_progress: bool=False) -> tuple[float, float]:
    """
    Find ICR and ISF for a given parameter set.

    Parameters:
    -----------
    p: ParameterSet
        Parameter set for the patient.
    cho_grams: float
        Carbohydrate grams to use for ICR calculation.
    print_progress: bool
        Whether to print progress during calculation.
    
    Returns:
    --------
    icr: float
        Insulin-to-carbohydrate ratio.
    isf: float
        Insulin sensitivity factor.
    """
    # Find ICR
    icr = find_icr(p, cho_grams=cho_grams, print_progress=print_progress)
    # Find ISF
    isf = find_isf(p, print_progress=print_progress)
    # Print results
    print(f'ICR object: {icr}')
    print(f'ISF object: {isf}')
    # Return the values
    return icr["icr_g_per_U"], isf["isf_mmol_per_U"]


if __name__ == "__main__":
    icr, isf = find_sensitivities(print_progress=True)
    print(f'ICR value: {icr}')
    print(f'ISF value: {isf}')
    