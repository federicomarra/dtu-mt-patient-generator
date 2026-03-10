# Test sensitivity script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensitivity import find_sensitivities

if __name__ == "__main__":
    icr, isf = find_sensitivities(print_progress=True)
    print(f'ICR value: {icr}')
    print(f'ISF value: {isf}')
    