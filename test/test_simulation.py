# Test script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation import run_simulation
from src.export import ExportConfig
from src.simulation import SimulationConfig
from test_sensitivity import find_sensitivities

if __name__ == "__main__":

    # Find sensitivities
    icr, isf = find_sensitivities(print_progress=True)

    # Simulation configuration
    test_sim_config = SimulationConfig(
        n_patients=10,
        n_days=1,
        international_unit=True,
        noise_std=0.10,
        noise_autocorr=0.7,  # Temporal correlation
        random_scenarios=False,
        clip_states=True,
        std_patient=True,
        insulin_sensitivity_factor=isf,
        insulin_carbohydrates_ratio=icr,
        random_seed=99921  # For reproducibility (change for different cohorts)
    )
    
    # Export configuration
    export_config = ExportConfig(
        export_to_parquet=True,
        export_to_csv=True
    )

    run_simulation(test_sim_config, export_config)