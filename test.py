# Test script

from src.simulation import run_simulation
from src.export import ExportConfig
from src.simulation import SimulationConfig

if __name__ == "__main__":
    # Simulation configuration
    test_sim_config = SimulationConfig(
        n_patients=10,
        n_days=1,
        international_unit=True,
        noise_std=0.10,
        noise_autocorr=0.7,  # Temporal correlation
        random_scenarios=False,
        clip_states=True,
        random_seed=99921  # For reproducibility (change for different cohorts)
    )
    
    # Export configuration
    export_config = ExportConfig(
        export_to_parquet=True,
        export_to_csv=True
    )

    run_simulation(test_sim_config, export_config)