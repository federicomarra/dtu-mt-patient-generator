# Hovorka Model Monte Carlo Simulation
# Main script

from src.simulation import run_simulation
from src.export import ExportConfig
from src.simulation import SimulationConfig

if __name__ == "__main__":
    # Configuration
    skip_terminal_input: bool = True
    
    # Simulation configuration
    sim_config = SimulationConfig(
        n_patients=50,
        n_days=5,
        international_unit=True,
        noise_std=0.10,
        noise_autocorr=0.7,  # Temporal correlation
        random_scenarios=True,
        clip_states=True,
        random_seed=999  # For reproducibility (change for different cohorts)
    )
    
    # Export configuration
    export_config = ExportConfig(
        export_to_parquet=True,
        export_to_csv=False
    )
    
    # Interactive input (if enabled)
    if not skip_terminal_input:
        unit_answer: str = input("Use international units, mmol/L instead of mg/dL? (y/n): ").strip().lower()
        sim_config.international_unit = unit_answer not in {"n", "no", "0", "false", "f"}
        
        sim_config.n_patients = int(input("Number of patients to simulate: "))
        sim_config.n_days = int(input("Number of days per patient: "))
        
        noise_answer = input(f"CGM noise std (mmol/L, default {sim_config.noise_std}): ").strip()
        if noise_answer:
            sim_config.noise_std = float(noise_answer)
    
    # Run simulation
    run_simulation(config=sim_config, export_config=export_config)