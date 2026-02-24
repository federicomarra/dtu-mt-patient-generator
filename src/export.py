import pandas as pd
from pathlib import Path

def export_to_formats(results_dict, n_patients, output_folder, export: list[bool] = [True, False]):
    """
    Exports simulation results to both Parquet and CSV files.

    Parameters:
    results_dict (dict): The nested dictionary containing patient results.
    n_patients (int): Number of patients simulated.
    output_folder (Path): The directory where the files will be saved.
    """
    records = []
    for p_id, p_data in results_dict.items():
        if p_data["days"] is None:
            continue
        for day, values in p_data["days"].items():
            # values is a numpy array of BG for each minute
            for minute, bg in enumerate(values):
                records.append({
                    "patient_id": p_id,
                    "day": day,
                    "minute": minute,
                    "blood_glucose": bg
                })
    
    df = pd.DataFrame(records)
    base_file_name = f"results_{n_patients}_patients"
    parquet_path = Path(output_folder) / f"{base_file_name}.parquet"
    csv_path = Path(output_folder) / f"{base_file_name}.csv"
    
    # Export to Parquet
    export_parquet, export_csv = export
    if export_parquet:
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"Data successfully exported in parquet format to {parquet_path}")
        except Exception as e:
            print(f"An error occurred while exporting to Parquet: {e}")

    # Export to CSV
    if export_csv:
        try:
            df.to_csv(csv_path, index=False)
            print(f"Data successfully exported in csv format to {csv_path}")
        except Exception as e:
            print(f"An error occurred while exporting to CSV: {e}")