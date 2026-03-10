# Export file functions

# Library imports
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Mapping, Sequence, SupportsFloat, TypedDict, cast, Any

# File imports
PatientId = int | str
DayValues = Mapping[int, Sequence[SupportsFloat] | np.ndarray]

@dataclass
class ExportConfig:
    """Export configuration with named parameters."""
    export_to_parquet: bool = True
    export_to_csv: bool = False
    
    def to_list(self) -> list[bool]:
        """Convert to list format for backward compatibility."""
        return [self.export_to_parquet, self.export_to_csv]

class PatientData(TypedDict):
    days: DayValues | None


ResultsDict = Mapping[PatientId, PatientData]


def _validate_results_dict(results_dict: object) -> ResultsDict:
    """
    Validates the structure of results_dict.
    
    Raises ValueError if structure is malformed.
    """
    if not isinstance(results_dict, Mapping):
        raise ValueError(f"results_dict must be a dict, got {type(results_dict)}")
    
    mapping_obj = cast(Mapping[object, object], results_dict)

    for p_id, p_data in mapping_obj.items():
        if not isinstance(p_data, dict):
            raise ValueError(f"Patient {p_id}: p_data must be dict, got {type(p_data)}")
        if "days" not in p_data:
            raise ValueError(f"Patient {p_id}: missing 'days' key")
        if p_data["days"] is not None:
            days_obj = cast(object, p_data["days"])
            if not isinstance(days_obj, Mapping):
                raise ValueError(f"Patient {p_id}: 'days' must be dict or None, got {type(days_obj)}")

    return cast(ResultsDict, results_dict)


def _minutes_to_clock_strings(absolute_minutes: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of absolute minutes into HH:MM clock strings.
    """
    minute_of_day = absolute_minutes % 1440
    hours = minute_of_day // 60
    mins = minute_of_day % 60

    hour_str = np.char.zfill(hours.astype(str), 2)
    min_str = np.char.zfill(mins.astype(str), 2)
    return np.char.add(np.char.add(hour_str, ":"), min_str)


def _format_patient_id(p_id: PatientId) -> str:
    try:
        p_id_int = int(p_id) if not isinstance(p_id, int) else p_id
        return f"{p_id_int:06d}"
    except (ValueError, TypeError):
        return str(p_id)


def _flatten_results(results_dict: ResultsDict) -> pd.DataFrame:
    """
    Flattens nested results dict into a single DataFrame.

    Uses block-wise DataFrame creation per patient/day to avoid per-row appends.
    Includes both `minute` (within day) and `absolute_minute` (global timeline).
    """
    blocks: List[pd.DataFrame] = []
    
    for p_id, p_data in results_dict.items():
        days = p_data.get("days")
        if days is None:
            continue
        p_id_str = _format_patient_id(p_id)
        
        for day, values in days.items():

            values_arr = np.asarray(values)
            if values_arr.size == 0:
                continue

            day_int = int(day)
            minutes = np.arange(values_arr.size, dtype=int)
            absolute_minutes = (day_int * 1440) + minutes
            times = _minutes_to_clock_strings(absolute_minutes)

            blocks.append(pd.DataFrame({
                "patient_id": p_id_str,
                "day": day_int,
                "minute": minutes,
                "absolute_minute": absolute_minutes,
                "time": times.tolist(),
                "blood_glucose": values_arr.astype(float)
            }))

    if not blocks:
        return pd.DataFrame(
            columns=["patient_id", "day", "minute", "absolute_minute", "time", "blood_glucose"]
        )

    df = pd.concat(blocks, ignore_index=True)
    
    # Normalize absolute_minute to start from 0
    if not df.empty:
        min_absolute = df["absolute_minute"].min()
        df["absolute_minute"] = df["absolute_minute"] - min_absolute
    
    return df


# Export functions
def export_to_formats(
    results_dict: object,
    n_patients: int,
    n_days: int,
    output_folder: Path,
    export: Optional[List[bool]] = None,
    config_metadata: Optional[dict[str, Any]] = None
) -> None:
    """
    Exports simulation results to both Parquet and CSV files with optional metadata.

    Parameters:
    results_dict (dict): The nested dictionary containing patient results.
    n_patients (int): Number of patients simulated.
    n_days (int): Number of days simulated.
    output_folder (Path): The directory where the files will be saved.
    export (list[bool] | None): [export_parquet, export_csv]. Defaults to [True, False].
    config_metadata (dict | None): Configuration parameters to log alongside results.
    
    Raises:
    ValueError: If results_dict has invalid structure.
    OSError: If output directory cannot be created.
    """
    # Handle mutable default argument
    if export is None:
        export = [True, False]
    
    # Validate export flags
    if len(export) != 2:
        raise ValueError(f"export must be a 2-element list [parquet, csv], got {export}")
    
    export_parquet, export_csv = export
    
    # Validate input structure
    try:
        validated_results_dict = _validate_results_dict(results_dict)
    except ValueError as e:
        raise ValueError(f"Invalid results_dict structure: {e}")
    
    # Ensure output directory exists
    output_path = Path(output_folder)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output directory {output_path}: {e}")
    
    # Flatten results into DataFrame
    df = _flatten_results(validated_results_dict)

    if df.empty:
        print("Warning: No records found in results_dict; skipping export.")
        return
    
    # Define file names and paths
    base_file_name = f"results_{n_patients}p_{n_days}d"
    parquet_path = output_path / f"{base_file_name}.parquet"
    csv_path = output_path / f"{base_file_name}.csv"
    
    # Export to Parquet
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
            
            # Write config metadata to separate file if provided
            if config_metadata:
                config_path = output_path / f"config_{n_patients}p_{n_days}d.txt"
                try:
                    with open(config_path, 'w') as f:
                        f.write("=== Simulation Configuration ===\n\n")
                        for key, value in sorted(config_metadata.items()):
                            f.write(f"{key}: {value}\n")
                    print(f"Configuration metadata saved to {config_path}")
                except Exception as meta_e:
                    print(f"Warning: Failed to write config metadata: {meta_e}")
        except Exception as e:
            print(f"An error occurred while exporting to CSV: {e}")