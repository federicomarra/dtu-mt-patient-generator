# Export file functions

# Library imports
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Mapping, Sequence, SupportsFloat, TypedDict, cast, Any

# File imports
PatientId = int | str
DaySeries = Sequence[SupportsFloat] | np.ndarray
DayRecord = Mapping[str, DaySeries]
DayValues = Mapping[int, DaySeries | DayRecord]

@dataclass
class ExportConfig:
    """Export configuration with named parameters."""
    export_to_parquet: bool = True
    export_to_csv: bool = False
    
    def to_list(self) -> list[bool]:
        """Convert to list format for backward compatibility."""
        return [self.export_to_parquet, self.export_to_csv]

class PatientData(TypedDict, total=False):
    days: DayValues | None
    params: Mapping[str, SupportsFloat] | Mapping[str, float]


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


def _align_label_list(raw: Any, default: Any, n: int) -> list[Any]:
    """Align a per-minute label list to exactly n elements, padding with default."""
    if raw is None:
        return [default] * n
    lst: list[Any] = list(raw)
    if len(lst) >= n:
        return lst[:n]
    return lst + [default] * (n - len(lst))


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
        params_obj = p_data.get("params")
        patient_age_years = np.nan
        if isinstance(params_obj, Mapping):
            age_raw = params_obj.get("age_years")
            if age_raw is not None:
                try:
                    patient_age_years = float(age_raw)
                except (TypeError, ValueError):
                    patient_age_years = np.nan
        
        for day, values in days.items():

            insulin_arr: np.ndarray
            cho_arr: np.ndarray
            scenario_id: Optional[int]
            missed_meal_id: Optional[int]
            late_bolus_id: Optional[int]
            late_bolus_ids_val: list[int]
            if isinstance(values, Mapping):
                values_arr = np.asarray(values.get("blood_glucose", []), dtype=np.float64)
                # Backward-compatible read: accept both legacy and explicit-unit keys.
                insulin_arr = np.asarray(
                    values.get("insulin_mU_min", values.get("insulin", [])),
                    dtype=np.float64,
                )
                cho_arr = np.asarray(
                    values.get("cho_mg_min", values.get("cho", [])),
                    dtype=np.float64,
                )
                # Deprecated scalar labels — kept for backward compat with analysis scripts.
                scenario_id = values.get("scenario_id", None)  # type: ignore[assignment]
                missed_meal_id = values.get("missed_meal_id", None)  # type: ignore[assignment]
                late_bolus_id = values.get("late_bolus_id", None)  # type: ignore[assignment]
                _raw_ids = values.get("late_bolus_ids", None)
                late_bolus_ids_val = list(int(x) for x in _raw_ids) if _raw_ids else []  # type: ignore[assignment]
                # New per-day metadata
                base_scenario_val = values.get("base_scenario", None)
                had_large_meal_val = values.get("had_large_meal", values.get("had_restaurant", None))
                had_missed_bolus_val = values.get("had_missed_bolus", None)
                n_late_boluses_val = values.get("n_late_boluses", None)
                exercise_overlay_val = values.get("exercise_overlay", None)
                # Per-minute ML label arrays (may be absent in legacy data)
                _bolus_status_raw = values.get("bolus_status", None)
                _meal_size_raw = values.get("meal_size", None)
                _exercise_type_raw = values.get("exercise_type", None)
            else:
                values_arr = np.asarray(values, dtype=np.float64)
                insulin_arr = np.full(values_arr.size, np.nan, dtype=np.float64)
                cho_arr = np.full(values_arr.size, np.nan, dtype=np.float64)
                scenario_id = None
                missed_meal_id = None
                late_bolus_id = None
                late_bolus_ids_val = []
                base_scenario_val = None
                had_large_meal_val = None
                had_missed_bolus_val = None
                n_late_boluses_val = None
                exercise_overlay_val = None
                _bolus_status_raw = None
                _meal_size_raw = None
                _exercise_type_raw = None

            if values_arr.size == 0:
                continue

            # Align optional input arrays to glucose length.
            if insulin_arr.size != values_arr.size:
                insulin_aligned = np.full(values_arr.size, np.nan, dtype=np.float64)
                common = min(insulin_arr.size, values_arr.size)
                if common > 0:
                    insulin_aligned[:common] = insulin_arr[:common]
                insulin_arr = insulin_aligned

            if cho_arr.size != values_arr.size:
                cho_aligned = np.full(values_arr.size, np.nan, dtype=np.float64)
                common = min(cho_arr.size, values_arr.size)
                if common > 0:
                    cho_aligned[:common] = cho_arr[:common]
                cho_arr = cho_aligned

            day_int = int(day)
            minutes = np.arange(values_arr.size, dtype=int)
            absolute_minutes = (day_int * 1440) + minutes
            times = _minutes_to_clock_strings(absolute_minutes)

            # Align per-minute ML label arrays to glucose length (pad with None/'none').
            n = values_arr.size
            bolus_status_col = _align_label_list(_bolus_status_raw, None, n)
            meal_size_col = _align_label_list(_meal_size_raw, None, n)
            exercise_type_col = _align_label_list(_exercise_type_raw, "none", n)

            blocks.append(pd.DataFrame({
                "patient_id": p_id_str,
                "patient_age_years": patient_age_years,
                "day": day_int,
                "minute": minutes,
                "absolute_minute": absolute_minutes,
                "time": times.tolist(),
                "blood_glucose": values_arr.astype(float),
                "cho_mg_min": cho_arr.astype(float),
                "insulin_mU_min": insulin_arr.astype(float),
                # Day-level scenario metadata (scalar broadcast to all rows).
                "base_scenario": base_scenario_val,
                "had_large_meal": had_large_meal_val,
                "had_missed_bolus": had_missed_bolus_val,
                "n_late_boluses": n_late_boluses_val,
                "exercise_overlay": exercise_overlay_val,
                # Per-minute ML labels.
                "bolus_status": bolus_status_col,
                "meal_size": meal_size_col,
                "exercise_type": exercise_type_col,
                # Deprecated scalar labels — kept for backward compat with analysis scripts.
                "scenario_id": scenario_id,
                "missed_meal_id": missed_meal_id,
                "late_bolus_id": late_bolus_id,
                "late_bolus_ids": [late_bolus_ids_val] * n,
            }))

    if not blocks:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "patient_age_years",
                "day",
                "minute",
                "absolute_minute",
                "time",
                "blood_glucose",
                "cho_mg_min",
                "insulin_mU_min",
                "base_scenario",
                "had_large_meal",
                "had_missed_bolus",
                "n_late_boluses",
                "exercise_overlay",
                "bolus_status",
                "meal_size",
                "exercise_type",
                "scenario_id",
                "missed_meal_id",
                "late_bolus_id",
                "late_bolus_ids",
            ]
        )

    df = pd.concat(blocks, ignore_index=True)
    
    # Normalize absolute_minute to start from 0
    if not df.empty:
        min_absolute = df["absolute_minute"].min()
        df["absolute_minute"] = df["absolute_minute"] - min_absolute
    
    return df


def _validate_parquet_output(parquet_path: Path, expected_rows: int) -> None:
    """Validate a parquet file footer and metadata after write."""
    import pyarrow.parquet as pq

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file was not created: {parquet_path}")
    if parquet_path.stat().st_size == 0:
        raise ValueError(f"Parquet file is empty: {parquet_path}")

    # Quick footer guard: valid parquet files end with PAR1.
    with parquet_path.open("rb") as fh:
        fh.seek(-4, os.SEEK_END)
        footer = fh.read(4)
    if footer != b"PAR1":
        raise ValueError(
            f"Parquet footer marker missing for {parquet_path}; file may be truncated"
        )

    metadata = pq.read_metadata(parquet_path)
    if metadata.num_rows != expected_rows:
        raise ValueError(
            f"Parquet row mismatch for {parquet_path}: "
            f"expected {expected_rows}, got {metadata.num_rows}"
        )


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
    parquet_tmp = output_path / f"{base_file_name}.parquet.tmp"
    csv_tmp = output_path / f"{base_file_name}.csv.tmp"
    
    # Export to Parquet
    if export_parquet:
        if parquet_tmp.exists():
            parquet_tmp.unlink()
        try:
            df.to_parquet(parquet_tmp, index=False)
            _validate_parquet_output(parquet_tmp, expected_rows=len(df))
            parquet_tmp.replace(parquet_path)
            print(f"Data successfully exported in parquet format to {parquet_path}")
        except Exception as e:
            if parquet_tmp.exists():
                parquet_tmp.unlink()
            raise RuntimeError(f"Parquet export failed: {e}") from e

    # Export to CSV
    if export_csv:
        if csv_tmp.exists():
            csv_tmp.unlink()
        try:
            df.to_csv(csv_tmp, index=False)
            if csv_tmp.stat().st_size == 0:
                raise ValueError(f"CSV file is empty: {csv_tmp}")
            csv_tmp.replace(csv_path)
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
            if csv_tmp.exists():
                csv_tmp.unlink()
            raise RuntimeError(f"CSV export failed: {e}") from e