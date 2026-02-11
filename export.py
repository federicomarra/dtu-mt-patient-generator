import parquet

def export_to_parquet(df, n_patients, output_path):
    """
    Exports a list to a Parquet file.

    Parameters:
    list (List): The DataFrame to export.
    n_patients (str): The path where the Parquet file will be saved.
    """
    output_path = f"monte_carlo_results_{n_patients}_patients.parquet"
    try:
        df.to_parquet(output_path, index=False)
        print(f"Data successfully exported to {output_path}")
    except Exception as e:
        print(f"An error occurred while exporting to Parquet: {e}")