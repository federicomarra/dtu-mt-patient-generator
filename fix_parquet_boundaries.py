#!/usr/bin/env python3
"""
Remove day-boundary duplicate rows from a simulation parquet file.

Each day was simulated with np.arange(0, 1441) — 1441 points — so the
last row of day N (absolute_minute=(N+1)*1440) is the same physiological
state as the first row of day N+1. This script drops those trailing rows,
leaving exactly 1440 rows per (patient, day).

Streams row-group by row-group: constant ~MB of RAM regardless of file size.

Usage:
    python fix_parquet_boundaries.py <input.parquet> [output.parquet]

If output is omitted, writes to <input>_clean.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]


def fix_boundaries(input_path: Path, output_path: Path) -> None:
    pf = pq.ParquetFile(input_path)
    # cast() pins pyarrow return values to known types; pyarrow has no stubs.
    n_groups = cast(int, pf.metadata.num_row_groups)  # type: ignore[union-attr]
    total_in = cast(int, pf.metadata.num_rows)         # type: ignore[union-attr]
    schema   = pf.schema_arrow                         # type: ignore[union-attr]

    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Row groups: {n_groups}   Total rows in: {total_in:,}")
    print(f"Expected rows out: {(total_in // 1441) * 1440:,}  "
          f"(removing 1 row per patient-day)")
    print()

    writer = None
    total_written = 0
    total_dropped = 0

    for rg_idx in range(n_groups):
        # Read as pandas so the filter expression is fully typed.
        chunk = cast(pd.DataFrame, pf.read_row_group(rg_idx).to_pandas())  # type: ignore[union-attr]

        # Drop the trailing boundary row of each day:
        # absolute_minute == (day + 1) * 1440 identifies the last row of day N,
        # which duplicates the first row of day N+1.
        keep: pd.Series = chunk["absolute_minute"] != (chunk["day"] + 1) * 1440
        filtered: pd.DataFrame = chunk[keep]

        dropped = len(chunk) - len(filtered)
        total_dropped += dropped
        total_written += len(filtered)

        # Convert back to Arrow for writing, preserving the original schema.
        table = pa.Table.from_pandas(filtered, schema=schema, preserve_index=False)  # type: ignore[attr-defined]
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema, compression="snappy")  # type: ignore[arg-type]
        writer.write_table(table)  # type: ignore[union-attr]

        if (rg_idx + 1) % 20 == 0 or rg_idx == n_groups - 1:
            print(f"  [{rg_idx+1:>3}/{n_groups}]  written {total_written:>12,}  dropped {total_dropped:>8,}",
                  flush=True)

    if writer:
        writer.close()  # type: ignore[union-attr]

    expected_out: int = (total_in // 1441) * 1440
    print(f"\nDone.")
    print(f"  Rows in:      {total_in:,}")
    print(f"  Rows dropped: {total_dropped:,}")
    print(f"  Rows out:     {total_written:,}")

    if total_written == expected_out:
        print(f"  Check: OK — exactly {total_written // 1440:,} patient-days × 1440 min")
    else:
        print(f"  Check: WARNING — expected {expected_out:,}, got {total_written:,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_parquet_boundaries.py <input.parquet> [output.parquet]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_stem(input_path.stem + "_clean")

    if output_path.exists():
        print(f"Error: {output_path} already exists — delete it first to avoid accidents")
        sys.exit(1)

    fix_boundaries(input_path, output_path)
