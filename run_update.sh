#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# 1) Ingest any new PBS Excel files found in ./raw into DuckDB
python3 pbs_ingest_duckdb.py --input_dir "./raw" --output_dir "./out"

# 2) Export the wide AEMP-by-month matrix (CSV + Excel)
python3 pbs_export_fixed_wide.py --db "./out/pbs_prices.duckdb" --output_dir "./out" --xlsx

echo "✅ Done. Open: $HOME/Downloads/KMC/out/aemp_fixed_wide.xlsx"
