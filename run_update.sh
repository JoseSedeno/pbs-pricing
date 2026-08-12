#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DB="./out/pbs_prices.duckdb"

# 1) Import the new PBS monthly file
python3 pbs_ingest_duckdb.py \
  --input_dir "./raw" \
  --output_dir "./out"

# Verify the database was actually produced and is non-empty
if [ ! -s "$DB" ]; then
  echo "ERROR: $DB was not created or is empty. Aborting before export." >&2
  exit 1
fi

# 2) Rebuild wide_fixed / wide_fixed_meta
python3 pbs_export_fixed_wide.py \
  --db "$DB" \
  --output_dir "./out" \
  --xlsx

# 3) Verify the database still exists after export
if [ ! -s "$DB" ]; then
  echo "ERROR: $DB missing after export step." >&2
  exit 1
fi

echo ""
echo "PBS update complete."
echo "Upload this file to Google Drive as a new version:"
echo "$(pwd)/out/pbs_prices.duckdb"
