#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "== 1) Ingest PBS AEMP (raw -> out/pbs_prices.duckdb) =="
python3 pbs_ingest_duckdb.py --input_dir "./raw" --output_dir "./out"

echo "== 2) Export PBS AEMP wide matrix (out/pbs_prices.duckdb -> out/) =="
python3 pbs_export_fixed_wide.py --db "./out/pbs_prices.duckdb" --output_dir "./out" --xlsx

echo "== 3) Ingest Chemo EFC (raw_chemo -> out_chemo/chemo_prices.duckdb) =="
python3 chemo_ingest_duckdb.py --input_dir "./raw_chemo" --output_dir "./out_chemo"

echo "✅ Done."
echo "PBS DB:   $(pwd)/out/pbs_prices.duckdb"
echo "CHEMO DB: $(pwd)/out_chemo/chemo_prices.duckdb"
echo "WIDE XLSX: $(pwd)/out/aemp_fixed_wide.xlsx"
