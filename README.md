# pbs-pricing

Simple pipeline to turn monthly PBS Excel releases into a clean **price-history matrix** (AEMP by month).

## What’s in here
- `pbs_ingest_duckdb.py` — loads raw PBS `.xlsx` files into a DuckDB file → `out/pbs_prices.duckdb`
- `pbs_export_fixed_wide.py` — exports a wide **AEMP per month** matrix → `out/aemp_fixed_wide.csv` (and optionally `.xlsx`)
- `archive/pbs_export_wide.py` — old exporter (kept for reference)

## Quick start

### Requirements (one-time)
```bash
pip install duckdb pandas openpyxl et-xmlfile

