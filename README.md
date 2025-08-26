# pbs-pricing

Simple pipeline to turn monthly PBS Excel releases into a clean price-history
matrix you can analyze or share.

## What’s in here
- `pbs_ingest_duckdb.py` — loads raw PBS `.xlsx` files into a DuckDB file
  (`out/pbs_prices.duckdb`).
- `pbs_export_fixed_wide.py` — exports a wide **AEMP per month** matrix to CSV/Excel.
- `archive/pbs_export_wide.py` — old exporter (kept for reference).

---

## Quick start

### 0) Requirements
- Python 3.9+  
- Install libs (once):
```bash
pip install duckdb pandas openpyxl et-xmlfile
# pbs-pricing
