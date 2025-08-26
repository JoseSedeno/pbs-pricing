#!/usr/bin/env python3
"""
Export AEMP history from DuckDB.

- Long format: one row per (product_line, date)
- Wide matrix: columns like 'AEMP Aug 13', pivoted by month
- Optional filters: --drug, --item_code, --program, --start_date, --end_date
"""

import os
import argparse
import duckdb
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to pbs_prices.duckdb")
    ap.add_argument("--output_dir", required=True, help="Folder for CSV exports")
    ap.add_argument("--drug", help="Filter: Legal Instrument Drug (text match)", default=None)
    ap.add_argument("--item_code", help="Filter: Item Code (exact)", default=None)
    ap.add_argument("--program", help="Filter: Program (exact)", default=None)
    ap.add_argument("--start_date", help="Inclusive start YYYY-MM-DD", default=None)
    ap.add_argument("--end_date", help="Inclusive end YYYY-MM-DD", default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    con = duckdb.connect(args.db)

    where = []
    params = {}

    if args.drug:
        where.append("LOWER(d.name_a) LIKE LOWER($drug)")
        params["drug"] = f"%{args.drug}%"
    if args.item_code:
        where.append("d.item_code_b = $b")
        params["b"] = args.item_code
    if args.program:
        where.append("COALESCE(d.attr_g,'') = $prog")
        params["prog"] = args.program
    if args.start_date:
        where.append("fm.snapshot_date >= DATE $start")
        params["start"] = args.start_date
    if args.end_date:
        where.append("fm.snapshot_date <= DATE $end")
        params["end"] = args.end_date

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    q = f"""
    SELECT
      fm.snapshot_date,
      d.name_a       AS A,
      d.item_code_b  AS B,
      d.attr_c       AS C,
      d.attr_f       AS F,
      d.attr_g       AS G,
      d.attr_j       AS J,
      d.variant_no   AS variant,
      fm.aemp        AS AEMP,
      fm.amt_pack_program,
      fm.brand_name,
      fm.manufacturer_code,
      fm.responsible_person,
      fm.pricing_qty,
      fm.max_qty,
      fm.max_repeats
    FROM fact_monthly fm
    JOIN dim_product_line d ON d.product_line_id = fm.product_line_id
    {where_sql}
    ORDER BY A, B, C, F, G, J, variant, fm.snapshot_date
    """
    df = con.execute(q, params).df()
    if df.empty:
        print("No rows matched your filters.")
        return

    # Long export
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    long_path = os.path.join(args.output_dir, "aemp_history_long.csv")
    df.to_csv(long_path, index=False)

    # Wide pivot: AEMP by month label
    df["MonthLabel"] = df["snapshot_date"].dt.strftime("AEMP %b %y")
    pivot = df.pivot_table(index=["A","B","C","F","G","J","variant"],
                           columns="MonthLabel",
                           values="AEMP",
                           aggfunc="last").sort_index()

    # Order columns chronologically
    ordered_cols = (df[["MonthLabel","snapshot_date"]].drop_duplicates()
                    .sort_values("snapshot_date")["MonthLabel"].tolist())
    pivot = pivot.reindex(columns=ordered_cols)

    wide_path = os.path.join(args.output_dir, "aemp_matrix_wide.csv")
    pivot.to_csv(wide_path)

    print("Exports written:")
    print(" -", long_path)
    print(" -", wide_path)

if __name__ == "__main__":
    main()

