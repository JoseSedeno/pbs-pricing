#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pbs_export_fixed_wide.py  (auto-mapping version)

Build a fixed-left, month-wide AEMP matrix from the DuckDB created by the ingest.

- Detects table names automatically
- Maps generic column names (item_code_b, name_a, attr_c, attr_f, attr_g, attr_j, group_key_no_b, …)
  to friendly PBS labels if canonical names are not present
- Produces one column per PBS snapshot month:  AEMP Sep 24, AEMP Oct 24, …

Usage:
  python3 pbs_export_fixed_wide.py \
    --db "./out/pbs_prices.duckdb" \
    --output_dir "./out" \
    --xlsx
"""

# ==============================
# SECTION 1 - Imports and helpers
# ==============================

import argparse
import os
import sys
from datetime import datetime

import duckdb
import pandas as pd


def _lower_set(iterable):
    return {str(x).lower() for x in iterable}


def list_tables(con):
    return [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main';"
        ).fetchall()
    ]


def list_columns(con, table):
    # DESCRIBE returns rows with first column as the column name
    return [r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()]


# =========================================
# SECTION 2 - Detect schema elements to use
# =========================================

def detect_schema(con):
    """
    Returns dict with:
      line_tbl, price_tbl, join_col, price_col, date_col
    """
    tables = list_tables(con)
    price_tbl = None
    price_col = None
    date_col = None

    # find a price table that has a date and an AEMP-like column
    for t in tables:
        cols = _lower_set(list_columns(con, t))
        has_date = any(c in cols for c in ["snapshot_date", "snapshot", "date"])
        has_aemp = any(c in cols for c in ["aemp", "exmanufacturerprice", "ex_manufacturer_price"])
        if has_date and has_aemp:
            price_tbl = t
            if "aemp" in cols:
                price_col = "aemp"
            elif "exmanufacturerprice" in cols:
                price_col = "exmanufacturerprice"
            else:
                price_col = "ex_manufacturer_price"
            if "snapshot_date" in cols:
                date_col = "snapshot_date"
            elif "snapshot" in cols:
                date_col = "snapshot"
            else:
                date_col = "date"
            break

    if not price_tbl:
        raise RuntimeError("Could not detect a price table with snapshot_date and AEMP.")

    # candidate join columns
    join_candidates = [
        "product_line_id",
        "product_id",
        "line_id",
        "product_line",
        "dim_product_line_id",
        "pl_id",
        "id",
    ]

    # find a line/product table sharing a join col
    line_tbl = None
    join_col = None
    price_cols = _lower_set(list_columns(con, price_tbl))
    for t in tables:
        cols = _lower_set(list_columns(con, t))
        if any(j in cols for j in join_candidates):
            for j in join_candidates:
                if j in cols and j in price_cols:
                    line_tbl, join_col = t, j
                    break
        if line_tbl:
            break

    if not line_tbl or not join_col:
        raise RuntimeError("Could not find a product table that shares a join column with the price table.")

    return {
        "line_tbl": line_tbl,
        "price_tbl": price_tbl,
        "join_col": join_col,
        "price_col": price_col,
        "date_col": date_col,
    }


# ============================================
# SECTION 3 - Main export logic and wide build
# ============================================

def main():
    # 3.1 - Arguments and IO setup
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to pbs_prices.duckdb")
    ap.add_argument("--output_dir", required=True, help="Where to write the export")
    ap.add_argument("--xlsx", action="store_true", help="Also write an .xlsx file")
    args = ap.parse_args()

    db = os.path.abspath(os.path.expanduser(args.db))
    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(db):
        print(f"ERROR: DB not found: {db}", file=sys.stderr)
        sys.exit(2)

    # 3.2 - Connect and detect schema
    con = duckdb.connect(db)
    schema = detect_schema(con)

    line_tbl = schema["line_tbl"]
    price_tbl = schema["price_tbl"]
    join_col = schema["join_col"]
    price_col = schema["price_col"]
    date_col = schema["date_col"]

    # 3.3 - Discover columns available in the product line table
    line_cols_all = list_columns(con, line_tbl)
    line_cols_lower = _lower_set(line_cols_all)

    # 3.4 - Friendly output labels and synonyms for the left block
    left_targets = [
        ("Item Code", ["item_code", "item_code_b", "item_code_a"]),
        ("Legal Instrument Drug", ["legal_instrument_drug", "name_a", "drug_name"]),
        ("Legal Instrument Form", ["legal_instrument_form", "attr_c", "form"]),
        ("Legal Instrument MoA", ["legal_instrument_moa", "attr_f", "moa"]),
        ("Brand Name", ["brand_name", "attr_g", "brand"]),                     # may be empty here
        ("Formulary", ["formulary", "attr_j"]),
        ("Program", ["program", "attr_b"]),
        ("Manufacturer Code", ["manufacturer_code", "group_key_no_b", "mfr_code"]),
        ("Responsible Person", ["responsible_person", "name_b"]),
        # AMT program string sometimes lives in price table, include here only if present on line table
        ("AMT Trade Product Pack Program", ["amt_trade_product_pack_program", "amt_pack_program"]),
    ]

    # 3.5 - Build select list from line table using synonyms
    select_pairs = []  # list of tuples (real_col_name_in_line, export_alias)
    for friendly, syns in left_targets:
        chosen = None
        for s in syns:
            if s.lower() in line_cols_lower:
                real = next((c for c in line_cols_all if c.lower() == s.lower()), s)
                chosen = real
                break
        if chosen:
            select_pairs.append((chosen, friendly))

    if not select_pairs:
        raise RuntimeError(
            f"Could not find any expected left columns in {line_tbl}. "
            f"Columns available: {line_cols_all}"
        )

    if join_col.lower() not in line_cols_lower:
        raise RuntimeError(f"Join column {join_col} not present in {line_tbl} (columns={line_cols_all})")

    # 3.6 - Build SELECT for the left block
    select_left = [f'l."{real}" AS "{alias}"' for (real, alias) in select_pairs]

    # 3.7 - Add snapshot attributes from the price table for Brand and AMT fields
    #       We take the latest row per item from the price table and join it to the left block.
    price_cols_all = [r[1] for r in con.execute(f"PRAGMA table_info('{price_tbl}')").fetchall()]
    lower_map = {c.lower(): c for c in price_cols_all}

    pm_aliases = {}  # map of source_col_in_price_tbl -> export_alias

    # Brand Name from price table, if present there
    if "brand_name" in lower_map:
        pm_aliases[lower_map["brand_name"]] = "Brand Name"

    # AMT Trade Product Pack (base field)
    if "amt_trade_product_pack" in lower_map:
        pm_aliases[lower_map["amt_trade_product_pack"]] = "AMT Trade Product Pack"
    elif "amt_trade_pack" in lower_map:
        pm_aliases[lower_map["amt_trade_pack"]] = "AMT Trade Product Pack"

    # Optional program text
    if "amt_pack_program" in lower_map:
        pm_aliases[lower_map["amt_pack_program"]] = "AMT Trade Product Pack Program"

    # Latest-per-item subquery from the price table
    p_meta = None
    if pm_aliases:
        pm_cols_select = [f'"{join_col}"'] + [f'"{c}"' for c in pm_aliases.keys()]
        p_meta = f"""
            SELECT {", ".join(pm_cols_select)}
            FROM (
                SELECT
                    "{join_col}",
                    {", ".join([f'"{c}"' for c in pm_aliases.keys()])},
                    ROW_NUMBER() OVER (PARTITION BY "{join_col}" ORDER BY "{date_col}" DESC) AS rn
                FROM "{price_tbl}"
            )
            WHERE rn = 1
        """

    # 3.8 - Build and run the main long query
    pm_selects = [f'pm."{src}" AS "{alias}"' for src, alias in pm_aliases.items()]
    left_select_clause = ", ".join(select_left + pm_selects) if pm_selects else ", ".join(select_left)

    if p_meta:
        q = f"""
        SELECT
          {left_select_clause},
          p."{date_col}"::DATE AS snapshot_date,
          p."{price_col}"     AS aemp_value
        FROM "{line_tbl}" l
        LEFT JOIN ({p_meta}) pm
          ON l."{join_col}" = pm."{join_col}"
        JOIN "{price_tbl}" p
          ON l."{join_col}" = p."{join_col}";
        """
    else:
        q = f"""
        SELECT
          {left_select_clause},
          p."{date_col}"::DATE AS snapshot_date,
          p."{price_col}"     AS aemp_value
        FROM "{line_tbl}" l
        JOIN "{price_tbl}" p
          ON l."{join_col}" = p."{join_col}";
        """

    df = con.execute(q).df()

    if df.empty:
        print("No data returned from the join. Check your DB contents.", file=sys.stderr)
        sys.exit(1)

    # 3.9 - Month label and pivot to wide
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["MonthLabel"] = df["snapshot_date"].dt.strftime("AEMP %b %y")

    # left headers = aliases from line table, plus pm aliases not already present
    left_from_line = [alias for (_, alias) in select_pairs]
    pm_headers = [alias for alias in pm_aliases.values() if alias not in left_from_line]
    left_export_headers = left_from_line + pm_headers

    pt = (
        df.pivot_table(
            index=left_export_headers,
            columns="MonthLabel",
            values="aemp_value",
            aggfunc="first",
        )
        .reset_index()
    )

    # 3.10 - Order the month columns chronologically
    month_cols = [c for c in pt.columns if str(c).startswith("AEMP ")]


    def _mkey(lbl):
        return datetime.strptime(lbl.replace("AEMP ", ""), "%b %y")


    month_cols_sorted = sorted(month_cols, key=_mkey)
    ordered_cols = left_export_headers + month_cols_sorted
    pt = pt[ordered_cols]

    # 3.11 - Write files
    csv_path = os.path.join(out_dir, "aemp_fixed_wide.csv")
    pt.to_csv(csv_path, index=False)

    if args.xlsx:
        xlsx_path = os.path.join(out_dir, "aemp_fixed_wide.xlsx")
        try:
            pt.to_excel(xlsx_path, index=False, engine="openpyxl")
        except Exception as e:
            print(f"WARNING: Excel export failed ({e}). CSV was written.", file=sys.stderr)

    # 3.12 - Console summary
    print("OK.")
    print("Detected:")
    print(f"  line_tbl  : {line_tbl}")
    print(f"  price_tbl : {price_tbl}")
    print(f"  join_col  : {join_col}")
    print(f"  date_col  : {date_col}")
    print(f"  price_col : {price_col}")
    print(f"Rows: {len(pt):,}   Month columns: {len(month_cols_sorted)}")
    print(f"CSV : {csv_path}")
    if args.xlsx:
        print(f"XLSX: {xlsx_path}")


# ===========================
# SECTION 4 - Entrypoint hook
# ===========================

if __name__ == "__main__":
    main()
