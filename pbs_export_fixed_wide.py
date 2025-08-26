#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pbs_export_fixed_wide.py  (auto-mapping version)

Build a fixed-left, month-wide AEMP matrix from the DuckDB created by the ingest.

- Detects table names automatically
- Maps generic column names (item_code_b, name_a, attr_c, attr_f, attr_g, attr_j, group_key_no_b, …)
  to friendly PBS labels, if the canonical names aren’t present.
- Produces one column per PBS snapshot month:  AEMP Sep 24, AEMP Oct 24, …

Usage:
  python3 pbs_export_fixed_wide.py \
    --db "./out/pbs_prices.duckdb" \
    --output_dir "./out" \
    --xlsx
"""

import argparse
import os
import sys
from datetime import datetime

import duckdb
import pandas as pd


def _lower_set(iterable):
    return {str(x).lower() for x in iterable}

def list_tables(con):
    return [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main';"
    ).fetchall()]

def list_columns(con, table):
    return [r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()]

def detect_schema(con):
    """
    Returns dict with:
      line_tbl, price_tbl, join_col, price_col, date_col
    """
    tables = list_tables(con)
    price_tbl = None
    price_col = None
    date_col  = None

    # find a price table that has a date and aemp-like column
    for t in tables:
        cols = _lower_set(list_columns(con, t))
        has_date = any(c in cols for c in ['snapshot_date', 'snapshot', 'date'])
        has_aemp = any(c in cols for c in ['aemp', 'exmanufacturerprice', 'ex_manufacturer_price'])
        if has_date and has_aemp:
            price_tbl = t
            if 'aemp' in cols:
                price_col = 'aemp'
            elif 'exmanufacturerprice' in cols:
                price_col = 'exmanufacturerprice'
            else:
                price_col = 'ex_manufacturer_price'
            if 'snapshot_date' in cols:
                date_col = 'snapshot_date'
            elif 'snapshot' in cols:
                date_col = 'snapshot'
            else:
                date_col = 'date'
            break

    if not price_tbl:
        raise RuntimeError("Could not detect a price table with snapshot_date+ aemp.")

    # candidate join columns
    join_candidates = ['product_line_id', 'product_id', 'line_id', 'product_line', 'dim_product_line_id', 'pl_id', 'id']

    # find a line/product table sharing a join col and likely item/drug columns
    line_tbl = None
    join_col = None
    for t in tables:
        cols = _lower_set(list_columns(con, t))
        if any(j in cols for j in join_candidates):
            # any shared join?
            for j in join_candidates:
                if j in cols and j in _lower_set(list_columns(con, price_tbl)):
                    line_tbl, join_col = t, j
                    break
        if line_tbl:
            break

    if not line_tbl or not join_col:
        raise RuntimeError("Could not find a product/line table that shares a join column with the price table.")

    return {
        'line_tbl': line_tbl,
        'price_tbl': price_tbl,
        'join_col':  join_col,
        'price_col': price_col,
        'date_col':  date_col
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True, help='Path to pbs_prices.duckdb')
    ap.add_argument('--output_dir', required=True, help='Where to write the export')
    ap.add_argument('--xlsx', action='store_true', help='Also write an .xlsx file')
    args = ap.parse_args()

    db = os.path.abspath(os.path.expanduser(args.db))
    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(db):
        print(f"ERROR: DB not found: {db}", file=sys.stderr)
        sys.exit(2)

    con = duckdb.connect(db)
    schema = detect_schema(con)

    line_tbl = schema['line_tbl']
    price_tbl = schema['price_tbl']
    join_col  = schema['join_col']
    price_col = schema['price_col']
    date_col  = schema['date_col']

    # Columns available in the line table
    line_cols_all   = list_columns(con, line_tbl)
    line_cols_lower = _lower_set(line_cols_all)

    # ---------- Friendly output labels and synonyms ----------
    # Left side we want (in this order), with synonym lists to match your ingest’s generic names.
    left_targets = [
        ("AMT Trade Product Pack_Program",  ['amt_trade_product_pack_program', 'amt_pack_program']),
        ("Item Code",                       ['item_code', 'item_code_b', 'item_code_a']),
        ("Legal Instrument Drug",           ['legal_instrument_drug', 'name_a', 'drug_name']),
        ("Legal Instrument Form",           ['legal_instrument_form', 'attr_c', 'form']),
        ("Legal Instrument MoA",            ['legal_instrument_moa', 'attr_f', 'moa']),
        ("Brand Name",                      ['brand_name', 'attr_g', 'brand']),
        ("Formulary",                       ['formulary', 'attr_j']),
        ("Program",                         ['program', 'attr_b']),
        ("Manufacturer Code",               ['manufacturer_code', 'group_key_no_b', 'mfr_code']),
        ("Responsible Person",              ['responsible_person', 'name_b'])
    ]

    # For each friendly label, pick the first synonym that exists in the table
    select_pairs = []   # (real_col, export_alias)
    for friendly, syns in left_targets:
        chosen = None
        for s in syns:
            if s.lower() in line_cols_lower:
                # get original case spelling
                real = next((c for c in line_cols_all if c.lower() == s.lower()), s)
                chosen = real
                break
        if chosen:
            select_pairs.append((chosen, friendly))
        # If none found, we just skip that column.

    if not select_pairs:
        raise RuntimeError(
            f"Could not find any of the expected left columns in {line_tbl}.\n"
            f"Columns available: {line_cols_all}"
        )

    # Also ensure we can join
    if join_col.lower() not in line_cols_lower:
        raise RuntimeError(f"Join column {join_col} not in {line_tbl} (columns={line_cols_all})")

    # Build SELECT for left side
    select_left = [f'l."{real}" AS "{alias}"' for (real, alias) in select_pairs]

    # Build & run query
    q = f"""
    SELECT
      {", ".join(select_left)},
      p."{date_col}"::DATE AS snapshot_date,
      p."{price_col}"     AS aemp_value
    FROM "{line_tbl}" l
    JOIN "{price_tbl}" p
      ON l."{join_col}" = p."{join_col}";
    """
    df = con.execute(q).df()

    # No rows?
    if df.empty:
        print("No data returned from the join. Check your DB contents.", file=sys.stderr)
        sys.exit(1)

    # Build MonthLabel and pivot
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['MonthLabel'] = df['snapshot_date'].dt.strftime('AEMP %b %y')

    left_export_headers = [alias for (_, alias) in select_pairs]

    pt = df.pivot_table(
        index=left_export_headers,
        columns='MonthLabel',
        values='aemp_value',
        aggfunc='first'
    ).reset_index()

    # Order the month columns chronologically
    month_cols = [c for c in pt.columns if str(c).startswith('AEMP ')]
    def _mkey(lbl):
        return datetime.strptime(lbl.replace('AEMP ', ''), '%b %y')
    month_cols_sorted = sorted(month_cols, key=_mkey)

    ordered = left_export_headers + month_cols_sorted
    pt = pt[ordered]

    # Write files
    csv_path = os.path.join(out_dir, 'aemp_fixed_wide.csv')
    pt.to_csv(csv_path, index=False)

    if args.xlsx:
        xlsx_path = os.path.join(out_dir, 'aemp_fixed_wide.xlsx')
        try:
            pt.to_excel(xlsx_path, index=False, engine='openpyxl')
        except Exception as e:
            print(f"WARNING: Excel export failed ({e}). CSV was written.", file=sys.stderr)

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


if __name__ == '__main__':
    main()
