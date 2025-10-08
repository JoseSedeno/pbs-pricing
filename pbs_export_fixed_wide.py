#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pbs_export_fixed_wide.py  (identifiers = A,B,C,E,F,I,X with dedup)

Left block identifiers in this exact order:
- Item Code                      (A)
- Legal Instrument Drug          (B)
- Legal Instrument Form          (C)
- Brand Name                     (E)  from latest snapshot
- Formulary                      (F)  from latest snapshot or attr_f
- Responsible Person             (I)
- AMT Trade Product Pack         (X)  from latest snapshot

Prevents double ups by:
1) sourcing Brand, Formulary and AMT from one place only (latest snapshot)
2) normalizing identifier strings
3) dropping duplicates per month before pivot

Outputs: out/aemp_fixed_wide.csv and .xlsx (if --xlsx)
"""

# ==============================
# SECTION 1: Imports and helpers
# ==============================

import argparse, os, sys, re
from datetime import datetime
import duckdb, pandas as pd


def _lower_set(xs):
    return {str(x).lower() for x in xs}


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


def _normalize_amt_trade_pack(x: str) -> str:
    """
    Normalize AMT Trade Product Pack:
    - remove commas only when they are between words (not numbers),
      e.g., 'injection, solution' -> 'injection solution'
      but '75 mg, 0.5 mL' stays as-is.
    - collapse whitespace.
    """
    if x is None:
        return x
    s = str(x).replace("\u00A0", " ")
    s = re.sub(r'(?<!\d)\s*,\s*(?!\d)', ' ', s)  # remove word-only commas
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# =========================================
# SECTION 2: Detect schema (tables/columns)
# =========================================

def detect_schema(con):
    tables = list_tables(con)

    # Find price table that has a date and an AEMP-like column
    price_tbl = price_col = date_col = None
    for t in tables:
        cols = _lower_set(list_columns(con, t))
        has_date = any(c in cols for c in ["snapshot_date", "snapshot", "date"])
        has_aemp = any(c in cols for c in ["aemp", "exmanufacturerprice", "ex_manufacturer_price"])
        if has_date and has_aemp:
            price_tbl = t
            price_col = (
                "aemp"
                if "aemp" in cols
                else ("exmanufacturerprice" if "exmanufacturerprice" in cols else "ex_manufacturer_price")
            )
            date_col = "snapshot_date" if "snapshot_date" in cols else ("snapshot" if "snapshot" in cols else "date")
            break
    if not price_tbl:
        raise RuntimeError("Could not detect price table with snapshot_date and AEMP.")

    # Find line/product table that shares a join column with the price table
    join_candidates = ["product_line_id", "product_id", "line_id", "product_line", "dim_product_line_id", "pl_id", "id"]
    line_tbl = join_col = None
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
        raise RuntimeError("Could not find a product table sharing a join column with the price table.")

    return {"line_tbl": line_tbl, "price_tbl": price_tbl, "join_col": join_col, "price_col": price_col, "date_col": date_col}


# ============================================
# SECTION 3: Build long frame and pivot to wide
# ============================================

def main():
    # 3.1 Args and IO
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

    # 3.2 Connect and detect schema
    con = duckdb.connect(db)
    s = detect_schema(con)
    line_tbl, price_tbl, join_col, price_col, date_col = s["line_tbl"], s["price_tbl"], s["join_col"], s["price_col"], s["date_col"]

    # 3.3 Columns in line table and identifier mapping (A,B,C,E,F,I)
    line_cols_all = list_columns(con, line_tbl)
    line_cols_lower = _lower_set(line_cols_all)

    # NOTE:
    # - Formulary MUST come from attr_f in dim (that is where Formulary lives).
    #   We still prefer the latest snapshot value from the price table if present.
    # - Removed attr_g from Brand mapping (attr_g is Program in our schema).
    left_targets = [
        ("Item Code",             ["item_code", "item_code_b", "item_code_a"]),      # A
        ("Legal Instrument Drug", ["legal_instrument_drug", "name_a", "drug_name"]), # B
        ("Legal Instrument Form", ["legal_instrument_form", "attr_c", "form"]),      # C
        ("Brand Name",            ["brand_name", "brand"]),                           # E (overridden by price table if present)
        ("Formulary",             ["attr_f", "formulary"]),                           # F (overridden by price table if present)
        ("Responsible Person",    ["responsible_person", "name_b"]),                  # I
        # AMT Trade Product Pack comes from price table latest snapshot (X)
    ]

    select_pairs = []
    for friendly, syns in left_targets:
        chosen = None
        for s_name in syns:
            if s_name.lower() in line_cols_lower:
                chosen = next((c for c in line_cols_all if c.lower() == s_name.lower()), s_name)
                break
        if chosen:
            select_pairs.append((chosen, friendly))
    if not select_pairs:
        raise RuntimeError(f"No expected left columns found in {line_tbl}.")
    if join_col.lower() not in line_cols_lower:
        raise RuntimeError(f"Join column {join_col} not present in {line_tbl}.")

    # 3.4 Snapshot attributes from price table for Brand, Formulary and AMT (E, F, X)
    price_cols_all = [r[1] for r in con.execute(f"PRAGMA table_info('{price_tbl}')").fetchall()]
    lmap = {c.lower(): c for c in price_cols_all}

    pm_aliases = {}

    # Brand from latest snapshot
    if "brand_name" in lmap:
        pm_aliases[lmap["brand_name"]] = "Brand Name"

    # Formulary from latest snapshot (overrides line-table value)
    if "formulary" in lmap:
        pm_aliases[lmap["formulary"]] = "Formulary"

    # AMT Trade Product Pack from latest snapshot
    if "amt_trade_product_pack" in lmap:
        pm_aliases[lmap["amt_trade_product_pack"]] = "AMT Trade Product Pack"
    elif "amt_trade_pack" in lmap:
        pm_aliases[lmap["amt_trade_pack"]] = "AMT Trade Product Pack"

    # Latest-per-item subquery for those fields
    p_meta = None
    if pm_aliases:
        pm_cols = [f'"{join_col}"'] + [f'"{c}"' for c in pm_aliases.keys()]
        p_meta = f"""
            SELECT {", ".join(pm_cols)}
            FROM (
                SELECT "{join_col}", {", ".join([f'"{c}"' for c in pm_aliases.keys()])},
                       ROW_NUMBER() OVER (PARTITION BY "{join_col}" ORDER BY "{date_col}" DESC) AS rn
                FROM "{price_tbl}"
            )
            WHERE rn = 1
        """

    # 3.5 Build SELECT lists. If an alias appears in pm_aliases, skip the line-table version so the price-table value wins.
    skip_aliases = set(pm_aliases.values())
    select_left = [f'l."{real}" AS "{alias}"' for (real, alias) in select_pairs if alias not in skip_aliases]
    pm_selects  = [f'pm."{src}" AS "{alias}"' for src, alias in pm_aliases.items()]

    # ---- DEBUG: show exactly which columns will be used ----
    chosen_map = {alias: real for (real, alias) in select_pairs}
    print("LINE-TABLE MAPPING:", chosen_map)
    print("PRICE OVERRIDES  :", pm_aliases)
    print("Will skip aliases:", set(pm_aliases.values()))
    # ---- END DEBUG ----

    left_select_clause = ", ".join(select_left + pm_selects) if pm_selects else ", ".join(select_left)

    # 3.6 Build long frame
    if p_meta:
        q = f"""
        SELECT
          {left_select_clause},
          p."{date_col}"::DATE AS snapshot_date,
          p."{price_col}"     AS aemp_value
        FROM "{line_tbl}" l
        LEFT JOIN ({p_meta}) pm ON l."{join_col}" = pm."{join_col}"
        JOIN "{price_tbl}" p    ON l."{join_col}" = p."{join_col}";
        """
    else:
        q = f"""
        SELECT
          {left_select_clause},
          p."{date_col}"::DATE AS snapshot_date,
          p."{price_col}"     AS aemp_value
        FROM "{line_tbl}" l
        JOIN "{price_tbl}" p ON l."{join_col}" = p."{join_col}";
        """
    df = con.execute(q).df()
    if df.empty:
        print("No data returned. Check DB contents.", file=sys.stderr)
        sys.exit(1)

    # ---- Normalize AMT punctuation safely (words-only commas) ----
    if "AMT Trade Product Pack" in df.columns:
        df["AMT Trade Product Pack"] = df["AMT Trade Product Pack"].map(_normalize_amt_trade_pack)

    # 3.7 Normalize identifiers and drop duplicates per month
    id_cols = [c for c in [
        "Item Code", "Legal Instrument Drug", "Legal Instrument Form",
        "Brand Name", "Formulary", "Responsible Person", "AMT Trade Product Pack"
    ] if c in df.columns]

    for c in id_cols:
        df[c] = (
            df[c].astype("string").fillna("")
                 .str.replace("\u00A0", " ", regex=False)  # non-breaking spaces
                 .str.replace(r"\s+", " ", regex=True)      # collapse whitespace
                 .str.strip()
        )

    # month key for dedup. keep the latest row in each month
    df["__ym"] = pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%m")
    df = (
        df.sort_values("snapshot_date")
          .drop_duplicates(subset=id_cols + ["__ym"], keep="last")
          .drop(columns="__ym")
    )

    # 3.8 Pivot wide using exactly the requested identifiers
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["MonthLabel"]    = df["snapshot_date"].dt.strftime("AEMP %b %y")

    desired_order = [
        "Item Code", "Legal Instrument Drug", "Legal Instrument Form",
        "Brand Name", "Formulary", "Responsible Person", "AMT Trade Product Pack"
    ]
    left_export_headers = [c for c in desired_order if c in df.columns]

    pt = (
        df.pivot_table(
            index=left_export_headers,
            columns="MonthLabel",
            values="aemp_value",
            aggfunc="first"
        )
        .reset_index()
    )

    # 3.9 Order month columns chronologically
    month_cols = [c for c in pt.columns if str(c).startswith("AEMP ")]
    def _mkey(lbl): return datetime.strptime(lbl.replace("AEMP ", ""), "%b %y")
    month_cols_sorted = sorted(month_cols, key=_mkey)
    pt = pt[left_export_headers + month_cols_sorted]

    # 3.10 Write files
    csv_path  = os.path.join(out_dir, "aemp_fixed_wide.csv")
    xlsx_path = os.path.join(out_dir, "aemp_fixed_wide.xlsx")
    pt.to_csv(csv_path, index=False)
    if args.xlsx:
        try:
            pt.to_excel(xlsx_path, index=False, engine="openpyxl")
        except Exception as e:
            print(f"WARNING: Excel export failed ({e}). CSV was written.", file=sys.stderr)

    # 3.11 Console summary
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
# SECTION 4: Entrypoint hook
# ===========================

if __name__ == "__main__":
    main()

