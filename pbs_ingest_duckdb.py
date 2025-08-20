#!/usr/bin/env python3
"""
PBS Ingest to DuckDB (AEMP-centric, variant-aware, snapshot attributes kept)

What this does
- Reads all files matching: ex-manufacturer-prices-non-efc-YYYY-MM-DD.xlsx
- Uses header NAMES (robust to column reordering) to find fields:
    Item Code, Legal Instrument Drug, Legal Instrument Form, Formulary, Program,
    Pack Quantity, AEMP, plus extra snapshot attributes:
    Legal Instrument MoA, Brand Name, Manufacturer Code, Responsible Person,
    Pricing Quantity, Maximum Quantity, Maximum Repeats, AMT Trade Product Pack
- Derives: amt_pack_program = [AMT Trade Product Pack] + "_" + [Program] (safely)
- Identity (base): (A,B,C,F,G,J) = (Drug, Code, Form, Formulary, Program, Pack Qty)
  If B (Item Code) changes → NEW product line
- VARIANTS:
  - If the SAME base identity appears multiple times in the SAME month with DIFFERENT AEMP,
    we treat each distinct AEMP as a different "variant" (separate product line) and KEEP them.
  - Variants PERSIST across months by matching a "variant_signature_base" AND the initial AEMP seen
    when the variant was created (variant_init_aemp). If the same kind of duplicate recurs later,
    the rows continue on the SAME variant line.
- Stores ONE price per (product_line_id, snapshot_date)
- Keeps extra fields as SNAPSHOT attributes for analytics
- Logs “collisions” (cases where a base identity had >1 AEMP in a month) to a table for review.

Tables created
- dim_product_line(product_line_id, item_code_b, name_a, attr_c, attr_f, attr_g, attr_j,
                   variant_no, variety_signature, group_key_no_b,
                   variant_signature_base, variant_init_aemp)
- fact_monthly(product_line_id, snapshot_date, aemp, source_file,
               moa, brand_name, manufacturer_code, responsible_person,
               pricing_qty, max_qty, max_repeats, amt_trade_pack, amt_pack_program)
- ingest_collisions(base_key, snapshot_date, distinct_aemps, example_aemps, source_file)

Run:
python C:\PBS\pbs_ingest_duckdb.py --input_dir "C:\PBS\raw" --output_dir "C:\PBS\out"
"""

import os, re, glob, argparse
import pandas as pd
import duckdb

# --------- Header matching (by NAME, not letter) ----------
REQUIRED = {
    "Item Code": ["item code", "pbs code", "code"],
    "Legal Instrument Drug": ["legal instrument drug", "drug name", "name"],
    "Legal Instrument Form": ["legal instrument form", "form"],
    "Formulary": ["formulary"],
    "Program": ["program"],
    "Pack Quantity": ["pack quantity", "pack qty"],
    "AEMP": ["aemp"]
}

# Optional snapshot fields (kept if present)
OPTIONAL = {
    "Legal Instrument MoA": ["legal instrument moa", "moa"],
    "Brand Name": ["brand name", "brand"],
    "Manufacturer Code": ["manufacturer code", "mfr code", "manufacturer"],
    "Responsible Person": ["responsible person", "sponsor"],
    "Pricing Quantity": ["pricing quantity"],
    "Maximum Quantity": ["maximum quantity", "max quantity", "max qty"],
    "Maximum Repeats": ["maximum repeats", "max repeats"],
    "AMT Trade Product Pack": ["amt trade product pack", "amt trade product", "trade product pack"]
}

def parse_date_from_filename(fname: str):
    m = re.search(r"ex-manufacturer-prices-non-efc-(\d{4}-\d{2}-\d{2})\.xlsx$", fname, flags=re.IGNORECASE)
    if m:
        return pd.to_datetime(m.group(1)).date()
    return None

def find_col(cols, aliases):
    """Exact (lowercased) then contains match."""
    cmap = {str(c).lower().strip(): c for c in cols}
    for a in aliases:
        if a in cmap:
            return cmap[a]
    for c in cols:
        cl = str(c).lower().strip()
        if any(a in cl for a in aliases):
            return c
    return None

def normalize_text(s):
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else None

def safe_concat(x, y, sep="_"):
    x = (x or "").strip()
    y = (y or "").strip()
    if x and y:
        return f"{x}{sep}{y}"
    return x or y  # whichever exists, or empty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help='Folder with "ex-manufacturer-prices-non-efc-*.xlsx"')
    ap.add_argument("--output_dir", required=True, help="Folder to write DuckDB file")
    args = ap.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "pbs_prices.duckdb")

    con = duckdb.connect(db_path)

    # ---------- Tables ----------
    con.execute("""
    CREATE TABLE IF NOT EXISTS dim_product_line (
        product_line_id INTEGER PRIMARY KEY,
        item_code_b TEXT NOT NULL,
        name_a TEXT NOT NULL,
        attr_c TEXT,
        attr_f TEXT,
        attr_g TEXT,
        attr_j TEXT,
        variant_no INTEGER NOT NULL,
        variety_signature TEXT,
        group_key_no_b TEXT,
        variant_signature_base TEXT,
        variant_init_aemp DECIMAL(18,6),
        UNIQUE(name_a, item_code_b, attr_c, attr_f, attr_g, attr_j, variant_no)
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS fact_monthly (
        product_line_id INTEGER NOT NULL,
        snapshot_date DATE NOT NULL,
        aemp DECIMAL(18,6) NOT NULL,
        source_file TEXT,
        -- snapshot attributes:
        moa TEXT,
        brand_name TEXT,
        manufacturer_code TEXT,
        responsible_person TEXT,
        pricing_qty TEXT,
        max_qty TEXT,
        max_repeats TEXT,
        amt_trade_pack TEXT,
        amt_pack_program TEXT,
        PRIMARY KEY (product_line_id, snapshot_date)
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS ingest_collisions (
        base_key TEXT,
        snapshot_date DATE,
        distinct_aemps INTEGER,
        example_aemps TEXT,
        source_file TEXT
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_dpl_base ON dim_product_line(name_a, item_code_b, attr_c, attr_f, attr_g, attr_j);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fm_line_date ON fact_monthly(product_line_id, snapshot_date);")

    files = sorted(glob.glob(os.path.join(input_dir, "ex-manufacturer-prices-non-efc-*.xlsx")))
    if not files:
        print("No files found in", input_dir)
        return

    problems = []
    total_rows = 0

    for f in files:
        try:
            df = pd.read_excel(f, sheet_name=0, dtype=str)
            cols = list(df.columns)

            # Map required headers by NAME
            req_map = {need: find_col(cols, aliases) for need, aliases in REQUIRED.items()}
            missing = [k for k, v in req_map.items() if v is None]
            print(f"\n>>> {os.path.basename(f)}")
            print("    header map (required):", req_map)
            if missing:
                print("    SKIP: missing headers:", missing)
                problems.append(f"{os.path.basename(f)} missing: {missing}")
                continue

            # Optional headers
            opt_map = {need: find_col(cols, aliases) for need, aliases in OPTIONAL.items()}
            print("    header map (optional):", opt_map)

            keep_cols = [req_map["Item Code"], req_map["Legal Instrument Drug"], req_map["Legal Instrument Form"],
                         req_map["Formulary"], req_map["Program"], req_map["Pack Quantity"], req_map["AEMP"]]
            # add optional if present
            for k,v in opt_map.items():
                if v is not None:
                    keep_cols.append(v)

            sub = df[keep_cols].copy()
            # rename to canonical names
            rename_map = {
                req_map["Item Code"]: "Item Code",
                req_map["Legal Instrument Drug"]: "Legal Instrument Drug",
                req_map["Legal Instrument Form"]: "Legal Instrument Form",
                req_map["Formulary"]: "Formulary",
                req_map["Program"]: "Program",
                req_map["Pack Quantity"]: "Pack Quantity",
                req_map["AEMP"]: "AEMP"
            }
            for k, want in opt_map.items():
                if want is not None:
                    rename_map[want] = k
            sub = sub.rename(columns=rename_map)

            # Clean basics
            for c in ["Item Code", "Legal Instrument Drug", "Legal Instrument Form", "Formulary", "Program", "Pack Quantity"]:
                sub[c] = sub[c].map(normalize_text)
            for c in ["Legal Instrument MoA", "Brand Name", "Manufacturer Code", "Responsible Person",
                      "Pricing Quantity", "Maximum Quantity", "Maximum Repeats", "AMT Trade Product Pack"]:
                if c in sub.columns:
                    sub[c] = sub[c].map(normalize_text)

            # Price numeric
            sub["AEMP"] = sub["AEMP"].astype(str).str.replace(",", "", regex=False).str.strip()
            sub["AEMP_num"] = pd.to_numeric(sub["AEMP"], errors="coerce")

            # Snapshot info
            snapshot_date = parse_date_from_filename(os.path.basename(f))
            sub["SnapshotDate"] = snapshot_date
            sub["SourceFile"] = os.path.basename(f)

            # Derived label
            if "AMT Trade Product Pack" in sub.columns:
                sub["amt_pack_program"] = sub.apply(
                    lambda r: safe_concat(r.get("AMT Trade Product Pack"), r.get("Program")), axis=1
                )
            else:
                sub["amt_pack_program"] = sub["Program"].fillna("")

            # Drop rows missing key parts
            sub = sub.dropna(subset=["Legal Instrument Drug", "Item Code", "SnapshotDate"])

            # ---- Build base identity + variant signature ----
            # Base identity (A,B,C,F,G,J)
            base_cols = ["Legal Instrument Drug","Item Code","Legal Instrument Form","Formulary","Program","Pack Quantity"]

            # Variant signature BASE = attributes likely to distinguish duplicates
            # (keep it stable across months; do NOT include AEMP here)
            sig_parts = []
            for c in ["Legal Instrument MoA","Brand Name","Manufacturer Code","Responsible Person",
                      "Pricing Quantity","Maximum Quantity","Maximum Repeats","AMT Trade Product Pack"]:
                if c in sub.columns:
                    sig_parts.append(sub[c].fillna(""))
                else:
                    sig_parts.append(pd.Series([""] * len(sub)))

            # Construct base signature as a single text
            sig_base = sig_parts[0].astype(str)
            for s in sig_parts[1:]:
                sig_base = sig_base + " | " + s.astype(str)
            sub["variant_signature_base"] = sig_base

            # We treat DIFFERENT AEMP in SAME month as separate variants:
            # per month, per base identity, per variant_signature_base, we keep DISTINCT AEMP_num
            # That means multiplying variants only when a conflict exists.
            grp_keys = base_cols + ["variant_signature_base", "SnapshotDate", "SourceFile"]
            cols_to_keep = base_cols + ["AEMP_num","SourceFile","SnapshotDate",
                                        "Legal Instrument MoA","Brand Name","Manufacturer Code","Responsible Person",
                                        "Pricing Quantity","Maximum Quantity","Maximum Repeats","AMT Trade Product Pack","amt_pack_program",
                                        "variant_signature_base"]

            sub_distinct = (sub[cols_to_keep]
                            .drop_duplicates()
                            .dropna(subset=["AEMP_num"]))  # we only store numeric AEMP

            # For collision logging: count distinct AEMP per base+sig_base in this month
            collisions = (sub_distinct
                          .groupby(base_cols + ["variant_signature_base","SnapshotDate","SourceFile"])["AEMP_num"]
                          .agg(["nunique", lambda s: ", ".join(sorted({f"{v:.6f}" for v in s}))])
                          .reset_index())
            collisions = collisions.rename(columns={"nunique":"distinct_aemps","<lambda_0>":"example_aemps"})

            # Save collisions with distinct_aemps > 1
            coll_to_log = collisions[collisions["distinct_aemps"] > 1].copy()
            if not coll_to_log.empty:
                con.register("tmp_coll", coll_to_log)
                con.execute("""
                INSERT INTO ingest_collisions (base_key, snapshot_date, distinct_aemps, example_aemps, source_file)
                SELECT
                  (COALESCE("Legal Instrument Drug",'') || ' | ' ||
                   COALESCE("Item Code",'') || ' | ' ||
                   COALESCE("Legal Instrument Form",'') || ' | ' ||
                   COALESCE("Formulary",'') || ' | ' ||
                   COALESCE("Program",'') || ' | ' ||
                   COALESCE("Pack Quantity",'')) AS base_key,
                  "SnapshotDate", "distinct_aemps", "example_aemps", "SourceFile"
                FROM tmp_coll;
                """)
                con.unregister("tmp_coll")

            # Register distinct rows for this file
            con.register("tmp_sub", sub_distinct)

            # --- Insert NEW variants into dim_product_line ---
            # We try to match an existing variant by base identity + variant_signature_base + variant_init_aemp == AEMP_num.
            # If not found, we create a NEW variant_no (1 + max for that base).
            con.execute("""
            INSERT INTO dim_product_line (
                product_line_id, item_code_b, name_a, attr_c, attr_f, attr_g, attr_j,
                variant_no, variety_signature, group_key_no_b,
                variant_signature_base, variant_init_aemp
            )
            SELECT
                (SELECT COALESCE(MAX(product_line_id),0) FROM dim_product_line)
                + ROW_NUMBER() OVER () AS new_id,
                s."Item Code",
                s."Legal Instrument Drug",
                s."Legal Instrument Form",
                s."Formulary",
                s."Program",
                s."Pack Quantity",
                -- new variant_no per base (A,B,C,F,G,J)
                COALESCE((
                    SELECT MAX(variant_no) FROM dim_product_line d
                    WHERE d.item_code_b = s."Item Code"
                      AND d.name_a     = s."Legal Instrument Drug"
                      AND COALESCE(d.attr_c,'') = COALESCE(s."Legal Instrument Form",'')
                      AND COALESCE(d.attr_f,'') = COALESCE(s."Formulary",'')
                      AND COALESCE(d.attr_g,'') = COALESCE(s."Program",'')
                      AND COALESCE(d.attr_j,'') = COALESCE(s."Pack Quantity",'')
                ), 0) + ROW_NUMBER() OVER (
                    PARTITION BY s."Item Code", s."Legal Instrument Drug", s."Legal Instrument Form",
                                 s."Formulary", s."Program", s."Pack Quantity"
                    ORDER BY s."AEMP_num" DESC NULLS LAST
                ) AS variant_no,
                -- display signature of the variety (without B)
                (COALESCE(s."Legal Instrument Form",'') || ' | ' ||
                 COALESCE(s."Formulary",'')          || ' | ' ||
                 COALESCE(s."Program",'')            || ' | ' ||
                 COALESCE(s."Pack Quantity",''))     AS variety_signature,
                -- group key without B (for context)
                (COALESCE(s."Legal Instrument Drug",'') || ' | ' ||
                 COALESCE(s."Legal Instrument Form",'') || ' | ' ||
                 COALESCE(s."Formulary",'')            || ' | ' ||
                 COALESCE(s."Program",'')              || ' | ' ||
                 COALESCE(s."Pack Quantity",''))       AS group_key_no_b,
                s."variant_signature_base",
                CAST(s."AEMP_num" AS DECIMAL(18,6)) AS variant_init_aemp
            FROM (
                -- candidates that DON'T already exist in dim as a variant with same base + sig_base + init_aemp
                SELECT DISTINCT
                    "Item Code","Legal Instrument Drug","Legal Instrument Form","Formulary","Program","Pack Quantity",
                    "variant_signature_base","AEMP_num"
                FROM tmp_sub
            ) s
            LEFT JOIN dim_product_line d
              ON d.item_code_b = s."Item Code"
             AND d.name_a     = s."Legal Instrument Drug"
             AND COALESCE(d.attr_c,'') = COALESCE(s."Legal Instrument Form",'')
             AND COALESCE(d.attr_f,'') = COALESCE(s."Formulary",'')
             AND COALESCE(d.attr_g,'') = COALESCE(s."Program",'')
             AND COALESCE(d.attr_j,'') = COALESCE(s."Pack Quantity",'')
             AND COALESCE(d.variant_signature_base,'') = COALESCE(s."variant_signature_base",'')
             AND d.variant_init_aemp = CAST(s."AEMP_num" AS DECIMAL(18,6))
            WHERE d.product_line_id IS NULL;
            """)

            # --- Insert monthly facts (one per (variant, date)) ---
            con.execute("""
            INSERT INTO fact_monthly (
                product_line_id, snapshot_date, aemp, source_file,
                moa, brand_name, manufacturer_code, responsible_person,
                pricing_qty, max_qty, max_repeats, amt_trade_pack, amt_pack_program
            )
            SELECT
                d.product_line_id,
                s."SnapshotDate",
                CAST(s."AEMP_num" AS DECIMAL(18,6)) AS aemp,
                s."SourceFile",
                s."Legal Instrument MoA"    AS moa,
                s."Brand Name"              AS brand_name,
                s."Manufacturer Code"       AS manufacturer_code,
                s."Responsible Person"      AS responsible_person,
                s."Pricing Quantity"        AS pricing_qty,
                s."Maximum Quantity"        AS max_qty,
                s."Maximum Repeats"         AS max_repeats,
                s."AMT Trade Product Pack"  AS amt_trade_pack,
                s."amt_pack_program"        AS amt_pack_program
            FROM tmp_sub s
            JOIN dim_product_line d
              ON d.item_code_b = s."Item Code"
             AND d.name_a     = s."Legal Instrument Drug"
             AND COALESCE(d.attr_c,'') = COALESCE(s."Legal Instrument Form",'')
             AND COALESCE(d.attr_f,'') = COALESCE(s."Formulary",'')
             AND COALESCE(d.attr_g,'') = COALESCE(s."Program",'')
             AND COALESCE(d.attr_j,'') = COALESCE(s."Pack Quantity",'')
             AND COALESCE(d.variant_signature_base,'') = COALESCE(s."variant_signature_base",'')
             AND d.variant_init_aemp = CAST(s."AEMP_num" AS DECIMAL(18,6))
            LEFT JOIN fact_monthly fm
              ON fm.product_line_id = d.product_line_id
             AND fm.snapshot_date   = s."SnapshotDate"
            WHERE fm.product_line_id IS NULL;
            """)

            total_rows += len(sub)
            con.unregister("tmp_sub")
            print("    inserted (this file): OK")

        except Exception as e:
            problems.append(f"{os.path.basename(f)} error: {e}")
            print("    ERROR:", e)

    # Summary
    total_lines = con.execute("SELECT COUNT(*) FROM dim_product_line").fetchone()[0]
    total_facts = con.execute("SELECT COUNT(*) FROM fact_monthly").fetchone()[0]
    print("\n=== SUMMARY ===")
    print("DuckDB file :", db_path)
    print("Product lines:", total_lines)
    print("Monthly rows :", total_facts)
    print("Raw rows read:", total_rows)
    if problems:
        print("Problems:")
        for p in problems:
            print(" -", p)

if __name__ == "__main__":
    main()

