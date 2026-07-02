#!/usr/bin/env python3
"""
Chemo EFC -> DuckDB (variant-aware, multi-metric prices)
"""

import os
import re
import glob
import argparse
import pandas as pd
import duckdb


def normalize_punct(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s)
    s = re.sub(r"[:;|/]", ",", s)
    s = re.sub(r",\s*,+", ",", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s or None


def normalize_text(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = re.sub(r"\s+", " ", str(s).strip())
    return s or None


def safe_concat(x, y, sep="_"):
    x = (x or "").strip()
    y = (y or "").strip()
    if x and y:
        return f"{x}{sep}{y}"
    return x or y


def parse_date_from_filename(fname: str):
    m = re.search(
        r"ex-manufacturer-prices-efc-(\d{4})-(\d{2})-(\d{1,2})\.[xX][lL][sS][xX]$",
        fname,
    )
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return pd.Timestamp(year=y, month=mo, day=d).date()


def find_col(cols, aliases):
    cmap = {str(c).lower().strip(): c for c in cols}
    for a in aliases:
        if a in cmap:
            return cmap[a]
    for c in cols:
        cl = str(c).lower().strip()
        if any(a in cl for a in aliases):
            return c
    return None


REQUIRED = {
    "Item Code": ["item code"],
    "Legal Instrument Drug": ["legal instrument drug", "drug name"],
    "Legal Instrument Form": ["legal instrument form"],
    "Formulary": ["formulary"],
    "Program": ["program"],
    "Pack Quantity": ["pack quantity", "pack qty"],
    "AEMP": ["aemp"],
}


OPTIONAL = {
    "Legal Instrument MoA": ["legal instrument moa", "moa"],
    "Brand Name": ["brand name", "brand"],
    "Manufacturer Code": ["manufacturer code", "mfr code"],
    "Responsible Person": ["responsible person", "sponsor"],
    "Pricing Quantity": ["pricing quantity"],
    "Vial Content": ["vial content", "vial contents"],
    "Maximum Amount": ["maximum amount", "max amount", "maximum quantity", "max quantity", "max qty"],
    "Number Repeats": ["number repeats", "max repeats", "maximum repeats"],

    # July 2026 fields
    "Dispensing Rule mnemonic": ["dispensing rule mnemonic"],
    "Dispensing setting": ["dispensing setting"],

    "AMT Trade Product Pack": [
        "amt trade product pack",
        "amt trade product",
        "trade product pack",
        "tpp",
    ],

    # metrics
    "PEMP": ["pemp"],
    "Ex-man Price per Vial": [
        "ex-man price per vial",
        "exman price per vial",
        "ex manufacturer price per vial",
    ],
    "DPMA": ["dpma"],
    "Claimed Price for Pack": ["claimed price for pack"],
    "Claimed Price for vial": ["claimed price for vial"],
    "Claimed DPMA": ["claimed dpma"],
    "Premium": ["premium", "premium value"],
    "Premium type": ["premium type"],
}


CHEMO_IDENTITY_COLS = [
    "Item Code",
    "Legal Instrument Drug",
    "Legal Instrument Form",
    "Legal Instrument MoA",
    "Brand Name",
    "Formulary",
    "Program",
    "Manufacturer Code",
    "Responsible Person",
    "Pack Quantity",
    "Pricing Quantity",
    "Vial Content",
    "Maximum Amount",
    "Number Repeats",
    "Dispensing Rule mnemonic",
    "Dispensing setting",
    "Premium type",
    "AMT Trade Product Pack",
]


PRICE_METRICS = [
    "AEMP",
    "PEMP",
    "Ex-man Price per Vial",
    "DPMA",
    "Claimed Price for Pack",
    "Claimed Price for vial",
    "Claimed DPMA",
    "Premium",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    input_dir = os.path.expanduser(os.path.normpath(args.input_dir))
    output_dir = os.path.expanduser(os.path.normpath(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, "chemo_prices.duckdb")
    con = duckdb.connect(db_path)

    files = sorted(
        glob.glob(os.path.join(input_dir, "ex-manufacturer-prices-efc-*.[xX][lL][sS][xX]"))
    )

    if not files:
        print(f"No files found in {input_dir}")
        return

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
        aemp DECIMAL(18,6),
        pemp DECIMAL(18,6),
        exman_price_per_vial DECIMAL(18,6),
        dpma DECIMAL(18,6),
        claimed_price_pack DECIMAL(18,6),
        claimed_price_vial DECIMAL(18,6),
        claimed_dpma DECIMAL(18,6),
        premium DECIMAL(18,6),
        source_file TEXT,
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

    con.execute("""
    CREATE INDEX IF NOT EXISTS idx_dpl_base
    ON dim_product_line(name_a, item_code_b, attr_c, attr_f, attr_g, attr_j);
    """)

    con.execute("""
    CREATE INDEX IF NOT EXISTS idx_fm_line_date
    ON fact_monthly(product_line_id, snapshot_date);
    """)

    problems = []
    total_rows = 0

    for f in files:
        try:
            df = pd.read_excel(f, sheet_name=0, dtype=str)
            df.columns = [str(c).strip() for c in df.columns]
            cols = list(df.columns)

            req_map = {need: find_col(cols, aliases) for need, aliases in REQUIRED.items()}
            missing = [k for k, v in req_map.items() if v is None]

            print(f"\n>>> {os.path.basename(f)}")
            print("    header map (required):", req_map)

            if missing:
                problems.append(f"{os.path.basename(f)} missing required: {missing}")
                print("    SKIP: missing headers:", missing)
                continue

            opt_map = {need: find_col(cols, aliases) for need, aliases in OPTIONAL.items()}
            print("    header map (optional):", opt_map)

            keep_cols = [req_map[k] for k in REQUIRED.keys()]
            for _, v in opt_map.items():
                if v is not None and v not in keep_cols:
                    keep_cols.append(v)

            sub = df[keep_cols].copy()

            rename_map = {req_map[k]: k for k in REQUIRED.keys()}
            for k, v in opt_map.items():
                if v is not None:
                    rename_map[v] = k

            sub = sub.rename(columns=rename_map)

            for c in CHEMO_IDENTITY_COLS:
                if c not in sub.columns:
                    sub[c] = pd.NA

            for c in CHEMO_IDENTITY_COLS:
                sub[c] = sub[c].map(normalize_text)

            for c in CHEMO_IDENTITY_COLS:
                if c in sub.columns:
                    sub[c] = sub[c].map(normalize_punct)

            for price_col in PRICE_METRICS:
                if price_col in sub.columns:
                    clean = (
                        sub[price_col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("$", "", regex=False)
                        .str.strip()
                    )
                    sub[price_col + "_num"] = pd.to_numeric(clean, errors="coerce")
                else:
                    sub[price_col + "_num"] = pd.NA

            snapshot_date = parse_date_from_filename(os.path.basename(f))
            sub["SnapshotDate"] = snapshot_date
            sub["SourceFile"] = os.path.basename(f)

            sub["amt_pack_program"] = sub.apply(
                lambda r: safe_concat(r.get("AMT Trade Product Pack"), r.get("Program")),
                axis=1,
            )

            sub = sub.dropna(subset=["Legal Instrument Drug", "Item Code", "SnapshotDate"])

            exclude = set(
                PRICE_METRICS
                + [m + "_num" for m in PRICE_METRICS]
                + ["SnapshotDate", "SourceFile"]
            )

            sig_cols = list(dict.fromkeys([c for c in sub.columns if c not in exclude]))
            sig_view = sub.loc[:, sig_cols].astype(str).fillna("")
            sub["variant_signature_base"] = sig_view.apply(
                lambda r: " | ".join(list(r.values)),
                axis=1,
            )

            base_cols = CHEMO_IDENTITY_COLS

            cols_to_keep = base_cols + [
                "SourceFile",
                "SnapshotDate",
                "amt_pack_program",
                "variant_signature_base",
            ]

            for m in PRICE_METRICS:
                cols_to_keep.append(m + "_num")

            existing_cols = [c for c in cols_to_keep if c in sub.columns]
            sub_distinct = sub[existing_cols].drop_duplicates()

            for m in PRICE_METRICS:
                col = m + "_num"
                if col not in sub_distinct.columns:
                    sub_distinct[col] = pd.NA

            if "AEMP_num" in sub_distinct.columns:
                collisions = (
                    sub_distinct
                    .groupby(base_cols + ["variant_signature_base", "SnapshotDate", "SourceFile"])["AEMP_num"]
                    .agg(
                        distinct_aemps="nunique",
                        example_aemps=lambda s: ", ".join(
                            sorted({f"{v:.6f}" for v in s if pd.notna(v)})
                        ),
                    )
                    .reset_index()
                )

                coll_to_log = collisions[collisions["distinct_aemps"] > 1].copy()

                if not coll_to_log.empty:
                    con.register("tmp_coll", coll_to_log)
                    con.execute("""
                        INSERT INTO ingest_collisions (
                            base_key,
                            snapshot_date,
                            distinct_aemps,
                            example_aemps,
                            source_file
                        )
                        SELECT
                            (
                                COALESCE("Legal Instrument Drug",'') || ' | ' ||
                                COALESCE("Item Code",'') || ' | ' ||
                                COALESCE("Legal Instrument Form",'') || ' | ' ||
                                COALESCE("Formulary",'') || ' | ' ||
                                COALESCE("Program",'') || ' | ' ||
                                COALESCE("Pack Quantity",'')
                            ) AS base_key,
                            "SnapshotDate",
                            "distinct_aemps",
                            "example_aemps",
                            "SourceFile"
                        FROM tmp_coll;
                    """)
                    con.unregister("tmp_coll")

            con.register("tmp_sub", sub_distinct)

            con.execute("""
            INSERT INTO dim_product_line (
                product_line_id,
                item_code_b,
                name_a,
                attr_c,
                attr_f,
                attr_g,
                attr_j,
                variant_no,
                variety_signature,
                group_key_no_b,
                variant_signature_base,
                variant_init_aemp
            )
            SELECT
                (SELECT COALESCE(MAX(product_line_id), 0) FROM dim_product_line)
                + ROW_NUMBER() OVER () AS new_id,
                s."Item Code",
                s."Legal Instrument Drug",
                s."Legal Instrument Form",
                s."Formulary",
                s."Program",
                s."Pack Quantity",
                COALESCE((
                    SELECT MAX(variant_no)
                    FROM dim_product_line d
                    WHERE d.item_code_b = s."Item Code"
                      AND d.name_a = s."Legal Instrument Drug"
                      AND COALESCE(d.attr_c,'') = COALESCE(s."Legal Instrument Form",'')
                      AND COALESCE(d.attr_f,'') = COALESCE(s."Formulary",'')
                      AND COALESCE(d.attr_g,'') = COALESCE(s."Program",'')
                      AND COALESCE(d.attr_j,'') = COALESCE(s."Pack Quantity",'')
                ), 0)
                + ROW_NUMBER() OVER (
                    PARTITION BY
                        s."Item Code",
                        s."Legal Instrument Drug",
                        s."Legal Instrument Form",
                        s."Formulary",
                        s."Program",
                        s."Pack Quantity"
                    ORDER BY s."AEMP_num" DESC NULLS LAST
                ) AS variant_no,
                (
                    COALESCE(s."Legal Instrument Form",'') || ' | ' ||
                    COALESCE(s."Formulary",'') || ' | ' ||
                    COALESCE(s."Program",'') || ' | ' ||
                    COALESCE(s."Pack Quantity",'')
                ) AS variety_signature,
                (
                    COALESCE(s."Legal Instrument Drug",'') || ' | ' ||
                    COALESCE(s."Legal Instrument Form",'') || ' | ' ||
                    COALESCE(s."Formulary",'') || ' | ' ||
                    COALESCE(s."Program",'') || ' | ' ||
                    COALESCE(s."Pack Quantity",'')
                ) AS group_key_no_b,
                s."variant_signature_base",
                CAST(s."AEMP_num" AS DECIMAL(18,6)) AS variant_init_aemp
            FROM (
                SELECT DISTINCT
                    "Item Code",
                    "Legal Instrument Drug",
                    "Legal Instrument Form",
                    "Formulary",
                    "Program",
                    "Pack Quantity",
                    "variant_signature_base",
                    "AEMP_num"
                FROM tmp_sub
            ) s
            LEFT JOIN dim_product_line d
              ON d.item_code_b = s."Item Code"
             AND d.name_a = s."Legal Instrument Drug"
             AND COALESCE(d.attr_c,'') = COALESCE(s."Legal Instrument Form",'')
             AND COALESCE(d.attr_f,'') = COALESCE(s."Formulary",'')
             AND COALESCE(d.attr_g,'') = COALESCE(s."Program",'')
             AND COALESCE(d.attr_j,'') = COALESCE(s."Pack Quantity",'')
             AND COALESCE(d.variant_signature_base,'') = COALESCE(s."variant_signature_base",'')
             AND (
                    (d.variant_init_aemp IS NULL AND s."AEMP_num" IS NULL)
                    OR d.variant_init_aemp = CAST(s."AEMP_num" AS DECIMAL(18,6))
                 )
            WHERE d.product_line_id IS NULL;
            """)

            con.execute("""
            INSERT INTO fact_monthly (
                product_line_id,
                snapshot_date,
                aemp,
                pemp,
                exman_price_per_vial,
                dpma,
                claimed_price_pack,
                claimed_price_vial,
                claimed_dpma,
                premium,
                source_file,
                moa,
                brand_name,
                manufacturer_code,
                responsible_person,
                pricing_qty,
                max_qty,
                max_repeats,
                amt_trade_pack,
                amt_pack_program
            )
            SELECT
                d.product_line_id,
                s."SnapshotDate",
                CAST(s."AEMP_num" AS DECIMAL(18,6)),
                CAST(s."PEMP_num" AS DECIMAL(18,6)),
                CAST(s."Ex-man Price per Vial_num" AS DECIMAL(18,6)),
                CAST(s."DPMA_num" AS DECIMAL(18,6)),
                CAST(s."Claimed Price for Pack_num" AS DECIMAL(18,6)),
                CAST(s."Claimed Price for vial_num" AS DECIMAL(18,6)),
                CAST(s."Claimed DPMA_num" AS DECIMAL(18,6)),
                CAST(s."Premium_num" AS DECIMAL(18,6)),
                s."SourceFile",
                s."Legal Instrument MoA",
                s."Brand Name",
                s."Manufacturer Code",
                s."Responsible Person",
                s."Pricing Quantity",
                s."Maximum Amount",
                s."Number Repeats",
                s."AMT Trade Product Pack",
                s."amt_pack_program"
            FROM tmp_sub s
            JOIN dim_product_line d
              ON d.item_code_b = s."Item Code"
             AND d.name_a = s."Legal Instrument Drug"
             AND COALESCE(d.attr_c,'') = COALESCE(s."Legal Instrument Form",'')
             AND COALESCE(d.attr_f,'') = COALESCE(s."Formulary",'')
             AND COALESCE(d.attr_g,'') = COALESCE(s."Program",'')
             AND COALESCE(d.attr_j,'') = COALESCE(s."Pack Quantity",'')
             AND COALESCE(d.variant_signature_base,'') = COALESCE(s."variant_signature_base",'')
             AND (
                    (d.variant_init_aemp IS NULL AND s."AEMP_num" IS NULL)
                    OR d.variant_init_aemp = CAST(s."AEMP_num" AS DECIMAL(18,6))
                 )
            LEFT JOIN fact_monthly fm
              ON fm.product_line_id = d.product_line_id
             AND fm.snapshot_date = s."SnapshotDate"
            WHERE fm.product_line_id IS NULL;
            """)

            total_rows += len(sub)
            con.unregister("tmp_sub")
            print("    inserted (this file): OK")

        except Exception as e:
            problems.append(f"{os.path.basename(f)} error: {e}")
            print("    ERROR:", e)

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
