import os
from pathlib import Path

import streamlit as st
import duckdb, pandas as pd
import altair as alt
import gdown

# ---- Page setup ----
st.set_page_config(page_title="PBS AEMP Viewer", layout="wide")
st.title("PBS AEMP Price Viewer")

def ensure_db() -> Path:
    """
    Use a local DuckDB. Order of preference:
      1) PBS_DB_PATH env var
      2) ./out/pbs_prices.duckdb (download from Drive if missing)
    """
    # 1) Env var override
    env = os.environ.get("PBS_DB_PATH")
    if env:
        p = Path(env)
        if p.exists() and p.stat().st_size > 1024:
            st.caption(f"Using DB (PBS_DB_PATH): {p}")
            return p

    # 2) Local ./out path, download if needed
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "pbs_prices.duckdb"

    if not db_path.exists():
        with st.spinner("Downloading database from Google Drive (first run only)…"):
            # Make sure this Drive file is set to: Anyone with the link → Viewer
            url = "https://drive.google.com/uc?id=1tVpP0p3XdSPyzn_GEs6T_q7I1Zkk3Veb&export=download"
            gdown.download(url, str(db_path), quiet=False)

    st.caption(f"DB path: {db_path}")
    if not db_path.exists():
        st.error("Database file not found after download."); st.stop()
        
    min_bytes = 100 * 1024 * 1024  # 100 MB safety threshold
    size_bytes = db_path.stat().st_size
    st.caption(f"DB size: {size_bytes:,} bytes")
    if size_bytes < min_bytes:
        st.error("Downloaded DB looks too small (likely failed download)."); st.stop()
    return db_path

DB_PATH = ensure_db()

# ---- Open DuckDB ----
try:
    con = duckdb.connect(str(DB_PATH), read_only=True)
except Exception as e:
    st.error(f"DuckDB couldn’t open the DB at {DB_PATH}.\n\n{e}")
    st.stop()

# ---------- helpers to adapt to schema ----------
@st.cache_data
def table_columns(table: str):
    rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    # (cid, name, type, notnull, dflt_value, pk)
    return {r[1].lower(): r[1] for r in rows}  # {lower: exact_case}

def pick_col(cols_map, *candidates, default="NULL"):
    """Return SQL identifier for the first existing candidate, else default literal."""
    for c in candidates:
        if c and c.lower() in cols_map:
            return f'"{cols_map[c.lower()]}"'
    return default

# Inspect available columns in both tables (schema-safe)
d_cols  = table_columns("dim_product_line")
fm_cols = table_columns("fact_monthly")

# Safe selections for left metadata
item_code_expr   = pick_col(d_cols, "item_code_b", "item_code")
form_expr        = pick_col(d_cols, "attr_c", "attr_f", "variant_signature_base", default="NULL")

line_brand_expr  = pick_col(d_cols, "brand_name", default="NULL")
line_formul_expr = pick_col(d_cols, "formulary", "attr_j", default="NULL")
resp_expr = pick_col(
    d_cols,
    "sponsor", "responsible_person_name", "responsible_person",
    "sponsor_name", "rp_name",
    "name_b", "resp_person", "contact_person",
    default="NULL"
)

# Latest snapshot fields from fact_monthly
fm_brand_expr  = pick_col(fm_cols, "brand_name", default="NULL")
fm_formul_expr = pick_col(fm_cols, "formulary", default="NULL")
fm_amt_expr    = pick_col(fm_cols, "amt_trade_product_pack", "amt_trade_pack", default="NULL")
fm_resp_expr   = pick_col(fm_cols, "responsible_person", "sponsor", "responsible_person_name", "rp_name", default="NULL")

# ---- Metadata (left block) using latest snapshot for Brand/Formulary/AMT ----
meta_sql = f"""
WITH d AS (
  SELECT
    "product_line_id"                           AS product_line_id,
    {item_code_expr}                            AS item_code_b,
    "name_a"                                    AS name_a,
    {form_expr}                                 AS form_src,
    {resp_expr}                                 AS responsible_person,
    {line_brand_expr}                           AS line_brand,       -- fallback only
    {line_formul_expr}                          AS line_formulary    -- fallback only
  FROM "dim_product_line"
),
fm AS (
  SELECT
    "product_line_id"                           AS product_line_id,
    {fm_brand_expr}                             AS fm_brand_name,
    {fm_formul_expr}                            AS fm_formulary,
    {fm_amt_expr}                               AS fm_amt_trade_product_pack,
    {fm_resp_expr}                              AS fm_responsible_person,
    ROW_NUMBER() OVER (
      PARTITION BY "product_line_id" ORDER BY "snapshot_date" DESC
    ) AS rn
  FROM "fact_monthly"
)
SELECT
  d.item_code_b                                 AS "Item Code",
  d.name_a                                      AS "Legal Instrument Drug",
  d.form_src                                    AS "Legal Instrument Form",
  COALESCE(CAST(fm.fm_brand_name AS VARCHAR), CAST(d.line_brand AS VARCHAR))     AS "Brand Name",
  COALESCE(
    NULLIF(CAST(d.line_formulary AS VARCHAR), ''),
    CASE
      WHEN fm.fm_formulary IN (1, '1')  THEN 'F1'
      WHEN fm.fm_formulary IN (60, '60') THEN 'F2'
      ELSE CAST(fm.fm_formulary AS VARCHAR)
    END
  )                                             AS "Formulary",
  COALESCE(
  CAST(d.responsible_person AS VARCHAR),
  CAST(fm.fm_responsible_person AS VARCHAR)
) AS "Responsible Person",
  fm.fm_amt_trade_product_pack                  AS "AMT Trade Product Pack"
FROM d
LEFT JOIN fm
  ON fm.product_line_id = d.product_line_id AND fm.rn = 1
WHERE lower(d.name_a) = lower(?)
ORDER BY 1;
"""

@st.cache_data
def get_drugs():
    sql = 'SELECT DISTINCT "name_a" AS drug FROM "dim_product_line" ORDER BY 1'
    return con.execute(sql).df()["drug"].tolist()

@st.cache_data
def get_series(drug: str, merge_codes: bool):
    if merge_codes:
        sql = """
        SELECT
          f.snapshot_date::DATE AS month,
          AVG(f.aemp) AS aemp
        FROM fact_monthly f
        JOIN dim_product_line d USING (product_line_id)
        WHERE lower(d.name_a) = lower(?)
        GROUP BY 1
        ORDER BY 1
        """
        df = con.execute(sql, [drug]).df()
    else:
        sql = """
        SELECT
          f.snapshot_date::DATE AS month,
          d.item_code_b         AS item_code,
          f.aemp
        FROM fact_monthly f
        JOIN dim_product_line d USING (product_line_id)
        WHERE lower(d.name_a) = lower(?)
        ORDER BY 1, 2
        """
        try:
            df = con.execute(sql, [drug]).df()
        except duckdb.BinderException:
            # fallback if item_code_b doesn't exist
            sql_fallback = sql.replace("d.item_code_b", "d.item_code")
            df = con.execute(sql_fallback, [drug]).df()
    return df

@st.cache_data
def build_export_table(drug: str) -> pd.DataFrame:
    # 1) Long series per item
    s = get_series(drug, merge_codes=False).copy()
    s["month"] = pd.to_datetime(s["month"], errors="coerce")
    s = s.dropna(subset=["month"])

    # 2) Month headers like "AEMP Aug 13"
    s["col_label"] = "AEMP " + s["month"].dt.strftime("%b %y")

    # 3) Wide pivot for prices
    wide = (
        s.pivot_table(
            index="item_code",
            columns="col_label",
            values="aemp",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"item_code": "Item Code"})
    )

    # 4) Left metadata (latest snapshot fields included)
    meta = con.execute(meta_sql, [drug]).df().drop_duplicates(subset=["Item Code"])

    # 5) Join meta + prices
    out = meta.merge(wide, on="Item Code", how="left")

    # 6) Fixed left columns (exact order) + months (chronological)
    fixed = [
        "Item Code",
        "Legal Instrument Drug",
        "Legal Instrument Form",
        "Brand Name",
        "Formulary",
        "Responsible Person",
        "AMT Trade Product Pack",
    ]
    month_cols = [c for c in out.columns if c.startswith("AEMP ")]
    month_cols = sorted(month_cols, key=lambda c: pd.to_datetime(c.replace("AEMP ", ""), format="%b %y"))

    return out[[c for c in fixed if c in out.columns] + month_cols]

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Filters")
    drugs = get_drugs()
    if not drugs:
        st.error("No drugs found in dim_product_line."); st.stop()
    drug = st.selectbox("Legal Instrument Drug", drugs, index=0)
    merge = st.checkbox("Merge Item Codes (treat as single product)", value=False)
    st.caption(
        "Tip: leave this OFF to see separate lines per Item Code, "
        "ON to view a single continuous series."
    )

# ---- Debug: Responsible Person column check (temporary) ----
st.markdown("### Debug: Responsible Person column check")
with st.expander("Debug: find Responsible Person column (temporary)"):
    cols_df = con.execute("""
        SELECT name
        FROM pragma_table_info('dim_product_line')
        ORDER BY name
    """).df()
    st.write("dim_product_line columns:", cols_df)

    sample = con.execute("""
        SELECT *
        FROM dim_product_line
        WHERE lower(name_a) = lower(?)
        LIMIT 1
    """, [drug]).df().T
    st.write("One sample row from dim_product_line:", sample)
    
# Also show fact_monthly columns + one sample row for the same drug
    fm_cols_df = con.execute("""
        SELECT name
        FROM pragma_table_info('fact_monthly')
        ORDER BY name
    """).df()
    st.write("fact_monthly columns:", fm_cols_df)

    fm_sample = con.execute("""
        SELECT *
        FROM fact_monthly f
        JOIN dim_product_line d USING (product_line_id)
        WHERE lower(d.name_a) = lower(?)
        ORDER BY snapshot_date DESC
        LIMIT 1
    """, [drug]).df().T
    st.write("One sample row from fact_monthly (joined):", fm_sample)

# ---- Series & chart ----
df = get_series(drug, merge)
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.sort_values("month")
if df.empty:
    st.warning("No data for this selection."); st.stop()

st.write(f"**Database:** `{DB_PATH}`")
st.write(f"**Drug:** {drug}")

if merge:
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", axis=alt.Axis(title="Month", format="%b %Y", labelAngle=0)),
            y=alt.Y("aemp:Q", title="AEMP"),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%Y-%m"),
                     alt.Tooltip("aemp:Q", title="AEMP")],
        )
        .properties(height=450, title=alt.TitleParams(f"{drug} — AEMP by month", anchor="start"))
        .interactive(bind_x=True)
    )
else:
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", axis=alt.Axis(title="Month", format="%b %Y", labelAngle=0)),
            y=alt.Y("aemp:Q", title="AEMP"),
            color=alt.Color("item_code:N", title="Item Code"),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%Y-%m"),
                     alt.Tooltip("item_code:N", title="Item"),
                     alt.Tooltip("aemp:Q", title="AEMP")],
        )
        .properties(height=450, title=alt.TitleParams(f"{drug} — AEMP by month", anchor="start"))
        .interactive(bind_x=True)
    )

st.altair_chart(chart, use_container_width=True)

# ---- Small table under the chart ----
df_small = df.copy()
df_small["Month"] = pd.to_datetime(df_small["month"], errors="coerce").dt.strftime("%b %Y")
small_cols = ["Month", "aemp"] if "item_code" not in df_small.columns else ["Month", "item_code", "aemp"]
st.dataframe(df_small[small_cols], use_container_width=True)

# ---- Wide table & download ----
st.markdown("### Item info + AEMP by month (wide)")
st.caption("Product columns first, then monthly AEMP columns in chronological order.")
export_df = build_export_table(drug)
st.dataframe(export_df, use_container_width=True)

export_csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download AEMP wide CSV — {drug}",
    data=export_csv,
    file_name=f"{drug.replace(' ','_').lower()}_aemp_wide.csv",
    mime="text/csv",
)





