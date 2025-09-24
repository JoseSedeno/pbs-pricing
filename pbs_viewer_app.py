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
  CASE
    WHEN COALESCE(CAST(d.line_formulary AS VARCHAR), CAST(fm.fm_formulary AS VARCHAR)) IN ('1','F1')  THEN 'F1'
    WHEN COALESCE(CAST(d.line_formulary AS VARCHAR), CAST(fm.fm_formulary AS VARCHAR)) IN ('60','F2') THEN 'F2'
    WHEN UPPER(COALESCE(CAST(d.line_formulary AS VARCHAR), CAST(fm.fm_formulary AS VARCHAR))) = 'CDL' THEN 'CDL'
    ELSE NULLIF(COALESCE(CAST(d.line_formulary AS VARCHAR), CAST(fm.fm_formulary AS VARCHAR)),'')
  END                                           AS "Formulary",
  COALESCE(
    CAST(d.responsible_person AS VARCHAR),
    CAST(fm.fm_responsible_person AS VARCHAR)
  )                                             AS "Responsible Person",
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

# ---- Wide table (unchanged) ----
@st.cache_data
def build_export_table(drug: str) -> pd.DataFrame:
    # 1) Long series per item
    s = get_series(drug, merge_codes=False).copy()
    s["month"] = pd.to_datetime(s["month"], errors="coerce")
    s = s.dropna(subset=["month"])

    # 2) Month headers like "AEMP Aug 13"
    s["col_label"] = "AEMP " + s["month"].dt.strftime("%b %y")

    # 3) Wide pivot for prices (one row per Item Code; one column per month)
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

    # 4) Left metadata (latest snapshot fields included via meta_sql)
    meta = (
        con.execute(meta_sql, [drug])
        .df()
        .drop_duplicates(subset=["Item Code"])
    )

    # 5) Join meta + prices
    out = meta.merge(wide, on="Item Code", how="left")

    # 6) Fixed left columns (exact order) + month columns (chronological)
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
    month_cols = sorted(
        month_cols,
        key=lambda c: pd.to_datetime(c.replace("AEMP ", ""), format="%b %y", errors="coerce")
    )

    return out[[c for c in fixed if c in out.columns] + month_cols]

# ---- Chart data from wide table (Month → Identifier → AEMP) ----
@st.cache_data
def build_chart_df(drug: str) -> pd.DataFrame:
    """
    Build a long dataframe for the chart with columns:
    month (datetime), display_name (new identifier), aemp (float).
    Uses the existing wide export table; does NOT change that table.
    """
    base = build_export_table(drug).copy()
    if base.empty:
        return base.assign(month=pd.NaT, display_name="", aemp=pd.NA).head(0)

    # Helper for safe strings
    def nz(col):
        return col.astype(str).replace({"None": "", "nan": ""})

    # New identifier = Item Code · Formulary · Brand Name · Legal Instrument Form · AMT Pack · Responsible Person
    base["display_name"] = (
        nz(base["Item Code"]) + " · " +
        nz(base["Formulary"]) + " · " +
        nz(base["Brand Name"]) + " · " +
        nz(base["Legal Instrument Form"]) + " · " +
        nz(base["AMT Trade Product Pack"]) + " · " +
        nz(base["Responsible Person"])
    ).str.replace(r"\s+·\s+$", "", regex=True)

    # Unpivot month columns
    month_cols = [c for c in base.columns if c.startswith("AEMP ")]
    long_df = base.melt(
        id_vars=["display_name"],
        value_vars=month_cols,
        var_name="month_label",
        value_name="aemp",
    )

    # Parse "AEMP Aug 13" → datetime
    long_df["month"] = pd.to_datetime(
        long_df["month_label"].str.replace("AEMP ", "", regex=False),
        format="%b %y",
        errors="coerce"
    )

    # Clean & order columns EXACTLY as requested: Month → Identifier → AEMP
    long_df = long_df.dropna(subset=["month"]).drop(columns=["month_label"])
    long_df = long_df.sort_values(["display_name", "month"])
    return long_df[["month", "display_name", "aemp"]]

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Filters")
    all_drugs = get_drugs()
    if not all_drugs:
        st.error("No drugs found in dim_product_line."); st.stop()

    # Multi-select up to 3 medicines
    selected_drugs = st.multiselect(
        "Legal Instrument Drug(s)",
        options=all_drugs,
        default=all_drugs[:1],
        max_selections=3
    )

    # Keep (even if chart path doesn’t use merge)
    merge = st.checkbox("Merge Item Codes (treat as single product)", value=False)

    st.caption(
        "Pick 1–3 drugs to combine on the chart. "
        "The wide table/export uses the first selected drug."
    )

# ---- Series & chart (compare across multiple drugs) ----
st.write(f"**Database:** `{DB_PATH}`")

# Build one long table across the selected drugs
frames = []
for d in (selected_drugs or []):
    df_d = build_chart_df(d)
    if not df_d.empty:
        df_d = df_d.copy()
        # Prefix identifier so the source drug is clear
        df_d["display_name"] = f"{d} · " + df_d["display_name"]
        df_d["__drug__"] = d
        frames.append(df_d)

if frames:
    chart_df = pd.concat(frames, ignore_index=True)
else:
    chart_df = pd.DataFrame(columns=["month", "display_name", "aemp", "__drug__"])

# Show which drugs are included
title_drug = ", ".join(selected_drugs) if selected_drugs else "(none)"
st.write(f"**Drug(s):** {title_drug}")

if chart_df.empty:
    st.warning("No data for the selected drug(s)."); st.stop()

# Time-range slider (uses chart_df to set bounds)
min_m, max_m = chart_df["month"].min(), chart_df["month"].max()
with st.sidebar:
    st.subheader("Time range")
    start_m, end_m = st.slider(
        "Select months",
        min_value=min_m.to_pydatetime(),
        max_value=max_m.to_pydatetime(),
        value=(min_m.to_pydatetime(), max_m.to_pydatetime()),
        format="MMM YYYY",
    )

# Apply time filter
mask = (chart_df["month"] >= pd.to_datetime(start_m)) & (chart_df["month"] <= pd.to_datetime(end_m))
chart_df = chart_df.loc[mask].copy()

# Pick up to 3 identifiers to compare
all_ids = sorted(chart_df["display_name"].unique().tolist())
with st.sidebar:
    st.subheader("Compare up to 3 products")
    picked = st.multiselect("Identifiers", options=all_ids, default=[], max_selections=3)

# If any picked, filter; otherwise show all
if picked:
    chart_df = chart_df[chart_df["display_name"].isin(picked)]

# Build chart
chart = (
    alt.Chart(chart_df.sort_values("month"))
    .mark_line(point=True)
    .encode(
        x=alt.X("month:T", axis=alt.Axis(title="Month", format="%b %Y", labelAngle=0)),
        y=alt.Y("aemp:Q", title="AEMP"),
        color=alt.Color("display_name:N", title="Identifier"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%Y-%m"),
            alt.Tooltip("display_name:N", title="Identifier"),
            alt.Tooltip("aemp:Q", title="AEMP"),
        ],
    )
    # ✅ show all selected drugs in the title
    .properties(height=450, title=alt.TitleParams(f"{title_drug} — AEMP by month", anchor="start"))
    .interactive(bind_x=True)
)

st.altair_chart(chart, use_container_width=True)

# ---- Tiny stats: latest AEMP and MoM% for currently visible identifiers ----
def latest_mom(df: pd.DataFrame) -> pd.DataFrame:
    # expects filtered chart_df columns: month, display_name, aemp
    out = []
    for name, g in df.sort_values("month").groupby("display_name", as_index=False):
        if g.shape[0] == 0:
            continue
        last = g.iloc[-1]
        prev = g.iloc[-2] if g.shape[0] >= 2 else None
        latest_val = float(last["aemp"]) if pd.notna(last["aemp"]) else None
        prev_val = float(prev["aemp"]) if (prev is not None and pd.notna(prev["aemp"])) else None
        if latest_val is not None and prev_val not in (None, 0):
            mom = (latest_val - prev_val) / prev_val * 100.0
        else:
            mom = None
        out.append(
            {
                "Identifier": name,
                "Latest Month": last["month"].strftime("%b %Y"),
                "AEMP (latest)": latest_val,
                "AEMP (prev)": prev_val,
                "MoM %": None if mom is None else round(mom, 2),
            }
        )
    return pd.DataFrame(out)

stats_df = latest_mom(chart_df)
if not stats_df.empty:
    st.markdown("#### Selected lines — latest & MoM")
    st.dataframe(stats_df, use_container_width=True)

# ---- Small table under the chart (Month → Identifier → AEMP) ----
st.dataframe(
    chart_df.assign(Month=chart_df["month"].dt.strftime("%b %Y"))[["Month", "display_name", "aemp"]],
    use_container_width=True
)

# ---- Wide table & download (respects time range; drop empty rows) ----
st.markdown("### Item info + AEMP by month (wide)")
st.caption("Product columns first, then monthly AEMP columns in chronological order.")

# Use the first selected drug for the wide table/export
export_base = selected_drugs[0] if selected_drugs else None
if not export_base:
    st.warning("Pick at least one drug to show the wide table/export."); st.stop()

# Full wide table
export_df = build_export_table(export_base)

# Build list of month columns within the slider range
start_dt = pd.to_datetime(start_m).to_period("M").to_timestamp()
end_dt   = pd.to_datetime(end_m).to_period("M").to_timestamp()

def _col_to_month(col: str) -> pd.Timestamp:
    return pd.to_datetime(col.replace("AEMP ", ""), format="%b %y", errors="coerce")

month_cols_all = [c for c in export_df.columns if c.startswith("AEMP ")]
kept_month_cols = [c for c in month_cols_all
                   if (_col_to_month(c) >= start_dt) and (_col_to_month(c) <= end_dt)]

# Keep only rows with at least one value in-range
fixed_cols = [
    "Item Code",
    "Legal Instrument Drug",
    "Legal Instrument Form",
    "Brand Name",
    "Formulary",
    "Responsible Person",
    "AMT Trade Product Pack",
]

if kept_month_cols:
    nonempty_mask = export_df[kept_month_cols].notna().any(axis=1)
    filtered_df = export_df.loc[nonempty_mask, [c for c in fixed_cols if c in export_df.columns] + kept_month_cols]
else:
    filtered_df = export_df[[c for c in fixed_cols if c in export_df.columns]].iloc[0:0]

st.dataframe(filtered_df, use_container_width=True)

# Download button (filename includes the range)
file_range = f"{start_dt:%Y-%m}_{end_dt:%Y-%m}"
export_csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download AEMP wide CSV — {export_base}",
    data=export_csv,
    file_name=f"{export_base.replace(' ','_').lower()}_{file_range}_aemp_wide.csv",
    mime="text/csv",
)








