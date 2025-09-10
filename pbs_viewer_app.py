import streamlit as st
import duckdb, pandas as pd
from pathlib import Path
import altair as alt
import gdown

# ---- Page setup ----
st.set_page_config(page_title="PBS AEMP Viewer", layout="wide")
st.title("PBS AEMP Price Viewer")

def ensure_db() -> Path:
    """Make sure ./out/pbs_prices.duckdb exists; download from Drive if missing."""
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "pbs_prices.duckdb"
    if not db_path.exists():
        with st.spinner("Downloading database from Google Drive (first run only)…"):
            # Make sure this Drive file is set to: Anyone with the link → Viewer
            url = "https://drive.google.com/file/d/1tVpP0p3XdSPyzn_GEs6T_q7I1Zkk3Veb/view?usp=sharing"
            gdown.download(url, str(db_path), quiet=False, fuzzy=True)

    st.caption(f"DB path: {db_path}")
    if not db_path.exists():
        st.error("Database file not found after download."); st.stop()

    size_bytes = db_path.stat().st_size
    st.caption(f"DB size: {size_bytes:,} bytes")
    if size_bytes < 1024:
        st.error("Downloaded DB looks too small (likely failed download)."); st.stop()
    return db_path

DB_PATH = ensure_db()

# ---- Open DuckDB ----
try:
    con = duckdb.connect(str(DB_PATH), read_only=True)
except Exception as e:
    st.error(f"DuckDB couldn’t open the DB at {DB_PATH}.\n\n{e}")
    st.stop()

# ---- Metadata (left block) using latest snapshot for Brand/Formulary/AMT ----
meta_sql = """
WITH d AS (
  SELECT
    product_line_id,
    item_code_b,
    name_a,
    attr_c AS form_c,
    attr_f AS form_f,
    name_b AS responsible_person,
    brand_name AS line_brand,
    formulary  AS line_formulary
  FROM dim_product_line
),
fm AS (
  SELECT
    product_line_id,
    brand_name,
    formulary,
    amt_trade_product_pack,
    ROW_NUMBER() OVER (PARTITION BY product_line_id ORDER BY snapshot_date DESC) AS rn
  FROM fact_monthly
)
SELECT
  d.item_code_b                                        AS "Item Code",
  d.name_a                                             AS "Legal Instrument Drug",
  COALESCE(d.form_c, d.form_f)                         AS "Legal Instrument Form",
  COALESCE(fm.brand_name, d.line_brand)                AS "Brand Name",
  COALESCE(fm.formulary,  d.line_formulary)            AS "Formulary",
  d.responsible_person                                 AS "Responsible Person",
  fm.amt_trade_product_pack                            AS "AMT Trade Product Pack"
FROM d
LEFT JOIN fm
  ON fm.product_line_id = d.product_line_id AND fm.rn = 1
WHERE lower(d.name_a) = lower(?)
ORDER BY 1;
"""

@st.cache_data
def get_drugs():
    sql = "SELECT DISTINCT name_a AS drug FROM dim_product_line ORDER BY 1"
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
        df = con.execute(sql, [drug]).df()
    return df

@st.cache_data
def build_export_table(drug: str) -> pd.DataFrame:
    # Long series per item
    s = get_series(drug, merge_codes=False).copy()
    s["month"] = pd.to_datetime(s["month"], errors="coerce")
    s = s.dropna(subset=["month"])

    # Month headers like "AEMP Aug 13"
    s["col_label"] = "AEMP " + s["month"].dt.strftime("%b %y")

    # Wide pivot for prices
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

    # Left metadata (latest snapshot fields included)
    meta = con.execute(meta_sql, [drug]).df()

    # Join meta + prices
    out = meta.merge(wide, on="Item Code", how="left")

    # Fixed left columns (exact order) + months (chronological)
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




