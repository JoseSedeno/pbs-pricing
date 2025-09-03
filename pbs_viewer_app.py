import streamlit as st
import duckdb, pandas as pd
from pathlib import Path
import altair as alt
import gdown

# Streamlit page config should be the first Streamlit call
st.set_page_config(page_title="PBS AEMP Viewer", layout="wide")
st.title("PBS AEMP Price Viewer")

def ensure_db() -> Path:
    """Ensure a local duckdb file exists in a writable place on Streamlit Cloud."""
    # Use a writable temp directory on Streamlit Cloud
    cache_dir = Path("/tmp/pbs_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    db_path = cache_dir / "pbs_prices.duckdb"

    # If missing or suspiciously small, (re)download from Google Drive
    if not db_path.exists() or db_path.stat().st_size < 1_000_000:
        with st.spinner("Downloading database from Google Drive (first run only)…"):
            url = "https://drive.google.com/uc?id=1A1xcx8b2Nl0v9X6gMx10jZI-XWheVsBn"
            gdown.download(url, str(db_path), quiet=False)

    return db_path

# Path to your DuckDB database (adjust if you moved it)
DB_PATH = ensure_db()  # <- new
    
# --- sanity checks before connecting ---
st.caption(f"DB path: {DB_PATH}")
if not DB_PATH.exists():
    st.error(
        "Database file not found after download.\n\n"
        "Likely causes:\n"
        "• Google Drive link is not set to “Anyone with the link – Viewer”\n"
        "• File ID in the URL is wrong\n"
        "• Download blocked by Drive confirmation"
    )
    st.stop()

size_bytes = DB_PATH.stat().st_size
st.caption(f"DB size: {size_bytes:,} bytes")
if size_bytes < 1024:   # tiny -> almost certainly a failed download
    st.error(
        "The downloaded database looks too small.\n\n"
        "Please check the Drive link permissions and file ID."
    )
    st.stop()

# ---- actually open the DB ----
try:
    con = duckdb.connect(str(DB_PATH), read_only=True)
except Exception as e:
    st.error(f"DuckDB couldn’t open the DB at {DB_PATH}.\n\n{e}")
    st.stop()

# Make a view with friendlier names so we can select by the labels we expect
meta_sql = """
WITH d AS (
  SELECT
    product_line_id,
    item_code_b,
    name_a,
    attr_f AS legal_instrument_form,
    attr_g AS legal_instrument_moa,
    brand_name,
    formulary,
    manufacturer_code
  FROM dim_product_line
)
SELECT
  d.item_code_b                AS "Item Code",
  d.name_a                     AS "Legal Instrument Drug",
  d.legal_instrument_form      AS "Legal Instrument Form",
  d.legal_instrument_moa       AS "Legal Instrument MoA",
  d.brand_name                 AS "Brand Name",
  d.formulary                  AS "Formulary",
  d.manufacturer_code          AS "Manufacturer Code"
FROM d
WHERE lower(d.name_a) = lower(?)
ORDER BY 1
"""

@st.cache_data
def get_drugs():
    sql = "SELECT DISTINCT name_a AS drug FROM dim_product_line ORDER BY 1"
    return con.execute(sql).df()["drug"].tolist()

@st.cache_data
def get_series(drug: str, merge_codes: bool):
    if merge_codes:
        # Treat all item codes for this drug as one series (average if multiple per month)
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
        # One line per Item Code (default behavior)
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
    # 1) Long series per item (ignore merge toggle for export)
    s = get_series(drug, merge_codes=False).copy()
    s["month"] = pd.to_datetime(s["month"], errors="coerce")
    s = s.dropna(subset=["month"])

    # 2) Month headers like "AEMP Aug 13"
    s["col_label"] = "AEMP " + s["month"].dt.strftime("%b %y")

    # 3) Pivot to wide (one row per Item Code, one col per month)
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

    # 4) Item metadata to appear before month columns
    meta = con.execute(meta_sql, [drug]).df()  # uses the CTE defined at the top of the file
    # st.write("META COLUMNS:", list(meta.columns))
    # st.write(meta.head())

    # 5) Join meta + prices
    out = meta.merge(wide, on="Item Code", how="left")

    # 6) Fixed columns first, then months in chronological order
    fixed = [
        "Item Code",
        "Legal Instrument Drug",
        "Legal Instrument Form",
        "Legal Instrument MoA",
        "Brand Name",
        "Formulary",
        "Manufacturer Code",
    ]
    month_cols = [c for c in out.columns if c.startswith("AEMP ")]
    month_cols = sorted(
        month_cols,
        key=lambda c: pd.to_datetime(c.replace("AEMP ", ""), format="%b %y"),
    )

    return out[[c for c in fixed if c in out.columns] + month_cols]

with st.sidebar:
    st.subheader("Filters")
    drugs = get_drugs()
    if not drugs:
        st.error("No drugs found in dim_product_line.")
        st.stop()
    drug = st.selectbox("Legal Instrument Drug", drugs, index=0)
    merge = st.checkbox("Merge Item Codes (treat as single product)", value=False)
    st.caption("Tip: leave this OFF to see separate lines per Item Code, "
               "ON to view a single continuous series.")

df = get_series(drug, merge)
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.sort_values("month")

if df.empty:
    st.warning("No data for this selection.")
    st.stop()

st.write(f"**Database:** `{DB_PATH}`")
st.write(f"**Drug:** {drug}")

if merge:
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month:T",
                axis=alt.Axis(title="Month", format="%b %Y", labelAngle=0)
            ),
            y=alt.Y("aemp:Q", title="AEMP"),
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%Y-%m"),
                alt.Tooltip("aemp:Q", title="AEMP"),
            ],
        )
        .properties(
            height=450,
            title=alt.TitleParams(f"{drug} — AEMP by month", anchor="start")
        )
        .interactive(bind_x=True)
    )
else:
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month:T",
                axis=alt.Axis(title="Month", format="%b %Y", labelAngle=0)
            ),
            y=alt.Y("aemp:Q", title="AEMP"),
            color=alt.Color("item_code:N", title="Item Code"),
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%Y-%m"),
                alt.Tooltip("item_code:N", title="Item"),
                alt.Tooltip("aemp:Q", title="AEMP"),
            ],
        )
        .properties(
            height=450,
            title=alt.TitleParams(f"{drug} — AEMP by month", anchor="start")
        )
        .interactive(bind_x=True)
    )

st.altair_chart(chart, use_container_width=True)

# ---- Small table under the chart (pretty Month)
df_small = df.copy()
df_small["Month"] = pd.to_datetime(df_small["month"], errors="coerce").dt.strftime("%b %Y")
if "item_code" in df_small.columns:
    small_cols = ["Month", "item_code", "aemp"]
else:
    small_cols = ["Month", "aemp"]
st.dataframe(df_small[small_cols], use_container_width=True)

# ===== SECTION: WIDE TABLE & DOWNLOAD (Item info + AEMP by month) =====
st.markdown("### Item info + AEMP by month (wide)")
st.caption("Product columns first, then monthly AEMP columns in chronological order.")

# Build the wide table
export_df = build_export_table(drug)
st.dataframe(export_df, use_container_width=True)

# Download (CSV of the same wide table shown above)
export_csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download AEMP wide CSV — {drug}",
    data=export_csv,
    file_name=f"{drug.replace(' ','_').lower()}_aemp_wide.csv",
    mime="text/csv",
)
# ===== END SECTION: WIDE TABLE & DOWNLOAD =====



