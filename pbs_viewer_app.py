import streamlit as st
import duckdb, pandas as pd
from pathlib import Path
import altair as alt

# Path to your DuckDB database (adjust if you moved it)
DB_PATH = Path("./out/pbs_prices.duckdb")

st.set_page_config(page_title="PBS AEMP Viewer", layout="wide")
st.title("PBS AEMP Price Viewer")

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

con = duckdb.connect(str(DB_PATH), read_only=True)

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
            x=alt.X("month:T", title="Month"),
            y=alt.Y("aemp:Q", title="AEMP"),
            tooltip=["month:T", "aemp:Q"],
        )
        .properties(height=450)
    )
else:
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("aemp:Q", title="AEMP"),
            color=alt.Color("item_code:N", title="Item Code"),
            tooltip=["month:T", "item_code:N", "aemp:Q"],
        )
        .properties(height=450)
    )

st.altair_chart(chart, use_container_width=True)

# Show a small table under the chart
st.dataframe(df)

# Download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name=f"{drug.replace(' ','_').lower()}_aemp_series.csv", mime="text/csv")

