import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import altair as alt
import gdown
st.set_page_config(page_title="PBS AEMP Viewer", layout="wide")
st.title("PBS AEMP Price Viewer")

# ---- Simple auth gate (must be above any data code) ----
from datetime import datetime

_auth = st.secrets.get("auth", {})
AUTH_NONCE = _auth.get("AUTH_NONCE", "")
PASSWORDS = dict(_auth.get("PASSWORDS", {}))  # {client_id: password}

def _is_authed() -> bool:
    tok = st.session_state.get("auth_ticket")
    return bool(tok and tok.get("nonce") == AUTH_NONCE and tok.get("client_id"))

def _login_ui():
    st.subheader("Secure login")
    with st.form("login"):
        client_id = st.text_input("Client ID")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        if client_id in PASSWORDS and pw == PASSWORDS[client_id]:
            st.session_state["auth_ticket"] = {
                "client_id": client_id,
                "nonce": AUTH_NONCE,
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
            }
            st.success("Signed in")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not _is_authed():
    _login_ui()
    st.stop()

with st.sidebar:
    if _is_authed():
        st.caption(f"Signed in as: {st.session_state['auth_ticket']['client_id']}")
        if st.button("Logout"):
            st.session_state.pop("auth_ticket", None)
            st.rerun()

def show_month_to_month_increases(con):
    # ---- Gate the whole section behind a sidebar toggle ----
    with st.sidebar:
        show_mom = st.toggle("Show month-to-month price increases", value=False, key="mom_show")
    if not show_mom:
        return

    # Local helper: only prefix with alias if it's a real column, not the literal NULL
    def _qualify(expr: str, alias: str) -> str:
        return f"{alias}.{expr}" if expr and expr.strip().upper() != "NULL" else "NULL"

    # Get available months from fact_monthly using snapshot_date
    months = (
        con.execute("""
            SELECT DISTINCT snapshot_date::DATE AS month
            FROM fact_monthly
            WHERE aemp IS NOT NULL
            ORDER BY 1
        """).df()["month"].tolist()
    )
    if not months:
        st.info("No months available in the database.")
        return

    latest = months[-1]
    prev = months[-2] if len(months) >= 2 else months[-1]

    # Default to latest two months but allow manual selection
    mode = st.radio(
        "Comparison range",
        ("Latest two months", "Pick months"),
        horizontal=True,
        key="mom_mode",
    )

    if mode == "Pick months":
        col_a, col_b = st.columns(2)
        with col_a:
            start_month = st.selectbox(
                "Start month",
                months,
                index=max(len(months) - 2, 0),
                format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
                key="mom_start",
            )
        with col_b:
            end_month = st.selectbox(
                "End month",
                months,
                index=len(months) - 1,
                format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
                key="mom_end",
            )
    else:
        start_month, end_month = prev, latest

    if pd.to_datetime(start_month) >= pd.to_datetime(end_month):
        st.warning("End month must be after start month.")
        return

    # Build safe column refs using your schema-adapted expressions defined earlier
    item_code_sql  = _qualify(item_code_expr,  "d")
    form_sql       = _qualify(form_expr,       "d")
    fm_brand_sql   = _qualify(fm_brand_expr,   "f")
    line_brand_sql = _qualify(line_brand_expr, "d")
    fm_resp_sql    = _qualify(fm_resp_expr,    "f")
    resp_sql       = _qualify(resp_expr,       "d")

    # Compare AEMP between the two months (schema-safe join to dim_product_line)
    sql = f"""
        SELECT
            {item_code_sql}                                                            AS item_code,
            d.name_a                                                                    AS legal_instrument_drug,
            {form_sql}                                                                  AS legal_instrument_form,
            COALESCE(CAST({fm_brand_sql} AS VARCHAR), CAST({line_brand_sql} AS VARCHAR)) AS brand_name,
            COALESCE(CAST({fm_resp_sql}  AS VARCHAR), CAST({resp_sql}       AS VARCHAR)) AS responsible_person,
            SUM(CASE WHEN f.snapshot_date::DATE = ? THEN f.aemp END) AS aemp_start,
            SUM(CASE WHEN f.snapshot_date::DATE = ? THEN f.aemp END) AS aemp_end
        FROM fact_monthly f
        JOIN dim_product_line d USING (product_line_id)
        WHERE f.snapshot_date::DATE IN (?, ?)
        GROUP BY 1,2,3,4,5
        HAVING aemp_start IS NOT NULL
           AND aemp_end   IS NOT NULL
           AND aemp_end > aemp_start
        ORDER BY (aemp_end - aemp_start) DESC
    """
    df = con.execute(sql, [start_month, end_month, start_month, end_month]).df()

    nice_start = pd.to_datetime(start_month).strftime("%b %Y")
    nice_end   = pd.to_datetime(end_month).strftime("%b %Y")

    # Section title (only shown when toggle is ON)
    st.markdown(f"### AEMP price increases: {nice_start} to {nice_end}")

    if df.empty:
        st.info(f"No increases found between {nice_start} and {nice_end}.")
        return

    # Compute deltas and round for display/CSV
    df["abs_change"] = (df["aemp_end"] - df["aemp_start"]).round(2)
    df["pct_change"] = (
        ((df["aemp_end"] - df["aemp_start"]) / df["aemp_start"]) * 100
    ).round(2)
    df["aemp_start"] = df["aemp_start"].round(2)
    df["aemp_end"]   = df["aemp_end"].round(2)

    # Final column order
    df = df[
        [
            "item_code",
            "legal_instrument_drug",
            "legal_instrument_form",
            "brand_name",
            "responsible_person",
            "aemp_start",
            "aemp_end",
            "abs_change",
            "pct_change",
        ]
    ]

    # Summary (plain text to match app body style)
    items_count = len(df)
    largest_inc = float(df["abs_change"].max())
    median_inc  = float(df["abs_change"].median())
    total_inc   = float(df["abs_change"].sum())
    st.text(
        f"Summary: {items_count} items increased. "
        f"Largest +${largest_inc:,.2f}, median +${median_inc:,.2f}, total +${total_inc:,.2f}."
    )

    # Pretty percent for on-screen table only; keep CSV numeric
    df_display = df.copy()
    df_display["pct_change"] = df_display["pct_change"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
    )

    st.text(f"Showing increases from {nice_start} to {nice_end}.")
    st.dataframe(df_display, use_container_width=True)

    st.download_button(
        f"Download CSV: increases {nice_start} to {nice_end}",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"aemp_increases_{pd.to_datetime(start_month).strftime('%Y-%m')}_to_{pd.to_datetime(end_month).strftime('%Y-%m')}.csv",
        mime="text/csv",
    )

# ---------------- Month-to-month DECREASES ----------------
def show_month_to_month_decreases(con):
    # Gate the whole section behind a sidebar toggle
    with st.sidebar:
        show_mom_dec = st.toggle(
            "Show month-to-month price decreases",
            value=False,
            key="momd_show",
        )
    if not show_mom_dec:
        return

    # Helper: only prefix with alias if it's a real column (not literal NULL)
    def _qualify(expr: str, alias: str) -> str:
        return f"{alias}.{expr}" if expr and expr.strip().upper() != "NULL" else "NULL"

    # Available months
    months = (
        con.execute("""
            SELECT DISTINCT snapshot_date::DATE AS month
            FROM fact_monthly
            WHERE aemp IS NOT NULL
            ORDER BY 1
        """).df()["month"].tolist()
    )
    if not months:
        st.info("No months available in the database.")
        return

    latest = months[-1]
    prev = months[-2] if len(months) >= 2 else months[-1]

    # Month picker (keys distinct from the increases section)
    mode = st.radio(
        "Comparison range",
        ("Latest two months", "Pick months"),
        horizontal=True,
        key="momd_mode",
    )
    if mode == "Pick months":
        col_a, col_b = st.columns(2)
        with col_a:
            start_month = st.selectbox(
                "Start month",
                months,
                index=max(len(months) - 2, 0),
                format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
                key="momd_start",
            )
        with col_b:
            end_month = st.selectbox(
                "End month",
                months,
                index=len(months) - 1,
                format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
                key="momd_end",
            )
    else:
        start_month, end_month = prev, latest

    if pd.to_datetime(start_month) >= pd.to_datetime(end_month):
        st.warning("End month must be after start month.")
        return

    # Schema-safe refs
    item_code_sql  = _qualify(item_code_expr,  "d")
    form_sql       = _qualify(form_expr,       "d")
    fm_brand_sql   = _qualify(fm_brand_expr,   "f")
    line_brand_sql = _qualify(line_brand_expr, "d")
    fm_resp_sql    = _qualify(fm_resp_expr,    "f")
    resp_sql       = _qualify(resp_expr,       "d")

    # Decreases only
    sql = f"""
        SELECT
            {item_code_sql}                                                            AS item_code,
            d.name_a                                                                    AS legal_instrument_drug,
            {form_sql}                                                                  AS legal_instrument_form,
            COALESCE(CAST({fm_brand_sql} AS VARCHAR), CAST({line_brand_sql} AS VARCHAR)) AS brand_name,
            COALESCE(CAST({fm_resp_sql}  AS VARCHAR), CAST({resp_sql}       AS VARCHAR)) AS responsible_person,
            SUM(CASE WHEN f.snapshot_date::DATE = ? THEN f.aemp END) AS aemp_start,
            SUM(CASE WHEN f.snapshot_date::DATE = ? THEN f.aemp END) AS aemp_end
        FROM fact_monthly f
        JOIN dim_product_line d USING (product_line_id)
        WHERE f.snapshot_date::DATE IN (?, ?)
        GROUP BY 1,2,3,4,5
        HAVING aemp_start IS NOT NULL
           AND aemp_end   IS NOT NULL
           AND aemp_end < aemp_start
        ORDER BY (aemp_start - aemp_end) DESC
    """
    df = con.execute(sql, [start_month, end_month, start_month, end_month]).df()

    nice_start = pd.to_datetime(start_month).strftime("%b %Y")
    nice_end   = pd.to_datetime(end_month).strftime("%b %Y")
    st.markdown(f"### AEMP price decreases: {nice_start} to {nice_end}")

    if df.empty:
        st.info(f"No decreases found between {nice_start} and {nice_end}.")
        return

    # Deltas
    df["abs_change"] = (df["aemp_start"] - df["aemp_end"]).round(2)  # positive drop size
    df["pct_change"] = (
        ((df["aemp_end"] - df["aemp_start"]) / df["aemp_start"]) * 100
    ).round(2)  # negative %

    df["aemp_start"] = df["aemp_start"].round(2)
    df["aemp_end"]   = df["aemp_end"].round(2)

    # Final column order
    df = df[
        [
            "item_code",
            "legal_instrument_drug",
            "legal_instrument_form",
            "brand_name",
            "responsible_person",
            "aemp_start",
            "aemp_end",
            "abs_change",
            "pct_change",
        ]
    ]

    # Summary
    items_count  = len(df)
    largest_drop = float(df["abs_change"].max())
    median_drop  = float(df["abs_change"].median())
    total_drop   = float(df["abs_change"].sum())
    st.text(
        f"Summary: {items_count} items decreased. "
        f"Largest −${largest_drop:,.2f}, median −${median_drop:,.2f}, total −${total_drop:,.2f}."
    )

    # On-screen pretty percent (CSV stays numeric)
    df_display = df.copy()
    df_display["pct_change"] = df_display["pct_change"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
    )

    st.text(f"Showing decreases from {nice_start} to {nice_end}.")
    st.dataframe(df_display, use_container_width=True)

    st.download_button(
        f"Download CSV: decreases {nice_start} to {nice_end}",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"aemp_decreases_{pd.to_datetime(start_month).strftime('%Y-%m')}_to_{pd.to_datetime(end_month).strftime('%Y-%m')}.csv",
        mime="text/csv",
    )

# ---- Page setup ----
def ensure_db() -> Path:
    """
    Use a local DuckDB. Order of preference:
      1) PBS_DB_PATH env var (absolute or relative)
      2) ./out/pbs_prices.duckdb (download from Drive if missing or forced)

    Force a re-download by setting env PBS_DB_FORCE=1 for one run.
    """
    # 1) Env var override
    env = os.environ.get("PBS_DB_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists() and p.stat().st_size > 1024:
            st.caption(f"Using DB (PBS_DB_PATH): {p}")
            return p
        else:
            st.warning(f"PBS_DB_PATH set but file missing/small: {p}")

    # 2) Local ./out path
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = (out_dir / "pbs_prices.duckdb").resolve()

    # Optional: force refresh
    force_refresh = os.environ.get("PBS_DB_FORCE", "").strip().lower() in {"1", "true", "yes"}
    if force_refresh and db_path.exists():
        try:
            db_path.unlink()
            st.caption("Removed existing local DB (PBS_DB_FORCE=1).")
        except Exception as e:
            st.warning(f"Could not remove existing DB: {e}")

    # Download if missing
if not db_path.exists():
    with st.spinner("Downloading database from Google Drive (first run only)…"):
        # Make sure this Drive file is shared: Anyone with link → Viewer
        drive_id = st.secrets.get("drive", {}).get("DB_FILE_ID")
        if not drive_id:
            st.error("Missing [drive].DB_FILE_ID in Secrets.")
            st.stop()
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, str(db_path), quiet=False)

    st.caption(f"DB path: {db_path}")
    if not db_path.exists():
        st.error("Database file not found after download.")
        st.stop()

    # Basic size sanity check
    min_bytes = 100 * 1024 * 1024  # 100 MB
    size_bytes = db_path.stat().st_size
    st.caption(f"DB size: {size_bytes:,} bytes")
    if size_bytes < min_bytes:
        st.error("Downloaded DB looks too small (likely incomplete).")
        st.stop()

    # Schema sanity check (must contain wide_fixed + wide_fixed_meta)
    try:
        con_check = duckdb.connect(str(db_path), read_only=True)
        have_wide = con_check.execute("""
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema='main' AND table_name='wide_fixed'
        """).fetchone()
        have_meta = con_check.execute("""
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema='main' AND table_name='wide_fixed_meta'
        """).fetchone()
        con_check.close()
    except Exception as e:
        st.error(f"Could not open DuckDB file: {e}")
        st.stop()

    if not (have_wide and have_meta):
        st.error(
            "DB is missing required tables (wide_fixed / wide_fixed_meta). "
            "Rebuild locally with the exporter and upload a new version to Drive (same file ID)."
        )
        st.stop()

    return db_path

DB_PATH = ensure_db()

# ---- Open DuckDB ----
try:
    con = duckdb.connect(str(DB_PATH), read_only=True)
except Exception as e:
    st.error(f"DuckDB couldn’t open the DB at {DB_PATH}.\n\n{e}")
    st.stop()
    
# ---- Load the exact wide table produced by the exporter ----
@st.cache_data(show_spinner=False)
def load_wide_from_db(db_path: str):
    import os
    mtime = os.path.getmtime(db_path)  # bust cache when DB is rebuilt
    con2 = duckdb.connect(str(db_path), read_only=True)
    df = con2.execute('SELECT * FROM wide_fixed').df()
    try:
        built_at = con2.execute('SELECT built_at FROM wide_fixed_meta').fetchone()[0]
    except Exception:
        built_at = None
    con2.close()
    return df, built_at, mtime

df_wide_all, wide_built_at, _ = load_wide_from_db(str(DB_PATH))
if wide_built_at:
    st.caption(f"Wide table built at: {wide_built_at}")

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

# --- Always-on, database-wide month-to-month increases (shown near top) ---
show_month_to_month_increases(con)
show_month_to_month_decreases(con)

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
  d.product_line_id                             AS product_line_id,
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
        sql = f"""
            SELECT
              f.snapshot_date::DATE AS month,
              d.product_line_id     AS product_line_id,
              {item_code_expr}      AS item_code,
              f.aemp
            FROM fact_monthly f
            JOIN dim_product_line d USING (product_line_id)
            WHERE lower(d.name_a) = lower(?)
            ORDER BY 1, 2, 3
        """
        df = con.execute(sql, [drug]).df()

    return df

# ---- Wide table (unchanged) ----
@st.cache_data
def build_export_table(drug: str) -> pd.DataFrame:
    # Use the exact wide table produced by the exporter
    df = df_wide_all.copy()
    if df.empty:
        return df

    # Filter by drug
    if "Legal Instrument Drug" in df.columns:
        df = df[df["Legal Instrument Drug"].str.lower() == (drug or "").lower()]
    else:
        return pd.DataFrame()

    # Month columns in chronological order
    month_cols = sorted(
        [c for c in df.columns if c.startswith("AEMP ")],
        key=lambda c: pd.to_datetime(c.replace("AEMP ", ""), format="%b %y", errors="coerce")
    )

    fixed = [
        "Item Code",
        "Legal Instrument Drug",
        "Legal Instrument Form",
        "Brand Name",
        "Formulary",
        "Responsible Person",
        "AMT Trade Product Pack",
    ]
    fixed = [c for c in fixed if c in df.columns]
    return df[fixed + month_cols].reset_index(drop=True)

# ---- Chart data from wide table (Month → Identifier → AEMP) ----
@st.cache_data
def build_chart_df(drug: str) -> pd.DataFrame:
    """
    Build a long dataframe for the chart with columns:
    month (datetime), display_name (identifier), aemp (float).
    Reads from the same wide table used for the export and
    gracefully handles missing identifier columns.
    """
    base = build_export_table(drug).copy()
    if base.empty:
        return base.assign(month=pd.NaT, display_name="", aemp=pd.NA).head(0)

    # Choose identifier columns that exist in wide_fixed
    id_candidates = [
        "Item Code",
        "Formulary",
        "Brand Name",
        "Legal Instrument Form",
        "AMT Trade Product Pack",
        "Responsible Person",  # ignored if missing
    ]
    id_cols = [c for c in id_candidates if c in base.columns]

    # Fallback guard
    if not id_cols:
        id_cols = [c for c in ["Item Code", "Brand Name"] if c in base.columns]

    # Build a single display label by joining the available columns
    base[id_cols] = base[id_cols].astype(str).replace({"None": "", "nan": ""})
    base["display_name"] = (
        base[id_cols]
        .agg(" · ".join, axis=1)
        .str.replace(r"( · )+$", "", regex=True)
    )

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
        errors="coerce",
    )

    # Clean and order
    long_df = long_df.dropna(subset=["month"]).drop(columns=["month_label"])
    long_df["aemp"] = pd.to_numeric(long_df["aemp"], errors="coerce")

    return long_df.sort_values(["display_name", "month"])[["month", "display_name", "aemp"]]

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
    
# Ensure clean data for chart
chart_df["month"] = pd.to_datetime(chart_df["month"], errors="coerce")
chart_df["aemp"]  = pd.to_numeric(chart_df["aemp"], errors="coerce")
chart_df = chart_df.dropna(subset=["month", "aemp"]).reset_index(drop=True)

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

# ---- Apply time filter to the combined chart data ----
mask = (chart_df["month"] >= pd.to_datetime(start_m)) & (chart_df["month"] <= pd.to_datetime(end_m))
chart_df = chart_df.loc[mask].copy()

# ---- Identifier picker (no hard cap) ----
all_ids = sorted(chart_df["display_name"].unique().tolist())
with st.sidebar:
    st.subheader("Compare products")
    select_all = st.checkbox("Select all identifiers", value=False)
    picked = st.multiselect("Identifiers", options=all_ids, default=[])

# If 'Select all' is on (or nothing picked), show everything
filtered_df = chart_df if (select_all or not picked) else chart_df[chart_df["display_name"].isin(picked)]

# ---- Chart ----
if filtered_df.empty:
    st.info("No series to plot with the current filters. Try widening the time range or clearing Identifier picks.")
else:
    chart = (
        alt.Chart(filtered_df.sort_values("month"))
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
        .properties(height=450, title=alt.TitleParams(f"{title_drug}: AEMP by month", anchor="start"))
        .interactive(bind_x=True)
    )
    st.altair_chart(chart, use_container_width=True)  # <-- this actually renders the chart

# ---- Small table under the chart (Month → Identifier → AEMP) ----
st.dataframe(
    filtered_df.assign(Month=filtered_df["month"].dt.strftime("%b %Y"))[["Month", "display_name", "aemp"]],
    use_container_width=True
)

# ---- Wide table & download (respects time range; drops empty rows) ----
st.markdown("### Item info + AEMP by month (wide)")
st.caption("Product columns first, then monthly AEMP columns in chronological order.")

# Use the first selected drug for the wide table/export
export_base = selected_drugs[0] if selected_drugs else None
if not export_base:
    st.warning("Pick at least one drug to show the wide table/export."); st.stop()

# Full wide table for that drug
export_df = build_export_table(export_base)

# Month columns within the slider range
start_dt = pd.to_datetime(start_m).to_period("M").to_timestamp()
end_dt   = pd.to_datetime(end_m).to_period("M").to_timestamp()

def _col_to_month(col: str) -> pd.Timestamp:
    return pd.to_datetime(col.replace("AEMP ", ""), format="%b %y", errors="coerce")

month_cols_all = [c for c in export_df.columns if c.startswith("AEMP ")]
kept_month_cols = [c for c in month_cols_all if (_col_to_month(c) >= start_dt) and (_col_to_month(c) <= end_dt)]

# Keep only rows with at least one non-null value in the kept months
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
    filtered_wide = export_df.loc[nonempty_mask, [c for c in fixed_cols if c in export_df.columns] + kept_month_cols]
else:
    # No months in range → show just headers (empty frame)
    filtered_wide = export_df[[c for c in fixed_cols if c in export_df.columns]].iloc[0:0]

st.dataframe(filtered_wide, use_container_width=True)

# Download button (filename includes the selected range)
file_range = f"{start_dt:%Y-%m}_{end_dt:%Y-%m}"
export_csv = filtered_wide.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download AEMP wide CSV: {export_base}",
    data=export_csv,
    file_name=f"{export_base.replace(' ','_').lower()}_{file_range}_aemp_wide.csv",
    mime="text/csv",
)
