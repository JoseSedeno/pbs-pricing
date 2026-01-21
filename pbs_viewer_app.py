import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import altair as alt
import gdown
import re
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0").strip().lower() in {"1", "true", "yes"}
st.set_page_config(page_title="PBS AEMP Viewer", layout="wide")
st.title("PBS AEMP Price Viewer")

# ---- Dataset selector ----
with st.sidebar:
    dataset = st.radio("Dataset", ["PBS AEMP", "Chemo EFC"], index=0)
    
# Map label -> (local filename, Streamlit secrets key)
DATASET_MAP = {
    "PBS AEMP": ("pbs_prices.duckdb", "DB_FILE_ID"),
    "Chemo EFC": ("chemo_prices.duckdb", "CHEMO_DB_FILE_ID"),
}

# ---- Simple auth gate (must be above any data code) ----
from datetime import datetime, timezone

# ---- Minimal access logger (CSV) ----
_LOG_PATH = Path("out/access_log.csv")

def _log_access(event: str, user: str):
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        is_new = not _LOG_PATH.exists()
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            if is_new:
                f.write("ts_utc,event,user\n")
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            f.write(f"{ts},{event},{user}\n")
    except Exception:
        # logging must never break the app
        pass

_auth = st.secrets.get("auth", {})
AUTH_NONCE = _auth.get("AUTH_NONCE", "")
PASSWORDS = dict(_auth.get("PASSWORDS", {}))      # {client_id: password}
EXPIRES   = dict(_auth.get("EXPIRES_UTC", {}))    # {client_id: ISO8601 UTC string}

def _is_authed() -> bool:
    tok = st.session_state.get("auth_ticket")
    return bool(tok and tok.get("nonce") == AUTH_NONCE and tok.get("client_id"))

def _is_expired(client_id: str) -> bool:
    exp = EXPIRES.get(client_id)
    if not exp:
        return False
    try:
        # Normalize: "Z" → "+00:00" and parse
        exp_s = str(exp).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(exp_s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > dt
    except Exception:
        return False

def _login_ui():
    st.subheader("Secure login")
    with st.form("login"):
        client_id = st.text_input("Client ID")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        cid = (client_id or "").strip()  # trim spaces
        # case-insensitive match for the ID
        lookup_id = cid if cid in PASSWORDS else next((k for k in PASSWORDS if k.lower() == cid.lower()), None)

        # expiry check (if configured)
        if lookup_id and _is_expired(lookup_id):
            st.error("This trial account has expired. Please contact us for access.")
            return

        if lookup_id and pw == PASSWORDS[lookup_id]:
            st.session_state["auth_ticket"] = {
                "client_id": lookup_id,
                "nonce": AUTH_NONCE,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _log_access("login", lookup_id)  # log successful sign-in
            st.success("Signed in")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not _is_authed():
    _login_ui()
    st.stop()

with st.sidebar:
    if _is_authed():
        user = st.session_state['auth_ticket']['client_id']
        st.caption(f"Signed in as: {user}")
        if st.button("Logout"):
            _log_access("logout", user)     # log sign-out
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
def ensure_db(dataset: str) -> Path:
    """
    Resolve a local DuckDB path for the selected dataset, downloading from Drive if missing.
    Env var overrides:
      - PBS_DB_PATH / PBS_DB_FORCE for "PBS AEMP"
      - CHEMO_DB_PATH / CHEMO_DB_FORCE for "Chemo EFC"
    """
    filename, id_key = DATASET_MAP[dataset]
    env_var = "PBS_DB_PATH" if dataset == "PBS AEMP" else "CHEMO_DB_PATH"
    force_var = "PBS_DB_FORCE" if dataset == "PBS AEMP" else "CHEMO_DB_FORCE"

    # 1) Env var override
    env = os.environ.get(env_var)
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists() and p.stat().st_size > 1024:
            st.caption(f"Using DB ({env_var}): {p}")
            return p
        else:
            st.warning(f"{env_var} set but file missing/small: {p}")

    # 2) Local ./out path
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = (out_dir / filename).resolve()

    # Optional: force refresh (env var or Secrets [drive])
    env_val = os.environ.get(force_var, "").strip().lower()
    secret_val = ""
    drive_secrets = st.secrets.get("drive", {})
    if isinstance(drive_secrets, dict):
        secret_val = str(drive_secrets.get(force_var, "")).strip().lower()

    force_refresh = env_val in {"1", "true", "yes"} or secret_val in {"1", "true", "yes"}
    if force_refresh and db_path.exists():
        try:
            db_path.unlink()
            st.caption(f"Removed existing local DB ({force_var}=1).")
        except Exception as e:
            st.warning(f"Could not remove existing DB: {e}")

    # Download if missing
    if not db_path.exists():
        drive_id = st.secrets.get("drive", {}).get(id_key)
        if not drive_id:
            st.error(f"Missing [drive].{id_key} in Secrets.")
            st.stop()

        with st.spinner(f"Downloading {dataset} database from Google Drive…"):
            gdown.download(id=drive_id, output=str(db_path), quiet=False)

        if DEBUG_MODE:
            st.caption(f"DB path: {db_path}")
        if not db_path.exists():
            st.error("Database file not found after download.")
            st.stop()

    # Basic size sanity check (chemo DBs are small but should be > ~1 MB)
    min_bytes = 1 * 1024 * 1024  # 1 MB
    try:
        size_bytes = db_path.stat().st_size
    except FileNotFoundError:
        st.error("Database file not found after download.")
        st.stop()

    st.caption(f"DB size: {size_bytes:,} bytes")
    if size_bytes < min_bytes:
        st.error("Downloaded DB looks too small (likely wrong/incomplete file).")
        st.stop()

    # Schema sanity check (dataset-aware)
    try:
        con_check = duckdb.connect(str(db_path), read_only=True)

        if dataset == "Chemo EFC":
            have_a = con_check.execute("""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema='main' AND table_name='dim_product_line'
            """).fetchone()
            have_b = con_check.execute("""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema='main' AND table_name='fact_monthly'
            """).fetchone()
            need_msg = "dim_product_line / fact_monthly"
        else:
            have_a = con_check.execute("""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema='main' AND table_name='wide_fixed'
            """).fetchone()
            have_b = con_check.execute("""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema='main' AND table_name='wide_fixed_meta'
            """).fetchone()
            need_msg = "wide_fixed / wide_fixed_meta"

        ok_a = bool(have_a and have_a[0] == 1)
        ok_b = bool(have_b and have_b[0] == 1)
        if not (ok_a and ok_b):
            st.error(f"DB is missing required tables ({need_msg}). "
                     "Rebuild locally and upload a new version to Drive (same file ID).")
            st.stop()
    except Exception as e:
        st.error(f"Could not open DuckDB file: {e}")
        st.stop()
    finally:
        try:
            con_check.close()
        except Exception:
            pass

    return db_path

DB_PATH = ensure_db(dataset)
DATA_VERSION = (dataset, os.path.getmtime(DB_PATH))

# ---- Open DuckDB ----
try:
    con = duckdb.connect(str(DB_PATH), read_only=True)
except Exception as e:
    st.error(f"DuckDB couldn’t open the DB at {DB_PATH}.\n\n{e}")
    st.stop()
    
# ---- Load a wide table for BOTH datasets (PBS uses stored wide; Chemo builds on the fly) ----
@st.cache_data(show_spinner=False)
def load_wide_from_db(db_path: str, dataset: str, _ver: tuple):
    import os
    mtime = os.path.getmtime(db_path)
    con2 = duckdb.connect(str(db_path), read_only=True)

    if dataset == "PBS AEMP":
        try:
            wide = con2.execute('SELECT * FROM wide_fixed').df()
            row = con2.execute('SELECT built_at FROM wide_fixed_meta').fetchone()
            built_at = row[0] if row else None

            # Normalize PBS wide column names so the viewer can find them
            def _pick(colnames, *opts):
                low = {c.strip().lower(): c for c in colnames}
                for o in opts:
                    k = o.strip().lower()
                    if k in low:
                        return low[k]
                return None

            # Canonicalize key columns and ensure presence
            colmap = {}

            # Responsible Person
            rp = _pick(
                wide.columns,
                "Responsible Person", "Responsible Person Name",
                "Sponsor", "Sponsor Name", "responsible_person"
            )
            if rp and rp != "Responsible Person":
                colmap[rp] = "Responsible Person"

            # AMT Trade Product Pack
            amt = _pick(
                wide.columns,
                "AMT Trade Product Pack", "AMT Trade Pack", "amt_trade_product_pack"
            )
            if amt and amt != "AMT Trade Product Pack":
                colmap[amt] = "AMT Trade Product Pack"

            # Item Code
            ic = _pick(wide.columns, "Item Code", "Item Code B", "item_code_b", "item_code")
            if ic and ic != "Item Code":
                colmap[ic] = "Item Code"

            # Legal Instrument Form
            lif = _pick(
                wide.columns,
                "Legal Instrument Form", "legal_instrument_form", "variant_signature_base"
            )
            if lif and lif != "Legal Instrument Form":
                colmap[lif] = "Legal Instrument Form"

            # Apply renames in one go
            if colmap:
                wide = wide.rename(columns=colmap)

            # Ensure presence and string dtype (keeps <NA>)
            for c in ["Responsible Person", "AMT Trade Product Pack", "Item Code", "Legal Instrument Form"]:
                if c not in wide.columns:
                    wide[c] = pd.NA
                wide[c] = wide[c].astype("string")

            return wide, built_at, mtime

        finally:
            try:
                con2.close()
            except Exception:
                pass

    # ---- Chemo EFC: build a PBS-compatible wide (AEMP by month) ----
    # Column maps (pick only if present)
    def _cols(tbl):
        return {r[1].lower(): r[1] for r in con2.execute(f'PRAGMA table_info("{tbl}")').fetchall()}
    def pick(cmap, *names):
        for n in names:
            if n and n.lower() in cmap:
                return cmap[n.lower()]
        return None

    d = _cols("dim_product_line")
    f = _cols("fact_monthly")

    item_code   = pick(d, "item_code_b", "item_code")
    form        = pick(d, "attr_c", "legal_instrument_form")
    brand_line  = pick(d, "brand_name")
    formulary_d = pick(d, "attr_f", "formulary")
    resp_line   = pick(d, "responsible_person", "sponsor", "name_b")
    brand_fm    = pick(f, "brand_name")
    formulary_f = pick(f, "formulary")
    resp_fm     = pick(f, "responsible_person", "sponsor", "responsible_person_name")
    amt_fm      = pick(f, "amt_trade_product_pack", "amt_trade_pack")

    sql = f"""
        SELECT
            d.product_line_id                                                AS product_line_id,
            d.name_a                                                         AS "Legal Instrument Drug",
            {('d."' + item_code   + '"') if item_code   else 'NULL'}         AS "Item Code",
            {('d."' + form        + '"') if form        else 'NULL'}         AS "Legal Instrument Form",
            COALESCE(CAST({('f."' + brand_fm    + '"') if brand_fm    else 'NULL'} AS VARCHAR),
                     CAST({('d."' + brand_line  + '"') if brand_line  else 'NULL'} AS VARCHAR))  AS "Brand Name",
            COALESCE(CAST({('d."' + formulary_d + '"') if formulary_d else 'NULL'} AS VARCHAR),
                     CAST({('f."' + formulary_f + '"') if formulary_f else 'NULL'} AS VARCHAR))  AS _formulary_src,
            COALESCE(CAST({('d."' + resp_line   + '"') if resp_line   else 'NULL'} AS VARCHAR),
                     CAST({('f."' + resp_fm     + '"') if resp_fm     else 'NULL'} AS VARCHAR))  AS "Responsible Person",
            {('f."' + amt_fm + '"') if amt_fm else 'NULL'}                  AS "AMT Trade Product Pack",
            DATE_TRUNC('month', f.snapshot_date)::DATE                       AS month,
            CAST(f.aemp AS DOUBLE)                                           AS aemp
        FROM fact_monthly f
        JOIN dim_product_line d USING (product_line_id)
    """
    long_df = con2.execute(sql).df()
    con2.close()

    if long_df.empty:
        return pd.DataFrame(), None, mtime

    def _norm_formulary(x):
        if x is None: return None
        s = str(x).strip()
        if s in ("1","F1"):  return "F1"
        if s in ("60","F2"): return "F2"
        if s.upper() == "CDL": return "CDL"
        return s or None

    long_df["Formulary"] = long_df["_formulary_src"].map(_norm_formulary)
    long_df.drop(columns=["_formulary_src"], inplace=True)
    long_df["month_label"] = pd.to_datetime(long_df["month"]).dt.strftime("AEMP %b %y")

    id_cols = [
        "Item Code","Legal Instrument Drug","Legal Instrument Form",
        "Brand Name","Formulary","Responsible Person","AMT Trade Product Pack",
    ]
    wide = (
        long_df
        .pivot_table(index=id_cols, columns="month_label", values="aemp", aggfunc="mean")
        .reset_index()
    )
    mcols = sorted(
        [c for c in wide.columns if c.startswith("AEMP ")],
        key=lambda c: pd.to_datetime(c.replace("AEMP ",""), format="%b %y", errors="coerce")
    )
    wide = wide[[c for c in id_cols if c in wide.columns] + mcols]
    return wide, None, mtime

# Use the unified loader
df_wide_all, wide_built_at, _ = load_wide_from_db(str(DB_PATH), dataset, DATA_VERSION)
if DEBUG_MODE and wide_built_at:
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

# --- Month-to-month sections (Chemo only) ---
if dataset == "Chemo EFC":
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

@st.cache_data(show_spinner=False)
def get_drugs(active_dataset: str):
    # Populate from the active dataset only
    if active_dataset == "PBS AEMP":
        return con.execute(
            'SELECT DISTINCT "Legal Instrument Drug" AS drug FROM wide_fixed ORDER BY 1'
        ).df()["drug"].tolist()
    else:  # Chemo EFC
        return con.execute(
            'SELECT DISTINCT name_a AS drug FROM dim_product_line ORDER BY 1'
        ).df()["drug"].tolist()

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

# Helper to canonicalise key parts for grouping (series_id)
def _canon_val(x: object) -> str:
    """
    Canonical string for identifier logic (NOT display):

    Purpose:
    - Treat cosmetic text differences as the SAME product
    - Preserve REAL differences like quantity / pack size

    Rules:
    - None / NaN / empty -> ""
    - lowercase
    - trim + collapse whitespace
    - remove punctuation that causes fake differences
    - normalise small wording noise only
    - DO NOT normalise quantities (numbers, mL, tablets count, pack counts)
    """
    if x is None:
        return ""

    s = str(x).strip().lower()

    if s in {"", "none", "nan", "<na>"}:
        return ""

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # remove punctuation that causes fake differences (keep hyphens)
    s = re.sub(r"[,\.;:]", "", s)

    # normalise safe wording noise only
    s = re.sub(r"\bfilm coated\b", "film-coated", s)
    s = re.sub(r"\btablets\b", "tablet", s)
    s = re.sub(r"\bcapsules\b", "capsule", s)

    # final cleanup in case punctuation removal created double spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s
    
# ---- Chart data from wide table (Month → Identifier → AEMP) ----
@st.cache_data
def build_chart_df(drug: str) -> pd.DataFrame:
    """
    Build a long dataframe for the chart with columns:
      month, display_name, aemp, Item Code, Responsible Person,
      AMT Trade Product Pack, series_id
    The three raw columns are passed through from the wide table so we can
    colour/track a stable series across months.
    """
    base = build_export_table(drug).copy()
    if base.empty:
        cols = ["month","display_name","aemp",
                "Item Code","Responsible Person","AMT Trade Product Pack","series_id"]
        return pd.DataFrame(columns=cols).head(0)

    # --- Build display label and stable series_id ---

    # Columns to show to the user in the label (use originals, hide empties)
    _disp_order = [
        "Item Code",
        "Brand Name",
        "Legal Instrument Form",
        "Formulary",
        "AMT Trade Product Pack",
        "Responsible Person",
    ]
    _disp_cols = [c for c in _disp_order if c in base.columns]

    # Safe string view with empties cleaned
    def _clean(s):
        if pd.isna(s):
            return ""
        t = str(s).strip()
        return "" if t in {"", "None", "nan", "<NA>"} else t

    base["display_name"] = base[_disp_cols].apply(
        lambda r: " · ".join([_clean(v) for v in r if _clean(v)]),
        axis=1
    )

    # Ensure raw key parts exist so schema is stable
    for c in ["Item Code", "Brand Name", "Legal Instrument Form", "Responsible Person", "AMT Trade Product Pack"]:
        if c not in base.columns:
            base[c] = pd.NA

    # Stable grouping key (do NOT include Item Code, so code changes do not create a new line)
    # Keep Legal Instrument Form so tablets vs liquids vs injections never merge
    base["series_id"] = (
        base["Legal Instrument Form"].map(_canon_val) + "|" +
        base["Brand Name"].map(_canon_val) + "|" +
        base["Responsible Person"].map(_canon_val) + "|" +
        base["AMT Trade Product Pack"].map(_canon_val)
    )

    # Unpivot month columns
    month_cols = [c for c in base.columns if c.startswith("AEMP ")]
    long_df = base.melt(
        id_vars=["display_name","Item Code","Responsible Person","AMT Trade Product Pack","series_id"],
        value_vars=month_cols,
        var_name="month_label",
        value_name="aemp",
    )

    # Parse month and clean
    long_df["month"] = pd.to_datetime(
        long_df["month_label"].str.replace("AEMP ", "", regex=False),
        format="%b %y",
        errors="coerce",
    )
    long_df["aemp"] = pd.to_numeric(long_df["aemp"], errors="coerce")

    long_df = (
        long_df.dropna(subset=["month","aemp"])
               .drop(columns=["month_label"])
               .sort_values(["series_id","display_name","month"])
               [["month","display_name","aemp","Item Code","Responsible Person","AMT Trade Product Pack","series_id"]]
               .reset_index(drop=True)
    )
    return long_df

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Filters")
    all_drugs = get_drugs(dataset)
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

# --- Diagnostics: check Responsible Person in PBS wide_fixed for the first selected drug ---
if DEBUG_MODE and dataset == "PBS AEMP":
    test_drug = (selected_drugs or [""])[0]

    with st.expander("Diagnostics: Responsible Person in wide_fixed (PBS path)", expanded=True):
        st.caption(f"Drug checked: {test_drug or '(none)'}")

        # Columns present in wide_fixed
        wf_cols_df = con.execute("PRAGMA table_info('wide_fixed')").df()
        wf_cols = [str(n) for n in wf_cols_df["name"].tolist()]
        wf_cols_lower = {c.lower(): c for c in wf_cols}

        # Candidate RP columns that might exist in some exports
        rp_candidates = [
            "Responsible Person", "Responsible Person Name",
            "Sponsor", "Sponsor Name", "responsible_person"
        ]
        present_rp_cols = [wf_cols_lower[c.lower()] for c in rp_candidates if c.lower() in wf_cols_lower]

        # Counts for the target drug
        total_rows = con.execute(
            'SELECT COUNT(*) FROM wide_fixed WHERE lower("Legal Instrument Drug") = lower(?)',
            [test_drug]
        ).fetchone()[0] if test_drug else 0

        # Count NULLs in the canonical target column if it exists
        nulls_in_rp = None
        if "Responsible Person".lower() in wf_cols_lower:
            rp_exact = wf_cols_lower["Responsible Person".lower()]
            nulls_in_rp = con.execute(
                f'SELECT COUNT(*) FROM wide_fixed '
                f'WHERE lower("Legal Instrument Drug") = lower(?) AND "{rp_exact}" IS NULL',
                [test_drug]
            ).fetchone()[0]

        # Also show non-null counts for any alternate RP columns if they exist
        alt_counts = {}
        for c in present_rp_cols:
            cnt = con.execute(
                f'SELECT COUNT(*) FROM wide_fixed '
                f'WHERE lower("Legal Instrument Drug") = lower(?) AND "{c}" IS NOT NULL',
                [test_drug]
            ).fetchone()[0]
            alt_counts[c] = int(cnt)

        st.write({
            "total_rows_for_drug": int(total_rows),
            "responsible_person_column_present": ("Responsible Person".lower() in wf_cols_lower),
            "nulls_in_responsible_person": (int(nulls_in_rp) if nulls_in_rp is not None else "column_missing"),
            "alternate_rp_columns_found": present_rp_cols,
            "non_null_counts_in_alternates": alt_counts,
        })

        # Show a small sample so you can eyeball what is present in the source
        sample_cols = [c for c in [
            "Item Code", "Brand Name", "Legal Instrument Form", "Formulary",
            "Responsible Person", "Responsible Person Name", "Sponsor", "Sponsor Name",
            "AMT Trade Product Pack"
        ] if c in wf_cols]

        if test_drug and sample_cols:
            cols_sql = ", ".join([f'"{c}"' for c in sample_cols])
            sql = (
                f'SELECT {cols_sql} '
                f'FROM wide_fixed '
                f'WHERE lower("Legal Instrument Drug") = lower(?) '
                f'LIMIT 50'
            )
            sample = con.execute(sql, [test_drug]).df()
            st.dataframe(sample, use_container_width=True)

# Build one long table across the selected drugs
frames = []
for d in (selected_drugs or []):
    df_d = build_chart_df(d)
    if not df_d.empty:
        df_d = df_d.copy()

        # Prefix identifier so the source drug is clear
        df_d["display_name"] = f"{d} · " + df_d["display_name"]
        df_d["__drug__"] = d

        # Make series_id unique across multiple drugs to prevent line merging
        df_d["series_id"] = f"{d}|" + df_d["series_id"]

        frames.append(df_d)

if frames:
    chart_df = pd.concat(frames, ignore_index=True)
else:
    chart_df = pd.DataFrame(
        columns=[
            "month",
            "display_name",
            "aemp",
            "Item Code",
            "Responsible Person",
            "AMT Trade Product Pack",
            "series_id",
            "__drug__",
        ]
    )
    
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

if pd.isna(min_m) or pd.isna(max_m):
    st.error("No valid months in chart data. Slider cannot be built.")
    st.stop()

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
if select_all or not picked:
    filtered_df = chart_df
else:
    filtered_df = chart_df[chart_df["display_name"].isin(picked)]

# ---- Range summaries (runs AFTER filtered_df is created) ----
if filtered_df.empty:
    st.caption("For the selected drug(s), no AEMP data is available for the current filters.")
else:
    # Ensure month is datetime for correct min/max
    _m = pd.to_datetime(filtered_df["month"], errors="coerce")

    first_month = _m.min()
    last_month  = _m.max()

    if pd.isna(first_month) or pd.isna(last_month):
        st.caption("For the selected drug(s), AEMP month data could not be parsed for the current filters.")
    else:
        st.caption(
            f"For the selected drug(s), AEMP data runs from {first_month:%b %Y} to {last_month:%b %Y} (current filters)."
        )

        coverage_df = (
            filtered_df.assign(_month=_m)
                      .dropna(subset=["_month"])
                      .groupby("series_id")["_month"]
                      .agg(first_month="min", last_month="max")
                      .reset_index()
        )
        coverage_df["First AEMP month"] = coverage_df["first_month"].dt.strftime("%b %Y")
        coverage_df["Last AEMP month"]  = coverage_df["last_month"].dt.strftime("%b %Y")

        with st.expander("First and last AEMP month for each identifier", expanded=False):
            st.dataframe(
                coverage_df[["series_id", "First AEMP month", "Last AEMP month"]],
                use_container_width=True,
            )

# ---- Chart ----
if filtered_df.empty:
    st.info("No series to plot with the current filters. Try widening the time range or clearing Identifier picks.")
else:
    chart = (
        alt.Chart(filtered_df.sort_values("month"))
        .transform_filter(alt.datum.aemp != None)
        .mark_line(point={"filled": True, "size": 30}, interpolate="linear", strokeWidth=2.5)
        .encode(
            x=alt.X("month:T", sort=None, axis=alt.Axis(title="Month", format="%b %Y", labelAngle=0)),
            y=alt.Y("aemp:Q", title="AEMP"),
            color=alt.Color("series_id:N", title="Identifier"),
            detail="series_id:N",
            order="month:T",
            tooltip=[
    alt.Tooltip("month:T", title="Month", format="%Y-%m"),
    alt.Tooltip("display_name:N", title="Identifier (label)"),
    alt.Tooltip("Item Code:N", title="Item Code"),
    alt.Tooltip("Responsible Person:N", title="Responsible Person"),
    alt.Tooltip("AMT Trade Product Pack:N", title="AMT Trade Product Pack"),
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
