#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_ingest_duckdb.py — PBS public data API -> out_api/pbs_api.duckdb

Sits next to pbs_ingest_duckdb.py / chemo_ingest_duckdb.py in the repo and
follows the same pattern: fetch raw data, load into a DuckDB file.

What it does:
  1. Asks the API which schedules (months) are available (~13 at any time).
  2. Downloads the endpoints in ENDPOINTS for the latest schedule
     (or every available schedule with --backfill).
  3. Saves each endpoint as a table in DuckDB, one month at a time.
     Re-running a month is safe: that month's rows are replaced.

Why it matters: the public API only keeps ~12-13 months. This script is
what builds the permanent archive. Run monthly (GitHub Actions later).

Verified against the live API (July 2026):
  - Base URL: https://data-api.health.gov.au/pbs/api/v3  (no auth needed)
  - Response shape: {"_meta": {...}, "_links": [...], "data": [...]}
  - schedule_code is a 4-digit code, e.g. 4706 = July 2026
  - AEMP lives in items.determined_price (verified: 10356C = 66.72)

Usage:
  python api_ingest_duckdb.py --output_dir ./out_api                 # latest month
  python api_ingest_duckdb.py --output_dir ./out_api --backfill      # all ~13 months
  python api_ingest_duckdb.py --output_dir ./out_api --schedule 4664 # one specific month

Requirements: pip install requests duckdb pandas
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

# ----------------------------- Config --------------------------------

BASE_URL = "https://data-api.health.gov.au/pbs/api/v3"

# Public subscription key (published by the Department for everyone).
SUBSCRIPTION_KEY = "2384af7c667342ceb5a736fe29f1dc6b"

# Core tables for the viewer rebuild. Add more later if needed
# (e.g. "restrictions", "atc-codes", "item-atc-relationships").
ENDPOINTS = [
    "items",                                # listings + prices (determined_price = AEMP)
    "organisations",                        # responsible person / sponsor
    "prescribers",                          # who can prescribe (nurse practitioner etc.)
    "dispensing-rules",                     # dispensing settings (community/private/public)
    "item-dispensing-rule-relationships",   # item <-> setting, with per-setting prices
    "amt-items",                            # AMT trade product pack
    "programs",                             # program code dictionary (S85 vs S100)
    "summary-of-changes",                   # PBS's official change log
]

DB_FILENAME = "pbs_api.duckdb"

# ------------------------- HTTP helpers ------------------------------


def polite_get(url: str, params: dict, sleep_s: float, max_tries: int = 5):
    """GET with a wait before every request (shared rate limit: 1 req / 20 s)."""
    last_err = None
    for attempt in range(1, max_tries + 1):
        time.sleep(sleep_s)
        try:
            resp = requests.get(url, params=params, timeout=180,
                                headers={"accept": "application/json",
                                         "subscription-key": SUBSCRIPTION_KEY})
        except requests.RequestException as e:
            last_err = f"network error: {e}"
            print(f"      attempt {attempt}: {last_err} - retrying")
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 204:
            # No content for this schedule/endpoint (e.g. oldest month has
            # no change log). Treat as empty, not as an error.
            return {"data": []}

        if resp.status_code == 429:
            wait = 30 * attempt
            print(f"      attempt {attempt}: rate-limited (429), waiting {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code in (500, 502, 503, 504):
            print(f"      attempt {attempt}: server error {resp.status_code} - retrying")
            continue

        # 400/404 etc: won't fix itself. Raise with the API's own message,
        # which is usually clear (we saw this with a bad schedule_code).
        raise RuntimeError(
            f"API returned {resp.status_code} for {url} params={params}. "
            f"Body starts: {resp.text[:300]}"
        )
    raise RuntimeError(f"Gave up on {url} after {max_tries} tries. Last: {last_err}")


def fetch_endpoint(endpoint: str, schedule_code, sleep_s: float, limit: int) -> pd.DataFrame:
    """Download every page of one endpoint for one schedule."""
    url = f"{BASE_URL}/{endpoint}"
    all_rows = []
    page = 1
    total = None

    while True:
        params = {"schedule_code": str(schedule_code), "limit": limit, "page": page}
        try:
            payload = polite_get(url, params, sleep_s)
        except RuntimeError as e:
            # If the API rejects our page size, drop to the safe default once.
            if "400" in str(e) and "limit" in str(e).lower() and limit > 100:
                print(f"      limit={limit} rejected, retrying with limit=100")
                limit = 100
                continue
            raise

        meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
        rows = payload.get("data", []) if isinstance(payload, dict) else payload

        if total is None:
            total = meta.get("total_records")
            if total is not None:
                print(f"      {total:,} records total")

        if not rows:
            break

        all_rows.extend(rows)
        print(f"      page {page}: +{len(rows)} rows ({len(all_rows):,} so far)")

        if len(rows) < limit:
            break  # last page
        if total is not None and len(all_rows) >= int(total):
            break
        page += 1
        if page > 2000:  # safety net
            print("      stopping: unexpectedly many pages")
            break

    return pd.DataFrame(all_rows)


# ------------------------- DuckDB helpers ----------------------------


def save(con, table: str, df: pd.DataFrame, schedule_code, effective_date):
    """Idempotent save: replace this schedule's rows in the given table."""
    if df.empty:
        # Still clear this schedule's old rows, otherwise a rerun leaves
        # stale data behind when an endpoint legitimately returns nothing.
        existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if table in existing:
            con.execute(f'DELETE FROM "{table}" WHERE _schedule_code = ?',
                        [str(schedule_code)])
        print(f"      nothing to save for {table}")
        return

    df = df.copy()
    # Drop response bookkeeping columns if they slipped into rows (CSV mode quirk).
    for junk in ("_meta", "_links"):
        if junk in df.columns:
            df = df.drop(columns=[junk])

    # Some API columns mix numbers and text in the same column (e.g. the
    # change log's old/new values). Convert all object columns to text so
    # DuckDB never has to guess.
    import json as _json
    KNOWN_NESTED_COLS = {"table_keys", "previous_detail", "change_detail"}
    for col in df.columns:
        if col in KNOWN_NESTED_COLS:
            # These fields hold a different set of keys on every row.
            # Always store them as a JSON text string, never as a
            # structured type, so DuckDB never has to guess a shape.
            df[col] = df[col].map(
                lambda v: _json.dumps(v) if isinstance(v, (dict, list)) else (
                    None if v is None else str(v)
                )
            )
            df[col] = df[col].astype("string")
        elif df[col].dtype == object:
            df[col] = df[col].map(
                lambda v: str(v) if isinstance(v, (dict, list)) else v
            )
            df[col] = df[col].astype("string")

    # Stamp every row so months are cleanly separable regardless of source columns.
    df["_schedule_code"] = str(schedule_code)
    df["_effective_date"] = str(effective_date) if effective_date else None
    df["_downloaded_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    con.register("df_new", df)
    existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if table not in existing:
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM df_new LIMIT 0')

    # The API occasionally adds columns between months; extend the table so
    # inserts never fail. Missing columns on either side are filled with NULL.
    table_cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()}
    for col in df.columns:
        if col not in table_cols:
            con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" VARCHAR')
            # New columns are created as text, so make the incoming values text
            # too, otherwise numeric data refuses the conversion.
            df[col] = df[col].astype("string")
            con.unregister("df_new")
            con.register("df_new", df)

    # Wrap delete+insert in one transaction: if the insert fails, the
    # delete is rolled back and the month's existing data survives.
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(f'DELETE FROM "{table}" WHERE _schedule_code = ?',
                    [str(schedule_code)])
        con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM df_new')
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("df_new")


def get_schedules(sleep_s: float) -> list[dict]:
    """Return available schedules, newest first (same list you saw in Postman)."""
    payload = polite_get(f"{BASE_URL}/schedules", {"limit": 100}, sleep_s)
    rows = payload.get("data", [])
    if not rows:
        raise RuntimeError("The /schedules endpoint returned no rows.")
    rows.sort(key=lambda r: str(r.get("effective_date", "")), reverse=True)
    return rows


# ------------------------------ Main ---------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest PBS public API data into DuckDB")
    ap.add_argument("--output_dir", default="./out_api", help="Where to write the DuckDB")
    ap.add_argument("--backfill", action="store_true",
                    help="Fetch ALL available schedules (~13 months), not just the latest")
    ap.add_argument("--schedule", default=None,
                    help="Fetch one specific schedule_code (e.g. 4664)")
    ap.add_argument("--sleep", type=float, default=21.0,
                    help="Seconds to wait before each request (public limit is 1/20s, shared)")
    ap.add_argument("--limit", type=int, default=1000,
                    help="Rows per page (auto-falls back to 100 if rejected)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / DB_FILENAME

    print(f"DB: {db_path}")
    con = duckdb.connect(str(db_path))
    try:
        return _run(con, db_path, args)
    finally:
        con.close()


def _run(con, db_path, args) -> int:

    print(f"Fetching available schedules (each request waits {args.sleep:.0f}s - normal)...")
    schedules = get_schedules(args.sleep)
    print(f"  {len(schedules)} schedules available: "
          f"{schedules[-1].get('effective_date')} .. {schedules[0].get('effective_date')}")

    # Keep the schedules list itself for reference (replaced wholesale).
    sched_df = pd.DataFrame(schedules)
    con.execute("CREATE OR REPLACE TABLE schedules AS SELECT * FROM sched_df")

    # Decide which schedules to pull.
    if args.schedule:
        targets = [s for s in schedules if str(s.get("schedule_code")) == str(args.schedule)]
        if not targets:
            print(f"ERROR: schedule {args.schedule} not in the available list "
                  f"(the public API only keeps ~13 months).", file=sys.stderr)
            return 2
    elif args.backfill:
        targets = list(reversed(schedules))  # oldest first, so history builds forward
    else:
        targets = [schedules[0]]  # latest only

    for sched in targets:
        code = sched.get("schedule_code")
        eff = sched.get("effective_date")
        print(f"\n=== Schedule {code} (effective {eff}) ===")
        for endpoint in ENDPOINTS:
            table = endpoint.replace("-", "_")
            print(f"   {endpoint} ...")
            df = fetch_endpoint(endpoint, code, args.sleep, args.limit)
            save(con, table, df, code, eff)
            print(f"   -> {len(df):,} rows into '{table}'")

    print("\nArchive contents:")
    for (tbl,) in sorted(con.execute("SHOW TABLES").fetchall()):
        try:
            n_sched, n_rows = con.execute(
                f'SELECT COUNT(DISTINCT _schedule_code), COUNT(*) FROM "{tbl}"'
            ).fetchone()
            print(f"   {tbl}: {n_rows:,} rows across {n_sched} schedule(s)")
        except Exception:
            n_rows = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            print(f"   {tbl}: {n_rows:,} rows")

    print(f"\nDone. Upload {db_path.name} to Google Drive when ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
