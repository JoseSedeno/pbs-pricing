#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
archive_ingest_duckdb.py — PBS archive ZIPs -> out_api/pbs_api.duckdb

The PBS publications archive publishes a "PBS API CSV files" ZIP for each
month. It holds 32 of the 34 API tables as CSVs, already stamped with
schedule_code. No rate limit, no waiting: one 5MB download per month.

Only summary-of-changes is missing from the ZIPs. That still needs the
live API (api_ingest_duckdb.py).

The archive lags roughly one month, so the newest schedule is usually
only available from the live API.

Usage:
  python archive_ingest_duckdb.py --output_dir ./out_api
  python archive_ingest_duckdb.py --output_dir ./out_api --months 2025-08-01,2025-09-01
  python archive_ingest_duckdb.py --output_dir ./out_api --download_only

Requirements: pip install requests duckdb
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests

# The 13 schedules currently in the API window. The archive should have
# all but the newest. Note 2025-12-10 is a mid-month revision.
DEFAULT_MONTHS = [
    "2025-08-01",
    "2025-09-01",
    "2025-10-01",
    "2025-11-01",
    "2025-12-01",
    "2025-12-10",
    "2026-01-01",
    "2026-02-01",
    "2026-03-01",
    "2026-04-01",
    "2026-05-01",
    "2026-06-01",
    "2026-07-01",
]

URL_TEMPLATE = (
    "https://www.pbs.gov.au/publication/schedule/"
    "{year}/{month}/{date}-PBS-API-CSV.zip?variant=3"
)

DB_FILENAME = "pbs_api.duckdb"


def download_zip(date_str: str, raw_dir: Path) -> Path | None:
    """Fetch one month's ZIP. Returns the path, or None if not published."""
    year, month, _ = date_str.split("-")
    url = URL_TEMPLATE.format(year=year, month=month, date=date_str)
    dest = raw_dir / f"{date_str}-PBS-API-CSV.zip"

    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"   already downloaded: {dest.name}")
        return dest

    print(f"   downloading {url}")
    try:
        resp = requests.get(url, timeout=180)
    except requests.RequestException as e:
        print(f"   network error: {e}")
        return None

    if resp.status_code == 404:
        print("   not published yet (404)")
        return None
    if resp.status_code != 200:
        print(f"   unexpected status {resp.status_code}")
        return None

    # A missing file sometimes returns an HTML error page with status 200.
    if not resp.content.startswith(b"PK"):
        print("   response is not a ZIP, skipping")
        return None

    dest.write_bytes(resp.content)
    print(f"   saved {len(resp.content):,} bytes")
    return dest


def extract_zip(zip_path: Path, raw_dir: Path) -> Path | None:
    """Unzip and return the folder holding the CSVs."""
    out_dir = raw_dir / zip_path.stem
    if not out_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)

    # The CSVs sit in a tables_as_csv subfolder.
    for candidate in out_dir.rglob("tables_as_csv"):
        if candidate.is_dir():
            return candidate

    # Fall back to wherever the CSVs actually are.
    for candidate in out_dir.rglob("*.csv"):
        return candidate.parent

    print(f"   no CSVs found in {zip_path.name}")
    return None


def load_csv(con, csv_path: Path, date_str: str) -> int:
    """Load one CSV into DuckDB, replacing this month's rows."""
    table = csv_path.stem.replace("-", "_")

    # Read everything as text. The CSVs use the literal string "null" for
    # missing values, and some columns mix numbers and text.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE staging AS
        SELECT * FROM read_csv_auto(
            '{csv_path.as_posix()}',
            all_varchar = true,
            nullstr = 'null',
            header = true
        )
    """)

    n = con.execute("SELECT COUNT(*) FROM staging").fetchone()[0]
    if n == 0:
        print(f"      {table}: empty, skipped")
        return 0

    stamped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE staging_stamped AS
        SELECT *,
               CAST(schedule_code AS VARCHAR) AS _schedule_code,
               '{date_str}' AS _effective_date,
               '{stamped_at}' AS _downloaded_at_utc
        FROM staging
    """)

    existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if table not in existing:
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM staging_stamped LIMIT 0')

    # Add any columns this month has that the table doesn't.
    table_cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()}
    stage_cols = {r[1] for r in con.execute("PRAGMA table_info(staging_stamped)").fetchall()}
    for col in stage_cols - table_cols:
        con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" VARCHAR')

    # Replace this month's rows in one transaction, so a failed insert
    # can't leave the month deleted.
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(f'DELETE FROM "{table}" WHERE _effective_date = ?', [date_str])
        con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM staging_stamped')
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    print(f"      {table}: {n:,} rows")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Load PBS archive ZIPs into DuckDB")
    ap.add_argument("--output_dir", default="./out_api")
    ap.add_argument("--raw_dir", default="./raw_api",
                    help="Where to keep the downloaded ZIPs")
    ap.add_argument("--months", default=None,
                    help="Comma-separated dates, e.g. 2025-08-01,2025-09-01")
    ap.add_argument("--download_only", action="store_true",
                    help="Fetch the ZIPs but don't load them")
    args = ap.parse_args()

    months = args.months.split(",") if args.months else DEFAULT_MONTHS

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / DB_FILENAME

    print(f"ZIPs:     {raw_dir}")
    print(f"Database: {db_path}")

    con = None
    if not args.download_only:
        con = duckdb.connect(str(db_path))

    try:
        got, missing = [], []

        for date_str in months:
            print(f"\n=== {date_str} ===")

            zip_path = download_zip(date_str, raw_dir)
            if zip_path is None:
                missing.append(date_str)
                continue

            if args.download_only:
                got.append(date_str)
                continue

            csv_dir = extract_zip(zip_path, raw_dir)
            if csv_dir is None:
                missing.append(date_str)
                continue

            total = 0
            for csv_path in sorted(csv_dir.glob("*.csv")):
                total += load_csv(con, csv_path, date_str)
            print(f"   {total:,} rows total")
            got.append(date_str)

        print(f"\nLoaded {len(got)} month(s): {', '.join(got) if got else 'none'}")
        if missing:
            print(f"Not available: {', '.join(missing)}")
            print("The archive lags about a month, so the newest schedule "
                  "usually has to come from the live API.")

        if con is not None:
            print("\nArchive contents:")
            for (tbl,) in sorted(con.execute("SHOW TABLES").fetchall()):
                try:
                    n_months, n_rows = con.execute(
                        f'SELECT COUNT(DISTINCT _effective_date), COUNT(*) FROM "{tbl}"'
                    ).fetchone()
                    print(f"   {tbl}: {n_rows:,} rows across {n_months} month(s)")
                except Exception:
                    n_rows = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
                    print(f"   {tbl}: {n_rows:,} rows")
    finally:
        if con is not None:
            con.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
