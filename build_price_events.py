"""
Build the price_events table from item_pricing_events.

Reads:  item_pricing_events, items   (in out_api/pbs_api.duckdb)
Writes: price_events                 (same database, replaced each run)

One row per pricing event, enriched with:
  - drug name, brand, PBS code, formulary (from items, same month)
  - price_before  (determined_price in the schedule before the event)
  - price_after   (determined_price in the event's schedule)
  - a readable label for documented event types; raw code otherwise

Labels follow PBS documentation:
  PD        -> Price disclosure            (Division 3B, NHA 1953)
  NB        -> First new brand             (Division 3A)
  5_YR_F1   -> 5-year anniversary (5%)     (Division 3A)
  10_YR_F1  -> 10-year anniversary (5%)    (Division 3A)
  15_YR_F1  -> 15-year anniversary (26.1%) (Division 3A)
  LEGIS     -> 15-year anniversary 1.48%   (s99ACP)
  APA, PD_FLOWON_SB -> not documented; raw code kept as label

Usage:
  python3 build_price_events.py [--db out_api/pbs_api.duckdb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb


def label_for(code: str) -> str:
    """Readable label for documented event types; raw code otherwise."""
    if "PD_FLOWON" in code:
        return code  # not documented
    if "APA" in code:
        return code  # not documented
    if "15_YR_F1" in code:
        return "15-year anniversary"
    if "10_YR_F1" in code:
        return "10-year anniversary"
    if "5_YR_F1" in code:
        return "5-year anniversary"
    if "LEGIS" in code:
        return "15-year anniversary (1.48%)"
    if code.endswith("_NB"):
        return "First new brand"
    if code.endswith("_PD"):
        return "Price disclosure"
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="out_api/pbs_api.duckdb")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    con = duckdb.connect(str(db_path))

    # Ordered schedule dates, from items (covers all 13 months incl. 2025-12-10)
    months = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT _effective_date FROM items ORDER BY 1"
        ).fetchall()
    ]
    prev_of = {m: (months[i - 1] if i > 0 else None) for i, m in enumerate(months)}

    # Register the label and prev-month mappings as small tables
    con.execute("CREATE OR REPLACE TEMP TABLE _prev(month VARCHAR, prev_month VARCHAR)")
    for m, p in prev_of.items():
        con.execute("INSERT INTO _prev VALUES (?, ?)", [m, p])

    codes = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT event_type_code FROM item_pricing_events"
        ).fetchall()
    ]
    con.execute("CREATE OR REPLACE TEMP TABLE _labels(code VARCHAR, label VARCHAR)")
    for c in codes:
        con.execute("INSERT INTO _labels VALUES (?, ?)", [c, label_for(c)])

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DROP TABLE IF EXISTS price_events")
        con.execute(
            """
            CREATE TABLE price_events AS
            SELECT
                e.li_item_id                                   AS li_item_id,
                cur.pbs_code                                   AS pbs_code,
                cur.li_drug_name                               AS li_drug_name,
                cur.brand_name                                 AS brand_name,
                cur.formulary                                  AS formulary,
                e._effective_date                              AS event_date,
                e.event_type_code                              AS event_type_code,
                l.label                                        AS event_label,
                (e.event_type_code LIKE '%COMBO%')             AS is_combination,
                CAST(e.percentage_applied AS DOUBLE)           AS percentage_applied,
                prev_i.determined_price                        AS price_before,
                cur.determined_price                           AS price_after
            FROM item_pricing_events e
            LEFT JOIN _labels l
                   ON l.code = e.event_type_code
            LEFT JOIN _prev p
                   ON p.month = e._effective_date
            LEFT JOIN items cur
                   ON cur.li_item_id = e.li_item_id
                  AND cur._effective_date = e._effective_date
            LEFT JOIN items prev_i
                   ON prev_i.li_item_id = e.li_item_id
                  AND prev_i._effective_date = p.prev_month
            """
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    # ---- Verification summary ----
    n = con.execute("SELECT COUNT(*) FROM price_events").fetchone()[0]
    src = con.execute("SELECT COUNT(*) FROM item_pricing_events").fetchone()[0]
    print(f"price_events rows: {n:,}  (source events: {src:,})")
    if n != src:
        print("WARNING: row count differs from source - investigate before using.")

    missing_name = con.execute(
        "SELECT COUNT(*) FROM price_events WHERE li_drug_name IS NULL"
    ).fetchone()[0]
    print(f"rows with no matching item in event month: {missing_name:,}")

    print("\nEvents per label:")
    for r in con.execute(
        "SELECT event_label, COUNT(*) FROM price_events GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall():
        print(f"   {r[0]}: {r[1]:,}")

    print("\nSpot check (Plerixafor 10083Q, Oct 2025):")
    for r in con.execute(
        """
        SELECT pbs_code, li_drug_name, brand_name, event_date, event_label,
               percentage_applied, price_before, price_after
        FROM price_events
        WHERE pbs_code = '10083Q' AND event_date = '2025-10-01'
        ORDER BY li_item_id LIMIT 3
        """
    ).fetchall():
        print("   ", r)

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
