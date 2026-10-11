#!/usr/bin/env python3
"""Pull VIC1 rooftop solar power for 2025 at 5-minute steps (13 API credits).

    rooftop: /v4/data/network/NEM   metrics=power, fueltech=solar_rooftop, interval=5m

The 5m interval is capped at 30 days per request, so the year is pulled in 13 chunks. Rooftop output is AEMO's
half-hourly estimate, so each value holds for six 5-minute steps.

Writes script/data/NEM_VIC1_rooftop_2025_5m.csv (MW, network time UTC+10).
Usage:  python script/pull_rooftop_5m.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_curtailment import API_BASE, MAX_RANGE_DAYS, chunk_ranges, load_api_key  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
START, END = datetime(2025, 1, 1), datetime(2026, 1, 1)


def main():
    key = load_api_key(None)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    values = {}
    with requests.Session() as s:
        for a, b in chunk_ranges(START, END, MAX_RANGE_DAYS["5m"]):
            params = [("interval", "5m"), ("date_start", a.strftime("%Y-%m-%dT%H:%M:%S")),
                      ("date_end", b.strftime("%Y-%m-%dT%H:%M:%S")), ("primary_grouping", "network_region"),
                      ("network_region", "VIC1"), ("metrics", "power"), ("fueltech", "solar_rooftop"),
                      ("secondary_grouping", "fueltech")]
            resp = s.get(f"{API_BASE}/data/network/NEM", params=params,
                         headers={"Authorization": f"Bearer {key}"}, timeout=120)
            payload = resp.json()
            if resp.status_code != 200 or not payload.get("success"):
                raise RuntimeError(f"{a:%Y-%m-%d}: HTTP {resp.status_code} {str(payload)[:300]}")
            (raw / f"NEM_VIC1_rooftop_5m_{a:%Y%m%d}_{b:%Y%m%d}.json").write_text(json.dumps(payload))
            for block in payload.get("data", []):
                for result in block.get("results", []):
                    for ts, v in result.get("data", []):
                        if v is not None:
                            values[ts] = values.get(ts, 0.0) + v
            print(f"{a:%Y-%m-%d} - {b:%Y-%m-%d}: {len(values)} values so far", flush=True)
    path = OUT / "NEM_VIC1_rooftop_2025_5m.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interval_start", "solar_rooftop_mw"])
        for ts in sorted(values):
            w.writerow([ts, values[ts]])
    print(f"wrote {path}: {len(values)} rows")


if __name__ == "__main__":
    main()
