#!/usr/bin/env python3
"""Pull VIC1 rooftop solar power and operational demand for 2025, hourly (2 API credits).

    rooftop: /v4/data/network/NEM   metrics=power, fueltech=solar_rooftop
    demand:  /v4/market/network/NEM metrics=demand

Writes script/data/NEM_VIC1_rooftop_demand_2025_1h.csv (MW, network time UTC+10).
Usage:  python script/pull_rooftop_demand.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_curtailment import API_BASE, load_api_key  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
START, END = datetime(2025, 1, 1), datetime(2026, 1, 1)


def get(session, key, url, params):
    resp = session.get(url, params=params, headers={"Authorization": f"Bearer {key}"}, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"API error: {payload.get('error')}")
    return payload


def series(payload):
    """{timestamp: value}, summed over every result series."""
    out = {}
    for block in payload.get("data", []):
        for result in block.get("results", []):
            for ts, value in result.get("data", []):
                if value is not None:
                    out[ts] = out.get(ts, 0.0) + value
    return out


def main():
    key = load_api_key(None)
    common = [("interval", "1h"), ("date_start", START.strftime("%Y-%m-%dT%H:%M:%S")),
              ("date_end", END.strftime("%Y-%m-%dT%H:%M:%S")),
              ("primary_grouping", "network_region"), ("network_region", "VIC1")]
    with requests.Session() as s:
        rooftop = get(s, key, f"{API_BASE}/data/network/NEM",
                      common + [("metrics", "power"), ("fueltech", "solar_rooftop"),
                                ("secondary_grouping", "fueltech")])
        demand = get(s, key, f"{API_BASE}/market/network/NEM", common + [("metrics", "demand")])
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "NEM_VIC1_rooftop_1h_2025.json").write_text(json.dumps(rooftop))
    (raw / "NEM_VIC1_demand_1h_2025.json").write_text(json.dumps(demand))

    roof, dem = series(rooftop), series(demand)
    path = OUT / "NEM_VIC1_rooftop_demand_2025_1h.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interval_start", "solar_rooftop_mw", "demand_mw"])
        for ts in sorted(set(roof) | set(dem)):
            w.writerow([ts, roof.get(ts, ""), dem.get(ts, "")])
    print(f"wrote {path}: {len(roof)} rooftop rows, {len(dem)} demand rows")


if __name__ == "__main__":
    main()
