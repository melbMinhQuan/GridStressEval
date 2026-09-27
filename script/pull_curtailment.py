#!/usr/bin/env python3
"""Pull curtailment time series from the Open Electricity API.

Defaults to Victoria (VIC1) utility solar curtailment for calendar year 2025 at
5-minute resolution, plus daily rollups.

API notes (verified against the live API, 2026-09-25):
  * Curtailment metrics are served by  /v4/market/network/{network_code}
    -- NOT by /v4/data/network/{network_code}, which rejects them with
    "Unsupported metrics" even though the OpenAPI schema advertises them.
  * The 5m interval is capped at a 30-day date range per request, so a full
    year has to be requested in chunks.
  * date_start/date_end are network-local time (NEM = UTC+10, no DST) and the
    range is half-open: [date_start, date_end).
  * Open Electricity derives curtailment from AEMO dispatch data as
    UIGF - CLEAREDMW, and shifts the series forward 5 minutes so it lines up
    with measured generation.

Usage:
    export OPENELECTRICITY_API_KEY=oe_xxx
    python script/pull_curtailment.py
    python script/pull_curtailment.py --region NSW1 --year 2024
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

API_BASE = "https://api.openelectricity.org.au/v4"
MARKET_ENDPOINT = API_BASE + "/market/network/{network_code}"

# One request returns every metric listed here, so pulling wind/total alongside
# solar costs no extra API credits.
DEFAULT_METRICS = [
    "curtailment_solar_utility",
    "curtailment_solar_utility_energy",
    "curtailment_wind",
    "curtailment_wind_energy",
    "curtailment",
    "curtailment_energy",
]

MAX_RANGE_DAYS = {"5m": 30, "1h": 365, "1d": 3650}
DEFAULT_OUT = Path(__file__).resolve().parent / "data"


def load_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    key = os.environ.get("OPENELECTRICITY_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENELECTRICITY_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit(
        "No API key. Set OPENELECTRICITY_API_KEY, pass --api-key, "
        f"or put OPENELECTRICITY_API_KEY=... in {env_file}"
    )


def chunk_ranges(start: datetime, end: datetime, max_days: int):
    """Yield half-open [a, b) windows no longer than max_days."""
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=max_days), end)
        yield cur, nxt
        cur = nxt


def fetch(session, api_key, network, metrics, interval, start, end, region, retries=5):
    params = [("metrics", m) for m in metrics]
    params += [
        ("interval", interval),
        ("date_start", start.strftime("%Y-%m-%dT%H:%M:%S")),
        ("date_end", end.strftime("%Y-%m-%dT%H:%M:%S")),
        ("primary_grouping", "network_region"),
    ]
    if region:
        params.append(("network_region", region))

    url = MARKET_ENDPOINT.format(network_code=network)
    delay = 2.0
    for attempt in range(1, retries + 1):
        resp = session.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        if resp.status_code == 200:
            payload = resp.json()
            if payload.get("success"):
                return payload
            raise RuntimeError(f"API error: {payload.get('error')}")
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            print(f"    HTTP {resp.status_code}, retry {attempt}/{retries - 1} "
                  f"in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    raise RuntimeError("exhausted retries")


def flatten(payload):
    """Turn an API payload into {(timestamp, region): {metric: value}} + units."""
    rows: dict[tuple[str, str], dict[str, float]] = {}
    units: dict[str, str] = {}
    for series in payload.get("data", []):
        metric = series["metric"]
        units[metric] = series.get("unit", "")
        for result in series.get("results", []):
            region = result.get("columns", {}).get("region", "")
            for timestamp, value in result.get("data", []):
                rows.setdefault((timestamp, region), {})[metric] = value
    return rows, units


def write_csv(path: Path, rows: dict, metrics: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["interval_start", "region", *metrics])
        for (timestamp, region) in sorted(rows):
            values = rows[(timestamp, region)]
            writer.writerow([timestamp, region, *(values.get(m, "") for m in metrics)])
    return len(rows)


def pull_interval(session, args, api_key, interval, out_dir, raw_dir):
    start = datetime(args.year, 1, 1)
    end = datetime(args.year + 1, 1, 1)
    max_days = MAX_RANGE_DAYS.get(interval, 30)

    all_rows: dict[tuple[str, str], dict[str, float]] = {}
    units: dict[str, str] = {}
    windows = list(chunk_ranges(start, end, max_days))
    print(f"\n[{interval}] {len(windows)} request(s) for {args.year} {args.region or 'all regions'}")

    for i, (a, b) in enumerate(windows, 1):
        print(f"  {i}/{len(windows)}  {a:%Y-%m-%d} -> {b:%Y-%m-%d}", end="", flush=True)
        payload = fetch(session, api_key, args.network, args.metrics,
                        interval, a, b, args.region)
        if raw_dir:
            tag = f"{args.network}_{args.region or 'ALL'}_{interval}_{a:%Y%m%d}_{b:%Y%m%d}.json"
            (raw_dir / tag).write_text(json.dumps(payload, indent=2))
        rows, chunk_units = flatten(payload)
        units.update(chunk_units)
        for key, values in rows.items():
            all_rows.setdefault(key, {}).update(values)
        print(f"  ({len(rows)} intervals)")
        if i < len(windows):
            time.sleep(args.sleep)

    name = f"{args.network}_{args.region or 'ALL'}_curtailment_{args.year}_{interval}.csv"
    path = out_dir / name
    n = write_csv(path, all_rows, args.metrics)
    print(f"  wrote {n} rows -> {path}")
    return all_rows, units, path


def summarise(rows, metrics, units, interval):
    """Totals per metric. Energy metrics sum; power metrics report the peak."""
    print(f"\n  Summary ({interval}):")
    for metric in metrics:
        values = [r[metric] for r in rows.values()
                  if r.get(metric) is not None]
        if not values:
            continue
        unit = units.get(metric, "")
        if metric.endswith("_energy"):
            print(f"    {metric:<36} total {sum(values):>14,.1f} {unit}")
        else:
            print(f"    {metric:<36} peak  {max(values):>14,.1f} {unit}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--network", default="NEM")
    p.add_argument("--region", default="VIC1",
                   help="NEM price zone (VIC1, NSW1, QLD1, SA1, TAS1); empty for all")
    p.add_argument("--intervals", nargs="+", default=["5m", "1d"],
                   choices=["5m", "1h", "1d"])
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--api-key", default=None)
    p.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between requests")
    p.add_argument("--no-raw", action="store_true",
                   help="skip saving raw JSON responses")
    args = p.parse_args()

    api_key = load_api_key(args.api_key)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = None if args.no_raw else out_dir / "raw"
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    results = {}
    for interval in args.intervals:
        rows, units, path = pull_interval(session, args, api_key, interval, out_dir, raw_dir)
        summarise(rows, args.metrics, units, interval)
        results[interval] = rows

    # Cross-check: 5m energy summed over the year should match the 1d rollup.
    if "5m" in results and "1d" in results:
        metric = "curtailment_solar_utility_energy"
        fine = sum(r.get(metric, 0.0) for r in results["5m"].values())
        coarse = sum(r.get(metric, 0.0) for r in results["1d"].values())
        if coarse:
            print(f"\n  Cross-check {metric}: 5m sum {fine:,.1f} vs 1d sum "
                  f"{coarse:,.1f} MWh ({100 * (fine - coarse) / coarse:+.3f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
