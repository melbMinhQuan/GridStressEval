# Open Electricity curtailment pull

`pull_curtailment.py` pulls curtailment time series from the
[Open Electricity API](https://docs.openelectricity.org.au/) (v4).

## Setup

The API key lives in `script/.env` (gitignored), or set it in the environment:

```bash
export OPENELECTRICITY_API_KEY=oe_...
```

## Run

```bash
# default: VIC1, calendar year 2025, 5-minute + daily
python script/pull_curtailment.py

# other region / year / resolution
python script/pull_curtailment.py --region NSW1 --year 2024 --intervals 5m 1h 1d
```

Outputs land in `script/data/` (gitignored; rerun the script to regenerate):

| file | contents |
| --- | --- |
| `NEM_VIC1_curtailment_2025_5m.csv` | 105,120 rows (365 × 288 dispatch intervals) |
| `NEM_VIC1_curtailment_2025_1d.csv` | 365 daily rows |
| `raw/*.json` | verbatim API responses, one per request (~55 MB) |

Columns: `interval_start, region, curtailment_solar_utility,
curtailment_solar_utility_energy, curtailment_wind, curtailment_wind_energy,
curtailment, curtailment_energy`. Power metrics are MW, `*_energy` are MWh.

## API notes (verified live, 2026-09-25)

* Curtailment is served by **`/v4/market/network/{network_code}`**. The
  `/v4/data/network/{network_code}` endpoint rejects these metrics with
  `Unsupported metrics: curtailment_solar_utility` even though its OpenAPI
  schema advertises them.
* The `5m` interval is capped at a **30-day range per request**, so a full year
  is chunked into 13 requests. `1d` fits in one request.
* `date_start` / `date_end` are network-local time (NEM = UTC+10, no DST) and
  the range is half-open: `[date_start, date_end)`.
* Open Electricity derives curtailment from AEMO dispatch data as
  `UIGF - CLEAREDMW`, and shifts the series forward 5 minutes so it aligns with
  measured generation.
* Curtailment is NEM-only (no WEM).
* Budget: one request = one API credit regardless of how many metrics it asks
  for, so all six curtailment metrics are pulled together. A full-year 5m + 1d
  pull costs 14 credits.

## 2025 VIC1 totals

| metric | value |
| --- | --- |
| utility solar curtailed | 472,394.6 MWh (peak 1,287.7 MW) |
| wind curtailed | 2,031,734.8 MWh (peak 3,941.6 MW) |
| total curtailed | 2,504,129.3 MWh (peak 4,886.2 MW) |

The 5m energy series sums to the 1d rollup to within 0.000%.
