"""Solar available to one circuit: x_d(t) = X_d(t) / C.

X is the VIC1 utility-solar curtailment (5 min), or, with config.SOLAR_SOURCE = "rooftop", the
VIC1 rooftop-solar shape (hourly, held for each 5 min) scaled to the same annual energy as the
curtailment, so only the timing changes; "rooftop_actual" uses the actual rooftop MW shared by
participating circuits; "rooftop_local" uses the actual rooftop MW shared by all VIC circuits. Data are in market time (UTC+10). With
config.LOCAL_TIME, slots are local clock time: on daylight-saving days local t = market t + 1 h.
"""
import csv
from functools import lru_cache

import numpy as np

from play import config


@lru_cache(maxsize=None)
def _profile_mw(source):
    """(days, slots) MW in market time, and the date of each day."""
    with open(config.SOLAR_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    mw = np.array([float(r["curtailment_solar_utility"]) for r in rows]).reshape(-1, config.SLOTS)
    days = np.array([r["interval_start"][:10] for r in rows[::config.SLOTS]])
    if source in ("rooftop_actual", "rooftop_local"):                 # actual rooftop MW, hourly held
        with open(config.ROOFTOP_DEMAND_CSV, newline="") as f:
            hourly = np.array([float(r["solar_rooftop_mw"]) for r in csv.DictReader(f)]).reshape(-1, 24)
        mw = np.repeat(hourly, config.SLOTS // 24, axis=1)
    if source == "rooftop_5m":                                       # actual rooftop MW, 5-minute steps
        with open(config.ROOFTOP_5M_CSV, newline="") as f:
            mw = np.array([float(r["solar_rooftop_mw"]) for r in csv.DictReader(f)]).reshape(-1, config.SLOTS)
    if source == "rooftop":
        with open(config.ROOFTOP_DEMAND_CSV, newline="") as f:
            hourly = np.array([float(r["solar_rooftop_mw"]) for r in csv.DictReader(f)]).reshape(-1, 24)
        shape = np.repeat(hourly, config.SLOTS // 24, axis=1)
        mw = shape * mw.sum() / shape.sum()                           # same annual energy as curtailment
    return mw, days


def to_local(series, days):
    """Shift (days, slots) market-time series to local clock time on daylight-saving days."""
    dst = (days < config.DST_END) | (days >= config.DST_START)
    out = series.copy()
    out[dst] = np.roll(series[dst], config.SLOTS // 24, axis=1)       # local t shows market t - 1 h
    return out


def excess_per_circuit(day=None, uptake=None):
    """(days, slots) kW of solar for one circuit: one day ("2025-11-06") or all of 2025."""
    mw, days = _profile_mw(config.SOLAR_SOURCE)
    if config.LOCAL_TIME:
        mw = to_local(mw, days)
    # rooftop_local: every circuit has its own panels -> share over ALL circuits (uptake has no effect)
    kw = mw * 1000 / (config.circuits(1.0) if config.SOLAR_SOURCE == "rooftop_local" else config.circuits(uptake))
    return kw if day is None else kw[days == day]
