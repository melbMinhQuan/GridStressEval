"""Report experiments (doc/power_saver_intervals_report.tex). All-On.

Intervals: 3 hours, starting every 0.5 h, inside 08:00-16:00, one set per season. For each season and K the
set with the highest solar sum S = sum_k sum_{t in I_k} mean solar(t) dt is chosen (ties: most solar
covered by the union). Structures: 1 interval; 2 intervals that do not overlap; 3 intervals
where the first and last do not overlap and the middle one overlaps both.

Allocation strategies (per season, every circuit composition n chooses an allocation a):
    lowest overload: a in C(n) = {peak <= T}, or the lowest-peak allocations if none; then most soak
    best soak:       most soak; among those, fewest overloads
    soak is maximised by coordinate ascent over compositions.
Overload: a circuit overloads if its allocation exceeds T in any season (full currents).
Soak, VIC totals over the hours U the design's intervals cover, with uptake u (share of circuits on the plan):
    L(t) = u (N_VIC / N) E_n[V A_a(n)(t)],  A = N I_S + added current      total load of the plan circuits
    soak = mean_d sum_{t in U} min(L(t), X_d(t)) dt                       X = VIC1 rooftop solar MW (SOLAR)
In code L = u N_VIC B + added, B = V I_S per customer; the seasonal weight w_s scales L.
Uptake: u is calibrated so the policy (one window, everyone in it) at today's mix peaks at the mean midday
peak (peak of each season's mean daily rooftop profile, averaged over the days of the year)
(calibrated_uptake() = 12.8%, rounded to UPTAKE = 13%; average-day peak). No seasonal scaling (SEASONAL_WEIGHTS = False).

Experiment: customer mix rho = p_M + p_L (p_M : p_L = 30 : 1) under both allocation strategies.

Run from the repo root:  python3 -m play.run_participation
"""
import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import clock, segments, splits
from play.experiments.common import weights
from play.soak import SLOT, in_range
from play.solar import _profile_mw

UPTAKE = None                                     # share of VIC circuits on the plan; calibrated_uptake() rounded
SOLAR = "rooftop_5m"                              # VIC1 rooftop solar generation, 5-minute steps (see play/solar.py)
SOLAR_NAME = "rooftop solar"                      # for labels; "curtailment" -> "utility solar curtailment"
B = config.load_kw()["S"]                         # kW per plan customer outside its extra draw (V I_S): total load
SEASONAL_WEIGHTS = False                          # no seasonal scaling: the plan load is the same in every season
SOAK_HOURS = "window"                             # soak counted over "own": the hours the design covers;
                                                  # "two": also the hours two intervals cover (small customers only);
                                                  # "fill": 08-16, a baseline load filling the solar 08-10 outside
                                                  # the intervals; "window": SOAK_WINDOW for every design
SOAK_WINDOW = (10, 15)                            # hours counted when SOAK_HOURS == "window"
REFERENCE = (10, 15)                              # hours of the rooftop solar soak is a share of (None: WINDOW)
NET_SOAK = False                                  # report soaked - excess (load not covered by solar) = 2 soak - energy
Q = 1.0                                           # share of the window load that is used (1 = All-On)
SEASONS = {"Summer": (12, 1, 2), "Autumn": (3, 4, 5), "Winter": (6, 7, 8), "Spring": (9, 10, 11)}
LENGTH = 3                                        # hours per free interval
WINDOW = (8, 16)                                  # intervals lie inside this window
STEP = 0.5                                        # hours between possible start times
TODAY = 0.31                                      # rho of today's mix (0.69 / 0.30 / 0.01)
RHO = 0.31 * np.arange(1, 33) / 16                # share of medium + large customers
RATIO = {"M": 30 / 31, "L": 1 / 31}               # p_M : p_L = 30 : 1
CAPS = (1, 5)                                     # overload caps reported, % of circuits
LABEL_AT = (0.155, 0.31, 0.465, 0.62)             # rho labelled on the curves
COLORS = {1: "#8a8984", 2: "#2a78d6", 3: "#eb6834"}
GRID = 1.0                                        # MW step of the soak tables
PASSES = 5                                        # coordinate-ascent passes (stops early when stable)
OUT = config.OUT_DIR / "report"


def mix(rho):
    return {"S": 1 - rho, "M": RATIO["M"] * rho, "L": RATIO["L"] * rho}


@lru_cache(maxsize=None)
def candidates(design, n_m, n_l):
    """Every allocation of composition (n_M, n_L), lowest peak first: added kW per segment, peak A, and the
    medium and large customers per interval."""
    k, n, amps, kw = len(design), config.N, config.CURRENT, config.load_kw()
    medium, large = np.array(splits(n_m, k)), np.array(splits(n_l, k))
    i, j = (x.ravel() for x in np.meshgrid(range(len(medium)), range(len(large)), indexing="ij"))
    cov = [c for _, _, c in segments(list(design))]
    m = np.stack([medium[i][:, c].sum(axis=1) for c in cov], axis=1)            # (allocations, segments)
    l = np.stack([large[j][:, c].sum(axis=1) for c in cov], axis=1)
    peak = (amps["S"] * (n - m - l) + amps["M"] * m + amps["L"] * l).max(axis=1)
    order = np.argsort(peak, kind="stable")
    return ((kw["M"] - kw["S"]) * m + (kw["L"] - kw["S"]) * l)[order], peak[order], medium[i][order], large[j][order]


def soak_curve(solar, a, b, base, xmax):
    """Grid X (MW) and f(X) = mean_d sum_{t in [a, b)} min(base + X, solar_d(t)) dt (MWh/day)."""
    r = np.sort(solar[:, in_range(a, b)], axis=0)
    days = len(r)
    cum = np.vstack([np.zeros(r.shape[1]), np.cumsum(r, axis=0)])
    x = np.arange(0, xmax + GRID, GRID)
    c = base + x
    f = np.zeros_like(x)
    for t in range(r.shape[1]):
        below = np.searchsorted(r[:, t], c)                                     # days with solar < base + X
        f += (cum[below, t] + c * (days - below)) / days
    return x, f * config.DT


@lru_cache(maxsize=None)
def solar_season(season):
    mw, days = _profile_mw(SOLAR)                                               # (days, slots) MW, all VIC
    months = np.array([int(d[5:7]) for d in days])
    return mw[np.isin(months, SEASONS[season])]


def covered(design):
    """(slots,) mask of the hours the design's intervals cover: where soak is counted."""
    mask = np.zeros(config.SLOTS, bool)
    for a, b in design:
        mask |= in_range(a, b)
    return mask


def net(load, solar, design, season=None):
    """Soak of a (slots,) plan load against (days, slots) solar over the counted hours (SOAK_HOURS), MWh/day:
    soaked, or soaked - excess if NET_SOAK. "fill": 08-16, and outside the intervals 08-10 a baseline load
    equal to the solar (all soaked, no excess)."""
    own = covered(design)
    plan = np.broadcast_to(load, solar.shape)
    if SOAK_HOURS == "window":
        counted = in_range(*SOAK_WINDOW)
    elif SOAK_HOURS == "fill":
        counted = in_range(8, 16)
        fill = in_range(8, 10) & ~own
        plan = plan.copy()
        plan[:, fill] = solar[:, fill]
    elif SOAK_HOURS == "two":
        counted = own | covered(choose(2, season))
    else:
        counted = own
    soaked = np.minimum(plan[:, counted], solar[:, counted])
    return ((2 * soaked - plan[:, counted]) if NET_SOAK else soaked).sum(axis=1).mean() * config.DT


def ideal(season=None):
    """Rooftop solar inside REFERENCE (WINDOW if None), mean MWh/day over the season's days (the year if None):
    the most any design could soak, and the unit soak is reported in."""
    days = [solar_season(season)] if season else [solar_season(s) for s in SEASONS]
    return np.vstack(days)[:, in_range(*(REFERENCE or WINDOW))].sum(axis=1).mean() * config.DT


def pct(soak, season=None):
    """Soak as % of the rooftop solar inside WINDOW."""
    return 100 * np.asarray(soak) / ideal(season)


SOAK_LABEL = "Soak (% of rooftop solar, 10–15)"


def counted(design, season=None):
    """(slots,) mask of the hours where soak is counted (SOAK_HOURS)."""
    if SOAK_HOURS == "window":
        return in_range(*SOAK_WINDOW)
    if SOAK_HOURS == "fill":
        return in_range(8, 16)
    if SOAK_HOURS == "two":
        return covered(design) | covered(choose(2, season))
    return covered(design)


def midday_peak(season=None):
    """Peak of the mean daily rooftop profile, MW: of the season, or the day-weighted mean over seasons."""
    if season:
        return solar_season(season).mean(axis=0).max()
    days = {s: len(solar_season(s)) for s in SEASONS}
    return sum(midday_peak(s) * days[s] for s in SEASONS) / sum(days.values())


def season_weight(season):
    """Seasonal scaling w_s = the season's midday peak / the mean over seasons (1 if SEASONAL_WEIGHTS is off).
    With the calibrated uptake the policy then peaks at each season's midday peak. It scales only the load
    counted for soak."""
    if not SEASONAL_WEIGHTS:
        return 1.0
    return midday_peak(season) / midday_peak()


def calibrated_uptake():
    """u such that the policy's load (everyone in one window, today's mix) peaks at the midday peak of the
    average day of the data (rooftop output averaged over all days at each time of day, at its highest)."""
    kw = config.load_kw()
    p = mix(TODAY)
    per_circuit = config.N * (B + p["M"] * (kw["M"] - kw["S"]) + p["L"] * (kw["L"] - kw["S"]))  # kW
    peak = np.vstack([solar_season(s) for s in SEASONS]).mean(axis=0).max()                   # MW
    return peak * 1000 / (config.VIC_CUSTOMERS / config.N * per_circuit)


def base_mw(season):
    """Normal load of all plan customers counted for soak in a season, VIC MW (times w_s)."""
    return UPTAKE * config.VIC_CUSTOMERS * B / 1000 * season_weight(season)


UPTAKE = round(calibrated_uptake(), 2)            # 12.8% -> 13%


def ascend(keys, full, limit, w, x, pick, soak, to_mw):
    """Coordinate ascent of soak: composition key may use its first limit[key] allocations (lowest peak
    first); equal soak goes to the lower peak."""
    for _ in range(PASSES):
        moved = False
        for key in keys:
            extra = full[key][0][: limit[key]]
            if len(extra) == 1:
                continue
            trial = x + to_mw * w[key] * (extra - extra[pick[key]])
            gain = soak(trial) - soak(x)
            best = np.flatnonzero(gain >= gain.max() - 1e-9)[0]
            if gain[best] > 1e-9 or (gain[best] > -1e-9 and best < pick[key]):
                x, pick[key], moved = trial[best], best, True
        if not moved:
            break
    return x, pick


def year_designs(k):
    """The chosen K-interval set of every season."""
    return {s: choose(k, s) for s in SEASONS}


@lru_cache(maxsize=64)
def year_allocation(k, rho_key, strategy, uptake, n, limit):
    """One allocation per circuit composition, kept in every season, so the plan load is the same all year:
    ({(n_M, n_L): overloads}, {(n_M, n_L): weight}, {(n_M, n_L): (medium, large) per interval}, MW per
    segment). Soak is maximised over the year (seasons weighted by their days). The seasons' interval sets
    share one structure (same segments), so an allocation means the same in every season."""
    designs = year_designs(k)
    rho = dict(rho_key) if isinstance(rho_key, tuple) else rho_key
    kw = config.load_kw()
    to_mw = Q * uptake * config.VIC_CUSTOMERS / n / 1000                       # added kW -> VIC MW
    days = {s: len(solar_season(s)) for s in SEASONS}
    share = {s: days[s] / sum(days.values()) for s in SEASONS}
    segs = {s: segments(list(d)) for s, d in designs.items()}
    pattern = {s: [tuple(np.atleast_1d(c).tolist()) for _, _, c in segs[s]] for s in SEASONS}
    ref = next(iter(SEASONS))
    assert all(pattern[s] == pattern[ref] for s in SEASONS), "seasons' interval sets differ in structure"
    clip = ((lambda a, b: (max(a, SOAK_WINDOW[0]), max(min(b, SOAK_WINDOW[1]), max(a, SOAK_WINDOW[0]))))
            if SOAK_HOURS == "window" else (lambda a, b: (a, b)))                # only counted hours raise soak
    curves = {s: [soak_curve(solar_season(s), *clip(a, b), base_mw(s), to_mw * n * (kw["L"] - kw["S"]))
                  for a, b, _ in segs[s]] for s in SEASONS}
    soak = lambda x: sum(share[s] * sum(np.interp(x[..., g], *curves[s][g]) for g in range(len(segs[s])))
                         for s in SEASONS)

    w = weights(n, rho if isinstance(rho, dict) else mix(rho))
    total = sum(w.values())
    w = {key: v / total for key, v in w.items()}
    keys = sorted(w, key=w.get, reverse=True)
    full = {key: candidates(designs[ref], *key) for key in keys}               # lowest peak first
    safe = {}
    for key in keys:                                                            # C(n): a prefix of full[key]
        peak = full[key][1]
        safe[key] = int((peak <= limit).sum()) or int((peak == peak[0]).sum())
    pick = {key: 0 for key in keys}
    x = sum(to_mw * w[key] * full[key][0][0] for key in keys)                   # added MW per segment
    x, pick = ascend(keys, full, safe, w, x, pick, soak, to_mw)
    if strategy == "best soak":
        # most soak with the fewest overloads: from the lowest-overload allocation, overload the circuits
        # that buy the most soak per unit of probability until no allocation adds soak
        top, _ = ascend(keys, full, {key: len(full[key][1]) for key in keys}, w, x, dict(pick), soak, to_mw)
        while soak(top) - soak(x) > 1e-6:
            best = None
            for key in keys:
                added, peak = full[key][:2]
                gain = soak(x + to_mw * w[key] * (added - added[pick[key]])) - soak(x)
                j = np.flatnonzero(gain >= gain.max() - 1e-9)[0]
                cost = w[key] if peak[j] > limit >= peak[pick[key]] else 0.0
                rate = np.inf if cost == 0 else gain[j] / cost
                if gain[j] > 1e-9 and (best is None or rate > best[0]):
                    best = (rate, key, j)
            if best is None:
                break
            _, key, j = best
            x = x + to_mw * w[key] * (full[key][0][j] - full[key][0][pick[key]])
            pick[key] = j
    over = {key: full[key][1][pick[key]] > limit for key in keys}
    alloc = {key: (full[key][2][pick[key]], full[key][3][pick[key]]) for key in keys}
    return over, w, alloc, x


def evaluate(design, season, rho, strategy):
    """({(n_M, n_L): overloads}, soak MWh/day over the season's days, {(n_M, n_L): weight},
    {(n_M, n_L): (medium, large) per interval}) of the year's allocation (year_allocation). rho is a share
    of M + L customers or a mix {S, M, L}."""
    k = len(design)
    if year_designs(k)[season] != tuple(design):
        raise ValueError(f"{design} is not the chosen {k}-interval set of {season}")
    key = tuple(sorted(rho.items())) if isinstance(rho, dict) else float(rho)
    over, w, alloc, x = year_allocation(k, key, strategy, UPTAKE, config.N, config.T)
    solar, base = solar_season(season), base_mw(season)
    segs = segments(list(design))
    load = np.full(config.SLOTS, base)
    for g, (a, b, _) in enumerate(segs):
        load[in_range(a, b)] += x[g]
    return over, net(load, solar, design, season), w, alloc


def combine(designs, rho, strategy):
    """(overload %, soak MWh/day) of one interval set per season: overload in any season, soak over the year."""
    days = {s: len(solar_season(s)) for s in SEASONS}
    res = {s: evaluate(designs[s], s, rho, strategy) for s in SEASONS}
    w = next(iter(res.values()))[2]
    over = sum(wi for key, wi in w.items() if any(res[s][0][key] for s in SEASONS))
    soak = sum(res[s][1] * days[s] for s in SEASONS) / sum(days.values())
    return over * 100, soak


def placements(k):
    """Interval sets of K 3-hour intervals inside WINDOW, starts every STEP hours."""
    starts = np.arange(WINDOW[0], WINDOW[1] - LENGTH + 1e-9, STEP)
    for s in combinations(starts, k):
        if k == 2 and s[1] < s[0] + LENGTH:                                     # 2: no overlap
            continue
        if k == 3 and not (s[2] >= s[0] + LENGTH and s[1] < s[0] + LENGTH and s[1] + LENGTH > s[2]):
            continue                                                            # 3: middle overlaps both
        yield tuple((float(a), float(a + LENGTH)) for a in s)


def solar_sum(design, profile):
    """(sum over intervals, sum over their union) of a mean daily profile, MWh/day."""
    inside = np.zeros(config.SLOTS, bool)
    for a, b in design:
        inside |= in_range(a, b)
    return sum(profile[in_range(a, b)].sum() for a, b in design) * config.DT, profile[inside].sum() * config.DT


@lru_cache(maxsize=None)
def choose(k, season):
    """Interval set with the highest solar sum in this season (ties: largest union)."""
    profile = solar_season(season).mean(axis=0)
    return max(placements(k), key=lambda d: solar_sum(d, profile))


def name(design):
    return " | ".join(f"{clock(a)}-{clock(b)}" for a, b in design)


def style(ax):
    ax.grid(color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def solar_figure(designs):
    """Mean daily solar per season, with the chosen intervals as bars under the curve."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True)
    top = max(solar_season(s).mean(axis=0).max() for s in SEASONS)
    for ax, season in zip(axes, SEASONS):
        ax.plot(SLOT, solar_season(season).mean(axis=0), color="#333333", lw=1.6)
        ax.axvspan(*WINDOW, color="#888", alpha=0.07, lw=0)
        for k in (1, 2, 3):
            for i, (a, b) in enumerate(designs[k][season]):
                y = -top * (0.08 * k + 0.025 * (k == 3) * (i == 1))             # 3: middle interval offset
                ax.plot([a, b], [y, y], color=COLORS[k], lw=4, solid_capstyle="butt",
                        label=f"{k} interval{'s' * (k > 1)}" if i == 0 else None)
        ax.set_xlim(6, 18)
        ax.set_title(season, loc="left", fontweight="bold")
        ax.set_xlabel("Hour (AEST)")
        style(ax)
    axes[0].set_ylabel(f"Mean {SOLAR_NAME} (MW)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_solar.png", dpi=150)
    plt.close(fig)


def weights_figure(designs):
    """Total load of two intervals at today's mix against mean rooftop solar per season, drawn over the
    hours where soak is counted."""
    kw = config.load_kw()
    to_mw = UPTAKE * config.VIC_CUSTOMERS / config.N / 1000
    loads = {}
    for season in SEASONS:
        _, _, w, alloc = evaluate(designs[2][season], season, TODAY, "lowest overload")
        per = to_mw * sum(w[key] * ((kw["M"] - kw["S"]) * m + (kw["L"] - kw["S"]) * l) for key, (m, l) in alloc.items())
        loads[season] = np.full(config.SLOTS, base_mw(season) / season_weight(season))   # unweighted
        for (a, b), mw in zip(designs[2][season], per):
            loads[season][in_range(a, b)] += mw
    for name, weighted, color in (("fig1b_load.png", SEASONAL_WEIGHTS, COLORS[2]),):
        fig, axes = plt.subplots(1, len(SEASONS), figsize=(14, 3.2), sharey=True)
        for ax, season in zip(axes, SEASONS):
            ws = season_weight(season) if weighted else 1.0
            ax.plot(SLOT, solar_season(season).mean(axis=0), color="#333333", lw=1.6, label="Rooftop solar")
            ax.plot(SLOT, np.where(counted(designs[2][season], season), ws * loads[season], np.nan), lw=2.2,
                    color=color, label="Load")                                  # drawn where soak is counted
            ax.set_xlim(6, 18)
            ax.set_xticks([6, 9, 12, 15, 18])
            ax.set_title(f"{season}, w = {ws:.2f}" if weighted else season, loc="left", fontweight="bold",
                         fontsize=10)
            style(ax)
            win = counted(designs[2][season], season)                          # share of the counted load soaked
            share = (np.minimum(ws * loads[season][win], solar_season(season)[:, win]).sum(axis=1).mean()
                     / (ws * loads[season][win]).sum())
            print(f"  {name:<26} {season:<7} w = {ws:.2f}  soaked {share:.1%}")
        axes[0].set_ylabel("MW")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.supxlabel("Hour", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=150)
        plt.close(fig)


def tradeoff_figure(curves):
    """Left: overload vs soak as rho grows, both strategies. Right: soak given up for the lowest overload."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    today = np.isclose(RHO, TODAY)
    for k, res in curves.items():
        lo, best = res["lowest overload"], res["best soak"]
        lab = f"{k} interval{'s' * (k > 1)}"
        ax.plot(lo[:, 0], lo[:, 1], "-o", ms=3.5, lw=2, color=COLORS[k], label=f"{lab}, lowest overload")
        ax.plot(best[:, 0], best[:, 1], "--", lw=1.5, color=COLORS[k], label=f"{lab}, best soak")
        ax.scatter(lo[today, 0], lo[today, 1], s=110, facecolor="none", edgecolor=COLORS[k], lw=2, zorder=5)
        for rho in LABEL_AT:
            i = np.flatnonzero(np.isclose(RHO, rho))[0]
            if lo[i, 0] < 90:                                                   # saturated points crowd near 100%
                ax.annotate(f"{RHO[i]:.0%}", (lo[i, 0], lo[i, 1]), xytext=(7, -3), textcoords="offset points",
                            fontsize=7.5, color="#52514e")
        bx.plot(RHO * 100, 100 * (best[:, 1] - lo[:, 1]) / best[:, 1], "-o", ms=3.5, lw=2, color=COLORS[k], label=lab)
    ax.scatter([], [], s=110, facecolor="none", edgecolor="#52514e", lw=2, label=f"today's mix (ρ = {TODAY:.0%})")
    ax.set_xlabel("Circuits that overload (%)")
    ax.set_ylabel(f"{SOLAR_NAME.capitalize()} soaked (MWh/day)")
    ax.set_title("Overload vs soak as ρ grows (labels: ρ)", loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    bx.set_xlabel("ρ = p_M + p_L (%)")
    bx.set_ylabel("Soak given up for the lowest overload (%)")
    bx.set_ylim(-0.01, max(0.1, bx.get_ylim()[1]))
    bx.set_title("Cost of allocating for safety", loc="left", fontweight="bold", fontsize=10)
    bx.legend(frameon=False, fontsize=8)
    for a in (ax, bx):
        style(a)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tradeoff.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    designs = {k: {s: choose(k, s) for s in SEASONS} for k in (1, 2, 3)}
    print("Seasonal weights:", {s: round(season_weight(s), 3) for s in SEASONS})
    print("Intervals (solar sum, union MWh/day)")
    for k, d in designs.items():
        for s, v in d.items():
            total, union = solar_sum(v, solar_season(s).mean(axis=0))
            print(f"  K={k} {s:<7} {name(v):<36} {total:6.0f} {union:6.0f}")
    solar_figure(designs)
    weights_figure(designs)

    curves = {k: {st: np.array([combine(d, rho, st) for rho in RHO]) for st in ("lowest overload", "best soak")}
              for k, d in designs.items()}
    tradeoff_figure(curves)
    i = np.flatnonzero(np.isclose(RHO, TODAY))[0]
    print("Customer mix (overload %, soak MWh/day)")
    for k, res in curves.items():
        lo, best = res["lowest overload"], res["best soak"]
        cost = 100 * (best[:, 1] - lo[:, 1]) / best[:, 1]
        caps = "  ".join(f"cap {c}%: rho {RHO[j]:.1%} -> {lo[j, 1]:,.1f}" for c in CAPS
                         for j in [np.flatnonzero(lo[:, 0] <= c)[-1]] if (lo[:, 0] <= c).any())
        print(f"  K={k}: today lowest {lo[i, 0]:.3f}% / {lo[i, 1]:,.1f}, best soak {best[i, 0]:.3f}% / {best[i, 1]:,.1f};"
              f"  {caps};  max soak given up {cost.max():.3f}% at rho {RHO[cost.argmax()]:.1%};"
              f"  max overload gap {np.max(best[:, 0] - lo[:, 0]):.2f} pts")
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
