#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster — dilution-aware log-time model.

Method in one paragraph: Exophase publishes trophy rates as of its own last
refresh, hours apart, so a read is a stale snapshot, not a live sample. When
the case history is flat (players tracked never moved), the page is stale and
a refresh is due: the median assumes one refresh multiplies players by about
six (the documented first-refresh jump, 638 -> 3,868) while new players hold
almost no trophies, so each rate dilutes to about a sixth of its old value.
When the history moves, each trophy is modelled as a growth line: the
logit of its rate is fit robustly (Theil-Sen, median pairwise slope) against
log-time since launch over the last 24 hours of reads and extrapolated to
the target. Theil-Sen matters because Exophase refreshes step rates down in
jumps, and a least-squares line would be dragged by them; the median slope
is not. A trophy with fewer than three reads in the window falls back to
persistence. The spread grows with the horizon: a random-walk term plus a
refresh-uncertainty term. On the trend path the fit contributes the standard
error of the extrapolated median (not the per-read scatter), and the walk
term is halved because the trend already captures systematic movement.
Quantiles come from a logit-normal around the median, so they stay ordered,
finite, and inside (0, 100). Stdlib only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist, median

# Rates live in percent; the scorer clamps answers to [0.01, 99.99].
RATE_LO = 0.01
RATE_HI = 99.99

# How much one Exophase refresh multiplies the tracked-player count. The kit
# documents the first refresh at 638 -> 3,868 players, about 6x.
REFRESH_MULT = 6.0

# Spread model (logit units). A refresh can move a rate by ~1.5 logit and its
# timing/size is uncertain, so 0.5 logit of refresh uncertainty is carried
# on every forecast; a random walk adds 0.5 * sqrt(days) over the horizon.
REFRESH_SIGMA = 0.5
WALK_SIGMA_PER_DAY = 0.5

_ND = NormalDist()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    """p in (0, 1) -> log odds. Caller keeps p away from 0 and 1."""
    return math.log(p / (1.0 - p))


def inv_logit(z):
    """Log odds -> p in (0, 1), stable for large |z|."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (all values here are positive)."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def parse_iso(s):
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def launch_time(case):
    """t=0 for the log-time axis: the game's release date at 00:00 UTC."""
    game = case.get("game") or {}
    rel = game.get("release_date")
    if rel:
        try:
            d = datetime.fromisoformat(str(rel))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d
        except ValueError:
            pass
    hist = case.get("history") or []
    if hist:
        return parse_iso(hist[0]["at"])
    return parse_iso(case["at"])


def theil_sen(xs, ys):
    """Robust line fit: median pairwise slope, median-residual intercept.

    Reads are subsampled before this is called, so the O(n^2) pair loop
    stays small (n <= 80). Deterministic: pairs are enumerated in order.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, ys[0]
    slopes = []
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n):
            dx = xs[j] - xi
            if dx != 0.0:
                slopes.append((ys[j] - yi) / dx)
    if not slopes:
        return 0.0, median(ys)
    b = median(slopes)
    a = median([y - b * x for x, y in zip(xs, ys)])
    return b, a


def robust_sigma(residuals):
    """1.4826 * MAD, the Gaussian-consistent spread of fit residuals."""
    if len(residuals) < 3:
        return 0.0
    med = median(residuals)
    return 1.4826 * median([abs(r - med) for r in residuals])


def horizon_days(case):
    mins = case.get("horizon_minutes")
    if mins is not None:
        return float(mins) / 1440.0
    at = parse_iso(case["at"])
    target = parse_iso(case["target"])
    return max(0.0, (target - at).total_seconds() / 86400.0)


def spread(days, fit_sigma=0.0):
    """Total logit spread for the dilution path: refresh + horizon walk.

    A refresh can move a rate by ~1.5 logit and its timing/size is
    uncertain, so 0.5 logit of refresh uncertainty is carried on every
    forecast; a random walk adds 0.5 * sqrt(days) over the horizon.
    """
    return math.sqrt(fit_sigma * fit_sigma
                     + REFRESH_SIGMA * REFRESH_SIGMA
                     + (WALK_SIGMA_PER_DAY ** 2) * days)


def trend_spread(days, pred_sigma):
    """Total logit spread for the trend path.

    The trend already captures systematic movement, so the walk term is
    halved (0.25 * sqrt(days)); pred_sigma is the standard error of the
    extrapolated median from the fit, not the per-read residual scatter —
    the median is far better determined than any single read.
    """
    return math.sqrt(pred_sigma * pred_sigma
                     + REFRESH_SIGMA * REFRESH_SIGMA
                     + (0.25 ** 2) * days)


def quantiles(median_pct, sigma, levels):
    """Logit-normal quantiles around a median rate (percent)."""
    m = clamp(median_pct, RATE_LO, RATE_HI) / 100.0
    mu = logit(m)
    out = []
    prev = None
    for tau in levels:
        z = _ND.inv_cdf(tau)
        q = round4(clamp(inv_logit(mu + z * sigma) * 100.0, RATE_LO, RATE_HI))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def subsample(points, cap):
    """Evenly spaced subset of at most `cap` points, first and last kept."""
    n = len(points)
    if n <= cap:
        return points
    idx = sorted({int(round(i * (n - 1) / (cap - 1))) for i in range(cap)})
    return [points[i] for i in idx]


def read_history(case):
    """History reads as dicts (ts, log-time x, players N, rates), oldest first."""
    launch = launch_time(case)
    reads = []
    for entry in case.get("history") or []:
        try:
            t = parse_iso(entry["at"])
            n = int(entry["players_tracked"])
            rates = entry.get("rates") or []
        except (KeyError, TypeError, ValueError):
            continue
        hours = max(0.0, (t - launch).total_seconds() / 3600.0)
        reads.append({"ts": t.timestamp(), "x": math.log1p(hours),
                      "n": n, "rates": rates})
    return reads


def latest_rate(reads, idx):
    """Most recent (rate, players) for one trophy, scanning newest first."""
    for read in reversed(reads):
        rates = read["rates"]
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx], read["n"]
    return None, None


def dilution_forecast(case, reads, levels):
    """Flat history: the page is stale, a refresh is due, rates dilute.

    Median = old earners / (players * 6), i.e. the old rate divided by six.
    A trophy everyone holds stays at 100%. Spread is the refresh term plus
    the horizon walk; it is wide enough to cover the no-refresh case at the
    upper quantiles.
    """
    days = horizon_days(case)
    sigma = spread(days)
    rows = []
    for idx in range(len(case["trophies"])):
        rate, n = latest_rate(reads, idx)
        if rate is None or n is None or n <= 0:
            rows.append(quantiles(0.5, 2.0, levels))
        elif rate >= RATE_HI:
            rows.append(quantiles(100.0, sigma, levels))
        else:
            earners = rate / 100.0 * n
            median_pct = earners / (n * REFRESH_MULT) * 100.0
            rows.append(quantiles(median_pct, sigma, levels))
    return rows


def trend_forecast(case, reads, levels):
    """Moving history: robust log-time growth line on the rate itself.

    Per trophy, logit(rate) is fit against log-time since launch with
    Theil-Sen over the last 24 hours of reads and read at the target. The
    median pairwise slope is robust to the downward steps a refresh writes
    into the rate series, where least squares would be dragged. Fewer than
    three reads in the window falls back to persistence for that trophy.
    Spread is the standard error of the extrapolated median plus the refresh
    and (halved) horizon-walk terms.
    """
    launch = launch_time(case)
    at = parse_iso(case["at"])
    target = parse_iso(case["target"])
    t_hours = max(0.0, (target - launch).total_seconds() / 3600.0)
    x_target = math.log1p(t_hours)
    days = horizon_days(case)
    window_start = at.timestamp() - 24.0 * 3600.0

    rows = []
    for idx in range(len(case["trophies"])):
        pts = []
        for read in reads:
            if read["ts"] < window_start:
                continue
            rates = read["rates"]
            if idx < len(rates) and rates[idx] is not None:
                pts.append((read["x"], logit(rates[idx] / 100.0)))
        if len(pts) < 3:
            rate, _n = latest_rate(reads, idx)
            if rate is None:
                rows.append(quantiles(0.5, 2.0, levels))
            else:
                rows.append(quantiles(rate, spread(days), levels))
            continue
        pts = subsample(pts, 80)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        b, a = theil_sen(xs, ys)
        median_pct = inv_logit(a + b * x_target) * 100.0
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        # Standard error of the extrapolated median: the per-read scatter
        # shrunk by the fit (n reads) and grown by the extrapolation
        # leverage. This is much smaller than the raw residual scatter.
        n = len(xs)
        x_mean = sum(xs) / n
        sxx = sum((x - x_mean) ** 2 for x in xs)
        leverage = 1.0 / n + (x_target - x_mean) ** 2 / sxx if sxx > 0 else 1.0
        pred_sigma = clamp(robust_sigma(resid) * math.sqrt(leverage), 0.15, 1.0)
        rows.append(quantiles(median_pct, trend_spread(days, pred_sigma),
                              levels))
    return rows


def forecast(case):
    """One case -> one row of quantiles per trophy, in trophy order."""
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    trophies = case.get("trophies") or []
    if not trophies:
        return []
    reads = read_history(case)
    if not reads:
        return [quantiles(1.0, 2.0, levels) for _ in trophies]
    # Players tracked never moved: the page is stale, a refresh is due.
    if len({r["n"] for r in reads}) <= 1:
        return dilution_forecast(case, reads, levels)
    return trend_forecast(case, reads, levels)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast(case)
        doc = {"trophies": [{"id": tr["id"], "q": row}
                            for tr, row in zip(case["trophies"], rows)]}
        out_name = fn[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, separators=(",", ":")) + "\n")
        sys.stderr.write(
            "%s: %d trophies, %d history reads -> %s\n"
            % (case.get("id", "?"), len(rows), len(case.get("history") or []),
               out_name))


if __name__ == "__main__":
    main()
