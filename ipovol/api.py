"""Generic entry point: price any newly listed (or about-to-list) name.

    from ipovol.api import price_target
    pt = price_target("CRWV US Equity", asof="2025-04-04", source=BloombergSource(), cal=cal, draws=draws)
    pt = price_target("targets/anthropic.json", source=BloombergSource(), cal=cal)

The target is a Bloomberg ticker (everything fetched) or a JSON profile (for names that have
not listed, or to override anything). Every input is resolved in the same order:
    profile value  >  data source  >  calibrated/default assumption (and a note is recorded).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import Calibration
from .config import DEFAULT_LOCKUP_CAL_DAYS, QUIET_PERIOD_CAL_DAYS, FEATURE_CLIP
from .features import compute_features, missing
from .pricer import IPOPricer, IPOInputs, SurfaceResult, realized_from_ohlc
from .sources import DataSource


@dataclass
class TargetSpec:
    name: str
    ticker: str | None = None
    listing_date: str | None = None
    offer_price: float | None = None
    shares_offered: float | None = None
    shares_out: float | None = None            # pre-listing: expected post-IPO share count
    valuation_usd: float | None = None         # pre-listing: expected market cap at listing
    expected_first_day_ret: float | None = None
    float_pct: float | None = None
    vc_backed: int | None = None
    tech: int | None = None
    vix: float | None = None
    features: dict | None = None               # full override of the feature vector
    peers: list | None = None
    n_peers: int = 6
    lockup_date: str | None = None
    earnings_dates: list | None = None         # future reaction dates
    borrow_now: float | None = None
    rate: float = 0.04
    div_yield: float = 0.0
    spot: float | None = None                  # pre-listing reference price (default offer price)
    extra_events: list = field(default_factory=list)
    event_premium: float = 1.0
    listing_day_D: float | None = None         # pre-listing only: day-one discovery, local-days
    use_quotes: bool = True
    expiries: list | None = None

    @classmethod
    def load(cls, target: "str | TargetSpec | dict") -> "TargetSpec":
        if isinstance(target, TargetSpec):
            return target
        if isinstance(target, dict):
            d = target
        elif str(target).endswith(".json") and Path(target).exists():
            d = json.load(open(target))
        else:                                    # a bare ticker
            return cls(name=str(target), ticker=str(target))
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class PricedTarget:
    spec: TargetSpec
    asof: pd.Timestamp
    inputs: IPOInputs
    result: SurfaceResult
    notes: list
    listed: bool
    peers: list
    realized_vol_fcst_21d: float


def standard_expiries(val, weeklies=2, monthlies=6, quarterlies_to_months=18) -> list:
    """Two weekly Fridays, the next six monthly (3rd-Friday) expiries, then quarterlies."""
    val = pd.Timestamp(val)
    fr = pd.date_range(val + pd.Timedelta(days=1), periods=weeklies, freq="W-FRI")
    third = [d for d in pd.date_range(val, val + pd.DateOffset(months=quarterlies_to_months), freq="WOM-3FRI")
             if d > val]
    monthly = third[:monthlies]
    quarterly = [d for d in third[monthlies:] if d.month in (3, 6, 9, 12)]
    return sorted(set(list(fr) + monthly + quarterly))


def estimated_earnings(listing, after, n=8) -> list:
    """First report ~45 days after the first quarter-end at least 20 days post-listing, then quarterly."""
    listing = pd.Timestamp(listing)
    q = listing + pd.offsets.QuarterEnd(0)
    first = q + pd.Timedelta(days=45)
    if first < listing + pd.Timedelta(days=20):
        first = q + pd.offsets.QuarterEnd(1) + pd.Timedelta(days=45)
    dates = [first + pd.DateOffset(months=3 * i) for i in range(n)]
    dates = [d + pd.offsets.BDay(0) for d in dates]
    return [d for d in dates if d > pd.Timestamp(after)]


def price_target(target, asof=None, source: DataSource | None = None, cal: Calibration | None = None,
                 draws: list[Calibration] | None = None, anchor: bool | None = None,
                 moneyness=np.linspace(0.5, 1.6, 45)) -> PricedTarget:
    spec = TargetSpec.load(target)
    notes = []
    src = source or DataSource()
    if cal is None:
        raise ValueError("pass a Calibration (Calibration.load('output/calibration.json'))")

    # ---- listing date & reference data ---------------------------------------------
    info = src.ipo_info(spec.ticker) if spec.ticker else {}
    listing = pd.Timestamp(spec.listing_date or info.get("ipo_date") or pd.NaT)
    if pd.isna(listing):
        raise ValueError(f"{spec.name}: no listing date (set listing_date in the profile)")
    offer = spec.offer_price or info.get("offer_price")
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()

    hist = src.history(spec.ticker, listing - pd.Timedelta(days=5), asof) if spec.ticker else None
    hist = hist.loc[listing:] if hist is not None and not hist.empty else None
    listed = hist is not None and len(hist) > 0 and asof >= listing

    # ---- valuation date --------------------------------------------------------------
    extra = list(spec.extra_events)
    if listed:
        val = hist.index[hist.index <= asof][-1]
        spot = float(hist.loc[val, "close"])
    else:
        val = listing - pd.offsets.BDay(1)
        spot = float(spec.spot or offer or 100.0)
        if asof > val:
            notes.append(f"{spec.name} has no trading data up to {asof.date()}; priced as of the day before listing.")
        else:
            notes.append(f"Pre-listing: priced as of {val.date()} (day before listing); options before that are forward-starting.")
        if spec.listing_day_D:
            extra.append(dict(date=listing, D=spec.listing_day_D, name="listing-day discovery"))
        else:
            notes.append("Listing-day price discovery is NOT included (set listing_day_D in the profile).")

    # ---- peers -------------------------------------------------------------------------
    peers = spec.peers or (src.suggest_peers(spec.ticker) if spec.ticker else [])
    if not peers:
        raise ValueError(f"{spec.name}: no peers (set 'peers' in the profile)")
    ok, young = [], []
    for p in peers:
        h = src.history(p, listing - pd.Timedelta(days=400), val)
        if h is None or h.empty or "iv_atm" not in h or h["iv_atm"].loc[:val].dropna().empty:
            continue
        ok.append(p)
        if h.index[0] > listing - pd.Timedelta(days=300):
            young.append(p)
    peers = ok[: spec.n_peers]
    if not peers:
        raise ValueError(f"{spec.name}: none of the peers has implied-vol data at {val.date()}")
    if not spec.peers:
        notes.append(f"Peers auto-selected: {', '.join(peers)} - review them.")
    if young:
        notes.append(f"Peers with <1y history before listing (their vol may itself carry an IPO premium): {', '.join(young)}")
    term, psi, gam = src.peer_term(peers, val)

    # ---- features ----------------------------------------------------------------------
    feats = src.deal_features(spec.ticker) if spec.ticker else None
    if not feats:
        d0 = hist.iloc[0] if listed else {}
        feats = compute_features(
            listing, offer_price=offer, first_close=d0.get("close") if listed else None,
            shares_offered=spec.shares_offered or info.get("shares_offered"),
            shares_out=spec.shares_out or (d0.get("sh_out") if listed else None),
            mkt_cap_usd=spec.valuation_usd or (d0.get("mkt_cap") if listed else None),
            vix=spec.vix if spec.vix is not None else src.vix(listing),
            vc_backed=spec.vc_backed, tech=spec.tech, float_pct=spec.float_pct,
            first_day_ret=None if listed else spec.expected_first_day_ret)
    if spec.features:
        feats = {**feats, **spec.features}
    miss = missing(feats)
    if miss:
        notes.append(f"Missing features priced as the average IPO: {', '.join(miss)}")
    raw_z = cal.scaler.transform(pd.DataFrame([feats])).iloc[0]
    clipped = {k: round(float(v), 1) for k, v in raw_z.items() if abs(v) > FEATURE_CLIP}
    if clipped:
        notes.append(f"Out-of-sample features clipped at ±{FEATURE_CLIP:.0f}σ (extrapolation): {clipped}")

    # ---- events --------------------------------------------------------------------------
    k0 = len(src.past_earnings(spec.ticker, listing, val)) if (listed and spec.ticker) else 0
    if spec.earnings_dates:
        e_dates = [pd.Timestamp(d) for d in spec.earnings_dates]
    else:
        nxt = src.next_earnings(spec.ticker, val) if spec.ticker else []
        if nxt:
            e_dates = [pd.Timestamp(nxt[0])] + [pd.Timestamp(nxt[0]) + pd.DateOffset(months=3 * i) for i in range(1, 6)] \
                if len(nxt) == 1 else [pd.Timestamp(d) for d in nxt]
            if len(nxt) == 1:
                notes.append("Earnings after the next one estimated at quarterly spacing.")
        else:
            e_dates = estimated_earnings(listing, val)
            notes.append(f"Earnings dates estimated (first ~{e_dates[0].date() if e_dates else 'n/a'}); set earnings_dates to override.")
    lockup = spec.lockup_date or info.get("lockup_date")
    if lockup is None or pd.isna(lockup):
        lockup = listing + pd.Timedelta(days=DEFAULT_LOCKUP_CAL_DAYS)
        notes.append(f"Lockup assumed at listing + {DEFAULT_LOCKUP_CAL_DAYS}d ({pd.Timestamp(lockup).date()}); check the S-1 for staged releases.")

    # ---- realised so far -------------------------------------------------------------------
    realized = None
    if listed and len(hist.loc[:val]) > 2:
        past_ev = [listing + pd.Timedelta(days=QUIET_PERIOD_CAL_DAYS)] + src.past_earnings(spec.ticker, listing, val)
        peer_frames = [src.history(p, listing - pd.Timedelta(days=5), val) for p in peers]
        realized = realized_from_ohlc(hist.loc[:val], peer_frames, listing, past_ev)

    # ---- listed quotes -------------------------------------------------------------------
    use_q = spec.use_quotes if anchor is None else anchor
    quotes = src.option_quotes(spec.ticker, val, spot) if (use_q and listed) else []
    if listed and use_q and not quotes:
        notes.append("No listed option quotes found: surface is pure model (no market anchoring).")

    inp = IPOInputs(valuation_date=val, listing_date=listing, spot=spot, features=feats, peer_term=term,
                    peer_psi=psi, peer_gamma=gam, rate=spec.rate, div_yield=spec.div_yield,
                    earnings_dates=e_dates, k0=k0, lockup_date=lockup, extra_events=extra,
                    borrow_now=spec.borrow_now, realized=realized, quotes=quotes,
                    event_premium=spec.event_premium)
    expiries = [pd.Timestamp(e) for e in spec.expiries] if spec.expiries else standard_expiries(val)
    pricer = IPOPricer(cal, draws)
    res = pricer.price(inp, expiries, moneyness=moneyness)
    rv = pricer.expected_realized(inp, 21)
    return PricedTarget(spec, val, inp, res, notes, listed, peers, rv)
