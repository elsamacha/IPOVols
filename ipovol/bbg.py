"""Bloomberg access (run on the terminal machine). Requires `pip install xbbg blpapi`.

Two uses:
* `run_pull(universe_csv)` - history for the calibration universe of past IPOs.
* `sources.BloombergSource` - live inputs for any single target name (built on these helpers).

Universe CSV (template: data/ipo_universe_template.csv):
    ticker, ipo_date, offer_price, shares_offered, peers (';'-separated), listing_type
    (ipo|direct|spac), lockup_date (optional), vc_backed, tech (0/1, optional)
Every field mnemonic lives in config.py - check each one in FLDS <GO>.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BBG_FIELDS, BBG_EARNINGS_BULK, BBG_VIX, HORIZON
from .features import compute_features

_IV_COLS = ("iv_atm", "iv_90", "iv_110", "iv_60", "iv_91", "iv_182", "iv_365")


def blp():
    try:
        from xbbg import blp as _b
    except ImportError as e:
        raise ImportError("pip install xbbg blpapi  (run on a machine with a Bloomberg terminal)") from e
    return _b


def load_universe(path) -> pd.DataFrame:
    u = pd.read_csv(path)
    lt = u["listing_type"] if "listing_type" in u else pd.Series("ipo", index=u.index)
    u = u[lt.fillna("ipo").str.lower() == "ipo"].copy()
    u["ipo_date"] = pd.to_datetime(u["ipo_date"])
    u["lockup_date"] = pd.to_datetime(u["lockup_date"], errors="coerce") if "lockup_date" in u else pd.NaT
    u["peers"] = u["peers"].fillna("").map(lambda s: [p.strip() for p in s.split(";") if p.strip()])
    return u.set_index("ticker")


def bdh(tickers, start, end, fields: dict = BBG_FIELDS) -> dict:
    """Daily history -> {ticker: DataFrame} with our column names, vols as decimals,
    shares and market cap in units (Bloomberg returns millions)."""
    raw = blp().bdh(tickers=list(tickers), flds=list(fields.values()), start_date=str(start)[:10],
                    end_date=str(end)[:10], CshAdjNormal=True, CshAdjAbnormal=True, CapChg=True)
    inv = {v: k for k, v in fields.items()}
    out = {}
    if raw is None or raw.empty:
        return out
    for tk in tickers:
        if tk not in raw.columns.get_level_values(0):
            continue
        df = raw[tk].rename(columns=inv)
        df.index = pd.to_datetime(df.index)
        for c in _IV_COLS:
            if c in df:
                df[c] = df[c] / 100.0
        for c in ("sh_out", "mkt_cap"):
            if c in df:
                df[c] = df[c] * 1e6
        out[tk] = df
    return out


def pull_prices(universe: pd.DataFrame, horizon_cal_days=int(HORIZON * 1.5) + 10, chunk=25) -> dict:
    """Daily OHLC + IVs for every IPO and peer over its window (peers from 400d before listing)."""
    jobs = {}
    for tk, row in universe.iterrows():
        s, e = row["ipo_date"] - pd.Timedelta(days=400), row["ipo_date"] + pd.Timedelta(days=horizon_cal_days)
        for name in [tk] + row["peers"]:
            a, b = jobs.get(name, (s, e))
            jobs[name] = (min(a, s), max(b, e))
    px, names = {}, list(jobs)
    for i in range(0, len(names), chunk):
        batch = names[i:i + chunk]
        s = min(jobs[n][0] for n in batch)
        e = min(max(jobs[n][1] for n in batch), pd.Timestamp.today())
        px.update(bdh(batch, s, e))
    return px


def reaction_dates(bulk: pd.DataFrame) -> list:
    """Parse an earnings-history bulk table into reaction dates
    (announcement day if before the open, otherwise next business day)."""
    if bulk is None or bulk.empty:
        return []
    dcol = next(c for c in bulk.columns if "date" in c.lower())
    tcol = next((c for c in bulk.columns if "time" in c.lower()), None)
    out = []
    for _, r in bulk.iterrows():
        dt = pd.to_datetime(r[dcol], errors="coerce")
        if pd.isna(dt):
            continue
        timing = str(r[tcol]).lower() if tcol else ""
        before_open = "bef" in timing or (":" in timing and timing < "09:30")
        out.append(dt if before_open else dt + pd.offsets.BDay(1))
    return sorted(set(out))


def pull_earnings(tickers) -> dict:
    out = {}
    for tk in tickers:
        try:
            out[tk] = reaction_dates(blp().bds(tk, BBG_EARNINGS_BULK))
        except Exception:
            out[tk] = []
    return out


def vix_series(start, end) -> pd.Series:
    v = blp().bdh(BBG_VIX, "PX_LAST", str(start)[:10], str(end)[:10])
    s = v.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s


def add_features(universe: pd.DataFrame, px: dict) -> pd.DataFrame:
    """Deal features for the calibration universe via the shared builder."""
    u = universe.copy()
    vix = vix_series(u["ipo_date"].min(), pd.Timestamp.today())
    rows = []
    for tk, r in u.iterrows():
        f = px.get(tk)
        d0 = f.loc[r["ipo_date"]:].iloc[0] if f is not None and not f.loc[r["ipo_date"]:].empty else {}
        rows.append(compute_features(
            r["ipo_date"], offer_price=r.get("offer_price"), first_close=d0.get("close"),
            shares_offered=r.get("shares_offered"), shares_out=d0.get("sh_out"),
            mkt_cap_usd=d0.get("mkt_cap"), vix=vix.asof(r["ipo_date"]),
            vc_backed=r.get("vc_backed"), tech=r.get("tech")))
    F = pd.DataFrame(rows, index=u.index)
    for c in F:
        u[c] = F[c]
    return u


def merge_borrow(px: dict, borrow_csv) -> dict:
    """Attach borrow-fee history (columns: ticker, date, fee as annual decimal).
    Source: desk securities-lending data or a Markit/S3 export; Bloomberg coverage is thin."""
    b = pd.read_csv(borrow_csv, parse_dates=["date"])
    for tk, g in b.groupby("ticker"):
        if tk in px:
            px[tk]["borrow_fee"] = g.set_index("date")["fee"].reindex(px[tk].index).ffill(limit=3)
    return px


def run_pull(universe_csv, out_path="data/bbg_raw.pkl"):
    u = load_universe(universe_csv)
    px = pull_prices(u)
    earnings = pull_earnings(list(u.index))
    deals = add_features(u, px)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(dict(deals=deals, px=px, earnings=earnings), fh)
    print(f"saved {len(px)} price series, {len(deals)} deals -> {out_path}")
    return deals, px, earnings
