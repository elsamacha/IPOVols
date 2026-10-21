"""Build the calibration panel (IPO x trading-day-since-listing) and the event table.

Inputs (identical for Bloomberg data and the synthetic generator):
    deals    : DataFrame indexed by ticker with columns
               ipo_date, peers (list of tickers), lockup_date (optional, NaT ok), FEATURES...
    px       : dict ticker -> DataFrame indexed by date with open, high, low, close and
               optionally iv_atm, iv_90, iv_110 (30d, vol units e.g. 0.55) and borrow_fee.
               Must contain every IPO and every peer.
    earnings : dict ticker -> list of *reaction* dates (first session that trades on the news)
"""
import numpy as np
import pandas as pd

from .config import (HORIZON, QUIET_PERIOD_CAL_DAYS, DEFAULT_LOCKUP_CAL_DAYS,
                     EVENT_HALF_WINDOW, LOCAL_VAR_WINDOW)
from .estimators import daily_variance, log_returns, smile_params


def _first_on_or_after(index: pd.DatetimeIndex, date) -> int | None:
    if date is None or pd.isna(date):
        return None
    pos = index.searchsorted(pd.Timestamp(date))
    return int(pos) if pos < len(index) else None


def _peer_block(px, peers, dates):
    vs, rs, ivs = [], [], {c: [] for c in ("iv_atm", "iv_90", "iv_110")}
    for p in peers:
        if p not in px:
            continue
        f = px[p]
        vs.append(daily_variance(f).reindex(dates))
        rs.append(log_returns(f["close"]).reindex(dates))
        for c in ivs:
            if c in f:
                ivs[c].append(f[c].reindex(dates))
    if not vs:
        raise ValueError(f"no peer data for peers {peers}")
    out = pd.DataFrame(index=dates)
    out["peer_var"] = pd.concat(vs, axis=1).mean(axis=1)
    R = pd.concat(rs, axis=1)
    out["peer_ret"] = R.mean(axis=1)                 # basket return (for abnormal returns)
    out["peer_ret2"] = (R ** 2).mean(axis=1)         # average single-name squared return
    for c, lst in ivs.items():
        # average *variance* across peers, report as vol
        out["peer_" + c] = np.sqrt((pd.concat(lst, axis=1) ** 2).mean(axis=1)) if lst else np.nan
    return out


def build_panel(deals: pd.DataFrame, px: dict, earnings: dict, horizon: int = HORIZON):
    rows, ev_rows = [], []
    for tk, deal in deals.iterrows():
        ipo_date = pd.Timestamp(deal["ipo_date"])
        f = px[tk].loc[ipo_date:].iloc[: horizon + 1].copy()
        if len(f) < 40:
            continue
        dates = f.index
        f["ticker"] = tk
        f["t"] = np.arange(len(f))
        f["var"] = daily_variance(f)
        f["ret"] = log_returns(f["close"])
        f.loc[f["t"] == 0, ["var", "ret"]] = np.nan          # listing day = allocation noise
        peers = deal["peers"] if isinstance(deal["peers"], (list, tuple)) else str(deal["peers"]).split(";")
        f = f.join(_peer_block(px, peers, dates))
        f["peer_var_smooth"] = f["peer_var"].rolling(21, center=True, min_periods=5).mean()

        # ---- events --------------------------------------------------------------
        events = []
        q = _first_on_or_after(dates, ipo_date + pd.Timedelta(days=QUIET_PERIOD_CAL_DAYS))
        if q is not None:
            events.append(("quiet", 0, q))
        lk_date = deal.get("lockup_date", pd.NaT)
        if pd.isna(lk_date):
            lk_date = ipo_date + pd.Timedelta(days=DEFAULT_LOCKUP_CAL_DAYS)
        lk = _first_on_or_after(dates, lk_date)
        if lk is not None and dates[lk] - pd.Timestamp(lk_date) <= pd.Timedelta(days=5):
            events.append(("lockup", 0, lk))
        e_dates = sorted(pd.Timestamp(x) for x in earnings.get(tk, []) if pd.Timestamp(x) > ipo_date)
        for i, ed in enumerate(e_dates, 1):
            pos = _first_on_or_after(dates, ed)
            if pos is not None and dates[pos] - ed <= pd.Timedelta(days=5):
                events.append(("earnings", i, pos))

        t_ev = np.array([e[2] for e in events]) if events else np.array([], int)
        t = f["t"].to_numpy()
        in_win = np.zeros(len(f), bool)
        for te in t_ev:
            in_win |= np.abs(t - te) <= EVENT_HALF_WINDOW
        f["event_window"] = in_win
        # IV exclusion: any IPO event reaction within the next 30 calendar days (inclusive of today)
        ahead = np.zeros(len(f), bool)
        for te in t_ev:
            dd = (dates[te] - dates).days
            ahead |= (dd >= 0) & (dd <= 30)
        f["event_ahead_30"] = ahead
        e_pos = sorted(e[2] for e in events if e[0] == "earnings")
        f["k"] = np.searchsorted(np.array(e_pos), t, side="left") if e_pos else 0

        # smile params
        if {"iv_atm", "iv_90", "iv_110"} <= set(f.columns):
            f["psi"], f["gamma"] = smile_params(f["iv_90"], f["iv_atm"], f["iv_110"])
        else:
            f["iv_atm"] = f.get("iv_atm", np.nan)
            f["psi"] = f["gamma"] = np.nan
        f["peer_psi"], f["peer_gamma"] = smile_params(f["peer_iv_90"], f["peer_iv_atm"], f["peer_iv_110"])
        if "borrow_fee" not in f:
            f["borrow_fee"] = np.nan
        rows.append(f)

        # ---- event returns -------------------------------------------------------
        close = f["close"].to_numpy()
        ret = f["ret"].to_numpy()
        pret = f["peer_ret"].to_numpy()
        var = f["var"].to_numpy()
        lo, hi = LOCAL_VAR_WINDOW
        for kind, seq, te in events:
            if te < 3 or te + 1 >= len(f):
                continue
            contaminated = any(abs(te - o) <= 3 for o in t_ev if o != te)
            r = np.log(close[te + 1] / close[te - 2])
            r_ab = r - np.nansum(pret[te - 1: te + 2])
            idx = np.r_[max(1, te - hi): te - lo + 1, te + lo: min(len(f), te + hi + 1)]
            idx = idx[~in_win[idx]]
            loc = np.nanmean(ret[idx] ** 2) if len(idx) >= 10 else np.nan
            loc_gk = np.nanmean(var[idx]) if len(idx) >= 10 else np.nan
            ev_rows.append(dict(ticker=tk, kind=kind, seq=seq, t=te, date=dates[te], r=r, r_ab=r_ab,
                                local_var=loc, local_var_gk=loc_gk, contaminated=contaminated))

    panel = pd.concat(rows).reset_index(names="date")
    events = pd.DataFrame(ev_rows)
    feats = deals.drop(columns=[c for c in ("ipo_date", "peers", "lockup_date") if c in deals])
    return panel, events, feats
