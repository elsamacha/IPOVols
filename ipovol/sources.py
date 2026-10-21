"""Data sources behind `price_target`. Same interface, two implementations:

* BloombergSource - live, any ticker (run on the terminal machine).
* PanelSource     - a stored universe (the Bloomberg pull pickle, or synthetic data). Used for
                    backtests and for re-pricing any past IPO as of any date, point in time.

Adding another vendor (QuantVol, an internal vol service) = implementing these methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, IV_TERM_COLS, BBG_REF_FIELDS, BBG_PEERS_BULK, BBG_OPT_CHAIN, BBG_OPT_FIELDS
from .estimators import smile_params


def term_from_frames(frames: list[pd.DataFrame], date) -> tuple[dict, float, float]:
    """Peer-average ATM term structure {cal_days: vol} (averaging variances) and the peer
    30d smile (psi, gamma), from each peer's last observation on or before `date`."""
    date = pd.Timestamp(date)
    rows = [f.loc[:date].iloc[-1] for f in frames if f is not None and not f.loc[:date].empty]
    if not rows:
        raise ValueError("no peer implied-vol data on or before the valuation date")
    R = pd.DataFrame(rows)
    term = {}
    for col, days in IV_TERM_COLS.items():
        if col in R and R[col].notna().any():
            term[days] = float(np.sqrt(np.nanmean(R[col].astype(float) ** 2)))
    if 30 not in term:
        raise ValueError("peers have no 30d ATM implied vol at the valuation date")
    psi, gam = np.nan, np.nan
    if {"iv_90", "iv_110"} <= set(R.columns):
        v90 = np.sqrt(np.nanmean(R["iv_90"].astype(float) ** 2))
        v110 = np.sqrt(np.nanmean(R["iv_110"].astype(float) ** 2))
        psi, gam = smile_params(v90, term[30], v110)
    return term, float(psi) if np.isfinite(psi) else -0.10, float(gam) if np.isfinite(gam) else 0.05


class DataSource:
    """Interface. Methods return None / [] when a source cannot answer."""

    def history(self, ticker, start, end) -> pd.DataFrame | None: ...
    def ipo_info(self, ticker) -> dict: return {}
    def past_earnings(self, ticker, start, end) -> list: return []
    def next_earnings(self, ticker, asof) -> list: return []
    def suggest_peers(self, ticker) -> list: return []
    def option_quotes(self, ticker, asof, spot) -> list: return []
    def deal_features(self, ticker) -> dict | None: return None
    def vix(self, date) -> float | None: return None

    def peer_term(self, peers, date):
        start = pd.Timestamp(date) - pd.Timedelta(days=10)
        return term_from_frames([self.history(p, start, date) for p in peers], date)


class PanelSource(DataSource):
    """Point-in-time view of a stored universe {deals, px, earnings}."""

    def __init__(self, deals: pd.DataFrame, px: dict, earnings: dict, vix: pd.Series | None = None):
        self.deals, self.px, self.earnings, self._vix = deals, px, earnings, vix

    @classmethod
    def from_pickle(cls, path):
        import pickle
        d = pickle.load(open(path, "rb"))
        return cls(d["deals"], d["px"], d["earnings"])

    def history(self, ticker, start, end):
        f = self.px.get(ticker)
        return None if f is None else f.loc[pd.Timestamp(start):pd.Timestamp(end)]

    def ipo_info(self, ticker):
        if ticker not in self.deals.index:
            return {}
        r = self.deals.loc[ticker]
        return {k: r[k] for k in ("ipo_date", "offer_price", "shares_offered", "lockup_date") if k in r and pd.notna(r[k])}

    def past_earnings(self, ticker, start, end):
        return [d for d in self.earnings.get(ticker, []) if pd.Timestamp(start) < d <= pd.Timestamp(end)]

    def next_earnings(self, ticker, asof):
        # Uses actual reaction dates: earnings dates are announced/estimated weeks ahead,
        # so this is a mild look-ahead on timing only, never on outcomes.
        return [d for d in self.earnings.get(ticker, []) if d > pd.Timestamp(asof)]

    def suggest_peers(self, ticker):
        return list(self.deals.loc[ticker, "peers"]) if ticker in self.deals.index else []

    def option_quotes(self, ticker, asof, spot):
        f = self.px.get(ticker)
        if f is None or "iv_atm" not in f or f.loc[:asof].empty:
            return []
        iv = f.loc[:asof, "iv_atm"].iloc[-1]
        if pd.isna(iv):
            return []
        return [(pd.Timestamp(asof) + pd.Timedelta(days=30), float(spot), float(iv))]

    def deal_features(self, ticker):
        if ticker not in self.deals.index:
            return None
        r = self.deals.loc[ticker]
        return {f: float(r[f]) for f in FEATURES if f in r and pd.notna(r[f])}

    def vix(self, date):
        return float(self._vix.asof(pd.Timestamp(date))) if self._vix is not None else None


class BloombergSource(DataSource):
    """Live Bloomberg via xbbg. Field mnemonics in config.py - verify in FLDS <GO>."""

    def __init__(self):
        from . import bbg
        self.bbg = bbg
        self._cache = {}

    def history(self, ticker, start, end):
        key = (ticker, str(start)[:10], str(end)[:10])
        if key not in self._cache:
            self._cache[key] = self.bbg.bdh([ticker], start, end).get(ticker)
        return self._cache[key]

    def ipo_info(self, ticker):
        try:
            r = self.bbg.blp().bdp(ticker, list(BBG_REF_FIELDS.values()))
        except Exception:
            return {}
        if r is None or r.empty:
            return {}
        inv = {v.lower(): k for k, v in BBG_REF_FIELDS.items()}
        row = r.iloc[0]
        out = {inv.get(c.lower(), c): row[c] for c in r.columns if pd.notna(row[c])}
        if "ipo_date" in out:
            out["ipo_date"] = pd.Timestamp(out["ipo_date"])
        return out

    def past_earnings(self, ticker, start, end):
        dates = self.bbg.pull_earnings([ticker]).get(ticker, [])
        return [d for d in dates if pd.Timestamp(start) < d <= pd.Timestamp(end)]

    def next_earnings(self, ticker, asof):
        nxt = self.ipo_info(ticker).get("next_earnings")
        if nxt is None or pd.isna(nxt):
            return []
        # timing unknown in the reference field: assume after the close -> react next session
        return [pd.Timestamp(nxt) + pd.offsets.BDay(1)]

    def suggest_peers(self, ticker):
        try:
            b = self.bbg.blp().bds(ticker, BBG_PEERS_BULK)
        except Exception:
            return []
        if b is None or b.empty:
            return []
        col = b.columns[0]
        return [t if t.endswith("Equity") else f"{t} Equity" for t in b[col].astype(str)]

    def option_quotes(self, ticker, asof, spot, n_expiries=4, band=(0.8, 1.2)):
        """Listed mid IVs, OTM side, near-dated expiries. Live only (asof ~ today)."""
        blp = self.bbg.blp()
        try:
            chain = blp.bds(ticker, BBG_OPT_CHAIN)
        except Exception:
            return []
        if chain is None or chain.empty:
            return []
        opts = chain.iloc[:, 0].astype(str).tolist()
        ref = blp.bdp(opts, list(BBG_OPT_FIELDS.values()))
        inv = {v.lower(): k for k, v in BBG_OPT_FIELDS.items()}
        ref = ref.rename(columns={c: inv.get(c.lower(), c) for c in ref.columns})
        ref["expiry"] = pd.to_datetime(ref["expiry"])
        ref = ref[(ref["strike"] / spot).between(*band) & ref["iv"].notna() & (ref.get("open_int", 1) > 0)]
        otm = ((ref["put_call"].str[0].str.upper() == "C") & (ref["strike"] >= spot)) | \
              ((ref["put_call"].str[0].str.upper() == "P") & (ref["strike"] < spot))
        ref = ref[otm]
        keep = sorted(ref["expiry"].unique())[:n_expiries]
        ref = ref[ref["expiry"].isin(keep)]
        return [(r.expiry, float(r.strike), float(r.iv) / 100.0) for r in ref.itertuples()]

    def vix(self, date):
        try:
            return float(self.bbg.vix_series(pd.Timestamp(date) - pd.Timedelta(days=7), date).iloc[-1])
        except Exception:
            return None
