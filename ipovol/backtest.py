"""Leave-one-fold-out backtest: does the model generalise to IPOs it has not seen?

For each fold, calibrate on the other IPOs, then price every held-out IPO with the same
`price_target` path used live, point in time at t0 trading days after listing, with no
market anchoring (the model's prior is what is being tested). Two targets:

* implied: the held-out IPO's actual 30d ATM implied vol at t0 (what the market charged)
* realised: its close-to-close realised vol over the next 21 trading days

Baselines, both fitted on the training folds only:
* B0 "peer vol":      peer 30d ATM implied (for realised: scaled to the peer realised level)
* B1 "peer x premium": B0 x the average IPO premium at that age (a flat multiplier)
The model has to beat B1 to justify its structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .api import TargetSpec, price_target
from .calibrate import calibrate, implied_blocks, realized_blocks
from .panel import build_panel
from .sources import PanelSource


def _qlike(real_vol, fc_vol):
    q = (real_vol / fc_vol) ** 2
    return q - np.log(q) - 1


def backtest(deals, px, earnings, origins=(5, 21), folds=5, horizon=21, seed=0, verbose=True):
    panel, events, feats = build_panel(deals, px, earnings)
    tickers = np.array(sorted(panel["ticker"].unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(tickers, rng.permutation(len(tickers)) % folds))
    src = PanelSource(deals, px, earnings)
    rows = []
    for f in range(folds):
        train = [t for t in tickers if fold_of[t] != f]
        test = [t for t in tickers if fold_of[t] == f]
        m = panel["ticker"].isin(train)
        cal = calibrate(panel[m], events[events["ticker"].isin(train)], feats.loc[train], verbose=False)
        ib, rb = implied_blocks(panel[m]), realized_blocks(panel[m])
        for o in origins:
            prem_I = float(ib.loc[ib["t"].between(o, o + 21), "R"].median())
            prem_R = float(rb.loc[rb["t"].between(o, o + horizon), "R"].mean())
            for tk in test:
                d = px[tk].loc[pd.Timestamp(deals.loc[tk, "ipo_date"]):]
                if len(d) <= o + horizon + 1:
                    continue
                asof = d.index[o]
                actual_iv = d["iv_atm"].iloc[o] if "iv_atm" in d else np.nan
                fut = np.log(d["close"]).diff().iloc[o + 1: o + 1 + horizon]
                real_rv = float(np.sqrt((fut ** 2).mean() * 252))
                spec = TargetSpec(name=tk, ticker=tk, use_quotes=False,
                                  expiries=[asof + pd.Timedelta(days=30)])
                try:
                    pt = price_target(spec, asof, src, cal, anchor=False)
                except Exception as e:           # missing peer data etc.
                    if verbose:
                        print(f"skip {tk} t={o}: {e}")
                    continue
                piv = pt.inputs.peer_term[30]
                rows.append(dict(
                    fold=f, ticker=tk, t0=o, asof=asof,
                    iv_actual=actual_iv, iv_model=float(pt.result.atm["atm_iv"].iloc[0]),
                    iv_B0=piv, iv_B1=piv * np.sqrt(prem_I),
                    rv_actual=real_rv, rv_model=pt.realized_vol_fcst_21d,
                    rv_B0=piv * np.sqrt(cal.peer_rv_iv), rv_B1=piv * np.sqrt(cal.peer_rv_iv * prem_R)))
        if verbose:
            print(f"fold {f + 1}/{folds} done ({len(test)} held-out IPOs)")
    res = pd.DataFrame(rows)
    return res, summarize(res)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for o, g in res.groupby("t0"):
        for target in ("iv", "rv"):
            a = g[f"{target}_actual"]
            ok = a.notna()
            for m in ("model", "B0", "B1"):
                fc = g.loc[ok, f"{target}_{m}"]
                err = fc - a[ok]
                r = dict(t0=o, target="implied 30d" if target == "iv" else "realised 21d", method=m,
                         n=int(ok.sum()), rmse_volpts=float(np.sqrt((err ** 2).mean()) * 100),
                         bias_volpts=float(err.mean() * 100),
                         mape=float((err.abs() / a[ok]).mean()))
                if target == "rv":
                    r["qlike"] = float(_qlike(a[ok], fc).mean())
                out.append(r)
    return pd.DataFrame(out)
