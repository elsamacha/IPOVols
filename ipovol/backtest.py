"""Leave-one-fold-out backtest: does the model generalise to IPOs it has not seen?

For each fold, calibrate on the other IPOs, then price every held-out IPO with the same
`price_target` path used live, point in time at t0 trading days after listing, with no
market anchoring (the model's prior is what is being tested). Two targets:

* implied: the held-out IPO's actual 30d ATM implied vol at t0 (what the market charged)
* realised: its close-to-close realised vol over the next 21 trading days

Benchmarks, all fitted on the training folds only:
* B0 "peer vol":       peer 30d ATM implied (for realised: scaled to the peer realised level)
* B1 "peer x premium": B0 x the average IPO premium at that age (a flat multiplier)
* LR "regression":     log(IPO var / peer var) on deal features + an event-in-window dummy
The full model has to beat LR to justify its structure for the ATM point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .api import TargetSpec, price_target
from .calibrate import calibrate
from .panel import build_panel
from .sources import PanelSource


def _qlike(real_vol, fc_vol):
    q = (real_vol / fc_vol) ** 2
    return q - np.log(q) - 1


def _observations(deals, px, earnings, src, tickers, origins, horizon):
    """Point-in-time facts for every IPO at every origin (identical for all folds)."""
    from .config import FEATURES
    obs = []
    for tk in tickers:
        ipo = pd.Timestamp(deals.loc[tk, "ipo_date"])
        d = px[tk].loc[ipo:]
        ev = list(earnings.get(tk, [])) + [ipo + pd.Timedelta(days=180), ipo + pd.Timedelta(days=25)]
        for o in origins:
            if len(d) <= o + horizon + 1:
                continue
            asof = d.index[o]
            try:
                piv = src.peer_term(src.suggest_peers(tk), asof)[0][30]
            except Exception:
                continue
            fut = np.log(d["close"]).diff().iloc[o + 1: o + 1 + horizon]
            obs.append(dict(ticker=tk, t0=o, asof=asof, piv=piv,
                            iv_actual=d["iv_atm"].iloc[o] if "iv_atm" in d else np.nan,
                            rv_actual=float(np.sqrt((fut ** 2).mean() * 252)),
                            ev30=float(any(asof < e <= asof + pd.Timedelta(days=30) for e in ev))))
    return pd.DataFrame(obs)


def _regression(train: pd.DataFrame, test: pd.DataFrame, Z: pd.DataFrame, target: str) -> np.ndarray:
    """Benchmark: log(IPO var / peer implied var) ~ deal features + event-in-window dummy.
    Returns vol forecasts, lognormal-mean corrected so the implied variance is unbiased."""
    ok = train[f"{target}_actual"].notna()
    tr = train[ok]
    X = np.c_[np.ones(len(tr)), Z.loc[tr["ticker"]].to_numpy(), tr["ev30"]]
    y = np.log(tr[f"{target}_actual"] ** 2 / tr["piv"] ** 2)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = float(np.var(y - X @ beta))
    Xt = np.c_[np.ones(len(test)), Z.loc[test["ticker"]].to_numpy(), test["ev30"]]
    return test["piv"].to_numpy() * np.sqrt(np.exp(Xt @ beta + s2 / 2))


def backtest(deals, px, earnings, origins=(5, 21), folds=5, horizon=21, seed=0, verbose=True):
    """Methods: model (full), LR (regression benchmark), B1 (peer x average premium at that age),
    B0 (raw peer vol). All benchmarks are fitted on the training folds only."""
    from .config import FEATURES
    panel, events, feats = build_panel(deals, px, earnings)
    tickers = np.array(sorted(panel["ticker"].unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(tickers, rng.permutation(len(tickers)) % folds))
    src = PanelSource(deals, px, earnings)
    obs = _observations(deals, px, earnings, src, tickers, origins, horizon)
    obs["fold"] = obs["ticker"].map(fold_of)
    Z = ((feats[FEATURES] - feats[FEATURES].mean()) / feats[FEATURES].std().replace(0, 1)).fillna(0)
    out = []
    for f in range(folds):
        train = [t for t in tickers if fold_of[t] != f]
        m = panel["ticker"].isin(train)
        cal = calibrate(panel[m], events[events["ticker"].isin(train)], feats.loc[train], verbose=False)
        for o in origins:
            tr = obs[(obs["fold"] != f) & (obs["t0"] == o)]
            te = obs[(obs["fold"] == f) & (obs["t0"] == o)].copy()
            if te.empty:
                continue
            te["iv_B0"] = te["piv"]
            te["rv_B0"] = te["piv"] * np.sqrt(cal.peer_rv_iv)
            te["iv_B1"] = te["piv"] * np.sqrt(np.nanmedian((tr["iv_actual"] / tr["piv"]) ** 2))
            te["rv_B1"] = te["piv"] * np.sqrt(np.mean((tr["rv_actual"] / tr["piv"]) ** 2))
            te["iv_LR"] = _regression(tr, te, Z, "iv")
            te["rv_LR"] = _regression(tr, te, Z, "rv")
            ivm, rvm = [], []
            for r in te.itertuples():
                spec = TargetSpec(name=r.ticker, ticker=r.ticker, use_quotes=False,
                                  expiries=[r.asof + pd.Timedelta(days=30)])
                try:
                    pt = price_target(spec, r.asof, src, cal, anchor=False)
                    ivm.append(float(pt.result.atm["atm_iv"].iloc[0]))
                    rvm.append(pt.realized_vol_fcst_21d)
                except Exception as e:
                    if verbose:
                        print(f"skip {r.ticker} t={o}: {e}")
                    ivm.append(np.nan)
                    rvm.append(np.nan)
            te["iv_model"], te["rv_model"] = ivm, rvm
            out.append(te)
        if verbose:
            print(f"fold {f + 1}/{folds} done")
    res = pd.concat(out, ignore_index=True)
    return res, summarize(res)


METHODS = ("model", "LR", "B1", "B0")


def summarize(res: pd.DataFrame, methods=METHODS) -> pd.DataFrame:
    out = []
    for o, g in res.groupby("t0"):
        for target in ("iv", "rv"):
            a = g[f"{target}_actual"]
            for m in methods:
                if f"{target}_{m}" not in g:
                    continue
                ok = a.notna() & g[f"{target}_{m}"].notna()
                fc, aa = g.loc[ok, f"{target}_{m}"], a[ok]
                err = fc - aa
                r = dict(t0=o, target="implied 30d" if target == "iv" else "realised 21d", method=m,
                         n=int(ok.sum()),
                         R2=float(1 - (err ** 2).sum() / ((aa - aa.mean()) ** 2).sum()),
                         rmse_volpts=float(np.sqrt((err ** 2).mean()) * 100),
                         bias_volpts=float(err.mean() * 100),
                         mape=float((err.abs() / aa).mean()),
                         within10=float((err.abs() / aa <= 0.10).mean()))
                if target == "rv":
                    r["qlike"] = float(_qlike(aa, fc).mean())
                out.append(r)
    return pd.DataFrame(out)
