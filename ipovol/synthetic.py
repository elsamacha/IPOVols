"""Synthetic IPO universe with known parameters, in exactly the raw format the Bloomberg
pull produces. Used to prove the calibration recovers the truth before trusting real data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, HORIZON, CAL_DAYS
from .estimators import smile_vols

TRUTH = dict(
    # realised multiplier
    beta_R=1.15, theta0_R=np.log(0.8), tau_R=60.0, delta=0.75,
    theta=dict(float_pct=-0.25, first_day_ret=0.20, log_mcap=-0.15, vc_backed=0.10,
               tech=0.15, vix_at_listing=0.10, post2023=0.0),
    # implied multiplier (IPO implied / peer implied)
    beta_I=1.25, a_I_scale=1.2, tau_I=50.0,
    idio_level_sd=0.20,           # persistent IPO-specific level shock (what the Bayes update learns)
    # events, in local daily-variance units
    D_earn1=25.0, D_earn2=15.0, earn_w=0.7, D_quiet=3.0,
    lock_p=0.35, lock_g_sd=8.0, lock_gauss_D=4.0,
    # smile (normalised z units)
    peer_psi=-0.12, peer_gamma=0.06, dpsi_c=0.30, dpsi_tau=40.0, dpsi_d=-0.02, dgamma=0.02,
    # borrow
    fee_inf=0.004, fee_0=0.124, fee_tau=35.0,
    peer_vrp=1.08,
)


def _ohlc(v_day, jumps, rng, steps=78, s0=100.0):
    """Simulate OHLC from daily variances; 20% overnight, 80% intraday; jumps hit the open."""
    n = len(v_day)
    on = rng.standard_normal(n) * np.sqrt(0.2 * v_day) - 0.1 * v_day + jumps
    dv = 0.8 * v_day / steps
    intr = rng.standard_normal((n, steps)) * np.sqrt(dv)[:, None] - 0.5 * dv[:, None]
    path = np.cumsum(intr, axis=1)
    lc = np.empty(n)
    lo_, hi_, op_ = np.empty(n), np.empty(n), np.empty(n)
    prev = np.log(s0)
    for i in range(n):
        o = prev + on[i]
        p = o + path[i]
        op_[i], hi_[i], lo_[i], lc[i] = o, max(o, p.max()), min(o, p.min()), p[-1]
        prev = lc[i]
    return pd.DataFrame(dict(open=np.exp(op_), high=np.exp(hi_), low=np.exp(lo_), close=np.exp(lc)))


def generate(n_ipo=160, seed=7, horizon=HORIZON, truth=TRUTH):
    rng = np.random.default_rng(seed)
    T = truth
    raw = pd.DataFrame(dict(
        float_pct=rng.uniform(0.05, 0.30, n_ipo),
        first_day_ret=np.clip(rng.normal(0.20, 0.25, n_ipo), -0.3, 1.5),
        log_mcap=rng.normal(np.log(5e9), 1.0, n_ipo),
        vc_backed=rng.binomial(1, 0.6, n_ipo).astype(float),
        tech=rng.binomial(1, 0.5, n_ipo).astype(float),
        vix_at_listing=rng.uniform(12, 35, n_ipo),
        post2023=rng.binomial(1, 0.3, n_ipo).astype(float),
    ))
    Z = (raw - raw.mean()) / raw.std()
    theta = np.array([T["theta"][f] for f in FEATURES])
    a_R = np.exp(T["theta0_R"] + Z[FEATURES].to_numpy() @ theta)
    a_I = a_R * T["a_I_scale"]
    eps = np.exp(rng.normal(0, T["idio_level_sd"], n_ipo) - 0.5 * T["idio_level_sd"] ** 2)

    ipo_dates = pd.to_datetime(rng.choice(pd.bdate_range("2012-01-03", "2024-06-28"), n_ipo, replace=False))
    deals, px, earnings = [], {}, {}
    for i in range(n_ipo):
        tk, pk = f"IPO{i:03d}", f"PEER{i:03d}"
        dates = pd.bdate_range(ipo_dates[i], periods=horizon + 1)
        n = len(dates)
        t = np.arange(n)
        # peer single-name vol: log-OU around 45%, with a year of history before the IPO
        PRE = 260
        lv = np.empty(PRE + n)
        lv[0] = np.log(0.45) + rng.normal(0, 0.15)
        for j in range(1, PRE + n):
            lv[j] = lv[j - 1] + 0.02 * (np.log(0.45) - lv[j - 1]) + 0.04 * rng.standard_normal()
        peer_v_full = np.exp(lv) ** 2 / 252
        peer_v = peer_v_full[PRE:]
        # events
        e1 = int(rng.integers(30, 56))
        e_t = [e for e in (e1, e1 + 63, e1 + 126, e1 + 189) if e < n - 2]
        lk_t = int(dates.searchsorted(ipo_dates[i] + pd.Timedelta(days=180)))
        q_t = int(dates.searchsorted(ipo_dates[i] + pd.Timedelta(days=25)))
        k = np.searchsorted(np.array(e_t), t, side="left")
        M_R = T["beta_R"] * (1 + a_R[i] * np.exp(-t / T["tau_R"]) * T["delta"] ** k) * eps[i]
        M_I = T["beta_I"] * (1 + a_I[i] * np.exp(-t / T["tau_I"]) * T["delta"] ** k) * eps[i]
        v = peer_v * M_R
        jumps = np.zeros(n)
        for j, te in enumerate(e_t):
            D = T["D_earn1"] if j == 0 else T["D_earn2"]
            J = D * v[te]
            m = np.sqrt(T["earn_w"] * J)
            jumps[te] = rng.choice([-m, m]) + rng.normal(0, np.sqrt((1 - T["earn_w"]) * J))
        if lk_t < n:
            s = np.sqrt(v[lk_t])
            jumps[lk_t] = (-T["lock_g_sd"] * s if rng.random() < T["lock_p"] else 0.0) \
                + rng.normal(0, np.sqrt(T["lock_gauss_D"]) * s)
        if q_t < n:
            jumps[q_t] = rng.normal(0, np.sqrt(T["D_quiet"] * v[q_t]))
        f = _ohlc(v, jumps, rng)
        g = _ohlc(peer_v_full, np.zeros(PRE + n), rng)
        f.index = dates
        g.index = pd.bdate_range(end=dates[-1], periods=PRE + n)
        # implied vols
        peer_iv = np.sqrt(peer_v * 252) * T["peer_vrp"]
        iv2 = peer_iv ** 2 * M_I
        for j, te in enumerate([*e_t, lk_t, q_t]):
            if te >= n:
                continue
            D = (T["D_earn1"] if j == 0 else T["D_earn2"]) if j < len(e_t) else \
                (T["lock_p"] * (1 - T["lock_p"]) * T["lock_g_sd"] ** 2 + T["lock_gauss_D"] if te == lk_t else T["D_quiet"])
            dd = (dates[te] - dates).days
            inwin = (dd >= 0) & (dd <= 30)
            iv2 = iv2 + inwin * D * v[te] / (30 / CAL_DAYS)
        iv_atm = np.sqrt(iv2) * np.exp(rng.normal(0, 0.03, n))
        psi = T["peer_psi"] + T["dpsi_c"] * np.exp(-t / T["dpsi_tau"]) + T["dpsi_d"] + rng.normal(0, 0.02, n)
        gam = T["peer_gamma"] + T["dgamma"] + rng.normal(0, 0.01, n)
        f["iv_90"], f["iv_110"] = smile_vols(iv_atm, psi, gam)
        f["iv_atm"] = iv_atm
        f.loc[f.index[:3], ["iv_atm", "iv_90", "iv_110"]] = np.nan     # options list on day 3
        piv_full = np.sqrt(peer_v_full * 252) * T["peer_vrp"]
        g["iv_atm"] = piv_full
        g["iv_90"], g["iv_110"] = smile_vols(piv_full, T["peer_psi"], T["peer_gamma"])
        lr = (0.45 * T["peer_vrp"]) ** 2                      # term structure mean-reverts to 45%
        for col, days in (("iv_60", 60), ("iv_91", 91), ("iv_182", 182), ("iv_365", 365)):
            w = (1 - np.exp(-0.02 * days * 252 / 365)) / (0.02 * days * 252 / 365)
            g[col] = np.sqrt(w * piv_full ** 2 + (1 - w) * lr)
        f["borrow_fee"] = (T["fee_inf"] + (T["fee_0"] - T["fee_inf"]) * np.exp(-t / T["fee_tau"])
                           * np.exp(rng.normal(0, 0.3))) * np.exp(rng.normal(0, 0.05, n))
        px[tk], px[pk] = f, g
        earnings[tk] = [dates[e] for e in e_t]
        deals.append(dict(ticker=tk, ipo_date=dates[0], peers=[pk], lockup_date=pd.NaT,
                          **raw.iloc[i].to_dict()))
    deals = pd.DataFrame(deals).set_index("ticker")
    return deals, px, earnings, dict(a_R=a_R, eps=eps, Z=Z)
