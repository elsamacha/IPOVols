"""Calibrate every model input from the panel of past IPOs.

Components
----------
1. Multiplier curves (realised and implied), variance ratio of IPO to peer single-name average:
       M(t, k; x) = beta_inf * (1 + a(x) * exp(-t/tau) * delta**k),   log a(x) = theta0 + x'theta
   t = trading days since listing, k = earnings reports already out, x = standardised deal
   features. The excess decays with time *and* with information (Pastor-Veronesi learning).
   Fitted by QLIKE (a Bregman loss, consistent for the conditional mean of a noisy variance
   ratio) with a ridge prior on theta = partial pooling toward the average IPO.
2. Event jumps in "local diffusive days": jump variance = D * local daily variance.
   Earnings (1st vs later), quiet-period end, lockup. Earnings get a bimodality weight from
   the 4th moment; lockup gets a gap-down mixture from a 2-component EM.
3. Smile: excess (IPO - peer) skew psi and convexity gamma, decaying curve.
4. Borrow fee curve (if data is supplied).
5. Prior weight n0 for the Bayesian update of the level, chosen out-of-sample (K-fold) to
   minimise QLIKE of 21-day-ahead realised variance forecasts.
6. Cluster bootstrap by IPO for parameter uncertainty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit

from .config import FEATURES, BLOCK, FEATURE_CLIP


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


# =============================================================================
# Feature scaling
# =============================================================================
@dataclass
class Scaler:
    mean: dict
    std: dict

    @classmethod
    def fit(cls, feats: pd.DataFrame, cols=FEATURES):
        m = feats[cols].mean()
        s = feats[cols].std().replace(0, 1.0).fillna(1.0)
        return cls(m.to_dict(), s.to_dict())

    def transform(self, feats: pd.DataFrame, clip=None) -> pd.DataFrame:
        cols = list(self.mean)
        z = (feats[cols] - pd.Series(self.mean)) / pd.Series(self.std)
        z = z.fillna(0.0)
        return z.clip(-clip, clip) if clip else z


# =============================================================================
# Multiplier curve
# =============================================================================
@dataclass
class Curve:
    log_beta: float
    theta0: float
    theta: list
    log_tau: float
    logit_delta: float

    @property
    def beta(self):
        return float(np.exp(self.log_beta))

    @property
    def tau(self):
        return float(np.exp(self.log_tau))

    @property
    def delta(self):
        return float(_expit(self.logit_delta))

    def a(self, z):
        return np.exp(self.theta0 + np.asarray(z, float) @ np.asarray(self.theta))

    def M(self, t, k, z):
        """z: (p,) for one IPO or (n,p) row-aligned with t,k."""
        a = self.a(z)
        return self.beta * (1 + a * np.exp(-np.asarray(t, float) / self.tau) * self.delta ** np.asarray(k, float))


def realized_blocks(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[(panel["t"] >= 1) & ~panel["event_window"] & panel["var"].notna()
              & panel["peer_var_smooth"].notna()].copy()
    p["block"] = (p["t"] - 1) // BLOCK
    b = p.groupby(["ticker", "block"]).agg(num=("var", "mean"), den=("peer_var_smooth", "mean"),
                                            n=("var", "size"), t=("t", "mean"), k=("k", "min"))
    b = b[(b["n"] >= 3) & (b["num"] > 0) & (b["den"] > 0)].reset_index()
    b["R"] = b["num"] / b["den"]
    return b


def _event_D(jumps: dict, kind: str, seq: int) -> float:
    key = {"quiet": "D_quiet", "lockup": "D_lockup"}.get(kind, "D_earn1" if seq == 1 else "D_earn2")
    d = jumps.get(key, 0.0)
    return float(d) if np.isfinite(d) else 0.0


def implied_blocks(panel: pd.DataFrame, events: pd.DataFrame | None = None,
                   jumps: dict | None = None) -> pd.DataFrame:
    """IPO/peer implied-variance ratio by 5-day block, *diffusive* part only.

    With events + jumps, each day's 30d implied variance is stripped of the expected variance of
    IPO events inside its window (J = D x local var, spread over 30/365), so every day is usable,
    including the first ~18 days whose window always contains the quiet-period end. Without them,
    falls back to dropping days with an event ahead (which leaves days 1-18 unobserved and makes
    the curve extrapolate into them - the source of an early-days bias).
    """
    p = panel[(panel["t"] >= 1) & panel["iv_atm"].notna() & panel["peer_iv_atm"].notna()].copy()
    p["iv2"], p["piv2"] = p["iv_atm"] ** 2, p["peer_iv_atm"] ** 2
    if events is None or jumps is None or events.empty:
        p = p[~p["event_ahead_30"]]
    else:
        p = p.reset_index(drop=True)
        adj = np.zeros(len(p))
        bad = np.zeros(len(p), bool)
        dates = p["date"].to_numpy()
        rows_by_tk = p.groupby("ticker").indices
        for ev in events.itertuples():
            rows = rows_by_tk.get(ev.ticker)
            if rows is None:
                continue
            dd = (np.datetime64(ev.date) - dates[rows]).astype("timedelta64[D]").astype(int)
            sel = rows[(dd >= 1) & (dd <= 30)]          # event still ahead at today's close
            if np.isfinite(ev.local_var) and ev.local_var > 0:
                adj[sel] += _event_D(jumps, ev.kind, ev.seq) * ev.local_var * 365.0 / 30.0
            else:
                bad[sel] = True
        iv2 = p["iv2"].to_numpy() - adj
        keep = ~bad & (iv2 > 0.25 * p["iv2"].to_numpy())
        p["iv2"] = iv2
        p = p[keep]
    p["block"] = (p["t"] - 1) // BLOCK
    b = p.groupby(["ticker", "block"]).agg(num=("iv2", "mean"), den=("piv2", "mean"),
                                            n=("iv2", "size"), t=("t", "mean"), k=("k", "min"))
    b = b[(b["n"] >= 2) & (b["num"] > 0)].reset_index()
    b["R"] = b["num"] / b["den"]
    return b


def fit_curve(blocks: pd.DataFrame, Z: pd.DataFrame, ridge=2.0, delta_prior=0.8,
              delta_prior_w=2.0, x0: Curve | None = None) -> Curve:
    Zm = Z.loc[blocks["ticker"].str.split("#").str[0]].to_numpy()
    t, k, R, w = (blocks[c].to_numpy(float) for c in ("t", "k", "R", "n"))
    p = Zm.shape[1]

    def unpack(v):
        return Curve(v[0], v[1], list(v[2:2 + p]), v[2 + p], v[3 + p])

    def obj(v):
        a =np.exp(np.clip(v[1] + Zm @ v[2:2 + p], -10, 5))
        M = np.exp(v[0]) * (1 + a * np.exp(-t / np.exp(v[2 + p])) * _expit(v[3 + p]) ** k)
        loss = np.sum(w * (R / M + np.log(M)))
        pen = ridge * np.sum(v[2:2 + p] ** 2) + delta_prior_w * (v[3 + p] - _logit(delta_prior)) ** 2
        return loss + pen

    if x0 is None:
        late = blocks.loc[blocks["t"] > 150, "R"]
        early = blocks.loc[blocks["t"] < 20, "R"]
        b0 = float(np.median(late)) if len(late) > 10 else float(np.median(blocks["R"]))
        a0 = max(float(np.mean(early)) / b0 - 1, 0.1) if len(early) else 0.5
        v0 = np.r_[np.log(b0), np.log(a0), np.zeros(p), np.log(50.0), _logit(delta_prior)]
    else:
        v0 = np.r_[x0.log_beta, x0.theta0, x0.theta, x0.log_tau, x0.logit_delta]
    bounds = [(-3, 3), (-6, 3)] + [(-3, 3)] * p + [(np.log(2), np.log(1000)), (-5, 5)]
    res = minimize(obj, v0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=2000))
    return unpack(res.x)


# =============================================================================
# Event jumps
# =============================================================================
def _D(sub: pd.DataFrame) -> float:
    """Jump variance in units of local daily variance, ratio of sums (3-day window)."""
    if len(sub) < 3:
        return np.nan
    num = (sub["r"] ** 2 - 3 * sub["local_var"]).sum()
    return float(max(num / sub["local_var"].sum(), 0.0))


def _em_gap(x: np.ndarray, iters=300):
    """2-component Gaussian mixture, shared variance. Returns (p_low, gap, s2) or None."""
    x = x[np.isfinite(x)]
    if len(x) < 15:
        return None
    p, m1, m2, s2 = 0.3, np.quantile(x, 0.15), np.median(x), np.var(x)
    for _ in range(iters):
        l1 = p * np.exp(-(x - m1) ** 2 / (2 * s2))
        l2 = (1 - p) * np.exp(-(x - m2) ** 2 / (2 * s2))
        r = l1 / np.maximum(l1 + l2, 1e-300)
        p = np.clip(r.mean(), 1e-3, 1 - 1e-3)
        m1 = np.sum(r * x) / np.sum(r)
        m2 = np.sum((1 - r) * x) / np.sum(1 - r)
        s2 = max(np.sum(r * (x - m1) ** 2 + (1 - r) * (x - m2) ** 2) / len(x), 1e-6)
    if m1 > m2:
        m1, m2, p = m2, m1, 1 - p
    return float(p), float(m2 - m1), float(s2)


def fit_jumps(events: pd.DataFrame) -> dict:
    e = events[~events["contaminated"] & events["local_var"].notna() & (events["local_var"] > 0)]
    earn = e[e["kind"] == "earnings"]
    out = dict(
        D_earn1=_D(earn[earn["seq"] == 1]),
        D_earn2=_D(earn[earn["seq"] >= 2]),
        D_quiet=_D(e[e["kind"] == "quiet"]),
        D_lockup=_D(e[e["kind"] == "lockup"]),
        n_earn1=int((earn["seq"] == 1).sum()), n_lockup=int((e["kind"] == "lockup").sum()),
    )
    # --- earnings bimodality from the 4th moment of scaled window returns --------------
    Dp = _D(earn)
    # scale by the range-based local variance: ~5x less noisy than squared returns, and its
    # small level bias is irrelevant because kurtosis is scale-free
    scale = earn["local_var_gk"] if "local_var_gk" in earn else earn["local_var"]
    x = (earn["r"] / np.sqrt(scale)).to_numpy()
    w = 0.0
    if len(x) >= 20 and Dp > 0:
        u = Dp / (Dp + 3)
        kap = np.mean(x ** 4) / np.mean(x ** 2) ** 2
        kY = (kap - 6 * u * (1 - u) - 3 * (1 - u) ** 2) / u ** 2
        w = float(np.sqrt(np.clip((3 - kY) / 2, 0.0, 0.9)))
    out["earn_bimodal_w"] = w
    # --- lockup gap-down mixture (abnormal returns, local-sd units) --------------------
    lk = e[e["kind"] == "lockup"]
    xl = (lk["r_ab"] / np.sqrt(lk["local_var"])).to_numpy()
    out["lockup_mean_abn"] = float(np.nanmean(lk["r_ab"])) if len(lk) else np.nan
    gap = _em_gap(xl)
    if gap and gap[1] > 1.5 * np.sqrt(gap[2]) and 0.05 <= gap[0] <= 0.7:
        out.update(lockup_gap_p=gap[0], lockup_gap_g=gap[1], lockup_gauss_D=max(gap[2] - 3, 0.5))
    else:
        out.update(lockup_gap_p=0.0, lockup_gap_g=0.0, lockup_gauss_D=out["D_lockup"])
    return out


# =============================================================================
# Smile and borrow
# =============================================================================
def _decay(t, c, tau, d):
    return c * np.exp(-t / tau) + d


def _fit_decay(t, y, p0):
    try:
        (c, tau, d), _ = curve_fit(_decay, t, y, p0=p0, bounds=([-1, 5, -1], [1, 400, 1]), maxfev=20000)
        return dict(c=float(c), tau=float(tau), d=float(d))
    except Exception:
        return dict(c=0.0, tau=50.0, d=float(np.nanmedian(y)) if len(y) else 0.0)


def skew_blocks(panel):
    p = panel[(panel["t"] >= 1) & ~panel["event_ahead_30"]].copy()
    p["dpsi"], p["dgamma"] = p["psi"] - p["peer_psi"], p["gamma"] - p["peer_gamma"]
    p = p[np.isfinite(p["dpsi"]) & np.isfinite(p["dgamma"])]
    p["block"] = (p["t"] - 1) // BLOCK
    return p.groupby(["ticker", "block"]).agg(t=("t", "mean"), dpsi=("dpsi", "median"),
                                              dgamma=("dgamma", "median")).reset_index()


def fit_skew(sb: pd.DataFrame) -> dict:
    if len(sb) < 20:
        return dict(psi=dict(c=0.0, tau=50.0, d=0.0), gamma=dict(c=0.0, tau=50.0, d=0.0))
    return dict(psi=_fit_decay(sb["t"], sb["dpsi"], (0.1, 40, 0.0)),
                gamma=_fit_decay(sb["t"], sb["dgamma"], (0.0, 40, 0.0)))


def fit_borrow(panel) -> dict | None:
    p = panel[panel["borrow_fee"].notna() & (panel["t"] >= 1)].copy()
    if p["ticker"].nunique() < 10:
        return None
    p["block"] = (p["t"] - 1) // BLOCK
    b = p.groupby(["ticker", "block"]).agg(t=("t", "mean"), f=("borrow_fee", "median")).reset_index()
    fit = _fit_decay(b["t"], b["f"], (0.1, 30, 0.005))
    return dict(f0=fit["c"] + fit["d"], f_inf=fit["d"], tau=fit["tau"])


# =============================================================================
# Prior weight n0 (out-of-sample)
# =============================================================================
def calibrate_n0(panel, rb, Z, origins=(5, 10, 21, 42, 63), horizon=21, folds=5, seed=0,
                 grid=np.geomspace(1, 1000, 49)):
    p = panel[(panel["t"] >= 1) & ~panel["event_window"] & panel["var"].notna()
              & panel["peer_var_smooth"].notna()]
    tickers = np.array(sorted(p["ticker"].unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(tickers, rng.integers(0, folds, len(tickers))))
    recs = []  # (rho, n, prior_fcst, realised)
    for f in range(folds):
        train = [t for t in tickers if fold_of[t] != f]
        test = [t for t in tickers if fold_of[t] == f]
        curve = fit_curve(rb[rb["ticker"].isin(train)], Z)
        for tk in test:
            d = p[p["ticker"] == tk]
            z = Z.loc[tk].to_numpy()
            base = d["peer_var_smooth"].to_numpy() * curve.M(d["t"].to_numpy(), d["k"].to_numpy(), z)
            tt, v = d["t"].to_numpy(), d["var"].to_numpy()
            for o in origins:
                past, fut = tt <= o, (tt > o) & (tt <= o + horizon)
                if past.sum() < 3 or fut.sum() < 10:
                    continue
                rho = v[past].mean() / base[past].mean()
                recs.append((rho, past.sum(), base[fut].mean(), v[fut].mean()))
    r = np.array(recs)
    losses = []
    for n0 in grid:
        fc = r[:, 2] * (n0 + r[:, 1] * r[:, 0]) / (n0 + r[:, 1])
        q = r[:, 3] / fc
        losses.append(np.mean(q - np.log(q) - 1))
    losses = np.array(losses)
    return float(grid[np.argmin(losses)]), pd.DataFrame(dict(n0=grid, qlike=losses))


# =============================================================================
# Top level
# =============================================================================
@dataclass
class Calibration:
    curve_R: Curve
    curve_I: Curve
    jumps: dict
    skew: dict
    borrow: dict | None
    n0: float
    scaler: Scaler
    n_ipos: int = 0
    peer_rv_iv: float = 1.0          # peer realised var / peer implied var (peer VRP), close-to-close
    meta: dict = field(default_factory=dict)

    def z(self, features: dict, clip=FEATURE_CLIP):
        f = pd.DataFrame([features])
        return self.scaler.transform(f, clip=clip).iloc[0].to_numpy()

    def to_dict(self):
        d = asdict(self)
        return json.loads(json.dumps(d, default=float))

    @classmethod
    def from_dict(cls, d):
        return cls(curve_R=Curve(**d["curve_R"]), curve_I=Curve(**d["curve_I"]), jumps=d["jumps"],
                   skew=d["skew"], borrow=d["borrow"], n0=d["n0"], scaler=Scaler(**d["scaler"]),
                   n_ipos=d.get("n_ipos", 0), peer_rv_iv=d.get("peer_rv_iv", 1.0), meta=d.get("meta", {}))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def peer_vrp_ratio(panel) -> float:
    """Peer realised / implied variance over the panel (close-to-close, so unbiased in level)."""
    col = "peer_ret2" if "peer_ret2" in panel else "peer_ret"
    p = panel[panel[col].notna() & panel["peer_iv_atm"].notna()]
    if len(p) < 500:
        return 1.0
    r2 = p[col] if col == "peer_ret2" else p[col] ** 2
    return float(r2.mean() * 252 / (p["peer_iv_atm"] ** 2).mean())


def calibrate(panel, events, feats, n0: float | None = None, ridge=2.0, verbose=True) -> Calibration:
    scaler = Scaler.fit(feats)
    Z = scaler.transform(feats)
    jumps = fit_jumps(events)                       # first: implied blocks strip event variance
    rb, ib = realized_blocks(panel), implied_blocks(panel, events, jumps)
    cR = fit_curve(rb, Z, ridge=ridge)
    cI = fit_curve(ib, Z, ridge=ridge) if len(ib) > 50 else cR
    skew = fit_skew(skew_blocks(panel))
    borrow = fit_borrow(panel)
    n0_curve = None
    if n0 is None:
        n0, n0_curve = calibrate_n0(panel, rb, Z)
    cal = Calibration(cR, cI, jumps, skew, borrow, float(n0), scaler, n_ipos=int(feats.shape[0]),
                      peer_rv_iv=peer_vrp_ratio(panel))
    if n0_curve is not None:
        cal.meta["n0_curve"] = n0_curve.to_dict(orient="list")
    if verbose:
        print(summary(cal))
    return cal


def bootstrap(panel, events, feats, base: Calibration, B=200, seed=1, ridge=2.0) -> list[Calibration]:
    """Cluster bootstrap by IPO. Keeps base scaler, n0 and borrow; refits curves, jumps, skew."""
    rng = np.random.default_rng(seed)
    Z = base.scaler.transform(feats)
    rb, ib, sb = realized_blocks(panel), implied_blocks(panel, events, base.jumps), skew_blocks(panel)
    g_rb, g_ib, g_sb = (dict(tuple(x.groupby("ticker"))) for x in (rb, ib, sb))
    g_ev = dict(tuple(events.groupby("ticker")))
    tickers = np.array(sorted(g_rb))
    out = []
    for b in range(B):
        draw = rng.choice(tickers, len(tickers), replace=True)

        def stack(g):
            parts = []
            for j, tk in enumerate(draw):
                if tk in g:
                    x = g[tk].copy()
                    x["ticker"] = f"{tk}#{j}"
                    parts.append(x)
            return pd.concat(parts) if parts else pd.DataFrame()

        rbb, ibb, sbb, evb = stack(g_rb), stack(g_ib), stack(g_sb), stack(g_ev)
        cR = fit_curve(rbb, Z, ridge=ridge, x0=base.curve_R)
        cI = fit_curve(ibb, Z, ridge=ridge, x0=base.curve_I) if len(ibb) > 50 else cR
        out.append(Calibration(cR, cI, fit_jumps(evb), fit_skew(sbb), base.borrow, base.n0,
                               base.scaler, base.n_ipos, base.peer_rv_iv))
    return out


def summary(cal: Calibration) -> str:
    lines = [f"IPOs: {cal.n_ipos}"]
    for name, c in (("realised", cal.curve_R), ("implied", cal.curve_I)):
        th = ", ".join(f"{f}={v:+.2f}" for f, v in zip(cal.scaler.mean, c.theta))
        lines.append(f"{name:8s} M: beta_inf={c.beta:.3f}  a_avg={np.exp(c.theta0):.3f}  "
                     f"tau={c.tau:.1f}d  delta={c.delta:.3f}\n          theta: {th}")
    j = cal.jumps
    lines.append(f"jumps (local-days): earn1={j['D_earn1']:.1f}  earn2+={j['D_earn2']:.1f}  "
                 f"quiet={j['D_quiet']:.1f}  lockup={j['D_lockup']:.1f}  bimodal_w={j['earn_bimodal_w']:.2f}")
    lines.append(f"lockup gap: p={j['lockup_gap_p']:.2f}  g={j['lockup_gap_g']:.1f} sd  "
                 f"mean abn ret={j['lockup_mean_abn']:+.2%}")
    s = cal.skew
    lines.append(f"excess skew psi: {s['psi']['c']:+.3f}*exp(-t/{s['psi']['tau']:.0f}) {s['psi']['d']:+.3f};  "
                 f"gamma: {s['gamma']['c']:+.3f}*exp(-t/{s['gamma']['tau']:.0f}) {s['gamma']['d']:+.3f}")
    if cal.borrow:
        b = cal.borrow
        lines.append(f"borrow: f0={b['f0']:.2%} -> {b['f_inf']:.2%}, tau={b['tau']:.0f}d")
    lines.append(f"prior weight n0 = {cal.n0:.0f} trading days;  peer realised/implied var = {cal.peer_rv_iv:.3f}")
    return "\n".join(lines)
