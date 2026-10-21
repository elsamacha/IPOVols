"""Price the implied-vol surface of a newly listed stock.

Construction, for each future trading day u (t = trading days since listing):

    diffusive daily var   v_I(u) = v_peer(u) * M_I(t, k; x) * lambda_bayes * lambda_diff
    physical daily var    v_R(u) = v_peer(u) * M_R(t, k; x) * lambda_bayes
    event jump variance   J_e    = D_e * v_R(u_e) * event_premium * lambda_jump

v_peer comes from the live peer ATM term structure (average of peer single-name variances).
Earnings are bimodal (+/- m plus Gaussian), lockup is a gap-down mixture, the rest Gaussian.
Each discrete jump state shifts the forward (martingale-compensated), the Gaussian parts add to
a base variance, and the base distribution carries a smile sigma(z) = sigma_atm (1 + psi z + gamma z^2)
with psi, gamma = peer smile + calibrated IPO excess. Option prices are the probability-weighted
sum over jump states, inverted back to Black vols on the true forward.

lambda_bayes shrinks the IPO's observed realised/prior ratio with weight n0 (calibrated).
lambda_diff, lambda_jump, dpsi are fitted to listed quotes when supplied (market anchoring).
Parameter uncertainty: rerun with each bootstrap calibration -> percentile bands.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .black import black_call, black_put, implied_vol_otm, black_delta_call
from .calibrate import Calibration, _decay
from .config import TRADING_DAYS, CAL_DAYS, QUIET_PERIOD_CAL_DAYS, DEFAULT_LOCKUP_CAL_DAYS, FEATURE_CLIP


@dataclass
class RealizedObs:
    """Observed history since listing (excluding listing day and event windows)."""
    n_days: int
    ipo_var_mean: float      # mean daily variance of the IPO (same estimator as calibration)
    peer_var_mean: float     # mean daily variance of the peers' single names, same days


@dataclass
class IPOInputs:
    valuation_date: str | pd.Timestamp
    listing_date: str | pd.Timestamp
    spot: float
    features: dict
    peer_term: dict                          # {calendar days: vol} peer-average ATM
    peer_psi: float = -0.10
    peer_gamma: float = 0.05
    rate: float = 0.04
    div_yield: float = 0.0
    earnings_dates: list = field(default_factory=list)   # future reaction dates
    k0: int = 0                              # earnings reports already out
    lockup_date: str | pd.Timestamp | None = None
    quiet_date: str | pd.Timestamp | None = None
    extra_events: list = field(default_factory=list)     # [{"date":..., "D":..., "name":...}]
    borrow_now: float | None = None          # current annual borrow fee, decays with calibrated tau
    realized: RealizedObs | None = None
    quotes: list = field(default_factory=list)            # [(expiry, strike, iv)]
    event_premium: float = 1.0               # multiplies event variance (implied > realised moves)


@dataclass
class SurfaceResult:
    expiries: list
    strikes: np.ndarray
    forwards: np.ndarray
    iv: np.ndarray                    # (n_exp, n_strike) point estimate
    iv_q: dict                        # quantile -> array
    atm: pd.DataFrame
    decomposition: pd.DataFrame
    info: dict

    def table(self) -> pd.DataFrame:
        rows = []
        for i, e in enumerate(self.expiries):
            for j, K in enumerate(self.strikes):
                r = dict(expiry=e, strike=K, moneyness_fwd=K / self.forwards[i], iv=self.iv[i, j])
                for q, a in self.iv_q.items():
                    r[f"iv_p{int(q * 100)}"] = a[i, j]
                rows.append(r)
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
def _peer_fwd_var(peer_term: dict, cal_off: np.ndarray, val: pd.Timestamp) -> np.ndarray:
    pts = np.array(sorted(peer_term), float)
    tv = np.array([peer_term[p] ** 2 * p / CAL_DAYS for p in sorted(peer_term)])
    pts0, tv0 = np.r_[0.0, pts], np.r_[0.0, tv]
    j = np.searchsorted(pts, cal_off, side="left")          # interval index (0..len)
    v = np.empty(len(cal_off))
    last_rate = None
    for i in range(len(pts)):
        m = j == i
        dv = tv0[i + 1] - tv0[i]
        if dv <= 0:
            dv = 0.2 * tv0[i + 1] * (pts0[i + 1] - pts0[i]) / pts0[i + 1]   # floor on inverted input
        # trading days in (val+pts0[i], val+pts0[i+1]] counted on the calendar, not the grid,
        # so a grid that stops mid-interval still gets the right per-day variance
        n = np.busday_count((val + pd.Timedelta(days=pts0[i] + 1)).date(),
                            (val + pd.Timedelta(days=pts0[i + 1] + 1)).date())
        v[m] = dv / max(n, 1)
        # per-trading-day rate for extrapolation
        last_rate = dv / max((pts0[i + 1] - pts0[i]) * TRADING_DAYS / CAL_DAYS, 1)
    v[j == len(pts)] = last_rate
    return v


def _discrete(kind, J, vR, jp, cal):
    """Return (values, probs, gaussian_var) for one event."""
    if kind == "earnings":
        w = cal.jumps.get("earn_bimodal_w", 0.0)
        m = np.sqrt(w * J)
        return (np.array([m, -m]), np.array([0.5, 0.5]), (1 - w) * J) if m > 1e-8 else (None, None, J)
    if kind == "lockup" and cal.jumps.get("lockup_gap_p", 0) > 0:
        p, g = cal.jumps["lockup_gap_p"], cal.jumps["lockup_gap_g"]
        s = np.sqrt(jp * vR)                        # local daily sd incl. scalings
        return np.array([-g * s, 0.0]), np.array([p, 1 - p]), cal.jumps["lockup_gauss_D"] * s * s
    return None, None, J


def svi_from_smile(Vb, psi, gam, kappa=1.5, max_wing=1.9):
    """Map the (psi, gamma) smile onto an SVI slice w(k) = a + b(rho k + sqrt(k^2 + s^2)).

    Matches total variance, slope and curvature at the money of
    sigma(z) = sigma_atm (1 + psi z + gamma z^2), z = k / sqrt(Vb), while forcing the wings to
    be linear in k with slopes b(1 +/- rho) < 2 (Roger Lee bound) - so no wing arbitrage however
    steep the calibrated skew is. kappa sets the curvature length scale in ATM standard deviations.
    """
    sq = np.sqrt(Vb)
    slope = 2 * psi * sq                     # dw/dk at k=0
    curv = 2 * (psi ** 2 + 2 * max(gam, 0.0))  # d2w/dk2 at k=0
    s = kappa * sq
    b = max(curv * s, abs(slope) / 0.95, 1e-6)
    b = min(b, max_wing / (1 + min(abs(slope) / b, 0.95)))
    rho = float(np.clip(slope / b, -0.95, 0.95))
    a = Vb - b * s
    wmin = a + b * s * np.sqrt(1 - rho ** 2)
    if wmin < 0.05 * Vb:                     # keep the minimum variance positive
        s_new = (Vb - 0.05 * Vb) / (b * (1 - np.sqrt(1 - rho ** 2)) + 1e-12)
        s = min(s, s_new)
        a = Vb - b * s
    return a, b, rho, s


class IPOPricer:
    def __init__(self, cal: Calibration, draws: list[Calibration] | None = None):
        self.cal, self.draws = cal, draws or []

    # ---------------------------------------------------------------------
    def _paths(self, cal: Calibration, inp: IPOInputs, horizon_end, lam_d=1.0):
        """Daily variance paths (diffusive implied v_I, physical v_R) and the event list."""
        val = pd.Timestamp(inp.valuation_date)
        lst = pd.Timestamp(inp.listing_date)
        days = pd.bdate_range(val + pd.offsets.BDay(1), pd.Timestamp(horizon_end))
        cal_off = (days - val).days.to_numpy().astype(float)
        t0 = int(np.busday_count(lst.date(), val.date()))          # negative before listing
        t_u = t0 + np.arange(1, len(days) + 1)

        e_dates = sorted(pd.Timestamp(d) for d in inp.earnings_dates if pd.Timestamp(d) > val)
        k_u = inp.k0 + np.searchsorted(np.array(e_dates, dtype="datetime64[ns]"),
                                       days.to_numpy(), side="left") if e_dates else np.full(len(days), inp.k0)
        z = cal.z(inp.features)
        vp = _peer_fwd_var(inp.peer_term, cal_off, val)
        tt = np.maximum(t_u, 0)
        MI = cal.curve_I.M(tt, k_u, z)
        MR = cal.curve_R.M(tt, k_u, z)

        lam_b, rho = 1.0, np.nan
        if inp.realized and inp.realized.n_days > 0 and t0 > 0:
            t_obs = np.arange(1, t0 + 1)
            m_obs = cal.curve_R.M(t_obs, np.full(t0, inp.k0), z).mean()
            rho = (inp.realized.ipo_var_mean / inp.realized.peer_var_mean) / m_obs
            n = inp.realized.n_days
            lam_b = (cal.n0 + n * rho) / (cal.n0 + n)

        vI = vp * MI * lam_b * lam_d
        # physical: peer implied variance -> peer realised (peer VRP), then IPO multiplier
        vR = vp * cal.peer_rv_iv * MR * lam_b

        def u_of(d):
            i = days.searchsorted(pd.Timestamp(d))
            return int(i) if i < len(days) else None

        evs = []
        for i, d in enumerate(e_dates):
            u = u_of(d)
            if u is not None:
                D = cal.jumps["D_earn1"] if inp.k0 + i == 0 else cal.jumps["D_earn2"]
                evs.append(("earnings", f"earnings #{inp.k0 + i + 1}", u, D))
        lk = pd.Timestamp(inp.lockup_date) if inp.lockup_date else lst + pd.Timedelta(days=DEFAULT_LOCKUP_CAL_DAYS)
        if lk > val and u_of(lk) is not None:
            evs.append(("lockup", "lockup expiry", u_of(lk), cal.jumps["D_lockup"]))
        qd = pd.Timestamp(inp.quiet_date) if inp.quiet_date else lst + pd.Timedelta(days=QUIET_PERIOD_CAL_DAYS)
        if qd > val and u_of(qd) is not None:
            evs.append(("quiet", "quiet-period end", u_of(qd), cal.jumps["D_quiet"]))
        for x in inp.extra_events:
            if pd.Timestamp(x["date"]) > val and u_of(x["date"]) is not None:
                evs.append(("custom", x.get("name", "custom"), u_of(x["date"]), float(x["D"])))
        return dict(val=val, lst=lst, days=days, t0=t0, t_u=t_u, vI=vI, vR=vR, evs=evs, z=z,
                    MI=MI, MR=MR, lam_b=lam_b, rho=rho)

    def expected_realized(self, inp: IPOInputs, horizon_days=21, cal: Calibration | None = None):
        """Physical-measure forecast of annualised close-to-close realised vol over the next
        `horizon_days` trading days, including scheduled event jumps (no event premium)."""
        cal = cal or self.cal
        end = pd.Timestamp(inp.valuation_date) + pd.offsets.BDay(horizon_days)
        P = self._paths(cal, inp, end)
        n = min(horizon_days, len(P["days"]))
        var = P["vR"][:n].sum() + sum(D * P["vR"][u] for _, _, u, D in P["evs"] if u < n)
        return float(np.sqrt(var / n * TRADING_DAYS))

    def _core(self, cal: Calibration, inp: IPOInputs, expiries, strikes, lam_d=1.0, lam_j=1.0,
              dpsi=0.0, detail=False):
        exp = [pd.Timestamp(e) for e in expiries]
        P = self._paths(cal, inp, max(exp), lam_d)
        val, days, t0, t_u, vI, vR, evs = (P[k] for k in ("val", "days", "t0", "t_u", "vI", "vR", "evs"))
        lam_b, rho, z, MI, MR = P["lam_b"], P["rho"], P["z"], P["MI"], P["MR"]
        jp = inp.event_premium * lam_j

        # ---- smile & carry ---------------------------------------------------
        sk = cal.skew
        psi = inp.peer_psi + _decay(t0, **sk["psi"]) + dpsi
        gam = max(inp.peer_gamma + _decay(t0, **sk["gamma"]), 0.0)
        if inp.borrow_now is not None and cal.borrow:
            fee = cal.borrow["f_inf"] + (inp.borrow_now - cal.borrow["f_inf"]) * np.exp(-(t_u - t0) / cal.borrow["tau"])
        elif inp.borrow_now is not None:
            fee = np.full(len(days), inp.borrow_now)
        elif cal.borrow:
            fee = cal.borrow["f_inf"] + (cal.borrow["f0"] - cal.borrow["f_inf"]) * np.exp(-t_u / cal.borrow["tau"])
        else:
            fee = np.zeros(len(days))

        strikes = np.asarray(strikes, float)
        iv = np.full((len(exp), strikes.shape[-1]), np.nan)
        fwd, rows = np.empty(len(exp)), []
        for i, e in enumerate(exp):
            n = int(days.searchsorted(e, side="right"))
            T = max((e - val).days, 1) / CAL_DAYS
            F = inp.spot * np.exp((inp.rate - inp.div_yield) * T - fee[:n].sum() / TRADING_DAYS)
            fwd[i] = F
            Vd = vI[:n].sum()
            shifts, probs, Vg, contrib = np.zeros(1), np.ones(1), 0.0, {}
            for kind, name, u, D in evs:
                if u >= n:
                    continue
                J = D * vR[u] * jp
                vals, pr, g = _discrete(kind, J, vR[u], jp, cal)
                Vg += g
                contrib[name] = J if vals is None else g + float(np.sum(pr * vals ** 2) - np.sum(pr * vals) ** 2)
                if vals is not None:
                    comp = -np.log(np.sum(pr * np.exp(vals)))
                    shifts = (shifts[:, None] + vals[None, :] + comp).ravel()
                    probs = (probs[:, None] * pr[None, :]).ravel()
            Vb = Vd + Vg
            Ks = strikes[i] if strikes.ndim == 2 else strikes
            Fs = F * np.exp(shifts)[:, None]
            a_, b_, r_, s_ = svi_from_smile(Vb, psi, gam)
            kk = np.log(Ks[None, :] / Fs)
            sig = np.sqrt(np.maximum(a_ + b_ * (r_ * kk + np.sqrt(kk * kk + s_ * s_)), 1e-8) / T)
            C = (probs[:, None] * black_call(Fs, Ks[None, :], sig, T)).sum(0)
            P = (probs[:, None] * black_put(Fs, Ks[None, :], sig, T)).sum(0)
            otm = np.where(Ks >= F, C, P)
            iv[i] = implied_vol_otm(otm, F, Ks, T)
            if detail:
                rows.append(dict(expiry=e.date(), cal_days=(e - val).days, forward=F,
                                 diffusive_vol=np.sqrt(Vd / T), total_vol_atm_approx=np.sqrt((Vd + sum(contrib.values())) / T),
                                 diffusive_var_share=Vd / (Vd + sum(contrib.values())),
                                 **{f"jump_{k}_move": np.sqrt(v) for k, v in contrib.items()}))
        info = dict(t0=t0, lambda_bayes=lam_b, rho_obs=rho, psi=psi, gamma=gam, z=z,
                    M_I_now=float(MI[0]), M_R_now=float(MR[0]))
        return iv, fwd, (pd.DataFrame(rows) if detail else None), info

    # ---------------------------------------------------------------------
    def _anchor(self, cal, inp):
        if not inp.quotes:
            return 1.0, 1.0, 0.0
        q = pd.DataFrame(inp.quotes, columns=["expiry", "strike", "iv"])
        q["expiry"] = pd.to_datetime(q["expiry"])
        exps = sorted(q["expiry"].unique())

        def resid(x):
            ld, lj, dp = np.exp(x[0]), np.exp(x[1]), x[2]
            out = []
            for e in exps:
                qq = q[q["expiry"] == e]
                iv, _, _, _ = self._core(cal, inp, [e], qq["strike"].to_numpy(), ld, lj, dp)
                out.append(np.nan_to_num(iv[0] - qq["iv"].to_numpy(), nan=1.0))
            # weak prior keeps unidentified directions at the model value
            return np.r_[np.concatenate(out), 0.02 * x[0], 0.02 * x[1], 0.02 * x[2]]

        r = least_squares(resid, np.zeros(3), bounds=([np.log(0.3)] * 2 + [-1.0], [np.log(3.0)] * 2 + [1.0]))
        return float(np.exp(r.x[0])), float(np.exp(r.x[1])), float(r.x[2])

    def price(self, inp: IPOInputs, expiries, moneyness=np.linspace(0.5, 1.6, 45),
              quantiles=(0.1, 0.25, 0.75, 0.9)) -> SurfaceResult:
        # strikes on a forward-moneyness grid per expiry, using the point-estimate forwards
        ld, lj, dp = self._anchor(self.cal, inp)
        _, fwd, _, _ = self._core(self.cal, inp, expiries, np.array([inp.spot]), ld, lj, dp)
        K = fwd[:, None] * np.asarray(moneyness)[None, :]
        iv, fwd, dec, info = self._core(self.cal, inp, expiries, K, ld, lj, dp, detail=True)
        info.update(lambda_diff=ld, lambda_jump=lj, dpsi_anchor=dp)
        raw = self.cal.scaler.transform(pd.DataFrame([inp.features])).iloc[0]
        info["clipped_features"] = {k: round(float(v), 2) for k, v in raw.items() if abs(v) > FEATURE_CLIP}

        draws_iv = []
        for c in self.draws:
            a = self._anchor(c, inp)
            draws_iv.append(self._core(c, inp, expiries, K, *a)[0])
        iv_q = {}
        if draws_iv:
            st = np.stack(draws_iv)
            iv_q = {q: np.nanquantile(st, q, axis=0) for q in quantiles}

        atm = self._atm_table(inp, expiries, fwd, K, iv, iv_q, moneyness)
        return SurfaceResult(list(expiries), np.asarray(moneyness), fwd, iv, iv_q, atm, dec, info)

    @staticmethod
    def _atm_table(inp, expiries, fwd, K, iv, iv_q, mny):
        val = pd.Timestamp(inp.valuation_date)
        rows = []
        for i, e in enumerate(expiries):
            T = max((pd.Timestamp(e) - val).days, 1) / CAL_DAYS
            atm = np.interp(1.0, mny, iv[i])
            dl = black_delta_call(fwd[i], K[i], iv[i], T)
            order = np.argsort(dl)
            c25 = np.interp(0.25, dl[order], iv[i][order])
            p25 = np.interp(0.75, dl[order], iv[i][order])
            r = dict(expiry=pd.Timestamp(e).date(), cal_days=(pd.Timestamp(e) - val).days, forward=fwd[i],
                     atm_iv=atm, rr25=c25 - p25, fly25=0.5 * (c25 + p25) - atm)
            for q, a in iv_q.items():
                r[f"atm_p{int(q * 100)}"] = np.interp(1.0, mny, a[i])
            rows.append(r)
        return pd.DataFrame(rows)


def realized_from_ohlc(ipo: pd.DataFrame, peers: list[pd.DataFrame], listing_date, event_dates=()):
    """Helper: build RealizedObs from daily OHLC of the IPO and its peers."""
    from .estimators import daily_variance
    d = ipo.loc[pd.Timestamp(listing_date):].copy()
    v = daily_variance(d).iloc[1:]
    pv = pd.concat([daily_variance(p).reindex(v.index) for p in peers], axis=1).mean(axis=1)
    mask = pd.Series(True, index=v.index)
    for ed in event_dates:
        mask &= ~((v.index >= pd.Timestamp(ed) - pd.offsets.BDay(1)) & (v.index <= pd.Timestamp(ed) + pd.offsets.BDay(1)))
    m = mask & v.notna() & pv.notna()
    return RealizedObs(int(m.sum()), float(v[m].mean()), float(pv[m].mean()))
