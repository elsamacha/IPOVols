import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ipovol.black import black_call, black_put, black_otm, implied_vol_otm
from ipovol.estimators import daily_variance, smile_params, smile_vols
from ipovol.synthetic import generate, TRUTH, _ohlc
from ipovol.panel import build_panel
from ipovol.calibrate import calibrate
from ipovol.pricer import IPOPricer, IPOInputs


def test_black_parity_and_inversion():
    F, T = 100.0, 0.5
    K = np.linspace(40, 200, 33)
    sig = 0.2 + 0.3 * np.abs(np.log(K / F))
    C, P = black_call(F, K, sig, T), black_put(F, K, sig, T)
    assert np.allclose(C - P, F - K, atol=1e-10)
    iv = implied_vol_otm(black_otm(F, K, sig, T), F, K, T)
    assert np.allclose(iv, sig, atol=1e-6)


def test_smile_params_roundtrip():
    lo, hi = smile_vols(0.6, 0.15, 0.04)
    psi, gam = smile_params(lo, 0.6, hi)
    assert abs(psi - 0.15) < 1e-10 and abs(gam - 0.04) < 1e-10


def test_daily_variance_unbiased():
    rng = np.random.default_rng(0)
    v = np.full(20000, 0.03 ** 2)
    f = _ohlc(v, np.zeros_like(v), rng, steps=390)
    est = daily_variance(f).iloc[1:].mean()
    assert -0.10 < est / v[0] - 1 < 0.0   # discrete-monitoring bias on H/L; cancels in IPO/peer ratios


@pytest.fixture(scope="module")
def calibrated():
    deals, px, earn, _ = generate(n_ipo=120, seed=11)
    panel, events, feats = build_panel(deals, px, earn)
    return calibrate(panel, events, feats, verbose=False)


def test_parameter_recovery(calibrated):
    c = calibrated
    assert abs(c.curve_R.beta / TRUTH["beta_R"] - 1) < 0.10
    assert abs(c.curve_R.tau / TRUTH["tau_R"] - 1) < 0.40
    assert abs(np.exp(c.curve_R.theta0) / np.exp(TRUTH["theta0_R"]) - 1) < 0.35
    assert abs(c.curve_I.beta / TRUTH["beta_I"] - 1) < 0.10
    assert c.curve_R.theta[0] < 0            # bigger float -> smaller IPO premium
    assert abs(c.jumps["D_earn1"] / TRUTH["D_earn1"] - 1) < 0.35
    assert abs(c.skew["psi"]["c"] - TRUTH["dpsi_c"]) < 0.08


def _inputs(**kw):
    base = dict(valuation_date="2027-01-19", listing_date="2027-01-11", spot=100.0,
                features=dict(float_pct=0.1, first_day_ret=0.3, log_mcap=np.log(5e10), vc_backed=1,
                              tech=1, vix_at_listing=18, post2023=1),
                peer_term={30: 0.55, 60: 0.54, 91: 0.53, 182: 0.51, 365: 0.50},
                earnings_dates=["2027-03-02", "2027-06-01"])
    base.update(kw)
    return IPOInputs(**base)


EXPS = pd.to_datetime(["2027-02-19", "2027-03-19", "2027-06-18", "2027-09-17", "2028-01-21"])


def test_no_arbitrage(calibrated):
    res = IPOPricer(calibrated).price(_inputs(), EXPS, moneyness=np.linspace(0.4, 2.0, 81))
    val = pd.Timestamp("2027-01-19")
    for i, e in enumerate(EXPS):
        T = (e - val).days / 365
        K = res.forwards[i] * res.strikes
        C = black_call(res.forwards[i], K, res.iv[i], T)
        assert np.all(np.diff(C) <= 1e-9), "calls must decrease in strike"
        assert np.all(np.diff(C, 2) >= -1e-7), "calls must be convex in strike"
    tv = res.atm["atm_iv"] ** 2 * res.atm["cal_days"]
    assert np.all(np.diff(tv) > 0), "ATM total variance must increase with maturity"


def test_market_anchoring_hits_quotes(calibrated):
    q = [("2027-02-19", 100.0, 0.70), ("2027-03-19", 100.0, 0.80)]
    res = IPOPricer(calibrated).price(_inputs(quotes=q), EXPS[:2], moneyness=np.linspace(0.7, 1.3, 61))
    for i, (_, _, iv) in enumerate(q):
        K = 100.0 / res.forwards[i]
        assert abs(np.interp(K, res.strikes, res.iv[i]) - iv) < 0.01


# ---- generic target API ---------------------------------------------------------------
from ipovol.api import price_target, standard_expiries
from ipovol.sources import PanelSource
from ipovol.backtest import backtest


@pytest.fixture(scope="module")
def universe():
    deals, px, earn, _ = generate(n_ipo=60, seed=21)
    return deals, px, earn


def test_price_any_listed_name(universe, calibrated):
    deals, px, earn = universe
    src = PanelSource(deals, px, earn)
    for tk, o in [("IPO001", 5), ("IPO030", 60), ("IPO055", 200)]:
        asof = px[tk].index[o]
        pt = price_target(tk, asof, src, calibrated)
        assert pt.listed and pt.result.info["t0"] == o
        assert np.isfinite(pt.result.atm["atm_iv"]).all()
        assert 0.1 < pt.realized_vol_fcst_21d < 3.0
        # anchored to the 30d ATM quote the panel source supplies
        T30 = asof + pd.Timedelta(days=30)
        spec = dict(name=tk, ticker=tk, expiries=[str(T30.date()), str((T30 + pd.Timedelta(days=60)).date())])
        pt30 = price_target(spec, asof, src, calibrated)
        assert abs(pt30.result.atm["atm_iv"].iloc[0] - px[tk]["iv_atm"].iloc[o]) < 0.01


def test_price_prelisting_profile(universe, calibrated):
    deals, px, earn = universe
    date = pd.Timestamp("2019-03-01")
    peers = [p for p in px if p.startswith("PEER") and px[p].index[0] < date - pd.Timedelta(days=30)
             and px[p].index[-1] > date][:3]
    assert peers, "need peers covering the date"
    spec = dict(name="pre", listing_date="2019-03-11", offer_price=40.0, valuation_usd=6e10,
                float_pct=0.08, expected_first_day_ret=0.3, vc_backed=1, tech=1, vix=14.0,
                peers=peers, listing_day_D=20)
    pt = price_target(spec, date, PanelSource(deals, px, earn), calibrated)
    assert not pt.listed and pt.asof < pd.Timestamp("2019-03-11")
    assert "listing-day discovery_move" in " ".join(pt.result.decomposition.columns)
    assert np.isfinite(pt.result.atm["atm_iv"]).all()


def test_standard_expiries():
    e = standard_expiries("2027-01-19")
    assert e == sorted(e) and len(e) >= 8 and all(d.weekday() == 4 for d in e)


def test_backtest_runs(universe):
    deals, px, earn = universe
    res, summ = backtest(deals, px, earn, origins=(10,), folds=3, verbose=False)
    assert len(res) > 40 and set(summ["method"]) == {"model", "B0", "B1"}
