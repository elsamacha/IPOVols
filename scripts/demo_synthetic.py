"""End-to-end demo on synthetic data with known parameters.

    python scripts/demo_synthetic.py

1. simulates 160 past IPOs (raw Bloomberg-shaped data), 2. calibrates + bootstraps,
3. prints recovered vs true parameters, 4. prices a hypothetical mega-cap AI listing,
5. writes CSV/JSON outputs and two charts to output/.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ipovol.synthetic import generate, TRUTH
from ipovol.panel import build_panel
from ipovol.calibrate import calibrate, bootstrap, realized_blocks, implied_blocks, Scaler
from ipovol.pricer import IPOPricer, IPOInputs, RealizedObs
from ipovol.config import FEATURES

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
                     "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

# ---------------------------------------------------------------- 1-2 calibrate
deals, px, earnings, truth_extra = generate(n_ipo=160, seed=7)
panel, events, feats = build_panel(deals, px, earnings)
cal = calibrate(panel, events, feats)
draws = bootstrap(panel, events, feats, cal, B=200)
cal.save(OUT / "calibration.json")
with open(OUT / "bootstrap_draws.pkl", "wb") as fh:
    pickle.dump([d.to_dict() for d in draws], fh)

# ---------------------------------------------------------------- 3 recovery table
def get(c, key):
    return {
        "realised beta_inf": c.curve_R.beta, "realised a (avg IPO)": np.exp(c.curve_R.theta0),
        "realised tau (days)": c.curve_R.tau, "delta (per earnings)": c.curve_R.delta,
        "implied beta_inf": c.curve_I.beta, "implied tau (days)": c.curve_I.tau,
        "theta float_pct": c.curve_R.theta[0], "theta first_day_ret": c.curve_R.theta[1],
        "theta log_mcap": c.curve_R.theta[2],
        "D earnings #1": c.jumps["D_earn1"], "D earnings #2+": c.jumps["D_earn2"],
        "D quiet end": c.jumps["D_quiet"], "earnings bimodal w": c.jumps["earn_bimodal_w"],
        "lockup gap prob": c.jumps["lockup_gap_p"], "lockup gap (sd)": c.jumps["lockup_gap_g"],
        "excess skew psi_0": c.skew["psi"]["c"], "skew decay tau": c.skew["psi"]["tau"],
    }[key]

true_vals = {
    "realised beta_inf": TRUTH["beta_R"], "realised a (avg IPO)": np.exp(TRUTH["theta0_R"]),
    "realised tau (days)": TRUTH["tau_R"], "delta (per earnings)": TRUTH["delta"],
    "implied beta_inf": TRUTH["beta_I"], "implied tau (days)": TRUTH["tau_I"],
    "theta float_pct": TRUTH["theta"]["float_pct"], "theta first_day_ret": TRUTH["theta"]["first_day_ret"],
    "theta log_mcap": TRUTH["theta"]["log_mcap"],
    "D earnings #1": TRUTH["D_earn1"], "D earnings #2+": TRUTH["D_earn2"], "D quiet end": TRUTH["D_quiet"],
    "earnings bimodal w": TRUTH["earn_w"], "lockup gap prob": TRUTH["lock_p"],
    "lockup gap (sd)": TRUTH["lock_g_sd"], "excess skew psi_0": TRUTH["dpsi_c"],
    "skew decay tau": TRUTH["dpsi_tau"],
}
rec = []
for k, tv in true_vals.items():
    bs = np.array([get(d, k) for d in draws], float)
    lo, hi = np.nanquantile(bs, [0.05, 0.95])
    rec.append(dict(parameter=k, truth=tv, estimate=get(cal, k), ci90_lo=lo, ci90_hi=hi,
                    truth_in_ci=bool(lo <= tv <= hi)))
rec = pd.DataFrame(rec)
rec.to_csv(OUT / "recovery_vs_truth.csv", index=False)
pd.set_option("display.width", 200)
print("\nRecovery vs truth (90% cluster-bootstrap CI):")
print(rec.round(3).to_string(index=False))

# ---------------------------------------------------------------- 4 price
listing = pd.Timestamp("2027-01-11")
val = listing + pd.offsets.BDay(5)
inp = IPOInputs(
    valuation_date=val, listing_date=listing, spot=100.0,
    features=dict(float_pct=0.06, first_day_ret=0.35, log_mcap=np.log(3.5e11), vc_backed=1, tech=1,
                  vix_at_listing=17.0, post2023=1),
    peer_term={30: 0.55, 60: 0.54, 91: 0.53, 182: 0.51, 365: 0.50},   # placeholder peer marks
    peer_psi=-0.10, peer_gamma=0.05, rate=0.04,
    earnings_dates=["2027-03-02", "2027-06-01", "2027-08-31"],
    borrow_now=0.15,
    realized=RealizedObs(n_days=4, ipo_var_mean=1.9 * 0.55 ** 2 / 252, peer_var_mean=0.55 ** 2 / 252),
)
expiries = pd.to_datetime(["2027-01-29", "2027-02-19", "2027-03-19", "2027-04-16", "2027-06-18",
                           "2027-07-16", "2027-09-17", "2028-01-21"])
pricer = IPOPricer(cal, draws)
res = pricer.price(inp, expiries)
res.table().to_csv(OUT / "surface.csv", index=False)
res.atm.to_csv(OUT / "atm_term_structure.csv", index=False)
res.decomposition.to_csv(OUT / "variance_decomposition.csv", index=False)
print("\nPricing info:", {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in res.info.items() if k != "z"})
print("\nATM term structure (mid = point calibration; p10-p90 = parameter uncertainty):")
print(res.atm.round(3).to_string(index=False))
print("\nVariance decomposition:")
print(res.decomposition.round(3).to_string(index=False))

# ---------------------------------------------------------------- 5 charts
Z = cal.scaler.transform(feats)
def binned(blocks, curve, truth_fn):
    b = blocks.copy()
    zz = Z.loc[b["ticker"]].to_numpy()
    b["fit"] = curve.M(b["t"], b["k"], zz)
    b["true"] = truth_fn(b)
    b["bin"] = (b["t"] // 10) * 10 + 5
    return b.groupby("bin")[["R", "fit", "true"]].mean()

a_true = pd.Series(truth_extra["a_R"], index=feats.index)
eps = pd.Series(truth_extra["eps"], index=feats.index)
tR = lambda b: (TRUTH["beta_R"] * (1 + a_true.loc[b["ticker"]].to_numpy() * np.exp(-b["t"] / TRUTH["tau_R"])
                * TRUTH["delta"] ** b["k"]) * eps.loc[b["ticker"]].to_numpy())
tI = lambda b: (TRUTH["beta_I"] * (1 + TRUTH["a_I_scale"] * a_true.loc[b["ticker"]].to_numpy()
                * np.exp(-b["t"] / TRUTH["tau_I"]) * TRUTH["delta"] ** b["k"]) * eps.loc[b["ticker"]].to_numpy())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (blocks, curve, fn, name) in zip(axes, [(realized_blocks(panel), cal.curve_R, tR, "Realised variance ratio"),
                                                (implied_blocks(panel), cal.curve_I, tI, "Implied variance ratio")]):
    g = binned(blocks, curve, fn)
    ax.scatter(g.index, g["R"], s=22, color=GRAY, label="data (10-day bins)", zorder=3)
    ax.plot(g.index, g["fit"], color=BLUE, lw=2, label="calibrated model")
    ax.plot(g.index, g["true"], color=ORANGE, lw=2, ls=(0, (4, 3)), label="true process")
    ax.set_title(f"{name}, IPO / peer", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("trading days since listing")
axes[0].set_ylabel("variance multiplier M(t)")
axes[0].legend(frameon=False)
fig.suptitle("Calibration recovers the synthetic truth (160 IPOs)", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "calibration_fit.png", dpi=160)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]
days = res.atm["cal_days"]
ax.fill_between(days, res.atm["atm_p10"], res.atm["atm_p90"], color=BLUE, alpha=0.15, lw=0,
                label="p10-p90 parameter band")
ax.plot(days, res.atm["atm_iv"], color=BLUE, lw=2, marker="o", ms=5, label="ATM implied vol")
ax.axhline(0.55, color=GRAY, lw=1, ls=(0, (2, 2)))
ax.text(days.iloc[-1], 0.555, "peer 1m ATM", color=MUTED, ha="right", va="bottom", fontsize=9)
for d, lab in [("2027-03-02", "earnings #1"), ("2027-07-12", "lockup")]:
    x = (pd.Timestamp(d) - val).days
    ax.axvline(x, color="#c9c8c2", lw=1)
    ax.text(x + 3, res.atm["atm_p90"].max(), lab, color=MUTED, fontsize=9, va="top")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("calendar days to expiry")
ax.set_title("ATM term structure", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, loc="upper right")
ax = axes[1]
for i, col in [(1, BLUE), (3, ORANGE), (7, "#1baf7a")]:
    m = res.strikes
    sel = (m >= 0.6) & (m <= 1.5)
    ax.plot(m[sel], res.iv[i][sel], color=col, lw=2, label=f"{res.atm['cal_days'][i]}d")
    if res.iv_q:
        ax.fill_between(m[sel], res.iv_q[0.1][i][sel], res.iv_q[0.9][i][sel], color=col, alpha=0.12, lw=0)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("strike / forward")
ax.set_title("Smiles: early call skew fades with maturity", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, title="expiry", title_fontsize=9)
fig.suptitle("Hypothetical mega-cap AI listing, day 5: priced surface (synthetic calibration)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "priced_surface.png", dpi=160)
print(f"\nwrote outputs to {OUT}")
