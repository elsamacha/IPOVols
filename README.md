# ipovol: implied-vol surface for newly listed stocks

This package prices a full implied-vol surface (term structure, skew and event bumps, with uncertainty bands) for a stock that has little or no trading history. It is calibrated on a panel of past IPOs.

## Quick start: price any name

```bash
pip install xbbg blpapi numpy pandas scipy matplotlib

# 1. calibrate once on your universe of past IPOs (refresh quarterly)
python -m ipovol calibrate --universe data/ipo_universe.csv --B 200

# 2. price any target
python -m ipovol price "CRWV US Equity"                   # listed: everything fetched from Bloomberg
python -m ipovol price "CRWV US Equity" --asof 2025-04-04 # point in time
python -m ipovol price targets/anthropic.json             # not yet listed: JSON profile

# 3. check it generalises (leave-fold-out vs peer-vol baselines)
python -m ipovol backtest --panel data/bbg_raw.pkl
```

Each `price` run writes `output/targets/<name>_<date>/` containing:
- `summary.md` (inputs used, flags and the ATM table)
- `surface.csv`, `atm_term_structure.csv` and `variance_decomposition.csv`
- `surface.png`

To try everything without Bloomberg, use a synthetic universe:

```bash
python -m ipovol calibrate --synthetic 160 --B 100
python -m ipovol price IPO012 --source panel --panel data/synthetic.pkl --asof 2013-01-11
python -m ipovol backtest --synthetic 160
```

In Python:

```python
from ipovol.api import price_target
from ipovol.calibrate import Calibration
from ipovol.sources import BloombergSource
pt = price_target("ARM US Equity", "2023-09-21", BloombergSource(), Calibration.load("output/calibration.json"))
pt.result.atm, pt.result.table(), pt.notes, pt.realized_vol_fcst_21d
```

### How inputs are resolved for a target

The order is always **profile value → data source → default assumption**. Whenever a default is used, the report records it under Flags.

| Input | Listed ticker (Bloomberg) | Pre-listing profile | Default if missing |
|---|---|---|---|
| Listing date, offer price | `EQY_INIT_PO_*` fields | `listing_date`, `offer_price` | required |
| Peers | `BLOOMBERG_PEERS`; keeps those with IV data and flags peers with less than 1y of history | `peers` | required |
| Peer term structure and smile | 30d–12m ATM plus 90/110 moneyness vols at the valuation date | same | |
| Features | derived from first-day close, shares, market cap and VIX via `features.py` | `valuation_usd`, `float_pct`, `expected_first_day_ret`, ... | average IPO (flagged) |
| Earnings | `EXPECTED_REPORT_DT`, then quarterly spacing | `earnings_dates` | quarter-end + 45d, quarterly |
| Lockup | profile | `lockup_date` | listing + 180d (flagged) |
| Realised so far | daily OHLC since listing vs peers, event windows excluded | none | none |
| Listed quotes | option chain mid IVs, OTM, 4 nearest expiries | none | pure model (flagged) |
| Borrow | profile | `borrow_now` | calibrated decay curve |
| Listing-day move | none | `listing_day_D` | not included (flagged) |
| Expiries | 2 weeklies, 6 monthlies, then quarterlies to 18m | `expiries` | same |

`targets/_template.json` lists every profile key.

### Design choices that make it reusable
- **Same feature code for calibration and pricing.** `features.py` builds the deal features for both the calibration universe and the target, so the fitted coefficients always see identically defined inputs.
- **Pluggable data sources.** `sources.py` defines one interface: `BloombergSource` for live data and `PanelSource` for a stored pull or synthetic data, which gives point-in-time re-pricing of any past IPO. Adding QuantVol or an internal vol service means implementing about eight methods.
- **The backtest uses the live code path.** It prices held-out IPOs through the same `price_target` function, so what gets validated is exactly what runs live.
- **Nothing name-specific lives in the code.** Anthropic is just `targets/anthropic.json`, with placeholder values to fill in.

## Pipeline

```
Bloomberg (xbbg) ──► panel of past IPOs ──► calibration (+ cluster bootstrap)
                                                   │
target ticker or profile ──► price_target: peers, features, events, realised-so-far, quotes
                                                   ▼
                                  surface: mid, p10/p25/p75/p90, ATM/RR/fly, variance decomposition
```

## Model

**1. Diffusive variance relative to peers.** For each future trading day u:

```
v_I(u) = v_peer(u) × M_I(t, k; x) × λ_bayes × λ_diff
M(t, k; x) = β∞ · (1 + a(x)·e^(−t/τ)·δ^k),   log a(x) = θ0 + x'θ
```

- `v_peer` is the forward variance from the live peer ATM term structure. The peer average is taken over *single-name variances*, not a basket. A basket would understate vol through diversification.
- `t` is trading days since listing and `k` is the number of earnings reports already out. The IPO premium decays both with time and with information flow; this is the Pástor–Veronesi learning channel.
- `x` holds the standardised deal features: float %, first-day return, log market cap, VC-backed, tech, VIX at listing, and a post-2023 dummy.
- Two curves are fitted:
  - `M_I` is the ratio of IPO implied variance to peer implied variance. It drives the marks.
  - `M_R` is the realised ratio. It drives the jump scale and the Bayesian update.

**2. Scheduled events as variance lumps.** Each event adds variance `J = D × local daily var`, where D is measured in local diffusive days.

| Event | Shape |
|---|---|
| Earnings #1, #2+ | Bimodal: ±m plus a Gaussian, with weight w taken from the 4th moment |
| Lockup expiry | Gap-down mixture: probability p of a −g·σ gap, fitted by 2-component EM on abnormal returns |
| Quiet-period end | Gaussian |
| Custom (model launch, big deal) | Gaussian with D you set |

Discrete jump states are martingale-compensated, so they shift the forward without changing it in expectation. That is what produces the W-shaped smile before earnings and the put skew before lockup.

**3. Smile.** Skew and convexity are the peer smile plus a calibrated IPO excess, `Δψ(t) = c·e^(−t/τs) + d`. Early IPOs are typically call-skewed. The smile is mapped onto an SVI slice whose wing slopes stay under the Lee bound, so the surface stays arbitrage-free however steep the calibrated skew is.

**4. Borrow.** The fee curve `f(t) = f∞ + (f0 − f∞)e^(−t/τb)` feeds the forward. In the synthetic demo the fee starts around 13% and decays toward 0.4%.

**5. Bayesian level update.** `λ_bayes = (n0 + n·ρ)/(n0 + n)`, where ρ is realised-to-date divided by the model's prediction. The prior weight n0 is chosen out-of-sample by 5-fold cross-fitting, minimising the QLIKE of 21-day-ahead realised variance.

**6. Market anchoring.** When listed quotes exist, the pricer fits `λ_diff`, `λ_jump` and `Δψ` to them. The model then acts as the interpolator and extrapolator for unquoted strikes and expiries, and the bands collapse near quoted points.

**7. Uncertainty.** A cluster bootstrap resamples whole IPOs, refits, and reprices each draw to give p10–p90 bands. A sensible desk policy is to mark sold vol at p75 and bought vol at p25, and treat the spread as a model reserve.

### Estimation choices that matter

- **Daily variance** is estimated as overnight² plus Garman–Klass intraday. It is unbiased and about 5× more efficient than a squared return, which matters for short histories.
- **Peer denominators** use a 21-day smoothed peer variance. Dividing by a noisy 5-day peer estimate would bias the ratio up by about 10% through Jensen's inequality.
- **QLIKE loss** is used for the multiplier fit. It is consistent for the mean of a noisy variance ratio; plain least squares or robust losses bias the level.
- **Ridge on θ** is Gaussian partial pooling: sparse feature combinations shrink toward the average IPO.
- **Implied ratios** exclude days with an IPO event within 30 calendar days, so `M_I` is a diffusive ratio.
- **Peer VRP:** the physical forecast converts peer *implied* variance into peer *realised* variance using the panel ratio (`peer_rv_iv`, computed close-to-close from single-name returns) before applying the IPO multiplier. Without this, realised forecasts and event sizes would be overstated by the peer variance risk premium.
- **Contaminated events** (another event within ±3 days) are dropped. Jump size is a ratio of sums: `D = Σ(r² − 3σ²_loc)/Σσ²_loc`.
- **Extrapolation guard:** standardised features are clipped at ±3σ, and the clip is reported in `info["clipped_features"]`. For a mega-cap listing like Anthropic, `log_mcap` will almost always be clipped. Read that as a warning that size is out of sample. It is the case for anchoring to the first listed quotes as fast as possible.

## Backtest on synthetic data (`python -m ipovol backtest --synthetic 160`)

The backtest runs 5 folds. Each held-out IPO is priced 5 or 21 trading days after listing, with no market anchoring. Errors are in vol points:

| Age | Target | Model | B0 peer vol | B1 peer × avg premium |
|---|---|---|---|---|
| day 5 | 30d implied, RMSE | **9.6** | 32.9 | 14.5 |
| day 5 | 21d realised, RMSE | **15.2** | 26.4 | 16.0 |
| day 21 | 30d implied, RMSE | **8.3** | 38.6 | 24.3 |
| day 21 | 21d realised, RMSE | **23.1** | 37.2 | 28.4 |

The synthetic data comes from the model's own equations, so a win here is expected. It proves the machinery works end to end; it doesn't prove the model fits reality. The real test is the same command on the Bloomberg pull.

Realised errors stay large for every method. Over 21 days, sampling noise alone is about ±15% of the vol level, and a single event can dominate the window. The realised B1 baseline has a slightly lower MAPE than the model at day 21 even though the model wins on RMSE and QLIKE. Read the model as better on average, not dominant.

## Validation on synthetic data (`scripts/demo_synthetic.py`)

The demo simulates 160 IPOs with known parameters, in the same raw format the Bloomberg pull produces, and then runs the full pipeline. Results are in `output/recovery_vs_truth.csv` and `output/calibration_fit.png`.

- **Recovered within the 90% bootstrap CI:** 14 of 17 parameters, including every multiplier-curve, feature, jump-size, lockup and skew parameter.
- **Implied β∞ and τ:** 1.226 vs 1.25 true (−2%) and 56 vs 50 days. The truth sits just outside the CI. The 2% level error is small, but the implied decay time is weakly identified.
- **Earnings bimodality w:** 0.56 vs 0.70 true. The estimator is biased low by noise in the local-vol scaling, so treat w as a lower bound.
- **Tests** (`pytest tests`) cover:
  - Black parity and implied-vol inversion
  - estimator bias
  - parameter recovery
  - no-arbitrage (calls decreasing and convex in strike, ATM total variance increasing in maturity)
  - quote anchoring

## Running on real data

Use the commands in the quick start above. Before the first real run:

1. **Universe.** Start from `data/ipo_universe_template.csv`. The four rows in it are examples only; verify their dates and prices before use. Build the full list from `IPO <GO>` or ECM, US listings of $2B+ from 2010 onward. Include busted and delisted names. Tag direct listings and SPACs through `listing_type` so they are excluded. Choose peers using only information available at the IPO date.
2. **Field mnemonics.** Check every one in `FLDS <GO>`: `config.BBG_FIELDS`, `BBG_TERM_FIELDS` and the earnings bulk field. The code assumes Bloomberg quotes vols in percent. Confirm whether `%MNY_DF` moneyness is relative to spot or forward; it matters when borrow is large.
3. **Lockups.** The default is listing + 180 days. Hand-check the large names against the S-1, since early-release and staged lockups are common now.
4. **Borrow.** Bloomberg coverage of borrow is thin. Get desk securities-lending or Markit/S3 history into `ticker,date,fee` format and pass it with `--borrow`.
5. **Profile inputs** (`targets/*.json`), only for what Bloomberg can't supply:
   - pre-listing valuation, float and assumed first-day pop
   - peers, if Bloomberg's suggested list is poor
   - lockup date
   - borrow quote
   - custom events such as model releases

## Known limitations

- **Earnings timing in backtests:** `PanelSource` uses actual earnings reaction dates. That is a mild look-ahead on timing, since dates are usually announced weeks ahead, but never on outcomes.
- **Auto-selected peers:** Bloomberg's peer list is a starting point. Always review the peers named in the Flags section; peer choice is the biggest judgement input.

- **Listing-day price discovery** is not modelled. Day 0 is excluded from calibration. For pre-listing OTC pricing, add a custom event on the listing day with D set by judgement.
- **Event premium:** implied earnings moves usually exceed realised ones, and the prior doesn't include that premium. `event_premium` is a manual knob (default 1.0); anchoring corrects it once options list.
- **Smile shape** (ψ, γ) is taken from 30-day data and applied at all maturities in normalised-moneyness terms.
- **Calendars:** business days ignore exchange holidays, and there is no vol-of-vol or stochastic vol. It is a surface-construction model, not a dynamics model, so do not use it for exotic pricing that depends on forward skew.
- **Abnormal returns** assume beta = 1 to the peer basket.
- **Scale:** the synthetic truth uses a D of about 25 for first earnings, which gives roughly 20% moves on a stock at around 85% vol. Real calibrated values for mega-caps will be smaller. Every number in `output/` comes from synthetic data and is illustrative.
