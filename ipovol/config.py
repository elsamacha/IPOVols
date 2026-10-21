"""Shared conventions and constants.

Time conventions
----------------
* Diffusive variance is measured per *trading day* (a daily variance, not annualised).
* Implied vols from Bloomberg are treated as calendar-time vols: total variance to an
  expiry = iv**2 * calendar_days / 365.
* Reported implied vols out of the pricer use the same calendar/365 convention, so
  they are directly comparable to Bloomberg / listed quotes.
* Event jumps are lumps of variance on their reaction day.
"""

TRADING_DAYS = 252
CAL_DAYS = 365.0

# Deal features used to condition the multiplier curve (standardised before fitting).
FEATURES = [
    "float_pct",        # shares offered / shares outstanding
    "first_day_ret",    # first close / offer price - 1
    "log_mcap",         # log market cap at first close (USD)
    "vc_backed",        # 0/1
    "tech",             # 0/1
    "vix_at_listing",   # VIX close on listing day
    "post2023",         # 0/1 regime dummy (AI / mega-cap concentration regime)
]

# Calibration horizon (trading days after listing) and block length for the curve fit.
HORIZON = 252
BLOCK = 5

# Event conventions
QUIET_PERIOD_CAL_DAYS = 25       # analyst initiations
DEFAULT_LOCKUP_CAL_DAYS = 180
EVENT_HALF_WINDOW = 1            # event window = [d-1, d+1] around the reaction day
LOCAL_VAR_WINDOW = (3, 15)       # local diffusive var from days d-15..d-3 and d+3..d+15

# Standardised features are clipped to this many sigmas at pricing time (extrapolation guard).
FEATURE_CLIP = 3.0

# Bloomberg field mnemonics. VERIFY EACH ONE IN FLDS <GO> BEFORE THE FIRST REAL RUN.
BBG_FIELDS = {
    "open": "PX_OPEN",
    "high": "PX_HIGH",
    "low": "PX_LOW",
    "close": "PX_LAST",
    "volume": "PX_VOLUME",
    "iv_atm": "30DAY_IMPVOL_100.0%MNY_DF",
    "iv_90": "30DAY_IMPVOL_90.0%MNY_DF",
    "iv_110": "30DAY_IMPVOL_110.0%MNY_DF",
    "iv_60": "60DAY_IMPVOL_100.0%MNY_DF",
    "iv_91": "3MTH_IMPVOL_100.0%MNY_DF",
    "iv_182": "6MTH_IMPVOL_100.0%MNY_DF",
    "iv_365": "12MTH_IMPVOL_100.0%MNY_DF",
    "mkt_cap": "CUR_MKT_CAP",
    "sh_out": "EQY_SH_OUT",
}
# Term-structure fields for the peer surface at the valuation date.
BBG_TERM_FIELDS = {
    30: "30DAY_IMPVOL_100.0%MNY_DF",
    60: "60DAY_IMPVOL_100.0%MNY_DF",
    91: "3MTH_IMPVOL_100.0%MNY_DF",
    182: "6MTH_IMPVOL_100.0%MNY_DF",
    365: "12MTH_IMPVOL_100.0%MNY_DF",
}
# Bulk field for historical earnings announcement dates + timing (before/after market).
BBG_EARNINGS_BULK = "EARN_ANN_DT_TIME_HIST_WITH_EPS"
BBG_VIX = "VIX Index"

# Column name -> calendar days, for building a term structure from stored history.
IV_TERM_COLS = {"iv_atm": 30, "iv_60": 60, "iv_91": 91, "iv_182": 182, "iv_365": 365}

# Reference data for a single target (VERIFY IN FLDS). Profile values always win.
BBG_REF_FIELDS = {
    "ipo_date": "EQY_INIT_PO_DT",
    "offer_price": "EQY_INIT_PO_PX",
    "shares_offered": "EQY_INIT_PO_SH_OFFERED",
    "next_earnings": "EXPECTED_REPORT_DT",
    "name": "NAME",
}
BBG_PEERS_BULK = "BLOOMBERG_PEERS"          # bulk field returning Bloomberg's peer list
BBG_OPT_CHAIN = "OPT_CHAIN"                 # bulk field: listed option tickers
BBG_OPT_FIELDS = {"strike": "OPT_STRIKE_PX", "expiry": "OPT_EXPIRE_DT", "iv": "IVOL_MID",
                  "put_call": "OPT_PUT_CALL", "open_int": "OPEN_INT"}
