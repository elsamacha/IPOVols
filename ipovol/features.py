"""One feature builder for both calibration and live pricing.

Training (past IPOs) and serving (the name you price) must compute features identically,
otherwise the fitted coefficients are applied to differently-scaled numbers. Every path in
the package goes through `compute_features`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES


def compute_features(listing_date, *, offer_price=None, first_close=None, shares_offered=None,
                     shares_out=None, mkt_cap_usd=None, vix=None, vc_backed=None, tech=None,
                     float_pct=None, first_day_ret=None) -> dict:
    """Return {feature: value}; values that cannot be derived are NaN (priced as neutral).

    Explicit overrides (float_pct, first_day_ret) win over derived values - use them for
    names that have not listed yet (e.g. expected float and an assumed first-day pop).
    """
    listing_date = pd.Timestamp(listing_date)
    nan = np.nan
    if float_pct is None:
        float_pct = shares_offered / shares_out if shares_offered and shares_out else nan
    if first_day_ret is None:
        first_day_ret = first_close / offer_price - 1 if offer_price and first_close else nan
    if mkt_cap_usd is None and first_close and shares_out:
        mkt_cap_usd = first_close * shares_out
    return dict(
        float_pct=float(float_pct),
        first_day_ret=float(first_day_ret),
        log_mcap=float(np.log(mkt_cap_usd)) if mkt_cap_usd else nan,
        vc_backed=float(vc_backed) if vc_backed is not None else nan,
        tech=float(tech) if tech is not None else nan,
        vix_at_listing=float(vix) if vix is not None else nan,
        post2023=float(listing_date >= pd.Timestamp("2023-01-01")),
    )


def missing(features: dict) -> list[str]:
    return [f for f in FEATURES if f not in features or pd.isna(features[f])]
