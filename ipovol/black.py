"""Black-76 pricing on the forward and a vectorised, robust implied-vol inversion."""
import numpy as np
from scipy.special import ndtr


def _d1d2(F, K, sd):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return d1, d1 - sd


def black_call(F, K, sigma, T):
    """Undiscounted Black call."""
    F, K, sigma, T = np.broadcast_arrays(*(np.asarray(a, float) for a in (F, K, sigma, T)))
    sd = sigma * np.sqrt(T)
    d1, d2 = _d1d2(F, K, sd)
    px = F * ndtr(d1) - K * ndtr(d2)
    return np.where(sd > 0, px, np.maximum(F - K, 0.0))


def black_put(F, K, sigma, T):
    """Undiscounted Black put (computed directly, not via parity)."""
    F, K, sigma, T = np.broadcast_arrays(*(np.asarray(a, float) for a in (F, K, sigma, T)))
    sd = sigma * np.sqrt(T)
    d1, d2 = _d1d2(F, K, sd)
    px = K * ndtr(-d2) - F * ndtr(-d1)
    return np.where(sd > 0, px, np.maximum(K - F, 0.0))


def black_otm(F, K, sigma, T):
    """Undiscounted out-of-the-money option (put for K<F, call for K>=F); numerically clean."""
    F, K, sigma, T = np.broadcast_arrays(*(np.asarray(a, float) for a in (F, K, sigma, T)))
    sd = sigma * np.sqrt(T)
    d1, d2 = _d1d2(F, K, sd)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    px = np.where(K >= F, call, put)
    return np.where(sd > 0, px, 0.0)


def implied_vol_otm(otm_price, F, K, T, lo=1e-4, hi=6.0, iters=80):
    """Invert undiscounted OTM prices to Black vols by vectorised bisection.

    Returns NaN where the price is outside no-arbitrage bounds.
    """
    p, F, K, T = np.broadcast_arrays(*(np.asarray(a, float) for a in (otm_price, F, K, T)))
    lo_a = np.full(p.shape, lo)
    hi_a = np.full(p.shape, hi)
    upper = np.where(K >= F, F, K)          # OTM call < F, OTM put < K
    bad = (p <= 0) | (p >= upper) | ~np.isfinite(p)
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        too_high = black_otm(F, K, mid, T) > p
        hi_a = np.where(too_high, mid, hi_a)
        lo_a = np.where(too_high, lo_a, mid)
    out = 0.5 * (lo_a + hi_a)
    return np.where(bad, np.nan, out)


def black_delta_call(F, K, sigma, T):
    sd = sigma * np.sqrt(T)
    d1, _ = _d1d2(F, K, sd)
    return ndtr(d1)
