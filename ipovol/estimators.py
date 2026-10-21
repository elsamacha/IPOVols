"""Range-based variance estimators and smile helpers."""
import numpy as np
import pandas as pd

LN2 = np.log(2.0)


def daily_variance(df: pd.DataFrame) -> pd.Series:
    """Per-day variance = overnight^2 + Garman-Klass intraday.

    Both parts are unbiased under driftless GBM, so the sum is an unbiased estimate of the
    close-to-close daily variance while being ~5x more efficient than a squared return.
    Falls back to the squared close-to-close return where OHL are missing.
    Columns required: open, high, low, close.
    """
    prev_close = df["close"].shift(1)
    overnight = np.log(df["open"] / prev_close)
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    gk = 0.5 * hl**2 - (2 * LN2 - 1) * co**2
    v = overnight**2 + gk
    cc = np.log(df["close"] / prev_close) ** 2
    return v.where(v.notna(), cc)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def yang_zhang(df: pd.DataFrame) -> float:
    """Yang-Zhang *daily* variance over a whole window (drift-robust, uses opens)."""
    d = df.dropna(subset=["open", "high", "low", "close"])
    if len(d) < 3:
        return np.nan
    o = np.log(d["open"] / d["close"].shift(1)).iloc[1:]
    c = np.log(d["close"] / d["open"]).iloc[1:]
    u = np.log(d["high"] / d["open"]).iloc[1:]
    dn = np.log(d["low"] / d["open"]).iloc[1:]
    n = len(o)
    rs = (u * (u - c) + dn * (dn - c)).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return o.var(ddof=1) + k * c.var(ddof=1) + (1 - k) * rs


# ---------- smile parameterisation ---------------------------------------------------
# sigma(z) = sigma_atm * (1 + psi*z + gamma*z^2),  z = ln(K/F) / sqrt(total variance)
# psi < 0 is ordinary put skew, psi > 0 call skew; gamma is convexity (wings).

def smile_params(iv90, iv100, iv110, T=30 / 365.0):
    """Recover (psi, gamma) from 90/100/110 moneyness vols at maturity T (years)."""
    iv90, iv100, iv110 = (np.asarray(a, float) for a in (iv90, iv100, iv110))
    sd = iv100 * np.sqrt(T)
    zp, zm = np.log(1.1) / sd, np.log(0.9) / sd
    a1, a2 = iv110 / iv100 - 1.0, iv90 / iv100 - 1.0
    det = zp * zm**2 - zm * zp**2
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = (a1 * zm**2 - a2 * zp**2) / det
        gamma = (zp * a2 - zm * a1) / det
    return psi, gamma


def smile_vols(iv100, psi, gamma, T=30 / 365.0):
    """Inverse of smile_params: 90/110 vols implied by (psi, gamma)."""
    sd = iv100 * np.sqrt(T)
    zp, zm = np.log(1.1) / sd, np.log(0.9) / sd
    return iv100 * (1 + psi * zm + gamma * zm**2), iv100 * (1 + psi * zp + gamma * zp**2)
