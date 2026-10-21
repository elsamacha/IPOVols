"""Write a per-name report: surface CSVs, a markdown summary and a chart."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def write_report(pt, out_dir="output/targets") -> Path:
    out = Path(out_dir) / f"{slug(pt.spec.name)}_{pt.asof.date()}"
    out.mkdir(parents=True, exist_ok=True)
    r = pt.result
    r.table().to_csv(out / "surface.csv", index=False)
    r.atm.to_csv(out / "atm_term_structure.csv", index=False)
    r.decomposition.to_csv(out / "variance_decomposition.csv", index=False)
    info = {k: v for k, v in r.info.items() if k != "z"}

    fmt = lambda x: f"{x:.1%}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else "-"
    atm = r.atm.copy()
    cols = ["expiry", "cal_days", "forward", "atm_iv", "rr25", "fly25"] + [c for c in atm if c.startswith("atm_p")]
    lines = [f"# {pt.spec.name}: IPO vol surface as of {pt.asof.date()}", "",
             f"- Status: {'listed' if pt.listed else 'pre-listing'}; listing date {pd.Timestamp(pt.inputs.listing_date).date()}; "
             f"trading days since listing: {info['t0']}",
             f"- Spot/reference: {pt.inputs.spot:.2f}; peers: {', '.join(pt.peers)}",
             f"- Peer ATM term: " + ", ".join(f"{d}d {v:.1%}" for d, v in sorted(pt.inputs.peer_term.items())),
             f"- IPO multiplier now: implied ×{info['M_I_now']:.2f} (variance), realised ×{info['M_R_now']:.2f}; "
             f"Bayes level ×{info['lambda_bayes']:.2f}",
             f"- Market anchoring: diffusive ×{info['lambda_diff']:.2f}, jumps ×{info['lambda_jump']:.2f}, "
             f"skew shift {info['dpsi_anchor']:+.3f} ({len(pt.inputs.quotes)} quotes)",
             f"- Expected realised vol, next 21 trading days: {pt.realized_vol_fcst_21d:.1%}", "",
             "## Flags", *([f"- {n}" for n in pt.notes] or ["- none"]), "",
             "## ATM term structure", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in atm[cols].iterrows():
        vals = [str(row["expiry"]), str(int(row["cal_days"])), f"{row['forward']:.2f}"] + \
               [fmt(row[c]) for c in cols[3:]]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "## Events in the surface", ""]
    lines.append("- Earnings: " + ", ".join(str(pd.Timestamp(d).date()) for d in pt.inputs.earnings_dates[:4]))
    lines.append(f"- Lockup: {pd.Timestamp(pt.inputs.lockup_date).date()}")
    for e in pt.inputs.extra_events:
        lines.append(f"- {e.get('name', 'custom')}: {pd.Timestamp(e['date']).date()} (D={e['D']})")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    _chart(pt, out / "surface.png")
    return out


def _chart(pt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    r = pt.result
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    d = r.atm["cal_days"]
    if "atm_p10" in r.atm:
        ax.fill_between(d, r.atm["atm_p10"], r.atm["atm_p90"], color=BLUE, alpha=0.15, lw=0, label="p10–p90")
    ax.plot(d, r.atm["atm_iv"], color=BLUE, lw=2, marker="o", ms=4, label="ATM implied vol")
    ax.axhline(pt.inputs.peer_term[30], color=GRAY, lw=1, ls=(0, (2, 2)))
    ax.text(d.iloc[-1], pt.inputs.peer_term[30], " peer 1m", color=MUTED, fontsize=9, va="bottom", ha="right")
    for (dt, lab) in [(pt.inputs.earnings_dates[0] if pt.inputs.earnings_dates else None, "earnings"),
                      (pt.inputs.lockup_date, "lockup")]:
        if dt is None:
            continue
        x = (pd.Timestamp(dt) - pd.Timestamp(pt.asof)).days
        if 0 < x < d.iloc[-1]:
            ax.axvline(x, color="#c9c8c2", lw=1)
            ax.text(x + 2, ax.get_ylim()[1], lab, color=MUTED, fontsize=9, va="top")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, 0))
    ax.set_xlabel("calendar days to expiry")
    ax.set_title("ATM term structure", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    m = r.strikes
    sel = (m >= 0.6) & (m <= 1.5)
    idx = sorted({min(1, len(r.expiries) - 1), len(r.expiries) // 2, len(r.expiries) - 1})
    for i, col in zip(idx, (BLUE, ORANGE, AQUA)):
        ax.plot(m[sel], r.iv[i][sel], color=col, lw=2, label=f"{int(r.atm['cal_days'][i])}d")
        if r.iv_q:
            ax.fill_between(m[sel], r.iv_q[0.1][i][sel], r.iv_q[0.9][i][sel], color=col, alpha=0.12, lw=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, 0))
    ax.set_xlabel("strike / forward")
    ax.set_title("Smiles", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, title="expiry", title_fontsize=9)
    fig.suptitle(f"{pt.spec.name}: implied vol surface as of {pt.asof.date()}", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
