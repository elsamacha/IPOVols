"""Command line.

  python -m ipovol calibrate --universe data/ipo_universe.csv      # Bloomberg pull + fit
  python -m ipovol calibrate --panel data/bbg_raw.pkl               # refit from a saved pull
  python -m ipovol calibrate --synthetic 160                        # synthetic universe (testing)

  python -m ipovol price "CRWV US Equity" --asof 2025-04-04         # any listed ticker, live Bloomberg
  python -m ipovol price targets/anthropic.json                     # pre-listing profile
  python -m ipovol price IPO012 --source panel --panel data/synthetic.pkl --asof 2016-03-01

  python -m ipovol backtest --panel data/bbg_raw.pkl                # leave-fold-out vs baselines
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


def _load_panel(path):
    d = pickle.load(open(path, "rb"))
    return d["deals"], d["px"], d["earnings"]


def cmd_calibrate(a):
    from .panel import build_panel
    from .calibrate import calibrate, bootstrap
    if a.universe:
        from .bbg import run_pull, merge_borrow
        deals, px, earnings = run_pull(a.universe, a.save_panel or "data/bbg_raw.pkl")
    elif a.panel:
        deals, px, earnings = _load_panel(a.panel)
    else:
        from .synthetic import generate
        deals, px, earnings, _ = generate(n_ipo=a.synthetic)
        path = Path(a.save_panel or "data/synthetic.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(dict(deals=deals, px=px, earnings=earnings), open(path, "wb"))
        print(f"synthetic universe saved to {path}")
    if a.borrow:
        from .bbg import merge_borrow
        px = merge_borrow(px, a.borrow)
    panel, events, feats = build_panel(deals, px, earnings)
    cal = calibrate(panel, events, feats)
    out = Path(a.cal_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal.save(out / "calibration.json")
    if a.B:
        draws = bootstrap(panel, events, feats, cal, B=a.B)
        pickle.dump([d.to_dict() for d in draws], open(out / "bootstrap_draws.pkl", "wb"))
    print(f"calibration saved to {out}")


def cmd_price(a):
    from .api import price_target
    from .calibrate import Calibration
    from .report import write_report
    cal_dir = Path(a.cal_dir)
    cal = Calibration.load(cal_dir / "calibration.json")
    draws = None
    if not a.no_bands and (cal_dir / "bootstrap_draws.pkl").exists():
        draws = [Calibration.from_dict(d) for d in pickle.load(open(cal_dir / "bootstrap_draws.pkl", "rb"))]
    if a.source == "bbg":
        from .sources import BloombergSource
        src = BloombergSource()
    else:
        from .sources import PanelSource
        src = PanelSource.from_pickle(a.panel)
    pt = price_target(a.target, a.asof, src, cal, draws, anchor=False if a.no_quotes else None)
    out = write_report(pt, a.out)
    print((out / "summary.md").read_text())
    print(f"report written to {out}")


def cmd_backtest(a):
    from .backtest import backtest
    if a.panel:
        deals, px, earnings = _load_panel(a.panel)
    else:
        from .synthetic import generate
        deals, px, earnings, _ = generate(n_ipo=a.synthetic)
    res, summ = backtest(deals, px, earnings, origins=tuple(int(x) for x in a.origins.split(",")),
                         folds=a.folds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "backtest_detail.csv", index=False)
    summ.to_csv(out / "backtest_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(f"written to {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ipovol")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--universe")
    g.add_argument("--panel")
    g.add_argument("--synthetic", type=int)
    c.add_argument("--borrow")
    c.add_argument("--B", type=int, default=200, help="bootstrap draws (0 = none)")
    c.add_argument("--cal-dir", default="output")
    c.add_argument("--save-panel")
    c.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("price")
    p.add_argument("target", help="Bloomberg ticker or path to a JSON profile")
    p.add_argument("--asof")
    p.add_argument("--source", choices=["bbg", "panel"], default="bbg")
    p.add_argument("--panel", default="data/bbg_raw.pkl")
    p.add_argument("--cal-dir", default="output")
    p.add_argument("--out", default="output/targets")
    p.add_argument("--no-quotes", action="store_true")
    p.add_argument("--no-bands", action="store_true")
    p.set_defaults(fn=cmd_price)

    b = sub.add_parser("backtest")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--panel")
    g.add_argument("--synthetic", type=int)
    b.add_argument("--origins", default="5,21")
    b.add_argument("--folds", type=int, default=5)
    b.add_argument("--out", default="output/backtest")
    b.set_defaults(fn=cmd_backtest)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
