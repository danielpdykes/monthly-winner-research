"""Compute the backtested rule's output for the latest data (network required).

Rule output only, not investment advice. Mirrors the backtest definition: rank current
S&P 500 members by total return from the second-to-last to the last COMPLETED month-end;
the top name is the rule's pick for the purchase month that follows. Also reports the
SPY drawdown regime at that month-end and which regime overlays (if any) passed the
robustness checks in data/downside_summary.json. Writes data/live/<as_of>/signal.json.
Refuses to name a pick if fewer than --min-coverage of members have usable prices.
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
YAHOO = 'https://query1.finance.yahoo.com/v8/finance/chart/{}?range=6mo&interval=1d'


def get(url, retries=3):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            return urllib.request.urlopen(req, timeout=20).read()
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def daily(symbol):
    raw = get(YAHOO.format(symbol.replace('.', '-')))
    if not raw:
        return symbol, None
    try:
        r = json.loads(raw)['chart']['result'][0]
        px = r['indicators']['adjclose'][0]['adjclose']
        s = pd.Series(px, index=pd.to_datetime(r['timestamp'], unit='s').normalize()).dropna()
        return symbol, s
    except Exception:
        return symbol, None


def month_ends(as_of):
    """(signal month-end, base month-end, purchase month) from the last COMPLETED month before as_of."""
    first = pd.Timestamp(as_of).to_period('M')
    return first - 1, first - 2, first


def run(as_of, min_coverage, out_root):
    sig_m, base_m, buy_m = month_ends(as_of)
    members = pd.read_csv(pd.io.common.BytesIO(get(MEMBERS)))
    symbols = sorted(members.Symbol)
    with ThreadPoolExecutor(8) as ex:
        series = dict(ex.map(daily, symbols + ['SPY']))
    spy = series['SPY']
    rows, missing = [], []
    for sym in symbols:
        s = series.get(sym)
        if s is None or s.empty:
            missing.append(sym)
            continue
        by = s.groupby(s.index.to_period('M')).last()
        if sig_m not in by.index or base_m not in by.index:
            missing.append(sym)
            continue
        rows.append((sym, float(by[sig_m] / by[base_m] - 1)))
    rows.sort(key=lambda r: (-r[1], r[0]))
    coverage = len(rows) / len(symbols)

    # regime at the signal month-end, from the frozen history extended with live closes
    hist = pd.read_csv(ROOT / 'data/spy_daily.csv', parse_dates=['date']).set_index('date').adj_close
    spy_all = pd.concat([hist[hist.index < spy.index.min()], spy])
    me = spy_all.groupby(spy_all.index.to_period('M')).last()
    me = me[me.index <= sig_m]
    dd = float(me.iloc[-1] / me.max() - 1)
    month_ret = float(me.iloc[-1] / me.iloc[-2] - 1)
    window = spy_all[spy_all.index <= me.index.max().to_timestamp(how='end')]
    crash = bool((window.pct_change() <= -0.05).groupby(window.index.to_period('M')).any().get(sig_m, False)
                 or ((window / window.rolling(20).max() - 1) <= -0.10).groupby(window.index.to_period('M')).any().get(sig_m, False))
    state = 'post_crash' if crash else 'bear' if dd <= -0.20 else 'correction' if dd <= -0.10 else 'normal'
    summary = json.loads((ROOT / 'data/downside_summary.json').read_text())
    robust = summary['robust_action_by_state'].get(state, 'winner') if state != 'normal' else 'winner'
    latest = spy.index.max()
    now_dd = float(spy_all.iloc[-1] / spy_all.cummax().iloc[-1] - 1)
    ok = coverage >= min_coverage and rows
    result = dict(
        as_of=str(as_of), latest_price_date=str(latest.date()), purchase_month=str(buy_m),
        signal_month_end=str(sig_m), base_month_end=str(base_m),
        disclaimer='Mechanical output of a backtested rule. Not investment advice.',
        members=len(symbols), priced=len(rows), coverage=coverage, missing=missing,
        status='ok' if ok else 'refused: insufficient price coverage',
        pick=dict(ticker=rows[0][0], prior_month_return=rows[0][1]) if ok else None,
        top5=[dict(ticker=t, prior_month_return=r) for t, r in rows[:5]] if ok else [],
        regime=dict(state=state, spy_drawdown_at_signal=dd, spy_month_return=month_ret,
                    spy_drawdown_now=now_dd, crash_in_signal_month=crash),
        robust_overlay=robust,
        action='standard deposit into pick' if ok and robust == 'winner' else
               f'overlay "{robust}" passed robustness checks' if ok else 'no output',
        experimental_overlays='See data/regime_strategies.csv: none passed leave-one-episode-out and placebo checks '
                              'at the time of writing; shown for information only.')
    out = out_root / str(as_of)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'signal.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--as-of', default=str(date.today()))
    ap.add_argument('--min-coverage', type=float, default=0.98)
    ap.add_argument('--output', type=Path, default=ROOT / 'data/live')
    a = ap.parse_args()
    print(json.dumps(run(a.as_of, a.min_coverage, a.output), indent=2))
