"""Long-history (1926-2026) test of the down-market allocation rules.

Uses Ken French's free data library: monthly + daily Mkt-RF/RF (total-return market, T-bill)
and the value-weighted decile portfolios formed on prior 1-month return. The top decile
("Hi PRIOR") is a diversified proxy for 'last month's winners'; it is NOT the single best
stock, so results indicate whether the allocation rules generalise, not what the stock
strategy would have earned. --fetch downloads the zips, each into its own new directory.
"""
import argparse
import csv
import re
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from alloc import grid
from downside import crash_flags, label_regimes

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
FILES = ['F-F_Research_Data_Factors_CSV', 'F-F_Research_Data_Factors_daily_CSV', '10_Portfolios_Prior_1_0_CSV']
OUT = ROOT / 'data/long_history'
STATES = ('correction', 'bear', 'post_crash')


def fetch(raw):
    for name in FILES:
        d = raw / name
        d.mkdir(parents=True, exist_ok=False)          # fresh, empty directory per download
        req = urllib.request.Request(BASE + name + '.zip', headers={'User-Agent': 'Mozilla/5.0'})
        (d / (name + '.zip')).write_bytes(urllib.request.urlopen(req).read())
        with zipfile.ZipFile(d / (name + '.zip')) as z:
            for member in z.namelist():
                if '/' in member or '..' in member:
                    raise ValueError('Unsafe archive member ' + member)
                z.extract(member, d)


def section(path, title=None):
    """Rows of the first (or titled) data section of a French CSV: [date, v1, v2, ...] until a blank line."""
    lines = path.read_text(errors='replace').splitlines()
    start = 0
    if title:
        start = next(i for i, l in enumerate(lines) if title in l)
    rows, header = [], None
    for l in lines[start:]:
        cells = next(csv.reader([l]))
        if header is None and cells and cells[0].strip() == '' and len(cells) > 1:
            header = [c.strip() for c in cells[1:]]
            continue
        if header is not None:
            if not (cells and re.fullmatch(r'\d{6,8}', cells[0].strip())):
                if rows:
                    break
                continue
            rows.append([cells[0].strip()] + [float(c) for c in cells[1:]])
    df = pd.DataFrame(rows, columns=['date'] + header).replace([-99.99, -999], np.nan)
    fmt = '%Y%m%d' if len(rows[0][0]) == 8 else '%Y%m'
    df['date'] = pd.to_datetime(df.date, format=fmt)
    return df.set_index('date') / 100.0


def prepare(raw):
    m = section(raw / FILES[0] / 'F-F_Research_Data_Factors.csv')
    d = section(raw / FILES[1] / 'F-F_Research_Data_Factors_daily.csv')
    p = section(raw / FILES[2] / '10_Portfolios_Prior_1_0.csv', 'Value Weighted Returns -- Monthly')
    monthly = pd.DataFrame({'market': m['Mkt-RF'] + m['RF'], 'rf': m['RF'], 'top_decile': p['Hi PRIOR'], 'bottom_decile': p['Lo PRIOR']})
    daily = pd.DataFrame({'ret': d['Mkt-RF'] + d['RF']})
    daily['adj_close'] = (1 + daily.ret).cumprod()
    OUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT / 'french_monthly.csv')
    daily.reset_index().rename(columns={'index': 'date'}).to_csv(OUT / 'french_daily_market.csv', index=False)
    return monthly, daily.reset_index()


def matrices(returns):
    """V[i, t] = value at end of month t of $1 invested at start of month i."""
    n = len(returns)
    v = np.zeros((n, n))
    g = np.cumprod(1 + returns)
    for i in range(n):
        v[i, i:] = g[i:] / (g[i - 1] if i else 1.0)
    return v


def run(monthly, daily, rule='daily'):
    mon = monthly.dropna(subset=['market', 'top_decile']).copy()
    mon.index = mon.index.to_period('M')
    idx = (1 + mon.market).cumprod()
    flags = crash_flags(rule, idx, daily[['date', 'adj_close']])
    reg = label_regimes(idx, crash=flags)
    state = reg.regime.where(~reg.crash_month, 'post_crash')
    # a purchase in month i is decided on the state at month i-1; first month has no signal
    st = state.shift(1).fillna('normal').to_numpy()
    st = np.where(st == 'crash', 'post_crash', st)
    win = matrices(mon.top_decile.to_numpy())
    spy = matrices(mon.market.to_numpy())
    rf = mon.rf.to_numpy()
    dates = mon.index.to_timestamp()
    out = []
    windows = {'full 1926-2026': (0, len(mon)),
               'before 1998': (0, int((mon.index < pd.Period('1998-01')).sum())),
               '1998 onward': (int((mon.index < pd.Period('1998-01')).sum()), len(mon))}
    for name, (a, b) in windows.items():
        asof = mon.index[b - 1].to_timestamp(how='end').normalize()
        g, base = grid(st[a:b], win[a:b, a:b], spy[a:b, a:b], dates[a:b], rf[a:b], asof, STATES)
        g.insert(0, 'window', name)
        out.append(g)
    return pd.concat(out, ignore_index=True), reg


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fetch', type=Path, help='download into this NEW directory, then use it as --raw')
    ap.add_argument('--raw', type=Path)
    ap.add_argument('--rule', default='daily')
    a = ap.parse_args()
    raw = a.fetch or a.raw
    if a.fetch:
        fetch(a.fetch)
    if raw:
        monthly, daily = prepare(raw)
    else:
        monthly = pd.read_csv(OUT / 'french_monthly.csv', index_col=0, parse_dates=[0])
        daily = pd.read_csv(OUT / 'french_daily_market.csv', parse_dates=['date'])
    res, reg = run(monthly, daily, a.rule)
    res.to_csv(OUT / 'long_history_grid.csv', index=False)
    pd.set_option('display.width', 220, 'display.max_columns', 30)
    print(res[['window', 'state', 'action', 'months_affected', 'episodes', 'money_weighted_return',
               'mwr_vs_baseline', 'placebo_mean_delta', 'placebo_pctile', 'worst_drop_one_episode', 'best_drop_one_episode']].round(4).to_string())
