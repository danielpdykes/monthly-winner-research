"""Down-market (correction / bear / crash) analysis of the monthly-winner backtest.

Monthly resolution only: the frozen inputs hold month-start and month-end quotes,
not daily history. Regimes come from SPY month-end drawdowns and are known at the
signal date (no look-ahead) when used to choose that month's purchase.
Reads the frozen archive plus reproduced/ (run scripts/reproduce.py --allow-incomplete
first). Writes data/regimes.csv, data/regime_lots.csv, data/episodes.csv,
data/regime_strategies.csv and data/downside_summary.json. Same incomplete-ranking
caveat as the base report applies to every number here.
"""
import argparse
import json
import zipfile
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from alloc import clusters, evaluate, grid

ROOT = Path(__file__).resolve().parents[1]
CORRECTION, BEAR, CRASH = -0.10, -0.20, -0.10   # drawdown / single-month thresholds
ACTIONS = ('winner', 'spy', 'cash', 'double')
STATES = ('correction', 'bear', 'post_crash')
NAMED = [('Dot-com bust', '2000-03', '2002-10'), ('Global financial crisis', '2007-10', '2009-03'),
         ('2011 debt-ceiling', '2011-05', '2011-10'), ('Q4 2018 selloff', '2018-10', '2018-12'),
         ('COVID crash', '2020-02', '2020-03'), ('2022 bear market', '2022-01', '2022-10')]


RULES = ('m10', 'm7', 'm5', 'd5', 'r20_10', 'daily')   # 'daily' = d5 or r20_10


def crash_flags(rule, spy_me, daily=None):
    """Bool Series by month: did a crash/sharp-drop event occur in that month?
    m<N>: calendar-month SPY fall of N% or more; d5: any daily fall of 5% or more;
    r20_10: 20-trading-day drawdown of 10% or more; daily: d5 or r20_10."""
    if rule.startswith('m'):
        return spy_me.pct_change() <= -int(rule[1:]) / 100
    px = daily.set_index('date').adj_close
    d5 = (px.pct_change() <= -0.05)
    r20 = (px / px.rolling(20).max() - 1) <= -0.10
    flag = {'d5': d5, 'r20_10': r20, 'daily': d5 | r20}[rule]
    return flag.groupby(flag.index.to_period('M')).any().reindex(spy_me.index, fill_value=False)


def label_regimes(spy, correction=CORRECTION, bear=BEAR, crash=None):
    """spy: month-end adjusted closes indexed by Period('M'). crash: optional bool Series of crash months."""
    peak = spy.cummax()
    dd = spy / peak - 1
    ret = spy.pct_change()
    state = pd.Series('normal', index=spy.index)
    state[dd <= correction] = 'correction'
    state[dd <= bear] = 'bear'
    out = pd.DataFrame({'spy_close': spy, 'spy_return': ret, 'drawdown': dd, 'regime': state})
    out['crash_month'] = (ret <= CRASH) if crash is None else crash
    return out


def signal_state(reg):
    """Regime known at month-end t, used to decide the purchase made in month t+1."""
    # a crash month inside a bear market is both; the crash flag wins as the sharper event
    return reg.regime.replace({'crash': 'post_crash'}).where(~reg.crash_month, 'post_crash')


def load(archive):
    prefix = 'outputs/backtest_1998/'
    with zipfile.ZipFile(archive) as z:
        read = lambda n, **k: pd.read_csv(BytesIO(z.read(prefix + n)), **k)
        marks = read('price_marks.csv', index_col=0, parse_dates=[0])
        ends = read('price_endpoints.csv', parse_dates=['date'])
        last = read('terminal_quotes.csv', parse_dates=['date']).set_index('ticker')
    return marks, ends, last


def lot_paths(ledger, monthly, marks, last, spy_open):
    """Value at each month-end of a $1 deposit into that month's winner / SPY."""
    ends = pd.to_datetime(monthly.date)
    n = len(ledger)
    win = np.zeros((n, n))
    spy = np.zeros((n, n))
    spy_close = marks['SPY']
    for i, lot in enumerate(ledger.itertuples()):
        units = 1.0 / lot.entry_adj_open
        spy_units = 1.0 / spy_open.loc[pd.Timestamp(lot.buy_date)]
        for j in range(i, n):
            e = ends[j]
            if last.loc[lot.ticker, 'date'] >= e:
                win[i, j] = units * marks.loc[e, lot.ticker]
            spy[i, j] = spy_units * spy_close.loc[e]
    return win, spy


def episodes(reg):
    rows = []
    for name, a, b in NAMED:
        r = reg.loc[a:b]
        rows.append(dict(episode=name, start=a, end=b, spy_drawdown_trough=float(r.drawdown.min()),
                         worst_month=float(r.spy_return.min()), months=len(r)))
    return pd.DataFrame(rows)


def strategy_grid(ledger, win, spy, dates, rf, asof, placebo=100):
    """Override the action in one state at a time (rest buy the winner), with leave-one-episode-out
    and placebo checks. 'Robust' = positive gain, still positive after dropping any one episode,
    and better than 90% of random same-size month sets."""
    g, base = grid(ledger.signal_state.to_numpy(), win, spy, dates, rf, asof, STATES, placebo=placebo)
    g.insert(0, 'strategy', [f'{a} when {s}' if s != '-' else 'baseline: always buy winner' for s, a in zip(g.state, g.action)])
    g['vs_baseline_value'] = g.ending_value - base['ending_value']
    g['robust'] = (g.mwr_vs_baseline > 0) & (g.worst_drop_one_episode > 0) & (g.placebo_pctile >= 0.9)
    best = {}
    for st in STATES:
        ok = g[(g.state == st) & g.robust].sort_values('mwr_vs_baseline', ascending=False)
        best[st] = ok.action.iloc[0] if len(ok) else 'winner'
    plan = [best.get(x, 'winner') for x in ledger.signal_state]
    combo = evaluate(plan, win, spy, dates, rf, asof)
    g = pd.concat([g, pd.DataFrame([dict(strategy='combined: robust actions only', state=json.dumps(best), action='mixed',
                   **combo, mwr_vs_baseline=combo['money_weighted_return'] - base['money_weighted_return'],
                   vs_baseline_value=combo['ending_value'] - base['ending_value'])])], ignore_index=True)
    return g, best, base


def top_lots(ledger, win, state, n=5):
    """Which purchases would carry an extra $1,000 into the winner in this state."""
    extra = pd.Series(win[:, -1] * 1000.0, index=ledger.index)[ledger.signal_state == state].sort_values(ascending=False)
    return [dict(buy_date=ledger.buy_date[i], ticker=ledger.ticker[i], extra_ending_value=float(v)) for i, v in extra.head(n).items()], \
        float(extra.head(3).sum() / extra.sum())


def sweep(ledger, win, spy, dates, rf, asof, spy_me, daily):
    """Re-run the per-state grid across crash definitions and correction/bear thresholds."""
    months = pd.PeriodIndex(pd.to_datetime(ledger.buy_date), freq='M')
    rows = []
    for rule in RULES:
        flags = crash_flags(rule, spy_me, daily)
        for corr, bear in [(-0.07, -0.15), (-0.10, -0.20), (-0.15, -0.25)]:
            sb = signal_state(label_regimes(spy_me, corr, bear, flags))
            led = ledger.assign(signal_state=[sb.get(m - 1, 'normal') for m in months])
            g, _, _ = strategy_grid(led, win, spy, dates, rf, asof, placebo=0)
            for r in g[g.state.isin(STATES)].itertuples():
                rows.append(dict(crash_rule=rule, correction=corr, bear=bear, state=r.state, action=r.action,
                                 months=r.months_affected, episodes=r.episodes, money_weighted_return=r.money_weighted_return,
                                 mwr_vs_baseline=r.mwr_vs_baseline, worst_drop_one_episode=r.worst_drop_one_episode,
                                 twr_max_drawdown=r.twr_max_drawdown))
    return pd.DataFrame(rows)


def load_rf(ledger):
    """Monthly T-bill rate for each purchase month (French RF; last known value carried forward)."""
    f = pd.read_csv(ROOT / 'data/long_history/french_monthly.csv', index_col=0, parse_dates=[0])
    f.index = f.index.to_period('M')
    months = pd.PeriodIndex(pd.to_datetime(ledger.buy_date), freq='M')
    return f.rf.reindex(months).ffill().to_numpy()


def run(archive, reproduced, out, daily_path, rule='daily'):
    marks, ends, last = load(archive)
    ledger = pd.read_csv(reproduced / 'purchase_ledger.csv')
    monthly = pd.read_csv(reproduced / 'monthly_results.csv')
    spy_e = ends[ends.ticker == 'SPY'].sort_values('date')
    spy_me = spy_e.groupby(spy_e.date.dt.to_period('M')).adj_close.last()
    daily = pd.read_csv(daily_path, parse_dates=['date'])
    reg = label_regimes(spy_me, crash=crash_flags(rule, spy_me, daily))
    state_by_month = signal_state(reg)
    # purchase i in month m is decided on the state at the end of month m-1
    months = pd.PeriodIndex(pd.to_datetime(ledger.buy_date), freq='M')
    ledger['signal_state'] = [state_by_month.get(m - 1, 'normal') for m in months]
    win, spy = lot_paths(ledger, monthly, marks, last, spy_e.set_index('date').adj_open)
    dates = ledger.buy_date

    asof = pd.Timestamp('2026-10-09')
    rf = load_rf(ledger)
    base = evaluate(['winner'] * len(ledger), win, spy, dates, rf, asof)
    assert abs(base['ending_value'] - monthly.strategy_marked_value.iloc[-1]) < 0.01, 'baseline does not reconcile'

    # per-regime lot statistics (what the winner buy actually did when bought in each state)
    ledger['multiple'] = ledger.ending_marked_value / ledger.contribution
    ledger['spy_multiple'] = ledger.spy_ending_value / ledger.contribution
    lots = ledger.groupby('signal_state').agg(
        purchases=('ticker', 'size'), median_multiple=('multiple', 'median'),
        mean_multiple=('multiple', 'mean'), median_spy_multiple=('spy_multiple', 'median'),
        beat_spy=('multiple', lambda s: float((s > ledger.loc[s.index, 'spy_multiple']).mean())),
        written_off=('status', lambda s: int(s.str.startswith('written').sum())),
        mean_prior_month_return=('previous_month_return', 'mean')).reset_index()

    strat, best, base = strategy_grid(ledger, win, spy, dates, rf, asof)

    out.mkdir(parents=True, exist_ok=True)
    reg.assign(month=reg.index.astype(str)).reset_index(drop=True).to_csv(out / 'regimes.csv', index=False)
    lots.to_csv(out / 'regime_lots.csv', index=False)
    episodes(reg).to_csv(out / 'episodes.csv', index=False)
    strat.to_csv(out / 'regime_strategies.csv', index=False)
    sweep(ledger, win, spy, dates, rf, asof, spy_me, daily).to_csv(out / 'regime_sweep.csv', index=False)
    top, top3 = top_lots(ledger, win, 'post_crash')
    n_ep = int(clusters(crash_flags(rule, spy_me, daily).to_numpy()).max())
    cm = {r: int(crash_flags(r, spy_me, daily).sum()) for r in RULES}
    summary = dict(months=len(reg), correction_months=int((reg.drawdown <= CORRECTION).sum()),
                   bear_months=int((reg.drawdown <= BEAR).sum()), crash_rule=rule, crash_months=int(reg.crash_month.sum()), crash_months_by_rule=cm, crash_episodes=n_ep, top3_share_of_double_uplift=top3, top_double_lots=top,
                   purchases_by_state=ledger.signal_state.value_counts().to_dict(), robust_action_by_state=best,
                   caveat='In-sample, 3 states, few independent episodes; 235/346 rankings incomplete.')
    (out / 'downside_summary.json').write_text(json.dumps(summary, indent=2))
    return summary, lots, strat


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, default=ROOT / 'data/Backtest_Data_and_Code.zip')
    ap.add_argument('--reproduced', type=Path, default=ROOT / 'reproduced')
    ap.add_argument('--daily', type=Path, default=ROOT / 'data/spy_daily.csv')
    ap.add_argument('--rule', choices=RULES, default='daily')
    ap.add_argument('--output', type=Path, default=ROOT / 'data')
    a = ap.parse_args()
    s, lots, strat = run(a.archive, a.reproduced, a.output, a.daily, a.rule)
    pd.set_option('display.width', 200, 'display.max_columns', 20)
    print(json.dumps(s, indent=2)); print(lots.round(3)); print(strat.drop(columns=['state']).round(3))
    
