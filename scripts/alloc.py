"""Shared allocation simulator for the down-market analyses.

A plan assigns each month's deposit one of: winner, spy (the benchmark), cash,
double (2x into the winner). Skipped (cash) deposits earn the monthly risk-free
rate and are released into the next non-cash month's purchase, so a pause is a
delay, not a loss. Deposits are made at the start of the month and marked at
month-end.
"""
import numpy as np
import pandas as pd


def twr_maxdd(values, deposits):
    prev = np.r_[0.0, values[:-1]]
    r = values / np.where(prev + deposits > 0, prev + deposits, np.nan) - 1
    idx = np.cumprod(1 + np.nan_to_num(r))
    return float((idx / np.maximum.accumulate(idx) - 1).min())


def evaluate(plan, win, spy, dates, rf, asof, amount=1000.0):
    """win[i, t] / spy[i, t]: value at month-end t of $1 deposited in month i (0 before i).
    rf[i]: risk-free return for month i (decimal). Returns summary metrics."""
    from reproduce import annual_return
    n = len(plan)
    w = np.zeros(n)
    s = np.zeros(n)
    pend = 0.0
    pend_path = np.zeros(n)
    dep = np.zeros(n)
    for i, act in enumerate(plan):
        dep[i] = amount * (2 if act == 'double' else 1)
        if act == 'cash':
            pend += amount
        else:
            put = dep[i] + pend
            pend = 0.0
            if act == 'spy':
                s[i] = put
            else:
                w[i] = put
        pend *= 1 + rf[i]
        pend_path[i] = pend
    values = (w[:, None] * win).sum(0) + (s[:, None] * spy).sum(0) + pend_path
    final = float(values[-1])
    mw = annual_return(dates, dep, pd.Timestamp(asof), final) if final > 0 else float('nan')
    return dict(ending_value=final, contributed=float(dep.sum()), money_weighted_return=mw,
                twr_max_drawdown=twr_maxdd(values, dep), cash_idle=float(pend))


def clusters(mask, gap=3):
    """Group the True positions of a boolean array into episodes: a new episode starts
    after `gap` or more consecutive False positions. Returns an int array (0 = not flagged)."""
    ids = np.zeros(len(mask), int)
    cur, quiet = 0, gap
    for i, f in enumerate(mask):
        if f:
            if quiet >= gap:
                cur += 1
            quiet = 0
            ids[i] = cur
        else:
            quiet += 1
    return ids


def grid(states, win, spy, dates, rf, asof, state_list, actions=('spy', 'cash', 'double'), placebo=100, seed=0):
    """Override the action in one state at a time (everything else buys the winner).
    Each row also carries leave-one-episode-out robustness for the money-weighted return gain, and a
    placebo: the same action applied to the same NUMBER of randomly chosen months. A rule only means
    something if its gain beats what the swap earns anywhere (e.g. 'spy' always wins if the winner
    asset lags the benchmark on average). placebo_pctile = share of random draws the real rule beat."""
    rng = np.random.default_rng(seed)
    states = np.asarray(states)
    n = len(states)
    base = evaluate(['winner'] * n, win, spy, dates, rf, asof)
    rows = [dict(state='-', action='winner', months_affected=0, episodes=0, **base,
                 mwr_vs_baseline=0.0, worst_drop_one_episode=0.0, best_drop_one_episode=0.0)]
    for st in state_list:
        mask = states == st
        ids = clusters(mask)
        for act in actions:
            plan = np.where(mask, act, 'winner').tolist()
            full = evaluate(plan, win, spy, dates, rf, asof)
            loo = []
            for k in range(1, ids.max() + 1):
                p = np.where(mask & (ids != k), act, 'winner').tolist()
                loo.append(evaluate(p, win, spy, dates, rf, asof)['money_weighted_return'] - base['money_weighted_return'])
            pl = []
            for _ in range(placebo):
                pick = np.zeros(n, bool)
                pick[rng.choice(n, int(mask.sum()), replace=False)] = True
                pl.append(evaluate(np.where(pick, act, 'winner').tolist(), win, spy, dates, rf, asof)['money_weighted_return']
                          - base['money_weighted_return'])
            gain = full['money_weighted_return'] - base['money_weighted_return']
            rows.append(dict(state=st, action=act, months_affected=int(mask.sum()), episodes=int(ids.max()), **full,
                             placebo_mean_delta=float(np.mean(pl)) if pl else float('nan'),
                             placebo_pctile=float(np.mean(np.array(pl) < gain)) if pl else float('nan'),
                             mwr_vs_baseline=full['money_weighted_return'] - base['money_weighted_return'],
                             worst_drop_one_episode=min(loo) if loo else 0.0,
                             best_drop_one_episode=max(loo) if loo else 0.0))
    return pd.DataFrame(rows), base
