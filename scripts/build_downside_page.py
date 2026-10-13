"""Generate downside.html (static, offline) from the data/ CSV/JSON outputs.
Run after downside.py, long_history.py and signal.py. Embeds the newest data/live/*/signal.json
(git-ignored); pass --no-live to build the public version without it."""
import html
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data'
pct = lambda x, d=1: f'{x*100:+.{d}f}' if pd.notna(x) else '–'
pts = lambda x: f'{x*100:+.2f} pts' if pd.notna(x) else '–'
usd = lambda x: f'${x:,.0f}'


def chart(reg):
    w, h, l, r, t, b = 1040, 300, 52, 14, 14, 30
    n = len(reg)
    xs = lambda i: l + (w - l - r) * i / (n - 1)
    ys = lambda v: t + (h - t - b) * (-v / 0.6)          # drawdown 0 .. -60%
    parts = []
    for gv in (0, -.2, -.4, -.6):
        parts.append(f'<line x1="{l}" x2="{w-r}" y1="{ys(gv):.1f}" y2="{ys(gv):.1f}" class="grid"/>'
                     f'<text x="{l-8}" y="{ys(gv)+4:.1f}" class="ax" text-anchor="end">{gv*100:.0f}%</text>')
    for yr in range(2000, 2027, 4):
        i = reg.index[reg.month == f'{yr}-01']
        if len(i):
            parts.append(f'<text x="{xs(i[0]):.1f}" y="{h-8}" class="ax" text-anchor="middle">{yr}</text>')
    pts_ = ' '.join(f'{xs(i):.1f},{ys(max(v, -.6)):.1f}' for i, v in enumerate(reg.drawdown))
    shade = []
    for i, row in enumerate(reg.itertuples()):
        if row.drawdown <= -0.10:
            cls = 'bear' if row.drawdown <= -0.20 else 'corr'
            shade.append(f'<rect x="{xs(i)-(w-l-r)/(n-1)/2:.1f}" y="{t}" width="{(w-l-r)/(n-1):.1f}" height="{h-t-b}" class="{cls}"><title>{row.month}: SPY {row.drawdown*100:.1f}% from peak</title></rect>')
    dots = ''.join(f'<circle cx="{xs(i):.1f}" cy="{ys(max(row.drawdown,-.6)):.1f}" r="4" class="crash"><title>{row.month}: crash signal month</title></circle>'
                   for i, row in enumerate(reg.itertuples()) if row.crash_month)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="SPY drawdown from peak with correction and bear regimes shaded" class="chart">'
            + ''.join(shade) + ''.join(parts) + f'<polyline points="{pts_}" class="line"/>' + dots + '</svg>')


def table(df, cols, heads, fmt):
    th = ''.join(f'<th>{h}</th>' for h in heads)
    body = ''.join('<tr>' + ''.join(f'<td>{fmt[c](getattr(r, c)) if c in fmt else html.escape(str(getattr(r, c)))}</td>' for c in cols) + '</tr>'
                   for r in df.itertuples())
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def main(use_live=True):
    live = sorted((D / 'live').glob('*/signal.json')) if use_live else []
    sig = json.loads(live[-1].read_text()) if live else None
    reg = pd.read_csv(D / 'regimes.csv').reset_index(drop=True)
    reg = reg[reg.month >= '1998-01'].reset_index(drop=True)
    summ = json.loads((D / 'downside_summary.json').read_text())
    strat = pd.read_csv(D / 'regime_strategies.csv')
    lh = pd.read_csv(D / 'long_history/long_history_grid.csv')
    lots = pd.read_csv(D / 'regime_lots.csv')

    if sig and sig['pick']:
        rg = sig['regime']
        top = ''.join(f"<li><b>{t['ticker']}</b> {pct(t['prior_month_return'])}%</li>" for t in sig['top5'])
        card = f"""<div class="card"><p class="eyebrow">Rule output · purchase month {sig['purchase_month']} · data as of {sig['latest_price_date']}</p>
<p class="big">Standard deposit into <b>{sig['pick']['ticker']}</b></p>
<p>Top S&amp;P 500 performer in {sig['signal_month_end']}: <b>{pct(sig['pick']['prior_month_return'])}%</b>. Market regime at that month-end: <b>{rg['state'].replace('_',' ')}</b>
(SPY {pct(rg['spy_drawdown_at_signal'])}% from its peak; {pct(rg['spy_drawdown_now'])}% today). Overlay that passed the robustness checks: <b>{sig['robust_overlay'].replace('winner','none')}</b>.</p>
<p class="small">Runners-up: </p><ol class="top">{top}</ol>
<p class="small">Coverage {sig['priced']}/{sig['members']} current members priced ({sig['coverage']*100:.1f}%){'; missing: ' + ', '.join(sig['missing']) if sig['missing'] else ''}. Single data source (Yahoo); not cross-checked. Run <code>python scripts/signal.py</code> then rebuild this page to refresh.</p></div>"""
    elif sig:
        card = f"<div class=\"card warn\"><p class=\"big\">No output</p><p>{html.escape(sig['status'])}. Coverage {sig['coverage']*100:.1f}%; missing {len(sig['missing'])} tickers.</p></div>"
    else:
        card = '<div class="card warn"><p class="big">No live data yet</p><p>Run <code>python scripts/signal.py</code>.</p></div>'

    sfmt = dict(mwr_vs_baseline=pts, worst_drop_one_episode=pts, money_weighted_return=lambda x: f'{x*100:.2f}%',
                placebo_pctile=lambda x: f'{x*100:.0f}%' if pd.notna(x) else '–', months_affected=lambda x: f'{x:.0f}' if pd.notna(x) else '–',
                episodes=lambda x: f'{x:.0f}' if pd.notna(x) else '–', twr_max_drawdown=lambda x: f'{x*100:.0f}%',
                robust=lambda x: 'yes' if x is True else 'no')
    s = strat[strat.state.isin(['correction', 'bear', 'post_crash', '-'])].copy()
    s['robust'] = s.robust.map({True: True, False: False})
    t1 = table(s, ['strategy', 'months_affected', 'episodes', 'money_weighted_return', 'mwr_vs_baseline', 'worst_drop_one_episode', 'placebo_pctile', 'twr_max_drawdown', 'robust'],
               ['Override', 'Months', 'Episodes', 'Money-weighted return', 'Gain vs baseline', 'Worst case, drop 1 episode', 'Beat random months', 'Max drawdown', 'Robust?'], sfmt)
    l2 = lh[lh.state != '-'].copy()
    t2 = table(l2, ['window', 'state', 'action', 'months_affected', 'episodes', 'mwr_vs_baseline', 'placebo_mean_delta', 'placebo_pctile', 'worst_drop_one_episode'],
               ['Window', 'State', 'Action', 'Months', 'Episodes', 'Gain vs baseline', 'Same swap, random months', 'Beat random', 'Worst, drop 1 episode'],
               dict(sfmt, placebo_mean_delta=pts))
    t3 = table(lots, ['signal_state', 'purchases', 'median_multiple', 'median_spy_multiple', 'beat_spy', 'written_off'],
               ['State at signal', 'Purchases', 'Median multiple', 'SPY median multiple', 'Beat SPY', 'Written off'],
               dict(median_multiple=lambda x: f'{x:.1f}x', median_spy_multiple=lambda x: f'{x:.1f}x', beat_spy=lambda x: f'{x*100:.0f}%'))
    cm = summ['crash_months_by_rule']
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Downside months</title><meta name="description" content="How the monthly-winner backtest behaves in corrections, bear markets and crashes, plus the current rule output.">
<style>
:root{{--paper:#f7f6f0;--ink:#172c29;--muted:#536460;--line:#d9ded4;--green:#007a61;--blue:#5877b1;--gold:#ba8033;--soft:#edf0e5;--amber:#f7ead3;--corr:#e9e4c9;--bear:#e3c9b8;--crash:#b3412e}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--paper:#101a18;--ink:#e6ece8;--muted:#9bb0aa;--line:#2b3b37;--green:#43c4a2;--blue:#8fa9d6;--gold:#d6a15f;--soft:#1a2a26;--amber:#2c2517;--corr:#3a3820;--bear:#4a2f22;--crash:#e8806c}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:auto;padding:0 16px 60px}}@media(min-width:700px){{main{{padding:0 32px 60px}}}}
h1{{font:normal clamp(34px,5vw,56px)/1.1 Georgia,serif;letter-spacing:-.03em;margin:48px 0 18px}}h2{{font:normal 30px/1.2 Georgia,serif;margin:0 0 14px}}
.section{{padding:44px 0;border-top:1px solid var(--line)}}.lead{{font-size:20px;color:var(--muted);max-width:780px}}.small{{font-size:14px;color:var(--muted)}}
.note{{padding:18px 22px;background:var(--amber);border-left:3px solid var(--gold);margin:0 0 28px}}.card{{border:1px solid var(--line);border-left:4px solid var(--green);padding:22px 26px;background:var(--soft);border-radius:4px}}
.card.warn{{border-left-color:var(--crash)}}.big{{font:normal 30px/1.2 Georgia,serif;margin:6px 0 12px}}.eyebrow{{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0}}
ol.top{{display:flex;flex-wrap:wrap;gap:6px 22px;padding-left:20px;margin:4px 0 14px}}.chart{{width:100%;height:auto;display:block}}.grid{{stroke:var(--line);stroke-width:1}}.ax{{fill:var(--muted);font-size:12px}}
.line{{fill:none;stroke:var(--blue);stroke-width:2}}rect.corr{{fill:var(--corr)}}rect.bear{{fill:var(--bear)}}.crash{{fill:var(--crash);stroke:var(--paper);stroke-width:2}}
.key{{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}}.key i{{display:inline-block;width:14px;height:14px;vertical-align:-2px;margin-right:6px;border:1px solid var(--line)}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:600}}
code{{font-size:.9em;background:var(--soft);padding:1px 5px;border-radius:3px}}a{{color:inherit}}li{{margin-bottom:6px}}
</style></head><body><main>
<p class="small" style="margin-top:22px"><a href="index.html">← Back to the main report</a></p>
<h1>What would the rule do in a down month?</h1>
<p class="lead">The original experiment buys the previous month's best S&amp;P 500 performer every month. This page asks whether corrections, bear markets or crashes should change that, and shows the rule's output for the latest data.</p>
<div class="note"><b>Not investment advice.</b> This is a research tool. The card below is the mechanical output of a backtested rule, not a recommendation to buy or sell anything, and it does not account for your circumstances, taxes or costs. Past results do not predict future ones.</div>
<section class="section" style="border-top:0;padding-top:0"><h2>Rule output for the latest data</h2>{card}</section>
<section class="section"><h2>The short answer</h2>
<p><b>Nothing tested beat simply buying the winner every month.</b> Across {summ['crash_episodes']} crash episodes, {summ['correction_months']} correction months and {summ['bear_months']} bear months (1998–2026), no regime override survived the checks below, so the rule's output is the standard deposit in every regime.</p>
<ul><li><b>Doubling down</b> looked good after crashes (+0.85 pts a year) but nearly all of it came from one purchase, Nvidia in March 2003. Without that episode the gain turns negative.</li>
<li><b>Pausing buys</b> in bear markets, with the money earning T-bill rates then invested next month, shows +3.0 pts, but it also fails when one episode is removed. It looks like timing luck, not a pattern.</li>
<li><b>Switching to SPY</b> lowers the maximum drawdown by a few points in bear months but does not reliably improve returns.</li>
<li>In 1926–1997 data for a diversified "last month's top decile" portfolio, swapping to the market during bear markets and after crashes did beat random months, by about 1 point a year. That did not hold after 1998 and the proxy is not a single stock, so it is a hint, not a rule.</li></ul></section>
<section class="section"><h2>When the market was down</h2>
<p class="small">SPY drawdown from its running peak at each month-end. Shaded: correction (10%+ below peak) and bear (20%+). Dots: months flagged as a crash (a one-day fall of 5%+ or a 10%+ fall within 20 trading days). Flagged months by definition: monthly -10% {cm['m10']}, -7% {cm['m7']}, -5% {cm['m5']}, daily -5% {cm['d5']}, 20-day -10% {cm['r20_10']}, combined {cm['daily']}.</p>
{chart(reg)}
<div class="key"><span><i style="background:var(--corr)"></i>Correction</span><span><i style="background:var(--bear)"></i>Bear</span><span><i style="background:var(--crash);border-radius:50%"></i>Crash month</span><span><i style="background:var(--blue)"></i>SPY drawdown</span></div></section>
<section class="section"><h2>What the winner did when bought in each regime</h2>
<p class="small">Regime is measured at the month-end before each purchase. Multiples are the value of $1,000 today per $1,000 spent.</p>{t3}</section>
<section class="section"><h2>Each override, tested</h2>
<p class="section-intro small">One regime changed at a time; every other month buys the winner. "Robust" requires a positive gain, still positive after dropping any single episode, and better than 90% of random same-size sets of months. Paused deposits earn T-bill rates and are invested next month. Money-weighted return; baseline 17.84%. 235 of 346 rankings in the underlying data are incomplete.</p>{t1}</section>
<section class="section"><h2>Longer history: 1926–2026</h2>
<p class="small">Ken French's data: total-return market (the "SPY" side) and the top decile of stocks by prior one-month return (the "winner" side, a diversified proxy). "Same swap, random months" is what the action earns when applied to the same number of random months, so a rule only counts if it beats that.</p>{t2}</section>
<section class="section"><h2>Caveats</h2><ul class="small"><li>The monthly data has no intra-month detail except for the daily SPY series used to flag crashes.</li><li>The stock-level results rest on a handful of purchases (Nvidia, AMD, Micron) and an incomplete candidate universe.</li><li>The live pick uses current S&amp;P 500 members from a public list and Yahoo prices; verify before relying on it.</li><li>No taxes, costs or slippage. Cash earns the T-bill rate; fractional shares are assumed.</li></ul></section>
</main></body></html>"""
    (ROOT / 'downside.html').write_text(page)
    print('wrote downside.html', len(page))


if __name__ == '__main__':
    import sys
    main('--no-live' not in sys.argv)
