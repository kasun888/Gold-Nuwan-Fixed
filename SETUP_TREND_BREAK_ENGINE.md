# Trend Break Engine — automation setup

This replaces the bot's CPR breakout signal source with a Python port of
your `Trend Break Engine v5` Pine script (order blocks, BOS, retest
confirmation, structural SL, **TP1 only** — no TP2/TP3, as requested).
Everything else — OANDA execution, session windows, daily/session loss
caps, spread guard, margin guard, break-even, Telegram alerts, SQLite
logging — is untouched.

## Files

- `trend_break_signals.py` — new module, drop into the repo root next to `signals.py`.
- `bot.py` — patched copy. Two changes only:
  1. Import swap: `SignalEngine` now points at `TrendBreakEngine` instead of the CPR engine.
  2. `compute_sl_usd()` gained a `"signal_native"` mode that trusts the engine's own structural stop instead of recomputing a generic ATR stop.
- `settings_additions.json` — keys to add/overwrite in your `settings.json`.

## ⚠️ One setting you must not skip

Your Pine script's default `tp1R = 1.0` means TP1 sits at **1×** the stop
distance (a 1:1 risk:reward trade — TP1 is meant to be the *first* target,
not the whole trade). The existing bot has a hard RR gate:

```python
if rr_ratio < _min_rr:   # _min_rr comes from settings["rr_ratio"], default 2.65
    # skip trade
```

Since the CPR engine's default `rr_ratio` is `2.65`, if you don't change
it, **every single Trend Break Engine signal will be silently skipped** —
the bot will run forever showing "WATCHING" and never place a trade.

Set `"rr_ratio"` in `settings.json` to be **at or below** whatever you set
`"tbe_tp1_r"` to (both default to `1.0` in `settings_additions.json`). If
you want a tighter RR filter, raise `tbe_tp1_r` (which changes where TP1
actually sits, matching the Pine input) rather than leaving `tbe_tp1_r` at
1.0 and raising `rr_ratio` — the latter will just cause the gate to reject
everything again.

## Steps

1. Copy `trend_break_signals.py` into your repo root.
2. Replace `bot.py` with the patched version (or apply the two diffs above by hand if you've since modified `bot.py`).
3. Merge `settings_additions.json`'s keys into your live `settings.json` (on Railway this is the persistent-volume copy, not the bundled default).
4. Set `demo_mode: true` and deploy. Watch Telegram/logs for a few sessions before considering live money.
5. Tune `tbe_swing_len`, `tbe_ob_search`, `tbe_sl_mult`, `tbe_tp1_r` etc. to match how you've been running the indicator on TradingView.

## What's intentionally different from the CPR engine

- **No graded 4/5/6 scoring.** Trend Break Engine is a binary "confirmed retest or not" signal, so it reports a fixed score of `6` on every confirmed BUY/SELL (always clears `signal_threshold`) and sizes every trade at `position_full_usd` — there's no partial-size tier.
- **No H1/H4 EMA or ADX trend filters.** Those were CPR-engine-specific. Your Pine script's own EMA-ribbon bias check (`tbe_use_trend`) plays that role here.
- **No internal "one trade at a time" bookkeeping.** The Pine script's `tradeActive` state doesn't need porting — `bot.py`'s existing `max_concurrent_trades` / open-position guard already blocks a second entry while one is open, so duplicating it here would just be two guards second-guessing each other.
- **Timeframe is explicit.** Pine ran on whatever chart timeframe you had open; the Python port fetches one fixed OANDA granularity (`tbe_timeframe`, default `M15`) since there's no "current chart" in an automated loop.

## Before risking real money

This is a first-pass, line-by-line port — I ran it against synthetic
price data to confirm the BOS → order-block → mitigation → signal
pipeline executes correctly end-to-end, but I have **not** backtested it
against real OANDA history or cross-checked its outputs bar-for-bar
against the actual Pine indicator on TradingView. Do that comparison
yourself before going anywhere near live funds: run both side by side on
the same historical window and confirm the BUY/SELL bars, SL, and TP1
line up. I'm not able to give trading or financial advice on whether the
strategy itself is sound — only on the engineering.
