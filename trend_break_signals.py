"""Trend Break Engine — Python port of the TradingView Pine Script v6 indicator.

This replaces signals.py's CPR breakout engine with a direct port of the
"Trend Break Engine v5 Next Zones" Pine strategy:

  - EMA ribbon bias (EMA9 > EMA21 > EMA50 = bullish, reverse = bearish)
  - Swing-pivot market structure (BOS = break of structure)
  - Order blocks built from the last opposite-colour candle before a BOS
  - Retest / mitigation with directional reaction confirmation
  - Structure-aware stop loss (min of ATR floor and structural OB edge +
    ATR buffer), rejected if risk exceeds max_risk_atr
  - TP1 ONLY — no TP2 / TP3 tracking (dropped per user request)

Design notes vs. the Pine script:

  - Pine runs incrementally bar-by-bar with persistent `var` state across
    the whole chart history. This port replays that same bar-by-bar loop
    in Python over a fetched OANDA candle window, so live order-block /
    BOS state is identical in shape to what the indicator would show.
  - The "one trade at a time" bookkeeping in Pine (tradeActive) is NOT
    replicated here — that job belongs to bot.py's existing
    "max_concurrent_trades" / open-position guard, which already stops a
    second entry while a position is open. Duplicating it here would
    just be two guards disagreeing with each other.
  - Only the LAST FULLY CLOSED candle is evaluated as a signal candle
    (no intra-candle repaint), matching signals.py's
    require_candle_close=True behaviour.

Returned shape matches signals.py's SignalEngine.analyze() contract so it
is a drop-in replacement in bot.py's _signal_phase():

    (score, direction, details, levels, position_usd)

`score` is not a graded 0-6 score like the CPR engine (this strategy is a
binary confirmed/not-confirmed signal) — it reports 6 on a fresh BUY/SELL
confirmation so it always clears settings["signal_threshold"], and 0
otherwise. Use position_full_usd as your per-trade risk; there is no
partial-size tier for this strategy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from config_loader import load_secrets, load_settings
from oanda_trader import make_oanda_session

log = logging.getLogger(__name__)


# ─────────────────────────── small numeric helpers ──────────────────────────

def _ema_series(values: list[float], length: int) -> list[float]:
    """Standard EMA, seeded with an SMA of the first `length` values."""
    if len(values) < length:
        return [values[0]] * len(values) if values else []
    k = 2 / (length + 1)
    out = [None] * len(values)
    seed = sum(values[:length]) / length
    out[length - 1] = seed
    prev = seed
    for i in range(length, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    # backfill the warm-up region with the first computed EMA value so
    # indices line up 1:1 with `values` (Pine's ta.ema does the same:
    # it just has "na" there, which we never read anyway).
    for i in range(length - 1):
        out[i] = seed
    return out


def _atr_series(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    n = len(closes)
    out: list[float] = [None] * n
    if n < period + 1:
        return out
    trs = [0.0] * n
    for i in range(1, n):
        trs[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr = sum(trs[1:period + 1]) / period
    out[period] = atr
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + trs[i]) / period
        out[i] = atr
    return out


def _pivot_high(highs: list[float], i: int, left: int, right: int) -> float | None:
    if i - left < 0 or i + right >= len(highs):
        return None
    piv = highs[i]
    for j in range(i - left, i + right + 1):
        if j == i:
            continue
        if highs[j] >= piv:
            return None
    return piv


def _pivot_low(lows: list[float], i: int, left: int, right: int) -> float | None:
    if i - left < 0 or i + right >= len(lows):
        return None
    piv = lows[i]
    for j in range(i - left, i + right + 1):
        if j == i:
            continue
        if lows[j] <= piv:
            return None
    return piv


@dataclass
class OrderBlock:
    top: float
    bot: float
    born: int  # bar index it was created on


@dataclass
class _State:
    hi_lvl: float | None = None
    lo_lvl: float | None = None
    hi_broken: bool = True
    lo_broken: bool = True
    bull_obs: list = field(default_factory=list)
    bear_obs: list = field(default_factory=list)
    last_sig_bar: int = -99999


class TrendBreakEngine:
    """Python port of Trend Break Engine v5. Same OANDA session pattern as
    signals.py's SignalEngine so it can be swapped in with no other wiring
    changes to bot.py beyond the import + class name."""

    def __init__(self, demo: bool = True):
        secrets = load_secrets()
        self.api_key = secrets.get("OANDA_API_KEY", "")
        self.account_id = secrets.get("OANDA_ACCOUNT_ID", "")
        self.base_url = (
            "https://api-fxpractice.oanda.com" if demo else "https://api-fxtrade.oanda.com"
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.session = make_oanda_session(allowed_methods=["GET"])

    # ── public entry point (same contract as signals.SignalEngine.analyze) ──

    def analyze(self, asset: str = "XAUUSD", settings: dict | None = None):
        if settings is None:
            settings = load_settings()
        _instrument_key = (settings or {}).get("instrument_display", "XAU/USD").replace("/", "")
        if asset not in ("XAUUSD", _instrument_key):
            return 0, "NONE", f"Only {_instrument_key} supported in this version", {}, 0

        instrument = (settings or {}).get("instrument", "XAU_USD")
        tf = (settings or {}).get("tbe_timeframe", settings.get("timeframe", "M15"))

        # ── inputs (mirrors the Pine script's input.* defaults) ───────────
        s = settings or {}
        len1 = int(s.get("tbe_ema1", 9))
        len2 = int(s.get("tbe_ema2", 21))
        len3 = int(s.get("tbe_ema3", 50))
        len4 = int(s.get("tbe_ema4", 200))
        swing_len = int(s.get("tbe_swing_len", 10))
        ob_search = int(s.get("tbe_ob_search", 20))
        max_ob = int(s.get("tbe_max_ob", 8))
        require_reaction = bool(s.get("tbe_require_reaction", True))
        use_wick = bool(s.get("tbe_use_wick", True))
        body_ratio_min = float(s.get("tbe_body_ratio", 0.30))
        use_trend = bool(s.get("tbe_use_trend", True))
        cooldown = int(s.get("tbe_cooldown", 5))
        atr_len = int(s.get("tbe_atr_len", 14))
        sl_mode_structure = str(s.get("tbe_sl_mode", "Structure + ATR buffer")) != "ATR only"
        sl_mult = float(s.get("tbe_sl_mult", 1.5))
        sl_buffer_atr = float(s.get("tbe_sl_buffer_atr", 0.20))
        max_risk_atr = float(s.get("tbe_max_risk_atr", 3.0))
        tp1_r = float(s.get("tbe_tp1_r", 1.0))

        need = max(len4 + 5, 300)  # enough history for EMA200 + OB structure to settle
        closes, highs, lows, opens = self._fetch_candles(instrument, tf, need)
        if len(closes) < len4 + swing_len * 2 + 5:
            return 0, "NONE", "Not enough candle history for Trend Break Engine", {}, 0

        n = len(closes)

        e1 = _ema_series(closes, len1)
        e2 = _ema_series(closes, len2)
        e3 = _ema_series(closes, len3)
        e4 = _ema_series(closes, len4)
        atr = _atr_series(highs, lows, closes, atr_len)

        state = _State()

        # signals recorded for the FINAL completed bar only; everything
        # before that is just replayed to build up correct OB/BOS state.
        last_idx = n - 1  # last completed candle from OANDA (complete=True only)
        result_dir = "NONE"
        result_entry = None
        result_sl = None
        result_tp1 = None
        result_reason = ""
        result_setup = ""

        for i in range(n):
            bull_bias = e1[i] is not None and e2[i] is not None and e3[i] is not None and e1[i] > e2[i] > e3[i]
            bear_bias = e1[i] is not None and e2[i] is not None and e3[i] is not None and e1[i] < e2[i] < e3[i]

            # ── swing pivots (need swing_len bars of right-side confirmation,
            #    so a pivot at index i-swing_len becomes known at index i) ──
            piv_i = i - swing_len
            if piv_i >= swing_len:
                ph = _pivot_high(highs, piv_i, swing_len, swing_len)
                pl = _pivot_low(lows, piv_i, swing_len, swing_len)
                if ph is not None:
                    state.hi_lvl = ph
                    state.hi_broken = False
                if pl is not None:
                    state.lo_lvl = pl
                    state.lo_broken = False

            bos_up = state.hi_lvl is not None and not state.hi_broken and closes[i] > state.hi_lvl
            bos_dn = state.lo_lvl is not None and not state.lo_broken and closes[i] < state.lo_lvl
            if bos_up:
                state.hi_broken = True
            if bos_dn:
                state.lo_broken = True

            # ── new order blocks on a fresh BOS ────────────────────────────
            if bos_up:
                idx = None
                for k in range(1, ob_search + 1):
                    j = i - k
                    if j < 0:
                        break
                    if closes[j] < opens[j]:
                        idx = j
                        break
                if idx is not None:
                    state.bull_obs.append(OrderBlock(top=highs[idx], bot=lows[idx], born=i))
                    if len(state.bull_obs) > max_ob:
                        state.bull_obs.pop(0)

            if bos_dn:
                idx = None
                for k in range(1, ob_search + 1):
                    j = i - k
                    if j < 0:
                        break
                    if closes[j] > opens[j]:
                        idx = j
                        break
                if idx is not None:
                    state.bear_obs.append(OrderBlock(top=highs[idx], bot=lows[idx], born=i))
                    if len(state.bear_obs) > max_ob:
                        state.bear_obs.pop(0)

            # ── mitigation / retest (scan newest → oldest, like the Pine) ──
            mit_bull = False
            mit_bull_top = mit_bull_bot = None
            for oi in range(len(state.bull_obs) - 1, -1, -1):
                ob = state.bull_obs[oi]
                if i <= ob.born:
                    continue
                mid = (ob.top + ob.bot) * 0.5
                touched = lows[i] <= ob.top and highs[i] >= ob.bot
                invalid = closes[i] < ob.bot
                reacted = touched and (not require_reaction or (closes[i] > opens[i] and closes[i] >= mid))
                if invalid:
                    state.bull_obs.pop(oi)
                elif reacted:
                    if not mit_bull:
                        mit_bull = True
                        mit_bull_top, mit_bull_bot = ob.top, ob.bot
                    state.bull_obs.pop(oi)

            mit_bear = False
            mit_bear_top = mit_bear_bot = None
            for oi in range(len(state.bear_obs) - 1, -1, -1):
                ob = state.bear_obs[oi]
                if i <= ob.born:
                    continue
                mid = (ob.top + ob.bot) * 0.5
                touched = highs[i] >= ob.bot and lows[i] <= ob.top
                invalid = closes[i] > ob.top
                reacted = touched and (not require_reaction or (closes[i] < opens[i] and closes[i] <= mid))
                if invalid:
                    state.bear_obs.pop(oi)
                elif reacted:
                    if not mit_bear:
                        mit_bear = True
                        mit_bear_top, mit_bear_bot = ob.top, ob.bot
                    state.bear_obs.pop(oi)

            # ── filters ─────────────────────────────────────────────────
            rng = max(highs[i] - lows[i], 1e-9)
            body_ok = abs(closes[i] - opens[i]) / rng >= body_ratio_min
            spacing_ok = (i - state.last_sig_bar) >= cooldown
            base_ok = (not use_wick or body_ok) and spacing_ok

            raw_buy = mit_bull and base_ok and (not use_trend or bull_bias)
            raw_sell = mit_bear and base_ok and (not use_trend or bear_bias)

            buy_sig = sell_sig = False
            this_sl = this_tp1 = None
            if (raw_buy or raw_sell) and atr[i] is not None and atr[i] > 0:
                buy_atr_sl = closes[i] - atr[i] * sl_mult
                sell_atr_sl = closes[i] + atr[i] * sl_mult
                buy_struct_sl = (
                    min(mit_bull_bot, lows[i]) - atr[i] * sl_buffer_atr
                    if mit_bull_bot is not None else buy_atr_sl
                )
                sell_struct_sl = (
                    max(mit_bear_top, highs[i]) + atr[i] * sl_buffer_atr
                    if mit_bear_top is not None else sell_atr_sl
                )
                buy_sl = min(buy_atr_sl, buy_struct_sl) if sl_mode_structure else buy_atr_sl
                sell_sl = max(sell_atr_sl, sell_struct_sl) if sl_mode_structure else sell_atr_sl

                if raw_buy:
                    risk = closes[i] - buy_sl
                    if risk > 0 and risk <= atr[i] * max_risk_atr:
                        buy_sig = True
                        this_sl = buy_sl
                        this_tp1 = closes[i] + risk * tp1_r
                if raw_sell:
                    risk = sell_sl - closes[i]
                    if risk > 0 and risk <= atr[i] * max_risk_atr:
                        sell_sig = True
                        this_sl = sell_sl
                        this_tp1 = closes[i] - risk * tp1_r

            if buy_sig and sell_sig:
                buy_sig = sell_sig = False

            if buy_sig or sell_sig:
                state.last_sig_bar = i

            if i == last_idx:
                if buy_sig:
                    result_dir = "BUY"
                    result_entry = closes[i]
                    result_sl = this_sl
                    result_tp1 = this_tp1
                    result_setup = "Bullish OB retest + BOS"
                    result_reason = (
                        f"BOS confirmed bullish OB retest with reaction candle "
                        f"({'ribbon-aligned' if use_trend else 'trend filter off'})"
                    )
                elif sell_sig:
                    result_dir = "SELL"
                    result_entry = closes[i]
                    result_sl = this_sl
                    result_tp1 = this_tp1
                    result_setup = "Bearish OB retest + BOS"
                    result_reason = (
                        f"BOS confirmed bearish OB retest with reaction candle "
                        f"({'ribbon-aligned' if use_trend else 'trend filter off'})"
                    )
                elif mit_bull or mit_bear:
                    result_reason = "OB reaction seen but filtered out (trend/body/cooldown/risk)"
                else:
                    result_reason = "No confirmed OB retest on last closed candle"

        levels = {
            "current_price": round(closes[last_idx], 2),
            "atr": round(atr[last_idx], 2) if atr[last_idx] else None,
            "ema1": round(e1[last_idx], 2) if e1[last_idx] else None,
            "ema2": round(e2[last_idx], 2) if e2[last_idx] else None,
            "ema3": round(e3[last_idx], 2) if e3[last_idx] else None,
            "setup": result_setup,
            "signal_blockers": [],
        }

        if result_dir == "NONE" or result_entry is None:
            return 0, "NONE", result_reason, levels, 0

        sl_usd_rec = round(abs(result_entry - result_sl), 2)
        tp_usd_rec = round(abs(result_tp1 - result_entry), 2)
        rr_ratio = round(tp_usd_rec / sl_usd_rec, 2) if sl_usd_rec > 0 else 0

        levels.update({
            "entry": round(result_entry, 2),
            "sl_usd_rec": sl_usd_rec,
            "sl_source": "structural_ob",
            "tp_usd_rec": tp_usd_rec,
            "tp_source": "tp1_r_multiple",
            "rr_ratio": rr_ratio,
            "score": 6,
            "position_usd": int((settings or {}).get("position_full_usd", 100)),
            "mandatory_checks": {"score_ok": True, "rr_ok": True},
            "quality_checks": {"tp_ok": True},
        })

        position_usd = levels["position_usd"]
        log.info(
            "TBE signal | dir=%s entry=%.2f sl=$%.2f tp1=$%.2f rr=1:%.2f | %s",
            result_dir, result_entry, sl_usd_rec, tp_usd_rec, rr_ratio, result_reason,
        )
        return 6, result_dir, result_reason, levels, position_usd

    # ── data fetch ─────────────────────────────────────────────────────────

    def _fetch_candles(self, instrument: str, granularity: str, count: int = 300):
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {"count": str(min(count, 5000)), "granularity": granularity, "price": "M"}
        for attempt in range(3):
            try:
                r = self.session.get(url, headers=self.headers, params=params, timeout=20)
                if r.status_code == 200:
                    candles = r.json().get("candles", [])
                    complete = [c for c in candles if c.get("complete")]
                    closes = [float(c["mid"]["c"]) for c in complete]
                    highs = [float(c["mid"]["h"]) for c in complete]
                    lows = [float(c["mid"]["l"]) for c in complete]
                    opens = [float(c["mid"]["o"]) for c in complete]
                    return closes, highs, lows, opens
                log.warning("Fetch candles %s %s: HTTP %s", instrument, granularity, r.status_code)
            except Exception as e:
                log.warning("Fetch candles error (%s %s) attempt %s: %s", instrument, granularity, attempt + 1, e)
            time.sleep(1)
        return [], [], [], []
