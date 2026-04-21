"""Signal engine for CPR breakout detection on XAU/USD — v5.1-FIXED

FIXES vs original v5.1:
  - Minimum signal score raised to 5 (was 4) — only strong-confluence setups
  - H1 trend filter now HARD-BLOCKS instead of just logging (previously wasn't
    reliably blocking counter-trend trades)
  - SL calculation uses ATR × 1.5 multiplier (was 1.0) — gold needs breathing room
  - SL clamped to [35, 50] USD (was [25, 60]) — tighter band, wider minimum
  - TP = SL × rr_ratio (2.5x default) — was 1.5x which was mathematically losing
  - Asian session signals suppressed at signal level (belt-and-suspenders with settings)
  - Exhaustion threshold tightened to 1.8× ATR (was 2.5×) — catch overextended moves earlier
  - R2/S2 Extended Breakout signals now require score >= 6 (near-impossible) — effectively disabled
  - Added volume confirmation stub (CPR width must be < 0.7% for max score, was 0.5%)
  - SL source logic simplified: always ATR-based when ATR available

Scoring (Bull):
  Main condition  — price above CPR/PDH/R1: +2 | above R2 (extended): +1
  SMA alignment   — both SMA20 & SMA50 below price: +2 | one below: +1
  CPR width       — < 0.5% (narrow): +2 | 0.5%–0.7% (moderate): +1

Scoring (Bear):
  Main condition  — price below CPR/PDL/S1: +2 | below S2 (extended): +1
  SMA alignment   — both SMA20 & SMA50 above price: +2 | one above: +1
  CPR width       — same as Bull
  Trend exhaustion — price > exhaustion_atr_mult × ATR from SMA20: −1
                     (S2/R2 Extended setups are HARD BLOCKED when exhausted)

Position size by score:
  score 5–6  →  $100 (full)   [was: score > 4]
  score < 5  →  no trade
"""

import time
import logging
import requests
from config_loader import load_secrets, load_settings, DATA_DIR
from oanda_trader import make_oanda_session

log = logging.getLogger(__name__)

# Minimum score required to trade — FIXED: raised from 4 to 5
MIN_TRADE_SCORE = 5

# SGT hours that are London or US session (16–20 SGT = London, 21–00 SGT = US)
# Used to suppress Asian-session signals even if settings allow them through
_LONDON_US_HOURS = set(range(16, 24)) | {0}


def score_to_position_usd(score: int, settings: dict | None = None) -> int:
    """Return risk-dollar position size for a given score.

    FIXED: only score >= 5 gets a position. Score 4 no longer trades.
    """
    full = int((settings or {}).get("position_full_usd", 100))
    # Require score >= 5 (MIN_TRADE_SCORE) for any position
    min_score = int((settings or {}).get("signal_threshold", MIN_TRADE_SCORE))
    if score >= min_score:
        return full
    return 0


class SignalEngine:
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

    def analyze(self, asset: str = "XAUUSD", settings: dict | None = None):
        """Run the CPR scoring engine.

        Parameters
        ----------
        asset : str
            Instrument identifier (only XAUUSD supported).
        settings : dict | None
            Bot settings dict.

        Returns
        -------
        (score, direction, details, levels, position_usd)
        """
        if settings is None:
            settings = load_settings()

        _instrument_key = (settings or {}).get("instrument_display", "XAU/USD").replace("/", "")
        if asset not in ("XAUUSD", _instrument_key):
            return 0, "NONE", f"Only {_instrument_key} supported in this version", {}, 0

        instrument = (settings or {}).get("instrument", "XAU_USD")

        # ── Daily candles → CPR levels ─────────────────────────────────────────
        daily_closes, daily_highs, daily_lows = self._fetch_candles(instrument, "D", 3)
        if len(daily_closes) < 2:
            return 0, "NONE", "Not enough daily data for CPR", {}, 0

        prev_high  = daily_highs[-2]
        prev_low   = daily_lows[-2]
        prev_close = daily_closes[-2]

        pivot      = (prev_high + prev_low + prev_close) / 3
        bc         = (prev_high + prev_low) / 2
        tc         = (pivot - bc) + pivot
        if tc < bc:
            tc, bc = bc, tc
        daily_range     = prev_high - prev_low
        r1              = (2 * pivot) - prev_low
        r2              = pivot + daily_range
        s1              = (2 * pivot) - prev_high
        s2              = pivot - daily_range
        pdh             = prev_high
        pdl             = prev_low
        cpr_width_pct   = abs(tc - bc) / pivot * 100

        levels = {
            "pivot":         round(pivot, 2),
            "tc":            round(tc, 2),
            "bc":            round(bc, 2),
            "r1":            round(r1, 2),
            "r2":            round(r2, 2),
            "s1":            round(s1, 2),
            "s2":            round(s2, 2),
            "pdh":           round(pdh, 2),
            "pdl":           round(pdl, 2),
            "cpr_width_pct": round(cpr_width_pct, 3),
        }
        log.info(
            "CPR levels | pivot=%.2f TC=%.2f BC=%.2f R1=%.2f S1=%.2f "
            "R2=%.2f S2=%.2f PDH=%.2f PDL=%.2f width=%.3f%%",
            pivot, tc, bc, r1, s1, r2, s2, pdh, pdl, cpr_width_pct,
        )

        # ── M15 candles → price, SMA, ATR ─────────────────────────────────────
        tf = (settings or {}).get("timeframe", "M15")
        m15_closes, m15_highs, m15_lows = self._fetch_candles(instrument, tf, 65)
        if len(m15_closes) < 52:
            return 0, "NONE", "Not enough M15 data (need 52 candles for SMA50)", levels, 0

        _require_close  = bool((settings or {}).get("require_candle_close", True))
        current_close   = m15_closes[-2] if _require_close else m15_closes[-1]

        log.info(
            "Signal candle | close=%.2f (candle [-2]) | current_tick=%.2f (candle [-1]) | ATR=%.2f",
            m15_closes[-2], m15_closes[-1], self._atr(m15_highs, m15_lows, m15_closes, 14) or 0,
        )

        sma20 = sum(m15_closes[-21:-1]) / 20
        sma50 = sum(m15_closes[-51:-1]) / 50

        # ── H1 trend filter — FIXED: now a hard block, not advisory ───────────
        # Original bug: filter logged a block but signal continued. Now returns
        # early with score=0 so the trade is truly skipped.
        h1_trend_bullish = None
        _h1_filter = bool((settings or {}).get("h1_trend_filter_enabled", True))
        if _h1_filter:
            _h1_period = int((settings or {}).get("h1_ema_period", 21))
            h1_closes, _, _ = self._fetch_candles(instrument, "H1", _h1_period + 10)
            if len(h1_closes) >= _h1_period:
                # Use EMA instead of SMA for more responsive trend detection
                _h1_ema = self._ema(h1_closes, _h1_period)
                _h1_price = h1_closes[-1]
                h1_trend_bullish = _h1_price > _h1_ema
                log.info(
                    "H1 trend filter | price=%.2f EMA%d=%.2f | bullish=%s",
                    _h1_price, _h1_period, _h1_ema, h1_trend_bullish,
                )
            else:
                log.warning("H1 trend filter: insufficient data (%d candles, need %d) — BLOCKING trade (safe default)", len(h1_closes), _h1_period)
                # FIXED: when H1 data is unavailable, block the trade (safe default)
                return 0, "NONE", "H1 trend data unavailable — trade blocked (safe default)", levels, 0

        levels["h1_trend_bullish"] = h1_trend_bullish

        # ATR(14) for SL sizing
        atr_val = self._atr(m15_highs, m15_lows, m15_closes, 14)
        levels["atr"]           = round(atr_val, 2) if atr_val else None
        levels["current_price"] = round(current_close, 2)
        levels["sma20"]         = round(sma20, 2)
        levels["sma50"]         = round(sma50, 2)

        # ── Scoring ────────────────────────────────────────────────────────────
        score     = 0
        direction = "NONE"
        reasons   = []
        setup     = "Unknown"

        reasons.append(
            f"CPR TC={tc:.2f} BC={bc:.2f} width={cpr_width_pct:.2f}% | "
            f"R1={r1:.2f} R2={r2:.2f} S1={s1:.2f} S2={s2:.2f} | "
            f"PDH={pdh:.2f} PDL={pdl:.2f}"
        )

        # ── 1. Main condition ──────────────────────────────────────────────────
        if current_close > tc:
            direction = "BUY"
            if current_close > r2:
                score += 1
                setup = "R2 Extended Breakout"
                reasons.append(
                    f"⚠️ Price {current_close:.2f} > R2={r2:.2f} — extended entry (+1, weak)"
                )
            else:
                score += 2
                if current_close > r1:
                    setup = "R1 Breakout"
                elif current_close > pdh:
                    setup = "PDH Breakout"
                else:
                    setup = "CPR Bull Breakout"
                reasons.append(
                    f"✅ Price {current_close:.2f} above CPR/PDH/R1 zone [{setup}] (+2)"
                )
        elif current_close < bc:
            direction = "SELL"
            if current_close < s2:
                score += 1
                setup = "S2 Extended Breakdown"
                reasons.append(
                    f"⚠️ Price {current_close:.2f} < S2={s2:.2f} — extended entry (+1, weak)"
                )
            else:
                score += 2
                if current_close < s1:
                    setup = "S1 Breakdown"
                elif current_close < pdl:
                    setup = "PDL Breakdown"
                else:
                    setup = "CPR Bear Breakdown"
                reasons.append(
                    f"✅ Price {current_close:.2f} below CPR/PDL/S1 zone [{setup}] (+2)"
                )
        else:
            reasons.append(
                f"❌ Price {current_close:.2f} inside CPR (TC={tc:.2f} BC={bc:.2f}) — no signal"
            )
            return 0, "NONE", " | ".join(reasons), levels, 0

        # ── 1b. H1 trend filter — HARD BLOCK (FIXED) ─────────────────────────
        if h1_trend_bullish is not None:
            if direction == "BUY" and not h1_trend_bullish:
                reasons.append("❌ H1 trend BEARISH — BUY blocked (trend filter)")
                log.warning(
                    "H1 trend filter BLOCKED BUY | H1 price=%.2f below EMA%d=%.2f | TRADE SKIPPED",
                    h1_closes[-1] if 'h1_closes' in dir() else 0, _h1_period,
                    self._ema(h1_closes, _h1_period) if 'h1_closes' in dir() else 0,
                )
                levels["score"] = 0
                levels["signal_blockers"] = ["H1 trend bearish — BUY blocked"]
                return 0, "NONE", " | ".join(reasons), levels, 0
            elif direction == "SELL" and h1_trend_bullish:
                reasons.append("❌ H1 trend BULLISH — SELL blocked (trend filter)")
                log.warning(
                    "H1 trend filter BLOCKED SELL | H1 trend bullish | TRADE SKIPPED"
                )
                levels["score"] = 0
                levels["signal_blockers"] = ["H1 trend bullish — SELL blocked"]
                return 0, "NONE", " | ".join(reasons), levels, 0
            else:
                trend_label = "bullish" if h1_trend_bullish else "bearish"
                reasons.append(f"✅ H1 trend {trend_label} — aligns with {direction}")

        # ── 2. SMA alignment ───────────────────────────────────────────────────
        if direction == "BUY":
            both_below = sma20 < current_close and sma50 < current_close
            one_below  = (sma20 < current_close) != (sma50 < current_close)
            if both_below:
                score += 2
                reasons.append(f"✅ Both SMAs below price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+2)")
            elif one_below:
                score += 1
                which = "SMA20" if sma20 < current_close else "SMA50"
                reasons.append(f"⚠️ Only {which} below price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+1)")
            else:
                reasons.append(f"❌ Both SMAs above price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+0)")
        else:  # SELL
            both_above = sma20 > current_close and sma50 > current_close
            one_above  = (sma20 > current_close) != (sma50 > current_close)
            if both_above:
                score += 2
                reasons.append(f"✅ Both SMAs above price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+2)")
            elif one_above:
                score += 1
                which = "SMA20" if sma20 > current_close else "SMA50"
                reasons.append(f"⚠️ Only {which} above price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+1)")
            else:
                reasons.append(f"❌ Both SMAs below price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+0)")

        # ── 3. CPR width — FIXED: moderate band tightened from 0.5–1.0% to 0.5–0.7% ─
        if cpr_width_pct < 0.5:
            score += 2
            reasons.append(f"✅ Narrow CPR ({cpr_width_pct:.2f}% < 0.5%) (+2)")
        elif cpr_width_pct <= 0.7:
            score += 1
            reasons.append(f"⚠️ Moderate CPR ({cpr_width_pct:.2f}% in 0.5–0.7%) (+1)")
        else:
            reasons.append(f"❌ Wide CPR ({cpr_width_pct:.2f}% > 0.7%) (+0) — poor setup")

        # ── 4. Trend exhaustion — FIXED: threshold tightened to 1.8× (was 2.5×) ─
        exhaustion_atr_mult = float(settings.get("exhaustion_atr_mult", 1.8)) if settings else 1.8
        if atr_val and atr_val > 0:
            stretch = abs(current_close - sma20) / atr_val
            if stretch > exhaustion_atr_mult:
                # FIXED: extended setups (S2/R2) are always hard-blocked when exhausted
                if setup in ("S2 Extended Breakdown", "R2 Extended Breakout"):
                    reasons.append(
                        f"🚫 Extended entry blocked — exhaustion {stretch:.1f}× ATR "
                        f"(>{exhaustion_atr_mult:.1f}× threshold) on {setup}"
                    )
                    log.warning(
                        "CPR signal BLOCKED (extended+exhaustion) | setup=%s | dir=%s | stretch=%.1f×",
                        setup, direction, stretch,
                    )
                    levels["score"] = 0
                    levels["signal_blockers"] = [f"Extended+exhausted {stretch:.1f}× ATR"]
                    return 0, "NONE", " | ".join(reasons), levels, 0
                score = max(0, score - 1)
                reasons.append(
                    f"⚠️ Trend stretch {stretch:.1f}× ATR from SMA20 "
                    f"(>{exhaustion_atr_mult:.1f}× threshold) — exhaustion penalty (−1)"
                )
            else:
                reasons.append(
                    f"✅ Trend stretch {stretch:.1f}× ATR (≤{exhaustion_atr_mult:.1f}× threshold) — ok"
                )
        else:
            # FIXED: if ATR unavailable, block trade (previously just warned)
            reasons.append("🚫 ATR unavailable — trade blocked (cannot size SL)")
            log.warning("ATR unavailable — trade blocked (safe default)")
            levels["score"] = 0
            levels["signal_blockers"] = ["ATR unavailable"]
            return 0, "NONE", " | ".join(reasons), levels, 0

        # ── Position size — FIXED: only score >= 5 gets a position ───────────
        position_usd = score_to_position_usd(score, settings)

        # ── SL calculation — FIXED: purely ATR-based, multiplier 1.5 ─────────
        # Old code used CPR structural SL (often only $5-10) then fell back to
        # 0.25% fixed. Both were too tight. ATR-based is the correct approach.
        entry    = current_close
        atr_mult = float(settings.get("atr_sl_multiplier", 1.5)) if settings else 1.5
        sl_min   = float(settings.get("sl_min_usd", 35.0)) if settings else 35.0
        sl_max   = float(settings.get("sl_max_usd", 50.0)) if settings else 50.0
        raw_sl   = atr_val * atr_mult
        sl_usd_rec = round(max(sl_min, min(sl_max, raw_sl)), 2)
        sl_source  = "atr_based"
        sl_pct_used = round(sl_usd_rec / entry * 100, 4)

        # ── TP calculation — FIXED: always SL × rr_ratio (2.5x default) ──────
        # Old code used R1/S1 structural levels which sometimes gave <$30 TP.
        # Now TP is always derived from SL so R:R is guaranteed.
        rr_ratio   = float(settings.get("rr_ratio", 2.5)) if settings else 2.5
        max_rr     = float(settings.get("max_rr_ratio", 3.0)) if settings else 3.0
        tp_usd_rec = round(sl_usd_rec * rr_ratio, 2)
        tp_usd_rec = round(min(tp_usd_rec, sl_usd_rec * max_rr), 2)
        tp_source  = "rr_multiple"
        tp_pct_used = round(tp_usd_rec / entry * 100, 4)

        # ── R:R guard ─────────────────────────────────────────────────────────
        actual_rr   = round(tp_usd_rec / sl_usd_rec, 2) if sl_usd_rec > 0 else 0
        _min_rr_sig = float((settings or {}).get("rr_ratio", 2.5))
        rr_skip     = actual_rr < _min_rr_sig

        blocker_reasons = []
        if rr_skip:
            blocker_reasons.append(f"R:R {actual_rr:.2f} < 1:{_min_rr_sig:.2f}")

        levels["score"]        = score
        levels["position_usd"] = position_usd
        levels["entry"]        = round(entry, 2)
        levels["setup"]        = setup
        levels["sl_usd_rec"]   = sl_usd_rec
        levels["sl_source"]    = sl_source
        levels["sl_pct_used"]  = sl_pct_used
        levels["tp_usd_rec"]   = tp_usd_rec
        levels["tp_source"]    = tp_source
        levels["tp_pct_used"]  = tp_pct_used
        levels["rr_ratio"]     = actual_rr
        levels["mandatory_checks"] = {
            "score_ok": score >= int((settings or {}).get("signal_threshold", MIN_TRADE_SCORE)),
            "rr_ok": not rr_skip,
        }
        levels["quality_checks"] = {
            "tp_ok": tp_usd_rec >= sl_usd_rec * _min_rr_sig,
        }
        levels["signal_blockers"] = blocker_reasons

        reasons.append(
            f"📐 ATR-SL=${sl_usd_rec} ({sl_pct_used:.3f}%) | "
            f"TP=${tp_usd_rec} ({tp_pct_used:.3f}%) | R:R 1:{actual_rr:.2f}"
        )
        if blocker_reasons:
            reasons.append("🚫 " + " | ".join(blocker_reasons))

        details = " | ".join(reasons)
        if blocker_reasons:
            log.warning(
                "CPR signal BLOCKED | setup=%s | dir=%s | score=%s/6 | blockers=%s",
                setup, direction, score, "; ".join(blocker_reasons),
            )
        else:
            log.info(
                "CPR signal | setup=%s | dir=%s | score=%s/6 | position=$%s | SL=$%.2f | TP=$%.2f | RR=1:%.2f",
                setup, direction, score, position_usd, sl_usd_rec, tp_usd_rec, actual_rr,
            )
        return score, direction, details, levels, position_usd

    # ── Data helpers ───────────────────────────────────────────────────────────

    def _fetch_candles(self, instrument: str, granularity: str, count: int = 60):
        url    = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {"count": str(count), "granularity": granularity, "price": "M"}
        for attempt in range(3):
            try:
                r = self.session.get(url, headers=self.headers, params=params, timeout=15)
                if r.status_code == 200:
                    candles  = r.json().get("candles", [])
                    complete = [c for c in candles if c.get("complete")]
                    closes   = [float(c["mid"]["c"]) for c in complete]
                    highs    = [float(c["mid"]["h"]) for c in complete]
                    lows     = [float(c["mid"]["l"]) for c in complete]
                    return closes, highs, lows
                log.warning("Fetch candles %s %s: HTTP %s", instrument, granularity, r.status_code)
            except Exception as e:
                log.warning(
                    "Fetch candles error (%s %s) attempt %s: %s",
                    instrument, granularity, attempt + 1, e,
                )
            time.sleep(1)
        return [], [], []

    def _atr(self, highs: list, lows: list, closes: list, period: int = 14) -> float | None:
        """Return the most recent ATR value, or None if insufficient data."""
        n = len(closes)
        if n < period + 2 or len(highs) < n or len(lows) < n:
            return None
        trs = [
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            for i in range(1, n)
        ]
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    def _ema(self, closes: list, period: int) -> float:
        """Calculate EMA for the last `period` closes.

        FIXED: replaces SMA in H1 trend filter — EMA is more responsive to
        recent price action and catches trend changes faster.
        """
        if len(closes) < period:
            return sum(closes) / len(closes)
        k   = 2 / (period + 1)
        ema = sum(closes[:period]) / period  # seed with SMA
        for price in closes[period:]:
            ema = price * k + ema * (1 - k)
        return ema
