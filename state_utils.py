from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pytz

# DATA_DIR is the single source of truth — defined in config_loader.py.
# Importing here avoids the duplicate Path(os.environ.get(...)) definition
# that previously existed in both modules.
from config_loader import DATA_DIR

logger = logging.getLogger(__name__)

SG_TZ = pytz.timezone('Asia/Singapore')

CALENDAR_CACHE_FILE = DATA_DIR / 'calendar_cache.json'
SCORE_CACHE_FILE    = DATA_DIR / 'signal_cache.json'
OPS_STATE_FILE      = DATA_DIR / 'ops_state.json'
TRADE_HISTORY_FILE  = DATA_DIR / 'trade_history.json'
RUNTIME_STATE_FILE  = DATA_DIR / 'runtime_state.json'
# Removed: TRADE_HISTORY_ARCHIVE_FILE — archival removed; 90-day rolling window is sufficient
# Removed: LAST_TRADE_CANDLE_FILE     — never used anywhere in the codebase


def load_json(path: Path, default: Any):
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(default, dict) and not isinstance(data, dict):
                    return default.copy()
                if isinstance(default, list) and not isinstance(data, list):
                    return default.copy()
                return data
    except Exception as exc:
        logger.warning('Failed to load %s: %s', path, exc)
    return default.copy() if isinstance(default, (dict, list)) else default


def save_json(path: Path, data: Any):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile('w', delete=False, dir=str(path.parent), encoding='utf-8') as tmp:
            json.dump(data, tmp, indent=2)
            temp_name = tmp.name
        os.replace(temp_name, path)
    except Exception as exc:
        logger.warning('Failed to save %s: %s', path, exc)


def update_runtime_state(**kwargs) -> None:
    state = load_json(RUNTIME_STATE_FILE, {})
    if not isinstance(state, dict):
        state = {}
    state.update(kwargs)
    state['updated_at_sgt'] = datetime.now(SG_TZ).strftime('%Y-%m-%d %H:%M:%S')
    save_json(RUNTIME_STATE_FILE, state)


def parse_sgt_timestamp(value: str | None) -> datetime | None:
    """Parse a SGT timestamp string into a timezone-aware datetime.

    Accepts both '%Y-%m-%d %H:%M:%S' and ISO '%Y-%m-%dT%H:%M:%S' formats.
    Returns None if value is falsy or unparseable.

    Canonical implementation — imported by bot.py and calendar_fetcher.py
    so the logic lives in exactly one place.
    """
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return SG_TZ.localize(datetime.strptime(value, fmt))
        except Exception:
            pass
    return None


# ── v2.0 — Win Candle Lock helpers ────────────────────────────────────────────
# After a TP (winning) close, the bot records the M15 candle boundary at which
# the win occurred.  _guard_phase() checks this before allowing a new entry:
# if the current candle is still the same candle the win closed on, entry is
# blocked.  No timer involved — the lock clears automatically when the next
# M15 candle opens.  This prevents re-entering on exhausted price moves in
# the seconds/minutes immediately after a TP hit.
#
# Key design decisions:
#   - Candle-boundary based, NOT time-based (no cooldown minutes in settings)
#   - Consistent with require_candle_close=True — both wait for candle boundaries
#   - Lock is stored in runtime_state.json so it survives process restarts
#   - Lock auto-expires: once the current candle != win candle, it clears itself
#   - Only TP wins set the lock; SL losses do NOT (losses use loss-streak logic)

def get_m15_candle_floor(dt: datetime) -> str:
    """Return the M15 candle floor timestamp string for a given SGT datetime.

    Examples:
        10:47 SGT  →  '2026-03-23 10:45'
        10:53 SGT  →  '2026-03-23 10:45'
        11:01 SGT  →  '2026-03-23 11:00'
        11:15 SGT  →  '2026-03-23 11:15'
    """
    floored_min = (dt.minute // 15) * 15
    candle_floor = dt.replace(minute=floored_min, second=0, microsecond=0)
    return candle_floor.strftime("%Y-%m-%d %H:%M")


def set_last_win_candle(dt: datetime) -> None:
    """Record the M15 candle floor at which a TP win was detected.

    Called from backfill_pnl() whenever pnl > 0 is first detected on a
    previously-open trade (i.e. the trade just closed as a winner).
    The candle floor — not the exact close time — is stored so that
    comparison in _guard_phase() is purely candle-index based.
    """
    candle_ts = get_m15_candle_floor(dt)
    update_runtime_state(last_win_candle_ts=candle_ts)
    logger.info("Win candle lock SET — candle=%s (no new entry until next candle)", candle_ts)


def get_last_win_candle() -> str | None:
    """Return the stored win-candle floor string, or None if not set / cleared."""
    state = load_json(RUNTIME_STATE_FILE, {})
    val = state.get("last_win_candle_ts")
    # Treat explicit None or empty string as "not set"
    return val if val else None


def clear_last_win_candle() -> None:
    """Explicitly clear the win candle lock.

    Called by _guard_phase() when it detects the current candle has advanced
    past the win candle — the lock is no longer needed and is removed from
    runtime_state.json so subsequent log output stays clean.
    """
    update_runtime_state(last_win_candle_ts=None)
    logger.info("Win candle lock CLEARED — new M15 candle confirmed, entries re-enabled")


# ── v2.1 — Post-Win Session Lock helpers ──────────────────────────────────────
# After a TP (winning) close, the bot records the macro session (Asian/London/US)
# and trading day in which the win occurred.  _guard_phase() checks this before
# allowing a new entry: if the current macro session is the SAME session the win
# closed in, entry is blocked for the rest of that session.
#
# Key design decisions:
#   - Session-boundary based: resumes naturally when a new macro session opens
#   - Paired with the existing win candle lock (both can be enabled independently)
#   - Stored in runtime_state.json so it survives process restarts
#   - Lock clears automatically when the active macro session changes
#   - Controlled by new setting: post_win_session_lock (bool, default True)

def set_win_session(macro_session: str, trading_day: str) -> None:
    """Record the macro session + trading day in which a TP win was detected.

    Called from backfill_pnl() immediately after set_last_win_candle().
    The combination of macro_session + trading_day uniquely identifies one
    session block so the lock does not bleed across days.
    """
    update_runtime_state(
        last_win_session=macro_session,
        last_win_session_day=trading_day,
    )
    logger.info(
        "Post-win SESSION lock SET — session=%s day=%s (no new entry until next session)",
        macro_session, trading_day,
    )


def get_win_session() -> tuple[str | None, str | None]:
    """Return (macro_session, trading_day) of the last TP win, or (None, None)."""
    state = load_json(RUNTIME_STATE_FILE, {})
    return state.get("last_win_session") or None, state.get("last_win_session_day") or None


def clear_win_session() -> None:
    """Clear the post-win session lock.

    Called by _guard_phase() when it detects the active macro session has
    changed since the win was recorded.
    """
    update_runtime_state(last_win_session=None, last_win_session_day=None)
    logger.info("Post-win SESSION lock CLEARED — new session detected, entries re-enabled")
