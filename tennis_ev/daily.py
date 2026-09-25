"""Find today's matches and value bets without any clicking.

Used by the dashboard's auto mode and by `daily_scan.py` (cron / Task Scheduler).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import data
from .betfair import BetfairClient, BetfairError, end_of_day

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULT_EXTS = (".csv", ".xlsx", ".xls")


def load_results_folder(folder: str | Path = DATA_DIR) -> tuple[pd.DataFrame, list[str]]:
    """Load every results file in a folder (e.g. your 30/90/365-day files)."""
    folder = Path(folder)
    frames, sources = [], []
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in RESULT_EXTS and not f.name.startswith(("~", ".")):
                df = data.read_any(f.read_bytes(), f.name)
                frames.append(df)
                sources.append(f"{f.name}: {len(df):,} matches")
    return data.combine(frames), sources


def credentials(getter=os.environ.get) -> dict:
    return {k: getter(f"BETFAIR_{k.upper()}") or "" for k in ("app_key", "username", "password")}


def fetch_today(client: BetfairClient, username: str = "", password: str = "",
                tz: str = "Europe/London", include_in_play: bool = False,
                hours_ahead: float | None = None) -> pd.DataFrame:
    """All tennis matches from now until midnight (or the next N hours).

    Logs in if there is no session token, and logs in again once if the
    token has expired.
    """
    until = None if hours_ahead else end_of_day(tz)

    def fetch():
        return client.tennis_schedule(hours_ahead=hours_ahead or 24,
                                      include_in_play=include_in_play, until=until)

    if not client.session_token:
        client.login(username, password)
        return fetch()
    try:
        return fetch()
    except BetfairError as e:
        expired = "INVALID_SESSION" in str(e) or "NO_SESSION" in str(e)
        if not (expired and username and password):
            raise
        client.login(username, password)
        return fetch()
