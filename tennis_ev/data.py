"""Load and normalise historical match results.

Accepts the tennis-data.co.uk layout (Date, Winner, Loser, WRank, LRank,
PSW, PSL, B365W, ...) as well as simple generic layouts. Everything is
mapped onto one canonical schema:

    date, tour, tournament, surface, round, best_of,
    winner, loser, winner_rank, loser_rank, odds_winner, odds_loser
"""
from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL = [
    "date", "tour", "tournament", "surface", "round", "best_of",
    "winner", "loser", "winner_rank", "loser_rank", "odds_winner", "odds_loser",
]

# lower-cased source column -> canonical column
ALIASES = {
    "date": "date", "match_date": "date", "tourney_date": "date",
    "tour": "tour", "circuit": "tour",
    "tournament": "tournament", "tourney_name": "tournament", "event": "tournament",
    "surface": "surface", "court_surface": "surface",
    "round": "round",
    "best of": "best_of", "best_of": "best_of", "bestof": "best_of",
    "winner": "winner", "winner_name": "winner",
    "loser": "loser", "loser_name": "loser",
    "wrank": "winner_rank", "winner_rank": "winner_rank",
    "lrank": "loser_rank", "loser_rank": "loser_rank",
    "odds_winner": "odds_winner", "winner_odds": "odds_winner",
    "odds_loser": "odds_loser", "loser_odds": "odds_loser",
}

# Bookmaker/exchange odds columns in order of preference (sharpest first).
ODDS_PAIRS = [
    ("bfew", "bfel"),   # Betfair exchange
    ("psw", "psl"),     # Pinnacle
    ("avgw", "avgl"),   # market average
    ("b365w", "b365l"),
    ("maxw", "maxl"),
]

SURFACES = ["Hard", "Clay", "Grass"]


def normalise_surface(value) -> str:
    s = str(value).strip().lower()
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    # carpet and indoor hard play very similarly to hard
    return "Hard"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalise(df: pd.DataFrame, tour: str | None = None) -> pd.DataFrame:
    """Map an arbitrary results frame onto the canonical schema."""
    lower = {c: str(c).strip().lower() for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for src, low in lower.items():
        canon = ALIASES.get(low)
        if canon and canon not in out:
            out[canon] = df[src]

    if "odds_winner" not in out:
        for w, l in ODDS_PAIRS:
            cols = {v: k for k, v in lower.items()}
            if w in cols and l in cols:
                out["odds_winner"] = df[cols[w]]
                out["odds_loser"] = df[cols[l]]
                break

    missing = [c for c in ("date", "winner", "loser") if c not in out]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            "Need at least Date, Winner and Loser."
        )

    if "tour" not in out:
        out["tour"] = tour or "ATP"
    for col, default in (("tournament", ""), ("surface", "Hard"), ("round", ""),
                         ("best_of", 3)):
        if col not in out:
            out[col] = default
    for col in ("winner_rank", "loser_rank", "odds_winner", "odds_loser"):
        if col not in out:
            out[col] = np.nan

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=_guess_dayfirst(out["date"]))
    out["winner"] = out["winner"].astype(str).str.strip()
    out["loser"] = out["loser"].astype(str).str.strip()
    out["surface"] = out["surface"].map(normalise_surface)
    out["best_of"] = _to_num(out["best_of"]).fillna(3).astype(int)
    for col in ("winner_rank", "loser_rank", "odds_winner", "odds_loser"):
        out[col] = _to_num(out[col])
    out["tour"] = out["tour"].astype(str).str.upper()

    out = out.dropna(subset=["date"])
    out = out[(out["winner"] != "") & (out["loser"] != "") &
              (out["winner"].str.lower() != "nan") & (out["loser"].str.lower() != "nan")]
    # Retirements/walkovers are kept only if a 'Comment' column says Completed or is absent
    comment_col = next((c for c, low in lower.items() if low == "comment"), None)
    if comment_col is not None:
        comments = df.loc[out.index, comment_col].astype(str).str.lower()
        out = out[~comments.str.contains("walkover|w/o", regex=True)]
    return out[CANONICAL].reset_index(drop=True)


def _guess_dayfirst(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    for v in sample:
        parts = v.replace("-", "/").split("/")
        if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) <= 2 and int(parts[0]) > 12:
            return True
    return False


def read_any(data: bytes | str, filename: str = "", tour: str | None = None) -> pd.DataFrame:
    """Read CSV or Excel bytes/path and normalise."""
    name = filename.lower() if filename else str(data).lower() if isinstance(data, str) else ""
    if isinstance(data, bytes):
        buf = io.BytesIO(data)
        raw = pd.read_excel(buf) if name.endswith((".xlsx", ".xls")) else pd.read_csv(buf)
    else:
        raw = pd.read_excel(data) if name.endswith((".xlsx", ".xls")) else pd.read_csv(data)
    if tour is None:
        tour = "WTA" if "wta" in name else "ATP"
    return normalise(raw, tour=tour)


def combine(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge the 30/90/365-day uploads, removing overlap between them."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=CANONICAL)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date", kind="stable")
    df = df.drop_duplicates(subset=["date", "winner", "loser"], keep="last")
    return df.reset_index(drop=True)
