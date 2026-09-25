"""Turn a match schedule with prices into a ranked list of +EV bets."""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ev
from .data import normalise_surface
from .model import RatingModel
from .names import NameMatcher

# Tournament keyword -> surface. Betfair does not publish surface.
SURFACE_KEYWORDS = {
    "Grass": ["wimbledon", "queens", "queen's", "halle", "s-hertogenbosch", "hertogenbosch",
              "eastbourne", "mallorca", "stuttgart", "birmingham", "nottingham", "newport",
              "berlin", "bad homburg", "ilkley", "grass"],
    "Clay": ["roland garros", "french open", "monte carlo", "monte-carlo", "madrid", "rome",
             "italian open", "internazionali", "barcelona", "hamburg", "gstaad", "kitzbuhel",
             "bastad", "umag", "estoril", "munich", "geneva", "lyon", "buenos aires",
             "rio de janeiro", "santiago", "houston", "marrakech", "bucharest", "cordoba",
             "charleston", "bogota", "palermo", "prague", "rabat", "strasbourg",
             "parma", "lausanne", "budapest", "iasi", "clay"],
}

SCHEDULE_ALIASES = {
    "player1": ["player1", "player_1", "player a", "player_a", "home", "p1"],
    "player2": ["player2", "player_2", "player b", "player_b", "away", "p2"],
    "back1": ["back1", "odds1", "odds_1", "price1", "back_1", "odds a", "odds_a"],
    "back2": ["back2", "odds2", "odds_2", "price2", "back_2", "odds b", "odds_b"],
    "lay1": ["lay1", "lay_1"],
    "lay2": ["lay2", "lay_2"],
    "surface": ["surface"],
    "tournament": ["tournament", "competition", "event"],
    "start": ["start", "date", "time", "start_time", "datetime"],
}


def infer_surface(tournament: str, default: str = "Hard") -> str:
    t = str(tournament).lower()
    for surface, words in SURFACE_KEYWORDS.items():
        # word boundaries so e.g. "Challenger" does not match "halle"
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            return surface
    return default


def read_schedule(data: bytes, filename: str = "") -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(data)) if filename.lower().endswith((".xlsx", ".xls")) \
        else pd.read_csv(io.BytesIO(data))
    lower = {str(c).strip().lower(): c for c in raw.columns}
    out = pd.DataFrame()
    for canon, options in SCHEDULE_ALIASES.items():
        for o in options:
            if o in lower:
                out[canon] = raw[lower[o]]
                break
    for col in ("player1", "player2", "back1", "back2"):
        if col not in out:
            raise ValueError(f"Schedule is missing column '{col}' "
                             "(need player1, player2, back1, back2)")
    for col in ("back1", "back2", "lay1", "lay2"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "surface" in out:
        out["surface"] = out["surface"].map(normalise_surface)
    if "start" in out:
        out["start"] = pd.to_datetime(out["start"], errors="coerce")
    return out


@dataclass
class ScanSettings:
    bankroll: float = 1000.0
    commission: float = 0.05
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.03
    min_ev: float = 0.03
    min_odds: float = 1.3
    max_odds: float = 6.0
    model_weight: float = 0.6
    min_matches: int = 8
    include_lays: bool = False
    default_surface: str = "Hard"
    min_liquidity: float = 0.0


def _grade(ev_value: float, n_min: int, edge: float) -> str:
    score = 0
    score += 2 if ev_value >= 0.08 else 1 if ev_value >= 0.04 else 0
    score += 2 if n_min >= 25 else 1 if n_min >= 12 else 0
    score += 1 if 0.03 <= edge <= 0.15 else 0  # huge edges usually mean missing info
    return "A" if score >= 4 else "B" if score >= 2 else "C"


def _valid(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(x) and x > 1.0


def evaluate(schedule: pd.DataFrame, model: RatingModel, s: ScanSettings,
             aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Evaluate every selection in the schedule. Returns all rows (filter with `value_bets`)."""
    matcher = NameMatcher(model.players.keys(), aliases)
    rows = []
    for m in schedule.to_dict("records"):
        surface = m.get("surface")
        if not isinstance(surface, str) or not surface:
            surface = infer_surface(m.get("tournament", ""), s.default_surface)
        n1, n2 = matcher.resolve(m["player1"]), matcher.resolve(m["player2"])
        when = m.get("start")
        when = pd.Timestamp(when).tz_localize(None) if isinstance(when, pd.Timestamp) and when.tzinfo \
            else (when if isinstance(when, pd.Timestamp) and not pd.isna(when) else None)
        pred = model.predict(n1 or m["player1"], n2 or m["player2"], surface, when)
        p1 = pred["prob"]
        b1, b2 = m.get("back1"), m.get("back2")
        p_mkt1 = ev.no_vig(b1, b2)[0] if _valid(b1) and _valid(b2) else None
        p1_final = ev.blend(p1, p_mkt1, s.model_weight)

        for side, name, resolved, p_model, p_final, p_mkt, n_self, n_opp, opp in (
            (1, m["player1"], n1, p1, p1_final, p_mkt1, pred["n_a"], pred["n_b"], m["player2"]),
            (2, m["player2"], n2, 1 - p1, 1 - p1_final,
             None if p_mkt1 is None else 1 - p_mkt1, pred["n_b"], pred["n_a"], m["player1"]),
        ):
            base = {
                "start": m.get("start"), "tournament": m.get("tournament", ""),
                "surface": surface, "selection": name, "opponent": opp,
                "matched_name": resolved or "", "model_prob": p_model,
                "market_prob": p_mkt, "prob": p_final,
                "fair_odds": ev.fair_odds(p_final),
                "data_matches": min(n_self, n_opp),
                "known": bool(n1 and n2),
                "market_id": m.get("market_id", ""),
                "match_volume": m.get("matched", np.nan),
            }
            back = m.get(f"back{side}")
            if _valid(back):
                e = ev.back_ev(p_final, back, s.commission)
                k = ev.kelly_back(p_final, back, s.commission) * s.kelly_fraction
                stake = min(k, s.max_stake_pct) * s.bankroll
                rows.append({**base, "bet": "BACK", "odds": back,
                             "liquidity": m.get(f"back{side}_size", np.nan),
                             "ev": e, "edge": p_final - (p_mkt if p_mkt else 1 / back),
                             "stake": stake, "liability": stake,
                             "exp_profit": stake * e})
            lay = m.get(f"lay{side}")
            if s.include_lays and _valid(lay):
                e = ev.lay_ev(p_final, lay, s.commission)
                k = ev.kelly_lay(p_final, lay, s.commission) * s.kelly_fraction
                liability = min(k, s.max_stake_pct) * s.bankroll
                rows.append({**base, "bet": "LAY", "odds": lay,
                             "liquidity": m.get(f"lay{side}_size", np.nan),
                             "ev": e, "edge": (p_mkt if p_mkt else 1 / lay) - p_final,
                             "stake": liability / (lay - 1), "liability": liability,
                             "exp_profit": liability * e})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["grade"] = [_grade(e, n, d) for e, n, d in zip(out["ev"], out["data_matches"], out["edge"])]
    return out.sort_values("ev", ascending=False).reset_index(drop=True)


def value_bets(evaluated: pd.DataFrame, s: ScanSettings) -> pd.DataFrame:
    if evaluated.empty:
        return evaluated
    df = evaluated
    mask = (
        (df["ev"] >= s.min_ev)
        & df["odds"].between(s.min_odds, s.max_odds)
        & (df["data_matches"] >= s.min_matches)
        & df["known"]
        & (df["stake"] > 0)
    )
    if s.min_liquidity > 0:
        mask &= df["liquidity"].fillna(0) >= s.min_liquidity
    picks = df[mask]
    # never recommend both sides of one match: keep the best per match
    key = picks["start"].astype(str) + "|" + picks[["selection", "opponent"]].apply(
        lambda r: "|".join(sorted(r)), axis=1)
    return picks.loc[~key.duplicated()].reset_index(drop=True)
