"""Expected value and staking on a betting exchange (commission on net winnings)."""
from __future__ import annotations

import math


def no_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Market-implied probabilities with the overround removed."""
    ia, ib = 1 / odds_a, 1 / odds_b
    s = ia + ib
    return ia / s, ib / s


def blend(p_model: float, p_market: float | None, model_weight: float) -> float:
    """Combine model and market in log-odds space.

    Exchange prices are sharp; leaning partly on them cuts false positives.
    model_weight=1 trusts the model entirely, 0 copies the market.
    """
    if p_market is None or not math.isfinite(p_market):
        return p_model
    lm = math.log(p_model / (1 - p_model))
    lk = math.log(p_market / (1 - p_market))
    x = model_weight * lm + (1 - model_weight) * lk
    return 1 / (1 + math.exp(-x))


def back_ev(p: float, odds: float, commission: float) -> float:
    """EV per 1 unit backed."""
    return p * (odds - 1) * (1 - commission) - (1 - p)


def lay_ev(p: float, odds: float, commission: float) -> float:
    """EV per 1 unit of liability when laying a selection with win prob p."""
    win = (1 - commission) / (odds - 1)  # profit per unit liability if selection loses
    return (1 - p) * win - p


def kelly_back(p: float, odds: float, commission: float) -> float:
    b = (odds - 1) * (1 - commission)
    if b <= 0:
        return 0.0
    return max(0.0, (p * b - (1 - p)) / b)


def kelly_lay(p: float, odds: float, commission: float) -> float:
    """Kelly fraction of bankroll to put up as lay liability."""
    b = (1 - commission) / (odds - 1)
    return max(0.0, ((1 - p) * b - p) / b)


def fair_odds(p: float) -> float:
    return 1 / p if p > 0 else float("inf")


def min_back_odds(p: float, commission: float) -> float:
    """Lowest back price that is still +EV after commission."""
    return 1 + (1 - p) / (p * (1 - commission))
