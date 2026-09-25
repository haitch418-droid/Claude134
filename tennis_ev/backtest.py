"""Walk-forward backtest on the uploaded results that include odds.

Weights are fitted only on matches before the split date, then the betting
rules are replayed on the later matches. Features are always pre-match, so
there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ev
from .model import FEATURES, RatingModel, sigmoid
from .scanner import ScanSettings


def run(model: RatingModel, s: ScanSettings, train_frac: float = 0.6,
        staking: str = "kelly") -> dict:
    h = model.history
    if h is None or h.empty:
        return {"error": "No data loaded."}
    dates = h["date"].sort_values()
    split = dates.iloc[int(len(dates) * train_frac)]

    saved = (model.weights.copy(), model.fitted, dict(model.fit_info))
    fit_info = model.fit(until=split)
    weights = model.weights.copy()
    model.weights, model.fitted, model.fit_info = saved

    test = h[(h["date"] >= split)].copy()
    test = test[np.isfinite(test["odds_a"]) & np.isfinite(test["odds_b"])
                & (test["odds_a"] > 1) & (test["odds_b"] > 1)]
    if test.empty:
        return {"error": "No matches with odds after the training period. "
                         "Upload results that include odds columns (e.g. PSW/PSL, B365W/B365L)."}

    X = test[[f"f_{k}" for k in FEATURES]].to_numpy(float)
    test["p_model"] = sigmoid(X @ weights)
    inv_a, inv_b = 1 / test["odds_a"], 1 / test["odds_b"]
    test["p_market"] = inv_a / (inv_a + inv_b)
    lm = np.log(test["p_model"] / (1 - test["p_model"]))
    lk = np.log(test["p_market"] / (1 - test["p_market"]))
    test["p"] = sigmoid(s.model_weight * lm + (1 - s.model_weight) * lk)

    bank = s.bankroll
    bets = []
    for r in test.itertuples(index=False):
        n_min = min(r.n_a, r.n_b)
        if n_min < s.min_matches:
            continue
        best = None
        for side, p, odds, won in ((r.player_a, r.p, r.odds_a, r.a_won == 1),
                                   (r.player_b, 1 - r.p, r.odds_b, r.a_won == 0)):
            if not (s.min_odds <= odds <= s.max_odds):
                continue
            e = ev.back_ev(p, odds, s.commission)
            if e >= s.min_ev and (best is None or e > best[2]):
                best = (side, p, e, odds, won)
        if best is None:
            continue
        side, p, e, odds, won = best
        if staking == "flat":
            stake = s.bankroll * 0.01
        else:
            frac = ev.kelly_back(p, odds, s.commission) * s.kelly_fraction
            stake = min(frac, s.max_stake_pct) * bank
        if stake <= 0:
            continue
        pnl = stake * (odds - 1) * (1 - s.commission) if won else -stake
        bank += pnl
        bets.append({"date": r.date, "selection": side, "odds": odds, "prob": p,
                     "ev": e, "stake": stake, "won": won, "pnl": pnl, "bankroll": bank})

    bets_df = pd.DataFrame(bets)
    y = test["a_won"].to_numpy(float)

    def logloss(p):
        p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    summary = {
        "split_date": split, "fit": fit_info, "test_matches": int(len(test)),
        "log_loss_model": logloss(test["p_model"]),
        "log_loss_market": logloss(test["p_market"]),
        "log_loss_blend": logloss(test["p"]),
        "bets": int(len(bets_df)),
    }
    if len(bets_df):
        peak = bets_df["bankroll"].cummax()
        summary.update({
            "staked": float(bets_df["stake"].sum()),
            "profit": float(bets_df["pnl"].sum()),
            "roi": float(bets_df["pnl"].sum() / bets_df["stake"].sum()),
            "hit_rate": float(bets_df["won"].mean()),
            "avg_odds": float(bets_df["odds"].mean()),
            "max_drawdown": float(((peak - bets_df["bankroll"]) / peak).max()),
            "final_bankroll": float(bank),
        })

    bins = pd.cut(test["p_model"], np.linspace(0, 1, 11))
    calib = test.groupby(bins, observed=True).agg(
        predicted=("p_model", "mean"), actual=("a_won", "mean"), n=("a_won", "size")
    ).reset_index(drop=True)
    return {"summary": summary, "bets": bets_df, "calibration": calib}
