"""Player rating model.

A single chronological pass over the results builds, for each player:
  * an overall Elo and a surface-specific Elo (seeded from ATP/WTA ranking so
    players are not all stuck at 1500 when history is only a year long),
  * a dated match log used for 30 / 90 / 365-day form windows and fatigue.

Before every match the pre-match feature vector (player A minus player B) is
recorded, so the combining weights can be fitted on the user's own data with
no look-ahead. The fitted weights are the "formula":

    logit P(A beats B) = w · features(A, B)
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LN10_400 = math.log(10) / 400.0

FEATURES = [
    "elo",          # overall Elo difference, as a logit
    "surface_elo",  # surface Elo difference, as a logit
    "form_30",      # shrunk win-rate difference, last 30 days
    "form_90",      # last 90 days
    "form_365",     # last 365 days
    "surface_form", # win-rate difference on this surface, last 365 days
    "rank",         # log(rank B) - log(rank A)
    "fatigue",      # matches played in last 10 days, A minus B
]

FEATURE_LABELS = {
    "elo": "Overall Elo",
    "surface_elo": "Surface Elo",
    "form_30": "Form (30 days)",
    "form_90": "Form (90 days)",
    "form_365": "Form (365 days)",
    "surface_form": "Surface form (365d)",
    "rank": "Ranking",
    "fatigue": "Fatigue (10 days)",
}

# Sensible priors, used until enough of the user's data exists to fit them.
DEFAULT_WEIGHTS = np.array([0.45, 0.45, 0.35, 0.45, 0.30, 0.30, 0.10, -0.03])

DEFAULT_RANK = 400
SHRINK_N = 6  # pseudo-matches pulling win-rates towards 50%


def seed_elo(rank: float | None) -> float:
    """Initial Elo from world ranking (rank 1 ~ 2150, rank 100 ~ 1690)."""
    if rank is None or not np.isfinite(rank) or rank <= 0:
        return 1500.0
    return 2150.0 - 100.0 * math.log(rank)


def k_factor(n_matches: int, best_of: int = 3) -> float:
    k = 250.0 / ((n_matches + 5) ** 0.4)
    return k * (1.1 if best_of == 5 else 1.0)


@dataclass
class Player:
    name: str
    elo: float
    surface_elo: dict = field(default_factory=dict)
    n: int = 0
    n_surface: dict = field(default_factory=dict)
    rank: float = float("nan")
    # parallel lists: date ordinal, won (1/0), surface
    dates: list = field(default_factory=list)
    results: list = field(default_factory=list)
    surfaces: list = field(default_factory=list)

    def window(self, today: int, days: int, surface: str | None = None):
        lo = bisect.bisect_left(self.dates, today - days)
        hi = bisect.bisect_left(self.dates, today)
        if surface is None:
            res = self.results[lo:hi]
        else:
            res = [r for r, s in zip(self.results[lo:hi], self.surfaces[lo:hi]) if s == surface]
        return sum(res), len(res)


def shrunk_rate(wins: int, n: int) -> float:
    return (wins + SHRINK_N / 2) / (n + SHRINK_N)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RatingModel:
    def __init__(self):
        self.players: dict[str, Player] = {}
        self.weights = DEFAULT_WEIGHTS.copy()
        self.fitted = False
        self.fit_info: dict = {}
        self.history: pd.DataFrame | None = None  # pre-match features per match
        self.last_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------ build
    def _player(self, name: str, rank: float) -> Player:
        p = self.players.get(name)
        if p is None:
            p = Player(name=name, elo=seed_elo(rank))
            self.players[name] = p
        if np.isfinite(rank) and rank > 0:
            p.rank = rank
        return p

    def _surface_elo(self, p: Player, surface: str) -> float:
        return p.surface_elo.get(surface, p.elo)

    def features_for(self, a: Player, b: Player, today: int, surface: str) -> np.ndarray:
        f = np.empty(len(FEATURES))
        f[0] = (a.elo - b.elo) * LN10_400
        f[1] = (self._surface_elo(a, surface) - self._surface_elo(b, surface)) * LN10_400
        for i, days in ((2, 30), (3, 90), (4, 365)):
            wa, na = a.window(today, days)
            wb, nb = b.window(today, days)
            f[i] = shrunk_rate(wa, na) - shrunk_rate(wb, nb)
        wa, na = a.window(today, 365, surface)
        wb, nb = b.window(today, 365, surface)
        f[5] = shrunk_rate(wa, na) - shrunk_rate(wb, nb)
        ra = a.rank if np.isfinite(a.rank) else DEFAULT_RANK
        rb = b.rank if np.isfinite(b.rank) else DEFAULT_RANK
        f[6] = math.log(rb) - math.log(ra)
        f[7] = a.window(today, 10)[1] - b.window(today, 10)[1]
        return f

    def build(self, matches: pd.DataFrame) -> "RatingModel":
        """Run through all results in date order, recording pre-match features."""
        self.players = {}
        df = matches.sort_values("date", kind="stable").reset_index(drop=True)
        rows = []
        for m in df.itertuples(index=False):
            today = m.date.toordinal()
            w = self._player(m.winner, m.winner_rank)
            l = self._player(m.loser, m.loser_rank)
            surface = m.surface

            # Orientation alternates so the dataset is balanced (label 1/0).
            flip = len(rows) % 2 == 1
            a, b = (l, w) if flip else (w, l)
            feats = self.features_for(a, b, today, surface)
            rows.append({
                "date": m.date, "surface": surface, "tournament": m.tournament,
                "player_a": a.name, "player_b": b.name, "a_won": 0 if flip else 1,
                "odds_a": m.odds_loser if flip else m.odds_winner,
                "odds_b": m.odds_winner if flip else m.odds_loser,
                "n_a": a.n, "n_b": b.n,
                **{f"f_{k}": v for k, v in zip(FEATURES, feats)},
            })

            # --- update ratings
            ew = 1 / (1 + 10 ** ((l.elo - w.elo) / 400))
            kw, kl = k_factor(w.n, m.best_of), k_factor(l.n, m.best_of)
            w.elo += kw * (1 - ew)
            l.elo -= kl * (1 - ew)

            sw, sl = self._surface_elo(w, surface), self._surface_elo(l, surface)
            es = 1 / (1 + 10 ** ((sl - sw) / 400))
            ksw = k_factor(w.n_surface.get(surface, 0), m.best_of)
            ksl = k_factor(l.n_surface.get(surface, 0), m.best_of)
            w.surface_elo[surface] = sw + ksw * (1 - es)
            l.surface_elo[surface] = sl - ksl * (1 - es)

            for p, won in ((w, 1), (l, 0)):
                p.n += 1
                p.n_surface[surface] = p.n_surface.get(surface, 0) + 1
                p.dates.append(today)
                p.results.append(won)
                p.surfaces.append(surface)

        self.history = pd.DataFrame(rows)
        self.last_date = df["date"].max() if len(df) else None
        return self

    # -------------------------------------------------------------------- fit
    def fit(self, min_prior_matches: int = 5, l2: float = 2.0,
            until: pd.Timestamp | None = None, min_rows: int = 300) -> dict:
        """Fit combining weights by L2-regularised logistic regression.

        Regularisation pulls towards DEFAULT_WEIGHTS, so small datasets stay
        close to the prior and large ones learn their own formula.
        """
        h = self.history
        if h is None or h.empty:
            self.fit_info = {"status": "no data"}
            return self.fit_info
        mask = (h["n_a"] >= min_prior_matches) & (h["n_b"] >= min_prior_matches)
        if until is not None:
            mask &= h["date"] < until
        train = h[mask]
        if len(train) < min_rows:
            self.weights = DEFAULT_WEIGHTS.copy()
            self.fitted = False
            self.fit_info = {"status": "default weights", "rows": int(len(train)),
                             "reason": f"need {min_rows}+ matches with history to fit"}
            return self.fit_info

        X = train[[f"f_{k}" for k in FEATURES]].to_numpy(float)
        y = train["a_won"].to_numpy(float)
        w = DEFAULT_WEIGHTS.copy()
        prior = DEFAULT_WEIGHTS
        for _ in range(50):  # Newton-Raphson
            p = sigmoid(X @ w)
            grad = X.T @ (p - y) + l2 * (w - prior)
            H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        p = np.clip(sigmoid(X @ w), 1e-9, 1 - 1e-9)
        p0 = np.clip(sigmoid(X @ DEFAULT_WEIGHTS), 1e-9, 1 - 1e-9)
        self.weights = w
        self.fitted = True
        self.fit_info = {
            "status": "fitted", "rows": int(len(train)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "log_loss_default": float(-np.mean(y * np.log(p0) + (1 - y) * np.log(1 - p0))),
            "accuracy": float(np.mean((p > 0.5) == (y == 1))),
        }
        return self.fit_info

    # ---------------------------------------------------------------- predict
    def predict(self, a_name: str, b_name: str, surface: str = "Hard",
                when: pd.Timestamp | None = None) -> dict:
        """P(a beats b) plus a per-feature breakdown."""
        when = when or (self.last_date + pd.Timedelta(days=1) if self.last_date is not None
                        else pd.Timestamp.today())
        today = pd.Timestamp(when).toordinal()
        a = self.players.get(a_name) or Player(a_name, 1500.0)
        b = self.players.get(b_name) or Player(b_name, 1500.0)
        feats = self.features_for(a, b, today, surface)
        contrib = feats * self.weights
        prob = float(sigmoid(contrib.sum()))
        return {
            "prob": prob, "features": dict(zip(FEATURES, feats)),
            "contributions": dict(zip(FEATURES, contrib)),
            "n_a": a.n, "n_b": b.n,
        }

    def history_probs(self, weights: np.ndarray | None = None) -> np.ndarray:
        w = self.weights if weights is None else weights
        X = self.history[[f"f_{k}" for k in FEATURES]].to_numpy(float)
        return sigmoid(X @ w)

    # ------------------------------------------------------------ inspection
    def player_table(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        as_of = as_of or self.last_date or pd.Timestamp.today()
        today = pd.Timestamp(as_of).toordinal() + 1
        rows = []
        for p in self.players.values():
            w30, n30 = p.window(today, 30)
            w90, n90 = p.window(today, 90)
            w365, n365 = p.window(today, 365)
            rows.append({
                "Player": p.name, "Rank": p.rank, "Elo": round(p.elo),
                "Hard Elo": round(p.surface_elo.get("Hard", p.elo)),
                "Clay Elo": round(p.surface_elo.get("Clay", p.elo)),
                "Grass Elo": round(p.surface_elo.get("Grass", p.elo)),
                "30d": f"{w30}-{n30 - w30}", "90d": f"{w90}-{n90 - w90}",
                "365d": f"{w365}-{n365 - w365}", "Matches": p.n,
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)


def logit(p: float) -> float:
    return _logit(p)
