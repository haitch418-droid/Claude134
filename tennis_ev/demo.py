"""Synthetic demo data so the app can be explored before real data is uploaded.

The numbers are invented; results and odds come from simulated player skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIRST = ["Alex", "Ben", "Carlos", "Daniil", "Elias", "Felix", "Gael", "Hugo", "Ivan", "Jonas",
         "Karl", "Luca", "Mateo", "Nico", "Oscar", "Pablo", "Quentin", "Rafael", "Stefan",
         "Tomas", "Ugo", "Viktor", "Wu", "Xavier", "Yannick", "Zane"]
LAST = ["Arden", "Brook", "Castell", "Dorn", "Eklund", "Falk", "Garros", "Holm", "Ivers",
        "Jansen", "Kovac", "Lindqvist", "Moreau", "Novak", "Ostrov", "Pires", "Quist", "Rossi",
        "Stahl", "Tanner", "Ulrich", "Varga", "Weiss", "Xenos", "Young", "Zeller", "Abbott",
        "Barros", "Cerny", "Duval", "Engel", "Ferro", "Gomez", "Hale", "Ionescu", "Jovic",
        "Keller", "Laurent", "Mertens", "Nadeau"]

TOURNAMENTS = [("Australian Open", "Hard", 5), ("Dubai", "Hard", 3), ("Miami", "Hard", 3),
               ("Monte Carlo", "Clay", 3), ("Madrid", "Clay", 3), ("Roland Garros", "Clay", 5),
               ("Halle", "Grass", 3), ("Wimbledon", "Grass", 5), ("Cincinnati", "Hard", 3),
               ("US Open", "Hard", 5), ("Shanghai", "Hard", 3), ("Paris", "Hard", 3)]


def make_players(n: int = 80, seed: int = 7):
    rng = np.random.default_rng(seed)
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(LAST)} {rng.choice(FIRST)[0]}.")
    names = sorted(names)
    skill = np.sort(rng.normal(0, 1, n))[::-1]
    players = pd.DataFrame({
        "name": names,
        "skill": skill,
        "clay": rng.normal(0, 0.35, n),
        "grass": rng.normal(0, 0.35, n),
    })
    players["rank"] = np.arange(1, n + 1) * 3
    return players


def _p_win(pa, pb, surface):
    adj = {"Clay": "clay", "Grass": "grass"}.get(surface)
    sa = pa.skill + (pa[adj] if adj else 0)
    sb = pb.skill + (pb[adj] if adj else 0)
    return 1 / (1 + np.exp(-1.1 * (sa - sb)))


def make_results(days: int = 365, end: pd.Timestamp | None = None, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    players = make_players(seed=seed)
    end = end or pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days)
    rows = []
    for d in pd.date_range(start, end - pd.Timedelta(days=1)):
        t_name, surface, best_of = TOURNAMENTS[(d.dayofyear // 30) % len(TOURNAMENTS)]
        order = rng.permutation(len(players))
        for k in range(rng.integers(8, 16)):
            i, j = order[2 * k], order[2 * k + 1]  # each player at most once per day
            pa, pb = players.iloc[i], players.iloc[j]
            p = _p_win(pa, pb, surface)
            a_wins = rng.random() < p
            # bookmaker odds: noisy view of the truth plus ~5% margin
            est = np.clip(p + rng.normal(0, 0.06), 0.03, 0.97)
            oa, ob = 1 / (est * 1.025), 1 / ((1 - est) * 1.025)
            w, l = (pa, pb) if a_wins else (pb, pa)
            ow, ol = (oa, ob) if a_wins else (ob, oa)
            rows.append({"Date": d.strftime("%d/%m/%Y"), "Tournament": t_name,
                         "Surface": surface, "Best of": best_of, "Winner": w["name"],
                         "Loser": l["name"], "WRank": w["rank"], "LRank": l["rank"],
                         "PSW": round(ow, 2), "PSL": round(ol, 2)})
    return pd.DataFrame(rows)


def to_full_name(short: str) -> str:
    """'Rossi C.' -> 'C Rossi' style to exercise the name matcher like Betfair names."""
    surname, initial = short.rsplit(" ", 1)
    first = next((f for f in FIRST if f[0] == initial[0]), initial[0])
    return f"{first} {surname}"


def make_schedule(n_matches: int = 24, seed: int = 11, surface: str = "Hard") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    players = make_players()
    now = pd.Timestamp.now().floor("h")
    rows = []
    for k in range(n_matches):
        i, j = rng.choice(len(players), 2, replace=False)
        pa, pb = players.iloc[i], players.iloc[j]
        p = _p_win(pa, pb, surface)
        est = np.clip(p + rng.normal(0, 0.07), 0.04, 0.96)
        b1, b2 = 1 / est, 1 / (1 - est)
        rows.append({
            "start": now + pd.Timedelta(minutes=45 * (k + 1)),
            "tournament": "Demo Open", "surface": surface,
            "player1": to_full_name(pa["name"]), "player2": to_full_name(pb["name"]),
            "back1": round(b1 * 0.99, 2), "lay1": round(b1 * 1.02, 2),
            "back2": round(b2 * 0.99, 2), "lay2": round(b2 * 1.02, 2),
            "back1_size": float(rng.integers(50, 3000)), "lay1_size": float(rng.integers(50, 3000)),
            "back2_size": float(rng.integers(50, 3000)), "lay2_size": float(rng.integers(50, 3000)),
        })
    return pd.DataFrame(rows)
