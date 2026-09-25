import math

import pandas as pd
import pytest

from tennis_ev import backtest, data, demo, ev, scanner
from tennis_ev.betfair import catalogue_to_schedule
from tennis_ev.model import RatingModel
from tennis_ev.names import NameMatcher, key


@pytest.fixture(scope="module")
def model():
    m = RatingModel().build(data.normalise(demo.make_results(days=240)))
    m.fit()
    return m


def test_normalise_tennis_data_layout():
    raw = pd.DataFrame({
        "Date": ["13/01/2026", "14/01/2026"], "Tournament": ["AO", "AO"], "Surface": ["Hard", "hard"],
        "Best of": [5, 5], "Winner": ["Sinner J.", "Alcaraz C."], "Loser": ["X A.", "Y B."],
        "WRank": [1, 2], "LRank": [50, "NR"], "B365W": [1.1, 1.2], "B365L": [6, 4],
        "PSW": [1.12, 1.22], "PSL": [7, 4.5], "Comment": ["Completed", "Walkover"],
    })
    df = data.normalise(raw)
    assert len(df) == 1  # walkover dropped
    row = df.iloc[0]
    assert row["date"] == pd.Timestamp("2026-01-13")
    assert row["odds_winner"] == 1.12  # Pinnacle preferred over B365
    assert row["winner_rank"] == 1


def test_combine_dedupes_overlapping_windows():
    df = data.normalise(demo.make_results(days=60))
    last30 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
    assert len(data.combine([last30, df])) == len(df)


@pytest.mark.parametrize("betfair,results", [
    ("Carlos Alcaraz", "Alcaraz C."),
    ("Alex de Minaur", "De Minaur A."),
    ("Felix Auger-Aliassime", "Auger-Aliassime F."),
    ("Jan-Lennard Struff", "Struff J.L."),
])
def test_name_matching(betfair, results):
    m = NameMatcher([results, "Someone Else X.", "Alcaraz Z."])
    assert m.resolve(betfair) == results


def test_name_key_formats_agree():
    assert key("Carlos Alcaraz") == key("Alcaraz C.")


def test_ev_formulas():
    assert ev.back_ev(0.5, 2.0, 0.0) == pytest.approx(0.0)
    assert ev.back_ev(0.5, 2.2, 0.05) == pytest.approx(0.5 * 1.2 * 0.95 - 0.5)
    # lay at fair price with no commission is zero EV
    assert ev.lay_ev(0.25, 4.0, 0.0) == pytest.approx(0.0)
    assert ev.kelly_back(0.5, 2.0, 0.0) == 0.0
    assert ev.kelly_back(0.6, 2.0, 0.0) == pytest.approx(0.2)
    # break-even price is exactly zero EV
    p, c = 0.4, 0.05
    assert ev.back_ev(p, ev.min_back_odds(p, c), c) == pytest.approx(0.0)
    pa, pb = ev.no_vig(1.9, 1.9)
    assert pa == pytest.approx(0.5) and pb == pytest.approx(0.5)
    assert ev.blend(0.7, 0.5, 1.0) == pytest.approx(0.7)
    assert ev.blend(0.7, 0.5, 0.0) == pytest.approx(0.5)


def test_model_is_symmetric_and_learns(model):
    names = list(model.players)
    a, b = names[0], names[1]
    p_ab = model.predict(a, b, "Clay")["prob"]
    p_ba = model.predict(b, a, "Clay")["prob"]
    assert p_ab + p_ba == pytest.approx(1.0)
    assert model.fitted
    assert model.fit_info["log_loss"] < math.log(2)


def test_features_have_no_lookahead():
    df = data.normalise(demo.make_results(days=30))
    m = RatingModel().build(df)
    first = m.history.iloc[0]
    # nobody has played before the first match
    assert first["n_a"] == 0 and first["n_b"] == 0
    assert first["f_form_30"] == 0


def test_scanner_finds_and_filters_bets(model):
    sched = demo.make_schedule()
    s = scanner.ScanSettings(include_lays=True, min_matches=0)
    out = scanner.evaluate(sched, model, s)
    assert set(out["bet"]) == {"BACK", "LAY"}
    assert out["known"].all()  # full names resolved to "Surname I." names
    picks = scanner.value_bets(out, s)
    assert (picks["ev"] >= s.min_ev).all()
    assert (picks["stake"] <= s.bankroll * s.max_stake_pct / (picks["odds"] - 1) + 1e-9).all() or \
        (picks["liability"] <= s.bankroll * s.max_stake_pct + 1e-9).all()
    # at most one pick per match
    pairs = picks.apply(lambda r: frozenset([r["selection"], r["opponent"]]), axis=1)
    assert pairs.is_unique


def test_surface_inference():
    assert scanner.infer_surface("ATP Wimbledon 2026") == "Grass"
    assert scanner.infer_surface("French Open Men") == "Clay"
    assert scanner.infer_surface("Some Challenger", "Hard") == "Hard"


def test_backtest_runs(model):
    res = backtest.run(model, scanner.ScanSettings())
    assert "summary" in res
    assert res["summary"]["test_matches"] > 0


def test_betfair_catalogue_parsing():
    catalogue = [{
        "marketId": "1.23", "marketStartTime": "2026-09-25T12:00:00.000Z",
        "competition": {"name": "ATP Tokyo"}, "event": {"name": "Fritz v Paul"},
        "runners": [{"selectionId": 1, "runnerName": "Taylor Fritz"},
                    {"selectionId": 2, "runnerName": "Tommy Paul"}],
    }]
    books = {"1.23": {"marketId": "1.23", "status": "OPEN", "inplay": False, "totalMatched": 5000,
                      "runners": [
                          {"selectionId": 1, "ex": {"availableToBack": [{"price": 1.8, "size": 100}],
                                                    "availableToLay": [{"price": 1.82, "size": 50}]}},
                          {"selectionId": 2, "ex": {"availableToBack": [{"price": 2.2, "size": 80}],
                                                    "availableToLay": [{"price": 2.24, "size": 60}]}},
                      ]}}
    df = catalogue_to_schedule(catalogue, books)
    assert len(df) == 1
    r = df.iloc[0]
    assert (r["player1"], r["back1"], r["lay2"], r["back2_size"]) == ("Taylor Fritz", 1.8, 2.24, 80)
