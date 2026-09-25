import datetime as dt

import pandas as pd
import pytest

import daily_scan
from tennis_ev import daily, demo
from tennis_ev.betfair import BetfairClient, BetfairError, end_of_day


def test_end_of_day_uses_local_midnight():
    now = dt.datetime(2026, 9, 25, 22, 30, tzinfo=dt.timezone.utc)
    # 22:30 UTC is already 08:30 on the 26th in Sydney
    assert end_of_day("Australia/Sydney", now).date() == dt.date(2026, 9, 27)
    eod = end_of_day("Europe/London", now)
    assert eod.date() == dt.date(2026, 9, 26) and eod.hour == 0


class FakeClient(BetfairClient):
    def __init__(self, expire_first=False):
        super().__init__(app_key="k")
        self.logins, self.calls, self.expire_first = 0, [], expire_first

    def login(self, username, password):
        self.logins += 1
        self.session_token = f"t{self.logins}"
        return self.session_token

    def tennis_schedule(self, hours_ahead=24, include_in_play=False, until=None):
        self.calls.append((hours_ahead, until))
        if self.expire_first and len(self.calls) == 1:
            raise BetfairError('listMarketCatalogue HTTP 400: {"faultcode":"INVALID_SESSION_INFORMATION"}')
        return demo.make_schedule(4)


def test_fetch_today_logs_in_and_uses_midnight():
    c = FakeClient()
    df = daily.fetch_today(c, "u", "p", tz="Europe/London")
    assert c.logins == 1 and len(df) == 4
    assert c.calls[0][1] is not None  # an end-of-day cut-off was sent


def test_fetch_today_next_hours_mode():
    c = FakeClient()
    daily.fetch_today(c, "u", "p", hours_ahead=6)
    assert c.calls[0] == (6, None)


def test_fetch_today_relogs_on_expired_session():
    c = FakeClient(expire_first=True)
    c.session_token = "old"
    daily.fetch_today(c, "u", "p")
    assert c.logins == 1 and len(c.calls) == 2


def test_fetch_today_raises_other_errors():
    class Broken(FakeClient):
        def tennis_schedule(self, *a, **k):
            raise BetfairError("HTTP 500")
    c = Broken()
    c.session_token = "t"
    with pytest.raises(BetfairError):
        daily.fetch_today(c, "u", "p")


def test_daily_scan_cli(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    demo.make_results(days=200).to_csv(data_dir / "atp_365d.csv", index=False)
    monkeypatch.setenv("BETFAIR_APP_KEY", "k")
    monkeypatch.setenv("BETFAIR_USERNAME", "u")
    monkeypatch.setenv("BETFAIR_PASSWORD", "p")
    monkeypatch.setattr(BetfairClient, "login", lambda self, u, p: setattr(self, "session_token", "t"))
    monkeypatch.setattr(BetfairClient, "tennis_schedule",
                        lambda self, hours_ahead=24, include_in_play=False, until=None: demo.make_schedule())
    rc = daily_scan.main(["--data", str(data_dir), "--out", str(tmp_path / "out"), "--min-matches", "0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "24 open tennis matches" in out
    files = list((tmp_path / "out").glob("value_bets_*.csv"))
    assert len(files) == 1
    assert len(pd.read_csv(files[0])) > 0


def test_app_finds_todays_games_automatically(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BETFAIR_APP_KEY", "k")
    monkeypatch.setenv("BETFAIR_USERNAME", "u")
    monkeypatch.setenv("BETFAIR_PASSWORD", "p")
    calls = []
    monkeypatch.setattr(BetfairClient, "login", lambda self, u, p: setattr(self, "session_token", "t"))

    def fake_schedule(self, hours_ahead=24, include_in_play=False, until=None):
        calls.append(until)
        return demo.make_schedule()
    monkeypatch.setattr(BetfairClient, "tennis_schedule", fake_schedule)
    monkeypatch.setattr(daily, "load_results_folder", lambda folder=None: (
        __import__("tennis_ev.data", fromlist=["normalise"]).normalise(demo.make_results(days=200)),
        ["demo.csv"]))

    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert len(calls) == 1 and calls[0] is not None  # fetched today's games with no clicks
    assert any("24 open tennis matches" in s.value for s in at.success)
    assert any("VALUE BETS" in m.value.upper() for m in at.markdown)
