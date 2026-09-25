"""Minimal Betfair Exchange API-NG client for tennis MATCH_ODDS markets.

Read-only: it logs in, lists upcoming tennis match-odds markets and pulls the
best available back/lay prices. It never places bets.

Docs: https://docs.developer.betfair.com/
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import requests

TENNIS_EVENT_TYPE = "2"

ENDPOINTS = {
    # region: (identity host, betting API host)
    "Global (.com)": ("https://identitysso.betfair.com", "https://api.betfair.com"),
    "Australia/NZ (.com.au)": ("https://identitysso.betfair.com.au", "https://api-au.betfair.com"),
    "Italy (.it)": ("https://identitysso.betfair.it", "https://api.betfair.com"),
    "Spain (.es)": ("https://identitysso.betfair.es", "https://api.betfair.com"),
}

# MarketBook weight: EX_BEST_OFFERS costs 5 points per market, 200 max per call.
BOOK_CHUNK = 40


class BetfairError(RuntimeError):
    pass


@dataclass
class BetfairClient:
    app_key: str
    session_token: str | None = None
    region: str = "Global (.com)"
    timeout: float = 20.0

    @property
    def _hosts(self):
        return ENDPOINTS.get(self.region, ENDPOINTS["Global (.com)"])

    # ------------------------------------------------------------------ auth
    def login(self, username: str, password: str) -> str:
        """Interactive (non-certificate) login. Returns and stores the session token."""
        identity, _ = self._hosts
        r = requests.post(
            f"{identity}/api/login",
            data={"username": username, "password": password},
            headers={"X-Application": self.app_key, "Accept": "application/json"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "SUCCESS":
            raise BetfairError(f"Login failed: {body.get('error', body)}")
        self.session_token = body["token"]
        return self.session_token

    def login_cert(self, username: str, password: str, cert_file: str, key_file: str) -> str:
        """Non-interactive certificate login (recommended for bots)."""
        identity, _ = self._hosts
        host = identity.replace("identitysso", "identitysso-cert")
        r = requests.post(
            f"{host}/api/certlogin",
            data={"username": username, "password": password},
            headers={"X-Application": self.app_key},
            cert=(cert_file, key_file), timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("loginStatus") != "SUCCESS":
            raise BetfairError(f"Cert login failed: {body.get('loginStatus', body)}")
        self.session_token = body["sessionToken"]
        return self.session_token

    def keep_alive(self) -> None:
        identity, _ = self._hosts
        requests.post(f"{identity}/api/keepAlive", headers=self._headers(), timeout=self.timeout)

    # --------------------------------------------------------------- betting
    def _headers(self) -> dict:
        if not self.session_token:
            raise BetfairError("Not logged in")
        return {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _call(self, method: str, params: dict):
        _, api = self._hosts
        r = requests.post(f"{api}/exchange/betting/rest/v1.0/{method}/",
                          json=params, headers=self._headers(), timeout=self.timeout)
        if r.status_code != 200:
            raise BetfairError(f"{method} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def list_tennis_markets(self, hours_ahead: int = 24, in_play: bool = False,
                            max_results: int = 500) -> list[dict]:
        now = dt.datetime.now(dt.timezone.utc)
        flt = {
            "eventTypeIds": [TENNIS_EVENT_TYPE],
            "marketTypeCodes": ["MATCH_ODDS"],
            "marketStartTime": {
                "from": (now - dt.timedelta(hours=6 if in_play else 0)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": (now + dt.timedelta(hours=hours_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        }
        if not in_play:
            flt["inPlayOnly"] = False
        return self._call("listMarketCatalogue", {
            "filter": flt,
            "maxResults": max_results,
            "sort": "FIRST_TO_START",
            "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "COMPETITION", "MARKET_START_TIME"],
        })

    def list_books(self, market_ids: list[str]) -> list[dict]:
        books = []
        for i in range(0, len(market_ids), BOOK_CHUNK):
            books += self._call("listMarketBook", {
                "marketIds": market_ids[i:i + BOOK_CHUNK],
                "priceProjection": {"priceData": ["EX_BEST_OFFERS"], "virtualise": True},
            })
        return books

    def tennis_schedule(self, hours_ahead: int = 24, include_in_play: bool = False) -> pd.DataFrame:
        """Upcoming tennis matches with best back/lay prices, one row per match."""
        catalogue = self.list_tennis_markets(hours_ahead, in_play=include_in_play)
        books = {b["marketId"]: b for b in self.list_books([m["marketId"] for m in catalogue])}
        return catalogue_to_schedule(catalogue, books, include_in_play=include_in_play)


def _best(prices: list[dict]) -> tuple[float | None, float]:
    if not prices:
        return None, 0.0
    return prices[0]["price"], prices[0]["size"]


def catalogue_to_schedule(catalogue: list[dict], books: dict[str, dict],
                          include_in_play: bool = False) -> pd.DataFrame:
    rows = []
    for m in catalogue:
        runners = m.get("runners", [])
        if len(runners) != 2:
            continue
        book = books.get(m["marketId"])
        if not book or book.get("status") != "OPEN":
            continue
        if book.get("inplay") and not include_in_play:
            continue
        by_id = {r["selectionId"]: r for r in book.get("runners", [])}
        row = {
            "market_id": m["marketId"],
            "start": pd.to_datetime(m.get("marketStartTime"), utc=True),
            "tournament": (m.get("competition") or {}).get("name", "")
            or (m.get("event") or {}).get("name", ""),
            "event": (m.get("event") or {}).get("name", ""),
            "matched": book.get("totalMatched", 0.0),
            "in_play": bool(book.get("inplay")),
        }
        for idx, r in enumerate(runners, start=1):
            ex = by_id.get(r["selectionId"], {}).get("ex", {})
            back, back_size = _best(ex.get("availableToBack", []))
            lay, lay_size = _best(ex.get("availableToLay", []))
            row.update({
                f"player{idx}": r["runnerName"], f"selection{idx}": r["selectionId"],
                f"back{idx}": back, f"back{idx}_size": back_size,
                f"lay{idx}": lay, f"lay{idx}_size": lay_size,
            })
        rows.append(row)
    return pd.DataFrame(rows)
