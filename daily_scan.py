"""Scan today's tennis on Betfair and save the value bets — no browser needed.

    python daily_scan.py                       # today's matches until midnight
    python daily_scan.py --hours 6             # next 6 hours instead
    python daily_scan.py --tz Australia/Sydney --min-ev 0.04

Reads results files from ./data and Betfair credentials from the environment
(BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD). Writes
output/value_bets_<date>.csv. Schedule it with cron or Windows Task Scheduler.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tennis_ev import daily, scanner
from tennis_ev.betfair import ENDPOINTS, BetfairClient
from tennis_ev.model import RatingModel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(daily.DATA_DIR), help="folder with results files")
    ap.add_argument("--out", default="output", help="folder for the bet slip CSV")
    ap.add_argument("--tz", default="Europe/London", help="your timezone, defines 'today'")
    ap.add_argument("--hours", type=float, default=None, help="scan the next N hours instead of today")
    ap.add_argument("--region", default="Global (.com)", choices=list(ENDPOINTS))
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--commission", type=float, default=0.05)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--max-stake", type=float, default=0.03, help="fraction of bankroll")
    ap.add_argument("--min-ev", type=float, default=0.03)
    ap.add_argument("--min-odds", type=float, default=1.3)
    ap.add_argument("--max-odds", type=float, default=6.0)
    ap.add_argument("--model-trust", type=float, default=0.6)
    ap.add_argument("--min-matches", type=int, default=8)
    ap.add_argument("--lays", action="store_true", help="include lay bets")
    args = ap.parse_args(argv)

    matches, sources = daily.load_results_folder(args.data)
    if matches.empty:
        print(f"No results files found in {args.data}. Put your 30/90/365-day files there.")
        return 1
    print("Data:", "; ".join(sources))

    creds = daily.credentials()
    if not creds["app_key"]:
        print("Set BETFAIR_APP_KEY, BETFAIR_USERNAME and BETFAIR_PASSWORD environment variables.")
        return 1

    model = RatingModel().build(matches)
    model.fit()

    client = BetfairClient(app_key=creds["app_key"], region=args.region)
    schedule = daily.fetch_today(client, creds["username"], creds["password"],
                                 tz=args.tz, hours_ahead=args.hours)
    window = f"next {args.hours:g}h" if args.hours else f"today ({args.tz})"
    print(f"Betfair: {len(schedule)} open tennis matches, {window}")

    settings = scanner.ScanSettings(
        bankroll=args.bankroll, commission=args.commission, kelly_fraction=args.kelly,
        max_stake_pct=args.max_stake, min_ev=args.min_ev, min_odds=args.min_odds,
        max_odds=args.max_odds, model_weight=args.model_trust, min_matches=args.min_matches,
        include_lays=args.lays)
    evaluated = scanner.evaluate(schedule, model, settings) if len(schedule) else pd.DataFrame()
    picks = scanner.value_bets(evaluated, settings) if len(evaluated) else evaluated

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"value_bets_{pd.Timestamp.now(tz=args.tz):%Y-%m-%d_%H%M}.csv"
    picks.to_csv(path, index=False)

    if picks.empty:
        print("No value bets pass the filters.")
    else:
        cols = ["grade", "start", "tournament", "bet", "selection", "opponent", "odds",
                "fair_odds", "ev", "stake"]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(picks[cols].round({"odds": 2, "fair_odds": 2, "ev": 3, "stake": 2})
                  .to_string(index=False))
    print(f"Saved {len(picks)} bets to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
