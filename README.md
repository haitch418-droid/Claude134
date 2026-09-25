# 🎾 Tennis EV+ Finder

Reads your historical tennis results (last 30 / 90 / 365 days), builds a rating model from them,
pulls the upcoming schedule and live prices from the **Betfair Exchange API**, and ranks the
**+EV back (and optionally lay) bets** in a dashboard.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, then:

1. **📂 Data**: upload your 30/90/365-day results files (CSV/Excel). Or click *Load demo data* to try it.
2. **📅 Schedule & Odds**: enter your Betfair app key + login and click *Fetch tennis markets*
   (or upload a schedule CSV / type matches in by hand). Check the guessed surfaces and map any
   player names that weren't matched automatically.
3. **💰 Value Bets**: ranked bets with odds, fair odds, EV, Kelly stake and a confidence grade.
   Download the bet slip as CSV.
4. **📈 Backtest**: run this before betting real money. It checks whether the formula beats the market on data it hasn't seen.

## Data format

The [tennis-data.co.uk](http://www.tennis-data.co.uk) layout works as-is (`Date, Tournament, Surface,
Best of, Winner, Loser, WRank, LRank, PSW, PSL, B365W, …`). Any file with at least `Date`, `Winner`,
`Loser` also works. Ranks, surface and odds columns are used when present. Odds are required for the backtest.
Overlapping matches between the 30/90/365 files are removed automatically.

Schedule CSV: `player1, player2, back1, back2` and optionally `lay1, lay2, surface, tournament, start`.
See `sample_data/` for examples (synthetic data).

## The formula

| Step | What it does |
|---|---|
| Ratings | Overall Elo and surface Elo (Hard/Clay/Grass), seeded from world ranking. |
| Form | Win rate over the last 30, 90 and 365 days and on the match surface, shrunk toward 50%. |
| Other | Ranking gap (log scale) and fatigue (matches played in the last 10 days). |
| Combine | `logit P(A) = Σ wᵢ·(Aᵢ − Bᵢ)`. The weights are fitted on your data with regularised logistic regression, using only information available before each match. |
| Market blend | `logit p = α·logit p_model + (1−α)·logit p_market`, where α is the *Model trust* setting. |
| EV (back) | `p·(odds−1)·(1−commission) − (1−p)` |
| EV (lay, per unit liability) | `(1−p)·(1−commission)/(odds−1) − p` |
| Stake | Fractional Kelly, capped at the max % of bankroll per bet. |

## Betfair

You need a Betfair account and an **application key** (see the
[Betfair developer docs](https://docs.developer.betfair.com/)). The app is **read-only**: it lists
tennis `MATCH_ODDS` markets and best back/lay prices. It never places bets.

To avoid typing credentials, create `.streamlit/secrets.toml` (gitignored):

```toml
BETFAIR_APP_KEY = "..."
BETFAIR_USERNAME = "..."
BETFAIR_PASSWORD = "..."
```

Environment variables with the same names also work. Certificate login is available in
`tennis_ev/betfair.py` (`BetfairClient.login_cert`) for unattended use.

## Honest caveats

* Exchange prices are sharp. The model only adds value if the backtest shows its blended log-loss
  **below the market's**. If it doesn't, the "value" bets are noise.
* The model doesn't know about injuries, withdrawals or motivation. Very large edges usually mean
  it's missing information, which is why the grades penalise them.
* Track closing-line value on real bets. It shows whether you have an edge much sooner than profit and loss does.

## Tests

```bash
pip install pytest && pytest -q
```
