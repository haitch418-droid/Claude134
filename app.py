"""Tennis EV+ Finder — Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tennis_ev import backtest, daily, data, demo, scanner
from tennis_ev.betfair import ENDPOINTS, BetfairClient
from tennis_ev.model import FEATURE_LABELS, FEATURES, RatingModel

st.set_page_config(page_title="Tennis EV+ Finder", page_icon="🎾", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.kpi {border-radius: 12px; padding: 14px 18px; background: rgba(127,127,127,.08);
      border: 1px solid rgba(127,127,127,.18);}
.kpi .label {font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .04em;}
.kpi .value {font-size: 1.6rem; font-weight: 700; margin-top: 2px;}
.kpi .sub {font-size: .8rem; opacity: .65;}
.betcard {border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
          border: 1px solid rgba(127,127,127,.22); background: rgba(127,127,127,.05);}
.betcard .title {font-size: 1.05rem; font-weight: 700;}
.betcard .meta {font-size: .82rem; opacity: .7;}
.tag {display:inline-block; padding: 2px 9px; border-radius: 999px; font-weight: 700;
      font-size: .75rem; margin-right: 6px;}
.tag.BACK {background:#1f6feb33; color:#58a6ff;} .tag.LAY {background:#db61a233; color:#f778ba;}
.tag.A {background:#23863633; color:#3fb950;} .tag.B {background:#9e6a0333; color:#d29922;}
.tag.C {background:#6e768133; color:#8b949e;}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("matches", None)
ss.setdefault("schedule", None)
ss.setdefault("sources", [])
ss.setdefault("aliases", {})

if ss.matches is None and not ss.get("data_dir_checked"):
    ss.data_dir_checked = True
    try:
        _m, _src = daily.load_results_folder()
        if len(_m):
            ss.matches, ss.sources = _m, [f"data/ {s}" for s in _src]
    except Exception as e:  # noqa: BLE001
        st.toast(f"Could not read data/ folder: {e}")


@st.cache_resource(show_spinner="Building ratings from match history…")
def build_model(matches: pd.DataFrame) -> RatingModel:
    m = RatingModel().build(matches)
    m.fit()
    return m


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='label'>{label}</div>"
                 f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>",
                 unsafe_allow_html=True)


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:
        return os.environ.get(name, "")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎾 Tennis EV+ Finder")
    st.caption("Ratings model × Betfair prices → ranked value bets")

    st.subheader("Bankroll & staking")
    bankroll = st.number_input("Bankroll", 10.0, 1e8, 1000.0, step=100.0)
    kelly = st.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05,
                      help="Share of the full Kelly stake to bet. 0.25 = quarter Kelly (recommended).")
    max_stake = st.slider("Max stake per bet (% bankroll)", 0.5, 10.0, 3.0, 0.5) / 100
    commission = st.slider("Betfair commission (%)", 0.0, 10.0, 5.0, 0.5,
                           help="Your market base rate. Charged on net winnings.") / 100

    st.subheader("Filters")
    min_ev = st.slider("Minimum EV (%)", 0.0, 25.0, 3.0, 0.5) / 100
    odds_range = st.slider("Odds range", 1.01, 20.0, (1.3, 6.0), 0.05)
    min_matches = st.slider("Min matches of data per player", 0, 50, 8,
                            help="Skip players the model barely knows.")
    min_liq = st.number_input("Min liquidity at price", 0.0, 1e6, 0.0, step=10.0)
    include_lays = st.toggle("Include LAY bets", value=False)

    st.subheader("Model")
    model_weight = st.slider(
        "Model trust vs market", 0.0, 1.0, 0.6, 0.05,
        help="1.0 = pure model, 0.0 = copy the market. Blending with the sharp exchange price "
             "filters out many false 'value' signals.")
    default_surface = st.selectbox("Default surface (when unknown)", data.SURFACES)

settings = scanner.ScanSettings(
    bankroll=bankroll, commission=commission, kelly_fraction=kelly, max_stake_pct=max_stake,
    min_ev=min_ev, min_odds=odds_range[0], max_odds=odds_range[1], model_weight=model_weight,
    min_matches=min_matches, include_lays=include_lays, default_surface=default_surface,
    min_liquidity=min_liq,
)

tab_bets, tab_sched, tab_data, tab_analyze, tab_bt, tab_help = st.tabs(
    ["💰 Value Bets", "📅 Schedule & Odds", "📂 Data", "🔍 Match Analyzer",
     "📈 Backtest", "ℹ️ How it works"])

model = build_model(ss.matches) if ss.matches is not None and len(ss.matches) else None

# --------------------------------------------------------------------- data
with tab_data:
    st.header("Historical results")
    st.write("Upload your **30-, 90- and 365-day** results files (CSV or Excel), or drop them in the "
             "project's `data/` folder so they load automatically every time the app opens. "
             "Overlapping matches between files are removed automatically. "
             "The tennis-data.co.uk layout works as-is; any file with `Date`, `Winner`, `Loser` "
             "columns works (ranks, surface and odds columns are used when present).")
    c1, c2, c3 = st.columns(3)
    up30 = c1.file_uploader("Last 30 days", type=["csv", "xlsx", "xls"], key="u30")
    up90 = c2.file_uploader("Last 90 days", type=["csv", "xlsx", "xls"], key="u90")
    up365 = c3.file_uploader("Last 365 days", type=["csv", "xlsx", "xls"], key="u365")
    extra = st.file_uploader("Additional files (e.g. separate ATP / WTA)", type=["csv", "xlsx", "xls"],
                             accept_multiple_files=True, key="uextra")
    tour = st.radio("Tour for files without a Tour column", ["Auto (from filename)", "ATP", "WTA"],
                    horizontal=True)

    b1, b2, b3 = st.columns([1, 1, 3])
    if b1.button("Load uploaded files", type="primary", width="stretch"):
        frames, sources = [], []
        for f in [up30, up90, up365, *(extra or [])]:
            if f is None:
                continue
            try:
                t = None if tour.startswith("Auto") else tour
                df = data.read_any(f.getvalue(), f.name, tour=t)
                frames.append(df)
                sources.append(f"{f.name}: {len(df):,} matches")
            except Exception as e:  # noqa: BLE001 - show any parse error to the user
                st.error(f"{f.name}: {e}")
        if frames:
            ss.matches = data.combine(frames)
            ss.sources = sources
            st.rerun()
        else:
            st.warning("Choose at least one file first.")
    if b2.button("Load demo data", width="stretch",
                 help="Synthetic players and odds so you can try every screen."):
        ss.matches = data.normalise(demo.make_results())
        ss.sources = ["Synthetic demo data (not real players)"]
        ss.schedule = demo.make_schedule()
        st.rerun()

    if ss.matches is not None and model is not None:
        m = ss.matches
        st.success(" · ".join(ss.sources))
        k = st.columns(5)
        kpi(k[0], "Matches", f"{len(m):,}", f"{m['date'].min():%d %b %Y} → {m['date'].max():%d %b %Y}")
        kpi(k[1], "Players", f"{len(model.players):,}")
        kpi(k[2], "With odds", f"{m['odds_winner'].notna().mean():.0%}", "needed for backtest")
        fi = model.fit_info
        kpi(k[3], "Formula", "Fitted" if model.fitted else "Default",
            f"{fi.get('rows', 0):,} training matches")
        kpi(k[4], "Accuracy", f"{fi['accuracy']:.1%}" if "accuracy" in fi else "—",
            "in-sample pick rate")

        st.subheader("Fitted formula weights")
        st.caption("logit P(A wins) = Σ weight × (A − B) for each factor. Learned from your data.")
        wdf = pd.DataFrame({"Factor": [FEATURE_LABELS[f] for f in FEATURES], "Weight": model.weights})
        fig = px.bar(wdf, x="Weight", y="Factor", orientation="h", color="Weight",
                     color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Player ratings")
        q = st.text_input("Search player")
        pt = model.player_table()
        if q:
            pt = pt[pt["Player"].str.contains(q, case=False)]
        st.dataframe(pt, width="stretch", height=420, hide_index=True)

        by_month = m.groupby([m["date"].dt.to_period("M").astype(str), "surface"]).size().reset_index(name="matches")
        fig = px.bar(by_month, x="date", y="matches", color="surface", title="Matches per month")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No data loaded yet. Upload your files or click **Load demo data**.")

# ----------------------------------------------------------------- schedule
with tab_sched:
    st.header("Match schedule & prices")
    source = st.radio("Source", ["Betfair Exchange (live)", "Upload schedule CSV", "Enter manually"],
                      horizontal=True)

    if source.startswith("Betfair"):
        c1, c2, c3 = st.columns([2, 1, 2])
        window = c1.radio("Window", ["Today (until midnight)", "Next N hours"], horizontal=True)
        hours = c2.number_input("Hours", 1, 72, 6, disabled=window.startswith("Today"))
        tz = c3.text_input("Your timezone", value=secret("TENNIS_TZ") or "Europe/London",
                           help="Defines 'today'. e.g. Europe/London, Australia/Sydney, America/New_York")
        c1, c2, c3 = st.columns([2, 1, 2])
        auto = c1.toggle("Auto-refresh", value=True,
                         help="Re-fetch prices and new matches while this page is open.")
        every = c2.number_input("Every (min)", 1, 120, 5, disabled=not auto)
        inplay = c3.checkbox("Include in-play markets", value=False)

        with st.expander("Betfair login", expanded=not secret("BETFAIR_APP_KEY")):
            with st.form("bf"):
                c1, c2 = st.columns(2)
                app_key = c1.text_input("App key", value=secret("BETFAIR_APP_KEY"), type="password")
                region = c2.selectbox("Region", list(ENDPOINTS))
                c1, c2 = st.columns(2)
                user = c1.text_input("Username", value=secret("BETFAIR_USERNAME"))
                pwd = c2.text_input("Password", value=secret("BETFAIR_PASSWORD"), type="password")
                saved = st.form_submit_button("Save login for this session")
            st.caption("Save `BETFAIR_APP_KEY`, `BETFAIR_USERNAME`, `BETFAIR_PASSWORD` in "
                       "`.streamlit/secrets.toml` and today's games load by themselves when the app opens.")
        if saved:
            ss.bf_creds = {"app_key": app_key, "username": user, "password": pwd, "region": region}
            ss.pop("bf_client", None)
        creds = ss.get("bf_creds") or {"app_key": app_key, "username": user, "password": pwd,
                                        "region": region}
        ready = bool(creds["app_key"] and ((creds["username"] and creds["password"])
                                           or ss.get("bf_client")))

        now = pd.Timestamp.now(tz="UTC")
        last = ss.get("bf_last_try")
        stale = last is None or (auto and now - last >= pd.Timedelta(minutes=every))
        fetch_now = st.button("🔄 Find today's games now" if window.startswith("Today")
                              else f"🔄 Find games in the next {hours}h", type="primary")
        if ready and (fetch_now or stale or ss.get("bf_window") != (window, hours, tz, inplay)):
            ss.bf_last_try = now
            ss.bf_window = (window, hours, tz, inplay)
            try:
                client = ss.get("bf_client") or BetfairClient(app_key=creds["app_key"],
                                                              region=creds["region"])
                with st.spinner("Finding today's tennis on Betfair…"):
                    sched = daily.fetch_today(
                        client, creds["username"], creds["password"], tz=tz,
                        include_in_play=inplay,
                        hours_ahead=None if window.startswith("Today") else hours)
                ss.bf_client = client
                ss.schedule = sched
                ss.bf_updated = now
                ss.pop("bf_error", None)
            except Exception as e:  # noqa: BLE001 - surface any login/API error
                ss.bf_error = str(e)
        elif not ready:
            st.info("Enter your Betfair app key, username and password above to find today's games.")

        if ss.get("bf_error"):
            st.error(f"Betfair: {ss.bf_error}")
        elif ss.get("bf_updated") is not None:
            st.success(f"{len(ss.schedule)} open tennis matches · updated "
                       f"{ss.bf_updated.tz_convert(tz):%H:%M:%S}"
                       + (f" · refreshes every {every} min" if auto else ""))

        if auto and ready:
            @st.fragment(run_every="30s")
            def _ticker():
                last_try = ss.get("bf_last_try")
                if last_try is not None and pd.Timestamp.now(tz="UTC") - last_try >= pd.Timedelta(minutes=every):
                    st.rerun(scope="app")
            _ticker()

    elif source.startswith("Upload"):
        st.write("Columns: `player1, player2, back1, back2` (required) and optionally "
                 "`lay1, lay2, surface, tournament, start`.")
        f = st.file_uploader("Schedule file", type=["csv", "xlsx", "xls"], key="usched")
        if f is not None and st.button("Load schedule", type="primary"):
            try:
                ss.schedule = scanner.read_schedule(f.getvalue(), f.name)
                st.success(f"Loaded {len(ss.schedule)} matches.")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    else:
        template = ss.schedule if ss.schedule is not None and not ss.schedule.empty else pd.DataFrame(
            {"start": [pd.Timestamp.now().floor("h")], "tournament": [""], "surface": [default_surface],
             "player1": [""], "player2": [""], "back1": [2.0], "back2": [2.0],
             "lay1": [np.nan], "lay2": [np.nan]})
        cols = [c for c in ["start", "tournament", "surface", "player1", "player2",
                            "back1", "back2", "lay1", "lay2"] if c in template]
        edited = st.data_editor(template[cols], num_rows="dynamic", width="stretch",
                                column_config={"surface": st.column_config.SelectboxColumn(
                                    options=data.SURFACES)})
        if st.button("Use this schedule", type="primary"):
            ss.schedule = edited.dropna(subset=["player1", "player2"])

    if ss.schedule is not None and not ss.schedule.empty:
        sched = ss.schedule.copy()
        if "surface" not in sched:
            sched["surface"] = [scanner.infer_surface(t, default_surface)
                                for t in sched.get("tournament", pd.Series([""] * len(sched)))]
        st.subheader(f"{len(sched)} matches")
        st.caption("Surface is guessed from the tournament name — correct it here if needed.")
        show = [c for c in ["start", "tournament", "surface", "player1", "back1", "lay1",
                            "player2", "back2", "lay2", "matched"] if c in sched]
        edited = st.data_editor(
            sched[show], width="stretch", hide_index=True, key="sched_edit",
            disabled=[c for c in show if c != "surface"],
            column_config={"surface": st.column_config.SelectboxColumn(options=data.SURFACES),
                           "matched": st.column_config.NumberColumn("Matched £", format="%.0f")})
        sched["surface"] = edited["surface"].values
        ss.schedule = sched

        if model is not None:
            unknown = sorted({n for n in pd.concat([sched["player1"], sched["player2"]])
                              if not scanner.NameMatcher(model.players.keys(), ss.aliases).resolve(n)})
            if unknown:
                with st.expander(f"⚠️ {len(unknown)} names not found in your data — map them"):
                    st.caption("Pick the matching player from your results files. "
                               "Unmapped players are excluded from value bets.")
                    known = [""] + sorted(model.players)
                    for n in unknown[:60]:
                        choice = st.selectbox(n, known, key=f"alias_{n}")
                        if choice:
                            ss.aliases[n] = choice

# ---------------------------------------------------------------- value bets
with tab_bets:
    if model is None:
        st.info("👈 Start in the **📂 Data** tab: upload your results files or load demo data.")
    elif ss.schedule is None or ss.schedule.empty:
        st.info("Now load upcoming matches in the **📅 Schedule & Odds** tab (Betfair, CSV or manual).")
    else:
        evaluated = scanner.evaluate(ss.schedule, model, settings, ss.aliases)
        picks = scanner.value_bets(evaluated, settings)

        k = st.columns(5)
        kpi(k[0], "Value bets", f"{len(picks)}", f"from {len(ss.schedule)} matches")
        kpi(k[1], "Total stake", f"{picks['liability'].sum():,.2f}" if len(picks) else "0",
            f"{picks['liability'].sum() / bankroll:.1%} of bankroll" if len(picks) else "")
        kpi(k[2], "Expected profit", f"{picks['exp_profit'].sum():+,.2f}" if len(picks) else "0")
        kpi(k[3], "Avg EV", f"{picks['ev'].mean():+.1%}" if len(picks) else "—")
        kpi(k[4], "Grade A", f"{(picks['grade'] == 'A').sum()}" if len(picks) else "0",
            "strong edge + solid data")
        st.write("")

        if picks.empty:
            st.warning("No bets pass your filters right now. Lower the minimum EV, widen the odds "
                       "range or raise model trust in the sidebar.")
        else:
            view = st.radio("View", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")
            if view == "Cards":
                cols = st.columns(2)
                for i, r in picks.iterrows():
                    start = pd.to_datetime(r["start"])
                    when = f"{start:%a %d %b %H:%M}" if pd.notna(start) else ""
                    mkt = f"{r['market_prob']:.0%}" if pd.notna(r["market_prob"]) else "—"
                    stake_txt = (f"Stake <b>{r['stake']:,.2f}</b>" if r["bet"] == "BACK" else
                                 f"Lay stake <b>{r['stake']:,.2f}</b> (liability {r['liability']:,.2f})")
                    cols[i % 2].markdown(f"""
<div class='betcard'>
  <span class='tag {r['bet']}'>{r['bet']}</span><span class='tag {r['grade']}'>Grade {r['grade']}</span>
  <span class='meta'>{when} · {r['tournament']} · {r['surface']}</span>
  <div class='title'>{r['selection']} <span style='opacity:.6;font-weight:400'>vs {r['opponent']}</span></div>
  <div style='margin-top:6px'>Odds <b>{r['odds']:.2f}</b> · Fair <b>{r['fair_odds']:.2f}</b> ·
  EV <b style='color:#3fb950'>{r['ev']:+.1%}</b></div>
  <div class='meta'>Model {r['model_prob']:.0%} · Market {mkt} · Blended {r['prob']:.0%} ·
  {stake_txt} · Exp. profit {r['exp_profit']:+,.2f}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.dataframe(
                    picks[["grade", "start", "tournament", "surface", "bet", "selection", "opponent",
                           "odds", "fair_odds", "prob", "market_prob", "ev", "stake", "liability",
                           "exp_profit", "liquidity", "data_matches"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "start": st.column_config.DatetimeColumn("Start", format="DD MMM HH:mm"),
                        "odds": st.column_config.NumberColumn("Odds", format="%.2f"),
                        "fair_odds": st.column_config.NumberColumn("Fair", format="%.2f"),
                        "prob": st.column_config.ProgressColumn("Win prob", min_value=0, max_value=1, format="%.2f"),
                        "market_prob": st.column_config.NumberColumn("Market", format="%.2f"),
                        "ev": st.column_config.NumberColumn("EV", format="%.3f"),
                        "stake": st.column_config.NumberColumn("Stake", format="%.2f"),
                        "liability": st.column_config.NumberColumn("Risk", format="%.2f"),
                        "exp_profit": st.column_config.NumberColumn("Exp. £", format="%.2f"),
                        "liquidity": st.column_config.NumberColumn("Avail.", format="%.0f"),
                        "data_matches": st.column_config.NumberColumn("Data n"),
                    })
            st.download_button("⬇️ Download bet slip (CSV)", picks.to_csv(index=False).encode(),
                               "value_bets.csv", "text/csv")

        with st.expander("All selections (including non-value)"):
            if not evaluated.empty:
                fig = px.scatter(evaluated, x="odds", y="ev", color="bet", symbol="grade",
                                 hover_data=["selection", "opponent", "prob", "market_prob"],
                                 log_x=True, title="EV vs odds for every selection")
                fig.add_hline(y=settings.min_ev, line_dash="dash", annotation_text="min EV")
                fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, width="stretch")
                st.dataframe(evaluated.drop(columns=["market_id"], errors="ignore"),
                             width="stretch", hide_index=True)

# ------------------------------------------------------------------ analyzer
with tab_analyze:
    st.header("Match analyzer")
    if model is None:
        st.info("Load data first.")
    else:
        names = sorted(model.players)
        c1, c2, c3 = st.columns([2, 2, 1])
        a = c1.selectbox("Player A", names, index=0)
        b = c2.selectbox("Player B", names, index=min(1, len(names) - 1))
        surf = c3.selectbox("Surface", data.SURFACES)
        c1, c2 = st.columns(2)
        oa = c1.number_input(f"Back odds {a}", 1.01, 1000.0, 2.0, 0.01)
        ob = c2.number_input(f"Back odds {b}", 1.01, 1000.0, 2.0, 0.01)
        pred = model.predict(a, b, surf)
        from tennis_ev import ev as evm
        pm = evm.no_vig(oa, ob)[0]
        pf = evm.blend(pred["prob"], pm, model_weight)
        k = st.columns(4)
        kpi(k[0], f"Model: {a}", f"{pred['prob']:.1%}", f"fair odds {1 / pred['prob']:.2f}")
        kpi(k[1], "Market (no-vig)", f"{pm:.1%}")
        kpi(k[2], "Blended", f"{pf:.1%}", f"fair {1 / pf:.2f} / {1 / (1 - pf):.2f}")
        ev_a, ev_b = evm.back_ev(pf, oa, commission), evm.back_ev(1 - pf, ob, commission)
        best = (a, ev_a) if ev_a >= ev_b else (b, ev_b)
        kpi(k[3], "Best back EV", f"{best[1]:+.1%}", best[0])

        contrib = pd.DataFrame({
            "Factor": [FEATURE_LABELS[f] for f in FEATURES],
            "A − B": [pred["features"][f] for f in FEATURES],
            "Contribution": [pred["contributions"][f] for f in FEATURES],
        })
        fig = go.Figure(go.Bar(x=contrib["Contribution"], y=contrib["Factor"], orientation="h",
                               marker_color=np.where(contrib["Contribution"] >= 0, "#3fb950", "#f85149")))
        fig.update_layout(title=f"Why the model rates {a} this way (→ favours {a}, ← favours {b})",
                          height=340, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        pt = model.player_table()
        st.dataframe(pt[pt["Player"].isin([a, b])], hide_index=True, width="stretch")

# ------------------------------------------------------------------ backtest
with tab_bt:
    st.header("Walk-forward backtest")
    st.write("Fits the formula on the earlier part of your data, then replays your current sidebar "
             "rules on the later part using the historical odds in your files. "
             "**Run this before staking real money** — it is the only evidence the edge is real.")
    if model is None:
        st.info("Load data first.")
    else:
        c1, c2 = st.columns(2)
        frac = c1.slider("Training share", 0.3, 0.9, 0.6, 0.05)
        staking = c2.radio("Staking", ["kelly", "flat"], horizontal=True,
                           format_func=lambda s: "Fractional Kelly" if s == "kelly" else "Flat 1%")
        if st.button("Run backtest", type="primary"):
            with st.spinner("Replaying…"):
                ss.bt = backtest.run(model, settings, frac, staking)
        res = ss.get("bt")
        if res and "error" in res:
            st.error(res["error"])
        elif res:
            s = res["summary"]
            k = st.columns(5)
            kpi(k[0], "Bets", f"{s['bets']:,}", f"from {s['test_matches']:,} test matches")
            if s["bets"]:
                kpi(k[1], "ROI", f"{s['roi']:+.1%}", f"on {s['staked']:,.0f} staked")
                kpi(k[2], "Profit", f"{s['profit']:+,.2f}", f"final bank {s['final_bankroll']:,.0f}")
                kpi(k[3], "Hit rate", f"{s['hit_rate']:.1%}", f"avg odds {s['avg_odds']:.2f}")
                kpi(k[4], "Max drawdown", f"{s['max_drawdown']:.1%}")
            st.caption(f"Test period from {s['split_date']:%d %b %Y}. Log-loss (lower is better): "
                       f"model {s['log_loss_model']:.4f} · market {s['log_loss_market']:.4f} · "
                       f"blend {s['log_loss_blend']:.4f}")
            if s["log_loss_blend"] >= s["log_loss_market"]:
                st.warning("The blended probability is not beating the market's own prices on this "
                           "data. Treat any 'value' as noise until more data or tuning fixes that.")
            bets = res["bets"]
            if len(bets):
                fig = px.line(bets, x="date", y="bankroll", title="Bankroll")
                fig.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, width="stretch")
            cal = res["calibration"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                     line=dict(dash="dash", color="gray")))
            fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["actual"], mode="markers+lines",
                                     name="model", marker=dict(size=np.sqrt(cal["n"]) + 4)))
            fig.update_layout(title="Calibration (test period)", xaxis_title="Predicted",
                              yaxis_title="Actual win rate", height=380, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")
            if len(bets):
                st.dataframe(bets, width="stretch", hide_index=True)

# ---------------------------------------------------------------------- help
with tab_help:
    st.header("How the formula works")
    st.markdown(r"""
**1. Ratings from your results.** Every match in your files is replayed in date order:

* **Elo** (overall) and **surface Elo** (Hard / Clay / Grass), seeded from world ranking so a
  year of data is enough. K-factor shrinks as a player's match count grows: $K = 250/(n+5)^{0.4}$.
* **Form** in the last **30, 90 and 365 days**, and on the match surface, as win-rates shrunk
  toward 50% so a 2–0 record is not treated like 20–0.
* **Ranking** gap (log scale) and **fatigue** (matches in the last 10 days).

**2. The formula.** The factors are combined as
$\text{logit}\,P(A) = \sum_i w_i\,(x_{A,i} - x_{B,i})$.
The weights $w$ are **fitted on your own data** (regularised logistic regression, using only
information available before each match). See them in the Data tab.

**3. Blend with the market.** Betfair prices are sharp. The final probability is
$\text{logit}\,p = \alpha\,\text{logit}\,p_{model} + (1-\alpha)\,\text{logit}\,p_{market}$
where $\alpha$ is *Model trust* in the sidebar.

**4. Expected value on Betfair** (commission $c$ on net winnings):

* Back at odds $O$: $EV = p\,(O-1)(1-c) - (1-p)$
* Lay at odds $L$ (per unit liability): $EV = (1-p)\,\frac{1-c}{L-1} - p$

**5. Stake** = fractional Kelly, capped at your max stake:
$f^* = \frac{p\,b - (1-p)}{b}$ with $b=(O-1)(1-c)$.

**Grades:** A = strong EV, plenty of data on both players and a believable edge (3–15%).
Very large edges usually mean the model is missing news (injury, withdrawal, retirement risk).

**Reality check.** No model guarantees profit. Use the Backtest tab: if the model does not beat the
market's log-loss on held-out data, the "value" is noise. Track closing-line value on real bets.
""")
