"""Backtest — every model scored out of sample, match date by match date (footsim/backtest.py)."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from footsim import backtest as B, data as D
from footsim import ui as UI

st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def run_team(seasons: tuple, n_prior: int, recency):
    return B.team_backtest(seasons, n_prior=n_prior, recency=recency)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def run_players(seasons: tuple, n_prior: int, recency, rounds: tuple, n_sims: int):
    return B.player_backtest(seasons, n_prior=n_prior, recency=recency, rounds=rounds, n_sims=n_sims)


st.sidebar.header("Setup")
options, defaults = D.season_choices()
season = st.sidebar.selectbox("Season to score", options, index=1 if len(options) > 1 else 0,
                              help="Every date of this season is predicted from matches before it.")
n_prior = st.sidebar.slider("Prior seasons in the window", 1, 3, 2)
preset = st.sidebar.select_slider("Recency", list(D.RECENCY_PRESETS), value="Balanced")
recency = D.RECENCY_PRESETS[preset]

st.title("🧪 Backtest")
st.write("For each match date, every rating and prior is refitted on matches played before it; that date's "
         "fixtures are then predicted. The closing market (Pinnacle where published, else the market average, "
         "overround removed) is the benchmark on the same matches.")

st.subheader("Team layer: 1X2 and goals")
with st.spinner("Refitting the team layer date by date…"):
    bt = run_team((int(season),), n_prior, recency)
if bt.empty:
    st.error("Nothing to score yet for this season."); st.stop()
m = B.team_metrics(bt)
st.dataframe(m.style.format("{:.4f}", na_rep="—"), width="stretch")
st.caption("log_loss / rps / brier: lower is better. gd_rmse: goal-difference error; shots_rmse: per-side shots error. "
           f"Matches: {len(bt)} predicted, {int(m.loc['model', 'n'])} with closing prices.")

left, right = st.columns(2)
with left:
    cal = B.calibration(bt, "p_home", 0)
    fig = go.Figure()
    fig.add_scatter(x=cal["predicted"], y=cal["actual"], mode="markers+lines", marker=dict(size=cal["n"] / 4 + 4),
                    name="home win", hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<extra></extra>")
    fig.add_scatter(x=[0.1, 0.8], y=[0.1, 0.8], mode="lines", line=dict(dash="dash", color=UI.GREY), name="perfect")
    fig.update_layout(height=320, xaxis_title="Predicted P(home win)", yaxis_title="Actual", margin=dict(t=20, b=45, l=60, r=20))
    st.plotly_chart(fig, width="stretch")
with right:
    slope = np.polyfit(bt["exp_margin"], bt["margin"], 1)[0]
    st.metric("Calibration slope (goal difference)", f"{slope:.2f}",
              help="Actual margin regressed on predicted. 1.0 = the right amount of shrinkage; above 1 = ratings too timid.")
    wk = bt.groupby("round").apply(lambda g: pd.Series(dict(
        n=len(g), model_ll=-np.log(np.clip(np.where(g["outcome"] == 0, g["p_home"], np.where(g["outcome"] == 1, g["p_draw"], g["p_away"])), 1e-6, 1)).mean(),
        market_ll=(-np.log(np.clip(np.where(g["outcome"] == 0, g["mk_home"], np.where(g["outcome"] == 1, g["mk_draw"], g["mk_away"])), 1e-6, 1)).mean()
                   if "mk_home" in g and g["mk_home"].notna().all() else np.nan))), include_groups=False)
    st.dataframe(wk.style.format({"n": "{:.0f}", "model_ll": "{:.3f}", "market_ll": "{:.3f}"}, na_rep="—"),
                 width="stretch", height=300)

with st.expander("Largest misses"):
    bt["surprise"] = -np.log(np.clip(np.where(bt["outcome"] == 0, bt["p_home"], np.where(bt["outcome"] == 1, bt["p_draw"], bt["p_away"])), 1e-6, 1))
    st.dataframe(bt.sort_values("surprise", ascending=False).head(12)[
        ["date", "home", "away", "hg", "ag", "exp_hg", "exp_ag", "p_home", "p_draw", "p_away"]]
        .style.format({"exp_hg": "{:.2f}", "exp_ag": "{:.2f}", "p_home": "{:.0%}", "p_draw": "{:.0%}", "p_away": "{:.0%}",
                       "hg": "{:.0f}", "ag": "{:.0f}", "date": lambda d: f"{d:%Y-%m-%d}"}),
        hide_index=True, width="stretch")

st.subheader("Player layer: shots, on target, goals, assists, key passes, saves")
st.write("History-only squads (the FPL feed cannot be replayed), every player with 5+ prior appearances, "
         "simulated **conditional on his actual role** (starter or sub) so the volume and quality models are "
         "scored apart from the lineup model — which is scored on its own below. The naive baseline is the "
         "player's trailing weighted average in the same role.")
rounds = st.slider("Rounds to score", 1, 38, (5, 38))
n_sims = st.select_slider("Simulations per player", [1000, 3000, 6000], value=3000)
if st.button("Run the player backtest (a few minutes)"):
    with st.spinner("Refitting and simulating every player-match…"):
        pb = run_players((int(season),), n_prior, recency, tuple(range(rounds[0], rounds[1] + 1)), n_sims)
    if pb.empty:
        st.warning("No player-matches scored.")
    else:
        pm = B.player_metrics(pb)
        st.dataframe(pm.style.format("{:.3f}"), width="stretch")
        st.caption("mae vs naive_mae is the test; cover_10_90 should be near 0.80; brier_ge1 vs base_ge1 (the base rate's "
                   "Brier is base × (1 − base)) for the anytime markets.")
        lm = B.lineup_metrics(pb)
        st.metric("Lineup model: Brier of P(start)", f"{lm['brier']:.3f}",
                  help=f"Base rate Brier {lm['base_brier']:.3f}; log-loss {lm['log_loss']:.3f}; {lm['n']} player-matches")
        stat = st.selectbox("Bias by projection quintile", list(pm.index))
        st.dataframe(B.quintile_bias(pb, stat).style.format("{:.3f}"), width="stretch")
        with st.expander("Largest misses"):
            pb["err"] = (pb["actual"] - pb["mean"]).abs()
            st.dataframe(pb.sort_values("err", ascending=False).head(15)[
                ["date", "player", "team", "opp", "stat", "started", "minutes", "mean", "actual", "naive"]]
                .style.format({"mean": "{:.2f}", "naive": "{:.2f}", "date": lambda d: f"{d:%Y-%m-%d}"}),
                hide_index=True, width="stretch")
