"""Team Strength — attack and defence solved from shots, xG per shot and finishing (footsim/teams.py)."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from footsim import data as D, teams as T
from footsim import ui as UI

st.set_page_config(page_title="Team Strength", page_icon="📊", layout="wide")

st.sidebar.header("Setup")
seasons, recency = UI.priors_picker()
ctx = UI.cached_context(tuple(seasons), recency)
UI.recency_caption(ctx)
r = ctx["ratings"]

st.title("📊 Team Strength")
st.caption(f"{r['n_matches']} matches, seasons {', '.join(str(s) for s in r['seasons'])} · "
           f"league: {r['lg_shots']:.2f} non-penalty shots per team-match, {r['lg_xgps']:.3f} xG per shot, "
           f"finishing {r['lg_fin']:.3f} goals per xG, on target {r['lg_sot']:.1%}, "
           f"penalties {r['lg_pen']:.3f}, own goals {r['lg_og']:.3f} → {r['lg_goals']:.2f} goals per team-match")

c1, c2, c3 = st.columns(3)
c1.metric("Home advantage on shots", f"×{np.exp(2 * r['home_vol']):.3f}", help="Home vs away, fitted jointly with the ratings")
c2.metric("…on xG per shot", f"×{np.exp(2 * r['home_q']):.3f}")
c3.metric("…on on-target rate", f"×{np.exp(2 * r['home_sot']):.3f}")
st.info("Home advantage is almost entirely **volume**: home sides take ~20% more shots at the same "
        "quality per shot. So the model fits it on the shot rate, never as a flat goals bonus.")

t = T.strength_table(r)
show = t.rename(columns={
    "xg_for": "Exp goals for", "xg_against": "Exp goals against", "net": "Net",
    "shots_for": "Shots for", "shots_against": "Shots against",
    "xgps_for": "xG/shot for", "xgps_against": "xG/shot against",
    "finishing": "Finishing", "keeping": "Keeping",
    "raw_goals_for": "Raw goals for", "raw_goals_against": "Raw goals against", "matches_w": "Weight"})
cols = ["Exp goals for", "Exp goals against", "Net", "Shots for", "Shots against",
        "xG/shot for", "xG/shot against", "Finishing", "Keeping", "Raw goals for", "Raw goals against", "Weight"]
st.subheader("Ratings")
st.caption("Expected goals for / against are what each side would produce against an average opponent at a "
           "neutral venue. Finishing and keeping are shrunk hard toward 1 — the data say they are mostly noise.")
st.dataframe(show[cols].style.format({c: "{:.2f}" for c in cols} | {"xG/shot for": "{:.3f}", "xG/shot against": "{:.3f}",
                                                                     "Weight": "{:.0f}"}),
             width="stretch", height=740)

st.subheader("Attack vs defence")
fig = go.Figure()
fig.add_scatter(x=t["xg_for"], y=t["xg_against"], mode="markers+text", text=t.index, textposition="top center",
                marker=dict(size=10, color=t["net"], colorscale="RdYlGn", showscale=False),
                hovertemplate="%{text}<br>for %{x:.2f}<br>against %{y:.2f}<extra></extra>")
fig.update_layout(height=520, xaxis_title="Expected goals for (vs average)", yaxis_title="Expected goals against",
                  yaxis_autorange="reversed", margin=dict(t=20, b=45, l=60, r=20))
st.plotly_chart(fig, width="stretch")

st.subheader("Fixture read")
teams = list(r["teams"])
ca, cb, cc = st.columns(3)
a = ca.selectbox("Home", teams, index=teams.index("MCI") if "MCI" in teams else 0)
b = cb.selectbox("Away", teams, index=teams.index("ARS") if "ARS" in teams else 1)
neutral = cc.toggle("Neutral venue", value=False)
if a != b:
    e = T.expected_goals(r, a, b, home=None if neutral else "a")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{a}", f"{e['goals_a']:.2f}", help=f"{e['shots_a']:.1f} shots, {e['a']['xgps']:.3f} xG/shot")
    m2.metric(f"{b}", f"{e['goals_b']:.2f}", help=f"{e['shots_b']:.1f} shots, {e['b']['xgps']:.3f} xG/shot")
    m3.metric("1X2", f"{e['win_a']:.0%} / {e['draw']:.0%} / {e['win_b']:.0%}")
    m4.metric("Over 2.5", f"{e['over25']:.0%}", help=f"BTTS {e['btts']:.0%}")
    st.caption("Analytic read (independent Poissons). The Match Simulation page runs the full engine, "
               "whose game-state feedback produces a few more draws than this.")

with st.expander("How the ratings are solved"):
    st.markdown(
        "Three multiplicative pieces per team-match, each solved by iterative proportional fitting "
        "(the SRS idea on the log scale) so a good attack that faced strong defences is not punished:\n\n"
        "- **shots** = league × attack volume × defence volume × home\n"
        "- **xG per shot** = league × attack quality × defence quality × home\n"
        "- **goals per xG** = league × finishing × keeping\n\n"
        f"Every rating is shrunk toward league by pseudo-matches at league average "
        f"({T.VOL_PRIOR_N:g} for volume, {T.QUAL_PRIOR_N:g} for quality, {T.FIN_PRIOR_XG:g} xG-worth for finishing), "
        "with recent matches weighted more. Penalties and own goals are separate small rates. "
        "The league finishing level is measured against actual goals, because Understat's xG runs above them.")
