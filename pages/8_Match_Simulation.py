"""Match Simulation — two squads in, a simulated box score out (footsim/game.py)."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from footsim import data as D, game as G, teams as T
from footsim import ui as UI

st.set_page_config(page_title="Match Simulation", page_icon="⚽", layout="wide")

st.sidebar.header("Setup")
seasons, recency = UI.priors_picker()
ctx = UI.cached_context(tuple(seasons), recency)
UI.recency_caption(ctx)
r = ctx["ratings"]

st.sidebar.divider()
from_fx = st.sidebar.toggle("Pick a fixture from the gameweek", value=True)
if from_fx:
    f = UI.fixture_picker(ctx, seasons, recency, key="ms")
    home, away, fx_row = f["home"], f["away"], f["row"]
    neutral = False
else:
    teams = list(r["teams"])
    home = st.sidebar.selectbox("Home", teams, index=teams.index("MCI") if "MCI" in teams else 0)
    away = st.sidebar.selectbox("Away", teams, index=teams.index("ARS") if "ARS" in teams else 1)
    neutral = st.sidebar.toggle("Neutral venue", value=False)
    fx_row = None
use_avail = UI.availability_toggle("ms")
n_sims = st.sidebar.select_slider("Simulations", [5000, 10000, 20000, 50000], value=20000)

st.title("⚽ Match Simulation")
if home == away:
    st.warning("Pick two different teams."); st.stop()
try:
    ra = UI.cached_roster(tuple(seasons), recency, home)
    rb = UI.cached_roster(tuple(seasons), recency, away)
except ValueError as e:
    st.error(str(e)); st.stop()

av_a = UI.cached_availability(tuple(seasons), recency, home) if use_avail else None
av_b = UI.cached_availability(tuple(seasons), recency, away) if use_avail else None
sim = G.simulate_match(ctx, ra, rb, home, away, home=None if neutral else "a", n_sims=n_sims, seed=11,
                       avail_a=av_a, avail_b=av_b)
s = G.summarize(sim)
e = T.expected_goals(r, home, away, home=None if neutral else "a", avail_a=av_a, avail_b=av_b)

st.caption(f"Priors from {', '.join(str(x) for x in ctx['seasons'])} · squads and availability from FPL · {n_sims:,} simulations")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{home}" + ("" if neutral else " (home)"), f"{s['mean_a']:.2f}")
c2.metric(f"{away}", f"{s['mean_b']:.2f}")
c3.metric("1X2 %", f"{s['win_a']:.0%}·{s['draw']:.0%}·{s['win_b']:.0%}".replace("%", ""),
          help=f"Fair decimal odds {D.decimal_odds(s['win_a'])} / {D.decimal_odds(s['draw'])} / {D.decimal_odds(s['win_b'])}")
c4.metric("Over 2.5", f"{s['over25']:.0%}", help=f"Over 1.5 {s['over15']:.0%} · Over 3.5 {s['over35']:.0%}")
c5.metric("BTTS", f"{s['btts']:.0%}", help=f"Clean sheet {home} {s['cs_a']:.0%} · {away} {s['cs_b']:.0%}")

fav = home if s["win_a"] >= s["win_b"] else away
st.info(f"**{fav} favoured.** Expected shots {s['shots_a']:.1f} – {s['shots_b']:.1f}, on target "
        f"{s['sot_a']:.1f} – {s['sot_b']:.1f}. Most likely scorelines: "
        + ", ".join(f"{k} ({p:.0%})" for k, p in s["scorelines"][:5]) + ". "
        + ("Neutral venue." if neutral else f"Home advantage: ×{np.exp(2 * r['home_vol']):.2f} on shots."))
if fx_row is not None and bool(fx_row.get("finished")) and pd.notna(fx_row.get("home_goals")):
    st.caption(f"**Final:** {home} {int(fx_row['home_goals'])} – {int(fx_row['away_goals'])} {away}.")

# --- availability ---------------------------------------------------------
notes = []
for team, ros in ((home, ra), (away, rb)):
    outs = ros[ros["status"].isin(["i", "s", "u"]) & (ros["games"] > 0)].sort_values("minutes_w", ascending=False)
    doubts = ros[ros["status"] == "d"]
    bits = []
    if len(outs):
        bits.append("out: " + ", ".join(f"{n} ({t.lower()})" for n, t in zip(outs["name"].head(6), outs["status_text"].head(6))))
    if len(doubts):
        bits.append("doubtful: " + ", ".join(f"{n} ({c:.0f}%)" for n, c in zip(doubts["name"], doubts["chance"].fillna(50))))
    notes.append(f"**{team}:** " + ("; ".join(bits) if bits else "no availability flags") + ".")
if use_avail:
    notes.append(UI.availability_caption(home, av_a) + "  \n" + UI.availability_caption(away, av_b))
st.caption("**Availability (FPL feed)** — " + "  \n".join(notes))

# --- score distribution ---------------------------------------------------
st.subheader("How the match finishes")
left, right = st.columns(2)
with left:
    margin = sim["goals_a"] - sim["goals_b"]
    ks = np.arange(-5, 6)
    p = np.array([(margin == k).mean() if abs(k) < 5 else ((margin <= k).mean() if k < 0 else (margin >= k).mean()) for k in ks])
    colors = [UI.ACCENT if k > 0 else UI.ACCENT2 if k < 0 else UI.GREY for k in ks]
    fig = go.Figure()
    fig.add_bar(x=[f"{k:+d}" if abs(k) < 5 else f"{k:+d}…" for k in ks], y=p, marker_color=colors,
                hovertemplate="margin %{x}<br>%{y:.1%}<extra></extra>")
    fig.update_layout(height=340, bargap=0.1, showlegend=False, xaxis_title=f"{home} goal difference",
                      yaxis_title="Share of simulations", yaxis_tickformat=".0%", margin=dict(t=20, b=45, l=60, r=20))
    st.plotly_chart(fig, **UI.WIDE)
with right:
    sc = pd.DataFrame(s["scorelines"], columns=["Scoreline", "Probability"])
    tot = sim["goals_a"] + sim["goals_b"]
    lines = pd.DataFrame({"Total goals": ["Over 0.5", "Over 1.5", "Over 2.5", "Over 3.5", "Over 4.5"],
                          "Probability": [(tot > k).mean() for k in (0.5, 1.5, 2.5, 3.5, 4.5)]})
    lines["Fair odds"] = lines["Probability"].map(D.decimal_odds)
    sc["Fair odds"] = sc["Probability"].map(D.decimal_odds)
    a1, a2 = st.columns(2)
    a1.dataframe(sc.style.format({"Probability": "{:.1%}"}), hide_index=True, **UI.WIDE)
    a2.dataframe(lines.style.format({"Probability": "{:.1%}"}), hide_index=True, **UI.WIDE)

# --- box scores ----------------------------------------------------------
st.subheader("Projected box score")
st.caption("Mean of every simulation. Player shots sum to team shots, goals to team goals (less own goals), "
           "assists sit inside key passes and the keeper's saves are the opponent's on-target shots minus goals — "
           "in every single simulation, not just on average.")
cols = st.columns(2)
for col, team, side in ((cols[0], home, sim["box_a"]), (cols[1], away, sim["box_b"])):
    with col:
        st.markdown(f"#### {team}")
        bs = G.box_score(side)
        show = bs[bs["Min"] >= 5][["Player", "Pos", "Start%", "Min", "Shots", "SoT", "Goals", "Anytime",
                                   "KP", "Assists", "Saves", "CS", "YC", "Status"]]
        st.dataframe(show.style.format({"Start%": "{:.0%}", "Min": "{:.0f}", "Shots": "{:.2f}", "SoT": "{:.2f}",
                                        "Goals": "{:.2f}", "Anytime": "{:.0%}", "KP": "{:.2f}", "Assists": "{:.2f}",
                                        "Saves": "{:.2f}", "CS": "{:.0%}", "YC": "{:.2f}"}),
                     hide_index=True, **UI.WIDE, height=560)
        st.caption(f"Team: {side['team_shots'].mean():.1f} shots, {side['team_sot'].mean():.1f} on target, "
                   f"{side['team_goals'].mean():.2f} goals (own goals {side['team_og'].mean():.2f}, "
                   f"penalties {side['team_pens'].mean():.2f})")

with st.expander("How a simulation runs, and what it does not model"):
    st.markdown(
        "1. **Lineups** — every player starts / comes on / stays unused with his own probabilities and minutes; "
        "exactly one goalkeeper.\n"
        "2. **Segments** — six 15-minute segments. Each side's shots are Poisson at its matchup rate × the segment's "
        "share × a shared *dominance* draw (the two sides' shot counts are negatively correlated in real matches) × "
        f"the game-state factor exp({G.LEAD_BETA_VOL:+.2f} × lead vs expected). A side that leads shoots less but "
        "better; on-target and conversion rise with the lead. Penalties and own goals are separate small events.\n"
        "3. **Allocation** — shots split across players on the pitch by minutes × share; on-target and goals handed "
        "out among them by each player's rates, capped so goals ≤ on target ≤ shots; assists drawn from teammates "
        "by key-pass weight.\n\n"
        "**Calibrated** on 2024–26 so the simulated draw rate, goal-difference spread, shot dispersion, the shot "
        "correlation between the sides and the shots-to-goals elasticity (0.33) match real matches "
        "(`python -m footsim.calibrate moments`).\n\n"
        "**Not modelled:** red cards, a referee effect on cards, in-match injuries, weather.")
st.caption("Self-contained — team strength is solved from shots, never from a bookmaker line. For research and entertainment.")
