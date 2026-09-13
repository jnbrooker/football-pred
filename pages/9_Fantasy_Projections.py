"""Fantasy Projections — every player on the gameweek's slate, scored on FPL rules per simulation (footsim/fantasy.py)."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from footsim import data as D, fantasy as F, game as G
from footsim import ui as UI

st.set_page_config(page_title="Fantasy Projections", page_icon="🏆", layout="wide")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Simulating the gameweek…")
def slate(seasons, recency, gw: int, rules_key: str, rules: dict, n_sims: int):
    ctx = UI.cached_context(tuple(sorted(seasons)), recency)
    fx = UI.cached_fpl_fixtures()
    fx = fx[fx["gw"] == gw].sort_values("kickoff")
    table, samples, _ = F.gameweek_projections(ctx, fx, rules, n_sims=n_sims)
    return table, samples


st.sidebar.header("Setup")
seasons, recency = UI.priors_picker()
ctx = UI.cached_context(tuple(seasons), recency)
UI.recency_caption(ctx)
st.sidebar.divider()
fx = UI.cached_fpl_fixtures()
gws = sorted(fx.dropna(subset=["gw"])["gw"].astype(int).unique().tolist())
cur, nxt = D.fpl_current_gw()
default = nxt if nxt in gws else cur if cur in gws else gws[0]
gw = st.sidebar.selectbox("Gameweek", gws, index=gws.index(default))
preset = st.sidebar.selectbox("Scoring", list(F.PRESETS))
rules = dict(F.PRESETS[preset])
with st.sidebar.expander("Edit scoring"):
    for k in ("goal_gk", "goal_def", "goal_mid", "goal_fwd", "assist", "cs_gk", "cs_def", "cs_mid",
              "saves_per", "gc_per2", "yellow", "defcon"):
        rules[k] = st.number_input(k, value=float(rules[k]), step=1.0, key=f"rule_{k}")
    rules["bonus"] = st.toggle("Bonus (approximate)", value=True)
n_sims = st.sidebar.select_slider("Simulations per match", [3000, 8000, 20000], value=8000)
pos_filter = st.sidebar.multiselect("Positions", ["GK", "DEF", "MID", "FWD"], default=["GK", "DEF", "MID", "FWD"])
min_start = st.sidebar.slider("Minimum start probability", 0.0, 1.0, 0.3, 0.05)

st.title("🏆 Fantasy Projections")
table, samples = slate(tuple(seasons), recency, int(gw), preset, rules, n_sims)
if table.empty:
    st.error("No fixtures simulated for this gameweek."); st.stop()

st.caption(f"Gameweek {gw} · {table['fixture'].nunique()} fixtures · {n_sims:,} simulations each · "
           f"scoring: {preset}. Bonus is rebuilt from BPS on the simulated line and is the least certain component.")
view = table[table["Pos"].isin(pos_filter) & (table["Start"] >= min_start)].copy()
view["Value"] = view["Proj"] / view["Price"].replace(0, np.nan)
ranked = view.sort_values("Proj", ascending=False).reset_index(drop=True)
show = ranked[
    ["Player", "Team", "Opp", "Pos", "Price", "Status", "Start", "Min", "Proj", "Floor", "Median", "Ceiling",
     "P6", "P10", "Goals", "Assists", "CS", "Saves", "pts_bonus", "pts_defcon", "Value"]]
st.dataframe(show.style.format({"Price": "{:.1f}", "Start": "{:.0%}", "Min": "{:.0f}", "Proj": "{:.2f}",
                                "Floor": "{:.0f}", "Median": "{:.0f}", "Ceiling": "{:.0f}", "P6": "{:.0%}", "P10": "{:.0%}",
                                "Goals": "{:.2f}", "Assists": "{:.2f}", "CS": "{:.0%}", "Saves": "{:.1f}",
                                "pts_bonus": "{:.2f}", "pts_defcon": "{:.2f}", "Value": "{:.2f}"}),
             hide_index=True, **UI.WIDE, height=640)

st.subheader("Where the points come from")
labels = (ranked["Player"] + " (" + ranked["Team"] + ")").tolist()
pick = st.selectbox("Player", labels, index=0)
row = ranked.iloc[labels.index(pick)]
comp = {F.COMPONENT_LABELS[k]: float(row[f"pts_{k}"]) for k in F.COMPONENT_LABELS if f"pts_{k}" in row}
left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_bar(x=list(comp), y=list(comp.values()), marker_color=UI.ACCENT,
                hovertemplate="%{x}: %{y:.2f}<extra></extra>")
    fig.update_layout(height=320, yaxis_title="Expected points", margin=dict(t=20, b=45, l=60, r=20))
    st.plotly_chart(fig, **UI.WIDE)
with right:
    x = samples.get(row["player_id"])
    if x is not None:
        ks = np.arange(int(x.min()), int(min(x.max(), 20)) + 1)
        p = [(x == k).mean() if k < 20 else (x >= 20).mean() for k in ks]
        fig2 = go.Figure()
        fig2.add_bar(x=[str(k) for k in ks], y=p, marker_color=UI.ACCENT, hovertemplate="%{x} pts: %{y:.1%}<extra></extra>")
        fig2.update_layout(height=320, xaxis_title="Points in the match", yaxis_title="Share of simulations",
                           yaxis_tickformat=".0%", margin=dict(t=20, b=45, l=60, r=20))
        st.plotly_chart(fig2, **UI.WIDE)
st.caption(f"{row['Player']}: projected {row['Proj']:.2f}, floor {row['Floor']:.0f}, ceiling {row['Ceiling']:.0f}; "
           f"starts {row['Start']:.0%}, {row['Min']:.0f} expected minutes.")
st.caption("Every player's points are scored per simulation, so teammates and opponents keep their correlation "
           "(a striker's goal is his keeper's lost clean sheet). For research and entertainment.")

# --- best squad under the rules -------------------------------------------------
st.subheader("Best squad for the gameweek")
st.write("The exact best 15 under the FPL rules (2 GK / 5 DEF / 5 MID / 3 FWD, at most three from a club), "
         "with the starting XI, bench and **captain** chosen to maximise the XI's expected points with the "
         "captain counted twice. Solved as an integer programme, not a greedy pick.")
b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
budget = b1.number_input("Budget (£m)", 50.0, 150.0, 100.0, 0.5)
objective = b2.selectbox("Maximise", ["Proj", "Median", "Ceiling", "Floor"], index=0,
                         help="Proj = expected points; Ceiling = 90th percentile (a differentials squad); Floor = 10th.")
bench_w = b3.slider("Bench weight", 0.0, 0.5, 0.1, 0.05,
                    help="How much a bench player's points count in the objective (0 = cheapest legal bench).")
all_labels = (table["Player"] + " (" + table["Team"] + ")").drop_duplicates().tolist()
lock = b4.multiselect("Must include", all_labels, key="lock")
excl = st.multiselect("Exclude", all_labels, key="excl")
lab2id = dict(zip(table["Player"] + " (" + table["Team"] + ")", table["player_id"]))
try:
    best = F.best_squad(table, budget=budget, bench_weight=bench_w, objective=objective,
                        locked=[lab2id[l] for l in lock], excluded=[lab2id[l] for l in excl])
except ValueError as e:
    st.error(str(e)); best = None
if best is not None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Captain", best["captain"])
    m2.metric("XI expected points", f"{best['points']:.1f}", help="Captain counted twice")
    m3.metric("Cost", f"£{best['cost']:.1f}m", help=f"£{budget - best['cost']:.1f}m in the bank")
    m4.metric("Bench expected", f"{best['bench_points']:.1f}")
    sq = best["squad"]
    cols = ["Role", "Player", "Team", "Opp", "Pos", "Price", "Proj", "Floor", "Ceiling", "Start", "Goals", "Assists", "CS", "Status"]
    cols = [c for c in cols if c in sq.columns]
    st.dataframe(sq[cols].style.format({"Price": "{:.1f}", "Proj": "{:.2f}", "Floor": "{:.0f}", "Ceiling": "{:.0f}",
                                        "Start": "{:.0%}", "Goals": "{:.2f}", "Assists": "{:.2f}", "CS": "{:.0%}"}),
                 hide_index=True, **UI.WIDE, height=560)
    st.caption("Start = modelled probability of starting; a player flagged doubtful is priced at his chance of playing. "
               "Prices are FPL's current values. Bench order is by expected points.")
