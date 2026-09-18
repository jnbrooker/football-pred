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

# --- season planner: the weeks ahead --------------------------------------------
from footsim import planner as PL

st.subheader("Season planner")
st.write("Projections for the gameweeks ahead from the same match engine, then two exact plans: the best **fixed** "
         "15 for the whole horizon (its XI and captain can change each week), and the best **transfer path** from "
         "the team you actually have — one free transfer a week banked to five, −4 for every extra, budget from your "
         "squad value plus bank. Later weeks are discounted because the engine's view of them is less certain.")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Simulating the gameweeks ahead…")
def horizon(seasons, recency, gws: tuple, rules_key: str, rules: dict, n_sims: int):
    ctx = UI.cached_context(tuple(sorted(seasons)), recency)
    return PL.horizon_projections(ctx, list(gws), rules, n_sims=n_sims)


p1, p2, p3, p4 = st.columns(4)
n_ahead = p1.slider("Gameweeks ahead", 2, 8, 5)
h_sims = p2.select_slider("Simulations per match (horizon)", [1000, 2000, 4000], value=2000)
discount = p3.slider("Weekly discount", 0.7, 1.0, 0.9, 0.05, help="Weight on next week's points relative to this week's.")
h_obj = p4.selectbox("Maximise", ["Proj", "Ceiling", "Floor"], index=0, key="h_obj")
gws_ahead = tuple(g for g in gws if g >= int(gw))[:n_ahead]
H = horizon(tuple(seasons), recency, gws_ahead, preset, rules, h_sims)
if H.empty:
    st.warning("No projections for the gameweeks ahead."); st.stop()
st.caption(f"Horizon: GW{gws_ahead[0]}–GW{gws_ahead[-1]} · {H['player_id'].nunique()} players · "
           f"{h_sims:,} simulations per match. Blank gameweeks count zero; double gameweeks are summed.")

tab_fixed, tab_plan = st.tabs(["Best fixed squad for the horizon", "Transfer plan from your team"])
with tab_fixed:
    f1, f2 = st.columns([1, 3])
    h_budget = f1.number_input("Budget (£m)", 50.0, 150.0, 100.0, 0.5, key="h_budget")
    h_lock = f2.multiselect("Must include", all_labels, key="h_lock")
    try:
        best = PL.best_season_squad(H, list(gws_ahead), budget=h_budget, discount=discount, objective=h_obj,
                                    locked=[lab2id[l] for l in h_lock])
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected XI points over the horizon", f"{best['total']:.0f}", help="Captain counted twice, no discount")
        m2.metric("Cost", f"£{best['weeks'][0]['cost']:.1f}m")
        m3.metric("Captains", " · ".join(dict.fromkeys(w["captain"] for w in best["weeks"])))
        fmt = {"Price": "{:.1f}", "Total": "{:.1f}"}
        st.dataframe(best["table"].style.format(fmt), hide_index=True, **UI.WIDE, height=560)
        st.caption("Per-week cells: expected points; (C) = captain that week; b = on the bench that week.")
    except ValueError as e:
        st.error(str(e))

with tab_plan:
    src = st.radio("Your team", ["Import from FPL team ID", "Pick the 15"], horizontal=True)
    cur_ids, bank, ft0 = [], 0.0, 1
    if src == "Import from FPL team ID":
        c1, c2 = st.columns([1, 3])
        entry = c1.number_input("FPL team ID", 1, 20_000_000, 1, 1, help="The number in your FPL URL: …/entry/<id>/…")
        if c1.button("Load team"):
            try:
                st.session_state["fpl_entry"] = PL.fpl_entry(int(entry))
            except Exception as e:
                st.error(f"Could not load team {entry}: {e}")
        ent = st.session_state.get("fpl_entry")
        if ent:
            id_by_fpl = dict(zip(H["fpl_id"].dropna().astype(int), H.loc[H["fpl_id"].notna(), "player_id"]))
            cur_ids = [id_by_fpl[i] for i in ent["fpl_ids"] if i in id_by_fpl]
            bank, ft0 = ent["bank"], ent["free_transfers"]
            c2.info(f"**{ent['team_name']}** ({ent['name']}) — picks from GW{ent['last_gw']}, value £{ent['value']:.1f}m, "
                    f"bank £{ent['bank']:.1f}m, free transfers {ent['free_transfers']}, overall rank {ent['overall_rank']:,}. "
                    f"{len(cur_ids)} of 15 players matched to projections.")
    else:
        picks = st.multiselect("Your 15", all_labels, key="my15")
        cur_ids = [lab2id[l] for l in picks]
        bank = st.number_input("Bank (£m)", 0.0, 50.0, 0.0, 0.1)
    o1, o2, o3 = st.columns(3)
    ft_have = o1.number_input("Free transfers you have", 0, 5, int(ft0), key=f"ft_have_{ft0}",
                              help="Prefilled from the import when a team is loaded (reconstructed from your "
                                   "transfer history); set it yourself if that is wrong.")
    max_tr = o2.selectbox("Max transfers a week", ["no limit", 1, 2, 3], index=0)
    hit = o3.number_input("Points per extra transfer", 0.0, 8.0, 4.0, 1.0)
    if len(cur_ids) == 15:
        try:
            plan = PL.plan_transfers(H, list(gws_ahead), cur_ids, bank=bank, free_transfers=int(ft_have),
                                     discount=discount, objective=h_obj, hit_cost=hit,
                                     max_transfers_per_gw=(None if max_tr == "no limit" else int(max_tr)))
            m1, m2, m3 = st.columns(3)
            m1.metric("Plan: expected points", f"{plan['total']:.0f}", help="XI points with captain doubled, minus hits, no discount")
            m2.metric("Do nothing", f"{plan['baseline_total']:.0f}", help="Keep the squad, best XI each week")
            m3.metric("Gain from transfers", f"{plan['total'] - plan['baseline_total']:+.1f}")
            wk = pd.DataFrame([dict(GW=w["gw"], Free=w["free_before"], Out=", ".join(w["transfers_out"]) or "—",
                                    In=", ".join(w["transfers_in"]) or "—", Hits=w["hits"], Captain=w["captain"],
                                    XI=round(w["xi_points"], 1), Cost=round(w["cost"], 1)) for w in plan["weeks"]])
            st.dataframe(wk, hide_index=True, **UI.WIDE)
            with st.expander("Squad by week"):
                st.dataframe(plan["table"].style.format({"Price": "{:.1f}", "Total": "{:.1f}"}), hide_index=True, **UI.WIDE)
            st.caption("Selling at current price (FPL's half-profit rule ignored); no chips. Free transfers are used before hits.")
        except ValueError as e:
            st.error(str(e))
    elif cur_ids:
        st.info(f"{len(cur_ids)} of 15 players selected.")
