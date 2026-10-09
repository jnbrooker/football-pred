"""Season Projection — the table this season, and the league five years out (footsim/season.py)."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from footsim import data as D, teams as T, season as SE
from footsim import ui as UI

st.set_page_config(page_title="Season Projection", page_icon="🏆", layout="wide")

st.sidebar.header("Setup")
seasons, recency = UI.priors_picker()
ctx = UI.cached_context(tuple(seasons), recency)
UI.recency_caption(ctx)
r = ctx["ratings"]

n_sims = st.sidebar.select_slider("Simulations", [2000, 5000, 10000, 20000, 50000], value=10000,
                                  help="Each one plays every fixture left this season, and every season after it.")
europe = st.sidebar.selectbox(
    "European places", [4, 5, 6, 7], index=1,
    help="League places only. The cup winners' places are not modelled — no cup fixtures exist in this "
         "suite — so in a season where the FA Cup or League Cup winner finishes high, one more league "
         "place goes into Europe than this says. England has had a fifth Champions League place in "
         "several recent seasons through the UEFA coefficient; that is why five is the default.")


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Playing out the season…")
def cached_table(seasons, recency, season, n_sims, shifts):
    ctx = UI.cached_context(tuple(seasons), recency)
    fx = SE.season_fixtures(season)
    return SE.simulate_table(ctx["ratings"], fx, n_sims=n_sims, seed=7,
                             shifts=dict(shifts) if shifts else None), fx


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Rating every past season…")
def cached_history(lo: int, hi: int):
    have = tuple(s for s in range(lo, hi) if (D.CACHE / f"shots_{D.LEAGUE}_{s}.parquet").exists())
    if not have:
        return pd.DataFrame(), have
    tm = D.load_team_matches(have)
    return SE.season_ratings(tm), have


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def cached_tiers(season, current):
    return SE.club_tiers(season, list(current))


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Playing out the seasons ahead…")
def cached_future(seasons, recency, season, years, n_sims, spend, tiers):
    ctx = UI.cached_context(tuple(seasons), recency)
    hist, _ = cached_history(2014, 2027)
    fx = SE.season_fixtures(season)
    return SE.simulate_future(ctx["ratings"], fx, hist, years=years, n_sims=n_sims,
                              seed=11, spend=dict(spend) if spend else None,
                              tiers=dict(tiers) if tiers else None)


cur_season = int(D.fpl_season())
sim, fx = cached_table(tuple(seasons), recency, cur_season, n_sims, ())
proj = SE.projected_table(sim, europe=europe)
played = int(sim["start"]["P"].max())

st.title("🏆 Season Projection")
st.caption(f"{SE.season_label(cur_season)} · {int(sim['start']['P'].sum() / 2)} matches played, "
           f"{sim['n_remaining']} to go · {n_sims:,} simulations of the rest of the season from the "
           f"same ratings the match pages use, each club carrying a strength offset of sd "
           f"{sim['drift_sd']:.2f} for what the ratings cannot yet know")

lead = proj.index[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Favourite", lead, f"{proj.at[lead, 'Title']:.0%} to win it")
c2.metric("Projected champion's points", f"{sim['points'].max(axis=1).mean():.0f}")
c3.metric("Points to stay up (17th)",
          f"{np.sort(sim['points'], axis=1)[:, SE.RELEGATED].mean():.0f}",
          help="The mean points of the club that finishes fourth from bottom.")
c4.metric("Most likely down", ", ".join(proj["Bottom3"].sort_values(ascending=False).head(3).index))

# ---------------------------------------------------------------------------
# The projected table
# ---------------------------------------------------------------------------

st.subheader("Projected final table")
show = proj.rename(columns={"Pts": "Proj Pts", "P10": "10th pct", "P90": "90th pct", "Pos": "Avg pos",
                            "UCL": "UCL (top 4)", "Europe": f"Europe (top {europe})", "Bottom3": "Relegated"})
pct = ["Title", "UCL (top 4)", f"Europe (top {europe})", "Top10", "Relegated"]
st.dataframe(
    show.style
        .format({"Proj Pts": "{:.1f}", "10th pct": "{:.0f}", "90th pct": "{:.0f}", "Avg pos": "{:.1f}",
                 "GD": "{:+.0f}", "Now": "{:.0f}", "P": "{:.0f}"} | {c: "{:.1%}" for c in pct})
        .background_gradient(subset=["Title", f"Europe (top {europe})"], cmap="Greens")
        .background_gradient(subset=["Relegated"], cmap="Reds"),
    **UI.WIDE, height=740)
st.caption("**Now** is the points already banked — those are facts, not simulated. Everything else is the "
           "distribution over the remaining fixtures: *Proj Pts* is the mean, the percentiles are where "
           "the middle 80% of the simulations land.")

# ---------------------------------------------------------------------------
# Where each club finishes
# ---------------------------------------------------------------------------

st.subheader("Where each club finishes")
pm = SE.position_matrix(sim)
fig = go.Figure(go.Heatmap(
    z=pm.values, x=list(pm.columns), y=list(pm.index), colorscale="Greens", zmin=0,
    hovertemplate="%{y} finishes %{x}: %{z:.1%}<extra></extra>", colorbar=dict(title="P")))
fig.add_vline(x=4.5, line_dash="dot", line_color=UI.GREY)
fig.add_vline(x=len(pm) - SE.RELEGATED + 0.5, line_dash="dot", line_color=UI.ACCENT2)
fig.update_layout(height=620, xaxis_title="Finishing position", yaxis_title="",
                  yaxis_autorange="reversed", margin=dict(t=20, b=45, l=60, r=20))
st.plotly_chart(fig, **UI.WIDE)

teams = list(proj.index)
pick = st.selectbox("Points distribution", teams, index=0, key="ptsclub")
pp = sim["points"][:, sim["teams"].index(pick)]
f2 = go.Figure()
f2.add_histogram(x=pp, nbinsx=int(max(pp) - min(pp)) or 10, marker_color=UI.ACCENT,
                 hovertemplate="%{x} points: %{y}<extra></extra>")
f2.add_vline(x=float(pp.mean()), line_color=UI.ACCENT2,
             annotation_text=f"mean {pp.mean():.0f}", annotation_position="top")
f2.update_layout(height=320, xaxis_title=f"{pick} final points", yaxis_title="simulations",
                 margin=dict(t=30, b=45, l=60, r=20), showlegend=False)
st.plotly_chart(f2, **UI.WIDE)

# ---------------------------------------------------------------------------
# The seasons after this one
# ---------------------------------------------------------------------------

st.header("The seasons after this one")
hist, have = cached_history(2014, 2027)
if hist.empty:
    st.warning("No completed seasons in the cache yet — run `python -m footsim.data 2018 … 2025` "
               "to rate the seasons the strength process is projected from.")
    st.stop()

st.caption(
    f"Rated on {hist['season'].nunique()} completed seasons ({SE.season_label(int(hist['season'].min()))}"
    f"–{SE.season_label(int(hist['season'].max()))}, {len(hist)} club-seasons). A club's strength splits "
    f"into a **level** that barely moves (sd {SE.LEVEL_SD:.2f} between clubs, drifting "
    f"{SE.LEVEL_INNOV_SD:.3f} a season) and a **season's form** that does not carry over at all "
    f"(sd {SE.FORM_SD:.2f}). Three go down each year and three come up at the promoted class's "
    f"strength ({SE.PROMOTED_NET:+.2f}), which club depending on how long it has been away.")

cc1, cc2 = st.columns([1, 2])
years = cc1.slider("Seasons ahead", 1, 8, 5, key="years")
fut_sims = cc1.select_slider("Simulations", [1000, 2000, 4000, 8000], value=4000, key="fsims")

lev = SE.club_levels(hist, SE.live_net(r).reindex(sim["teams"]).dropna(), played=played)
with cc2.expander("Investment: move a club's level", expanded=False):
    st.caption(
        "In level innovations a season: **+1** is a club doing to itself, every summer for the whole "
        "horizon, as much as a typical club does in a year. It is a standing shift, so it compounds — "
        f"about {SE.LEVEL_INNOV_SD * SE.POINTS_PER_NET:.0f} points a season after one, "
        f"{SE.LEVEL_INNOV_SD * SE.POINTS_PER_NET * (1 - SE.LEVEL_PSI ** 5) / (1 - SE.LEVEL_PSI):.0f} "
        "after five. This is the handle for a takeover, a spending spree, a sell-off or a points "
        "deduction: the suite has no financial data, so the size is yours to set. Zero everywhere is "
        "the fitted model.")
    ed = pd.DataFrame(dict(Club=sim["teams"],
                           Level=[float(lev["level"].get(t, 0.0)) for t in sim["teams"]],
                           Spend=0.0)).set_index("Club")
    ed = st.data_editor(
        ed, **UI.WIDE, height=320, disabled=["Level"],
        column_config={
            "Level": st.column_config.NumberColumn("Level now", format="%.2f",
                                                   help="The filtered club level, in log expected-goal difference."),
            "Spend": st.column_config.NumberColumn("Spend", min_value=-3.0, max_value=3.0, step=0.5,
                                                   format="%.1f", help="Level innovations a season.")})
spend = tuple((t, float(v)) for t, v in ed["Spend"].items() if abs(float(v)) > 1e-9)

TIER_NAMES = {1: "Premier League", 2: "Championship", 3: "League One", 4: "League Two"}
base_tiers, feed_live = cached_tiers(cur_season, tuple(sim["teams"]))
outside = [t for t in lev.index if t not in sim["teams"]]
outside = sorted(outside, key=lambda t: (-int(lev.at[t, "last_season"]), t))
with cc2.expander("Divisions: where the clubs outside the league actually are", expanded=not feed_live):
    st.caption(
        ("Read from football-data.co.uk. " if feed_live else
         ":red[The division feed could not be reached], so every club outside the top flight is "
         "assumed to be in the Championship. ") +
        "Only a **Championship** club can be promoted next May — a club that has been relegated "
        "twice has to climb back a division at a time — so correct anything wrong here. The model "
        "has no rating for these divisions, so within a division it has no opinion about which club "
        "goes up; a fallen giant comes back at the rate a random club in its division does.")
    te = pd.DataFrame(dict(Club=outside,
                           Division=[TIER_NAMES.get(base_tiers.get(t, 2), "Championship") for t in outside],
                           Last=[int(lev.at[t, "last_season"]) for t in outside])).set_index("Club")
    te = st.data_editor(
        te, **UI.WIDE, height=320, disabled=["Last"],
        column_config={
            "Division": st.column_config.SelectboxColumn(
                "Division now", options=list(TIER_NAMES.values()), required=True),
            "Last": st.column_config.NumberColumn("Last top-flight season", format="%d")})
inv = {v: k for k, v in TIER_NAMES.items()}
tiers = tuple(sorted((t, inv[v]) for t, v in te["Division"].items()))

fut = cached_future(tuple(seasons), recency, cur_season, years, fut_sims, spend, tiers)

seen = np.max([fut["in_league"][y].mean(axis=0) for y in range(years + 1)], axis=0)
options = [t for i, t in enumerate(fut["universe"])
           if not t.startswith("club ") and (t in sim["teams"] or seen[i] >= 0.02)]
club = st.selectbox(
    "Club", options, index=options.index(lead), key="outlook",
    help="Clubs currently in the league, plus any with a real chance of coming up within the horizon.")
out = SE.club_outlook(fut, club, europe=europe)
ever = out.attrs["ever"]
if club not in sim["teams"]:
    st.caption(f"**{club}** — **{TIER_NAMES.get(fut['tier0'].get(club, 2), 'Championship')}**, "
               f"last in the top flight in {SE.season_label(int(lev.at[club, 'last_season']))}. "
               "It has to win promotion before any of this applies"
               + (", and from below the Championship, one division at a time."
                  if fut["tier0"].get(club, 2) > 2 else "."))
m1, m2, m3, m4 = st.columns(4)
span = f"in at least one of the next {years + 1} seasons, this one included"
m1.metric(f"{club} relegated", f"{ever['Relegated']:.0%}",
          help=f"Going down from the Premier League {span}.")
m2.metric(f"In Europe (top {europe})", f"{ever['Europe']:.0%}", help=f"Finishing top {europe} {span}.")
m3.metric("In the Champions League", f"{ever['UCL']:.0%}", help=f"Finishing top 4 {span}.")
m4.metric("Champions", f"{ever['Title']:.0%}", help=f"Winning the league {span}.")

f3 = go.Figure()
for col, colour, dash in (("Europe", UI.ACCENT, "solid"), ("UCL", UI.ACCENT, "dash"),
                          ("Relegated", UI.ACCENT2, "solid"), ("Title", UI.GREY, "dot")):
    f3.add_scatter(x=list(out.index), y=out[col], mode="lines+markers", name=col,
                   line=dict(color=colour, dash=dash),
                   hovertemplate="%{x}: %{y:.1%}<extra>" + col + "</extra>")
f3.add_scatter(x=list(out.index), y=out["In league"], mode="lines", name="In the league",
               line=dict(color="#bbbbbb", width=1),
               hovertemplate="%{x}: %{y:.1%}<extra>in the league</extra>")
f3.update_layout(height=380, yaxis_tickformat=".0%", yaxis_title=f"P ({club})",
                 margin=dict(t=30, b=45, l=60, r=20), legend=dict(orientation="h", y=1.12))
st.plotly_chart(f3, **UI.WIDE)

def dashed(df, cols=("Pts", "Pos")):
    """Seasons the club spent outside the league have no points and no position;
    Streamlit renders a NaN as "None", so they are spelled out as a dash."""
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = [("—" if pd.isna(v) else f"{v:.1f}") for v in d[c]]
    return d


st.dataframe(dashed(out).style.format({"In league": "{:.1%}", "Title": "{:.1%}", "UCL": "{:.1%}",
                                       "Europe": "{:.1%}", "Relegated": "{:.1%}"}), **UI.WIDE)
st.caption("Each probability is unconditional — a season outside the top flight counts as neither "
           "European nor relegated, and *In league* says how often the club is there at all. *Pts* and "
           "*Pos* are averages over the simulations where it is.")

st.subheader(f"The league in {SE.season_label(fut['season0'] + years)}")
ft = SE.group_generic(SE.future_table(fut, years, europe=europe))
st.dataframe(
    dashed(ft).style.format({"In league": "{:.1%}", "Title": "{:.1%}", "UCL": "{:.1%}",
                             "Europe": "{:.1%}", "Relegated": "{:.1%}"}, na_rep="—")
      .background_gradient(subset=["Europe"], cmap="Greens")
      .background_gradient(subset=["Relegated"], cmap="Reds"),
    **UI.WIDE, height=620)
st.caption("*(second tier)* gathers the clubs with no top-flight record in the window — the model has no "
           "opinion about which of them comes up, only how many and how good they will be.")

with st.expander("How this is built, and what it cannot know"):
    st.markdown(f"""
**This season.** The points on the board are facts; every remaining fixture is played from the same
ratings the Match Simulation page uses, `{n_sims:,}` times. Each simulation also gives every club a
strength offset for the rest of the season (sd {sim['drift_sd']:.2f} in log expected-goal difference):
split-sample on 2014–2026 says that is how far a club's strength over the rest of a season sits from a
rating fitted on what it has played, beyond the rating's own sampling noise. Without it the table looks
far more certain than a season ever is.

**The seasons after.** A club's season strength is a slow **level** plus a **season of form**. Fitted on
the autocovariance of {len(hist)} club-seasons: the level has sd {SE.LEVEL_SD:.2f} between clubs and moves
{SE.LEVEL_INNOV_SD:.3f} a season (ψ {SE.LEVEL_PSI:.2f}); form has sd {SE.FORM_SD:.2f} and does not carry
over at all. A plain AR(1) on the season rating fits one year ahead and then forgets far too fast — it
implies a correlation of 0.31 five seasons out where the data say 0.53. Each club's level is filtered
from its own seasons; a club out of the league keeps its level, widened, but comes back at the promoted
class's strength, because what a promoted club did in the top flight years ago does not predict how it
does on its return (t = 0.8 on 17 returns — measured, not priced).

**The divisions below.** Only a Championship club can be promoted into the Premier League, so the model
is told which division each club is in (football-data.co.uk, correctable by hand above). Below the top
flight it has no ratings at all, so clubs move a division at a time at the mechanical rate — of 24 clubs
three go up and three or four go down — drawn independently. The rate is right; *which* club it is, is
not. A fallen giant in League One therefore climbs back at the rate a random League One club does, and
that is the single biggest thing a ratings feed for those divisions would fix.

**What else it cannot know.** No cup or European fixtures, so no congestion and no cup-winners' European
places. No ownership, takeovers, financial rules or points deductions — the *Spend* dial is the handle
for all of those. No knowledge of which second-tier club is good this year either: promoted sides all
arrive at the class average ({SE.PROMOTED_NET:+.2f}, sd {SE.PROMOTED_NET_SD:.2f}), and the classes have
been getting weaker, which the recency-weighted constant follows. Five seasons out, the width of these
numbers is the honest part; `python -m footsim.season backtest` is what says how honest.
""")
