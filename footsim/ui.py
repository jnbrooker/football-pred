"""
footsim/ui.py — the sidebar widgets and the generic single-stat page that
pages 1–6 share, so "pick a player from the live squad" behaves identically
everywhere.

A single-stat page differs from the others only in which simulated array it
shows (shots, on-target, goals, assists, saves, cards): the picker, the
Game / Season view, the "assume he starts" control, the distribution chart
and the line odds are all one function, `stat_page`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from . import data as D
from . import players as P
from . import roster as RO
from . import teams as T

ACCENT = "#2e7d5b"
ACCENT2 = "#c0563b"
GREY = "#8a8a8a"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Loading shots, squads and ratings…")
def cached_context(seasons: tuple[int, ...], recency) -> dict:
    return P.prepare(tuple(sorted(seasons)), recency)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner="Building squad…")
def cached_roster(seasons: tuple[int, ...], recency, team: str) -> pd.DataFrame:
    return P.roster_for(cached_context(tuple(sorted(seasons)), recency), team)


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def cached_fpl_fixtures() -> pd.DataFrame:
    return D.fpl_fixtures()


@st.cache_data(ttl=D.REFRESH_HOURS * 3600, show_spinner=False)
def cached_availability(seasons: tuple[int, ...], recency, team: str) -> dict:
    ctx = cached_context(tuple(sorted(seasons)), recency)
    ros = cached_roster(tuple(sorted(seasons)), recency, team)
    return RO.availability_factor(ctx["live"], team, ctx["lg"], roster=ros)


def availability_toggle(key: str) -> bool:
    return st.sidebar.toggle(
        "Price the squad available", value=False, key=f"{key}_avail",
        help="Scales each attack by the expected shot production of the players available "
             "(FPL status) over the typical squad. OFF by default: fitted on 2024–26 with hindsight "
             "lineups, the coefficient is zero — the factor moves attacks by ±8% and does not predict "
             "goals out of sample. Shown, not priced (ROADMAP §6.13).")


def availability_caption(team: str, f: dict) -> str:
    miss = ", ".join(f.get("missing", [])[:5]) or "nobody of note"
    return (f"**{team}** attack ×{f.get('xg', 1.0):.2f} for availability "
            f"(volume ×{f.get('volume', 1.0):.2f}, quality ×{f.get('quality', 1.0):.2f}); out: {miss}.")


# ---------------------------------------------------------------------------
# Sidebar pieces
# ---------------------------------------------------------------------------

def priors_picker(key: str = "seasons") -> tuple:
    names = list(D.RECENCY_PRESETS)
    choice = st.sidebar.select_slider(
        "How much to trust this season", names, value="Balanced", key=f"{key}_recency",
        help="**Long memory**: seasons count as flat blocks (×0.7 a year).  \n"
             "**Balanced**: ×0.85 a year, and a match 15 back counts half.  \n"
             "**Recent form**: ×0.7 a year, a match 8 back counts half.  \n"
             "Applies to every rate, share and team rating on every page.")
    recency = D.RECENCY_PRESETS[choice]
    options, defaults = D.season_choices()
    with st.sidebar.expander("Advanced: seasons in the window"):
        seasons = st.multiselect("Seasons used to build priors", options, default=defaults, key=key,
                                 help="Seasons are named by their start year: 2025 = 2025/26.")
    if not seasons:
        st.sidebar.error("Pick at least one season."); st.stop()
    return tuple(int(s) for s in seasons), recency


def recency_caption(ctx: dict) -> None:
    shares = D.weight_shares(ctx["live"]["tm"])
    if not shares:
        return
    latest = max(shares)
    parts = ", ".join(f"{s}/{str(s + 1)[-2:]}: {v:.0%}" for s, v in sorted(shares.items(), reverse=True))
    st.sidebar.caption(f"Weight by season — {parts}. This season is **{shares[latest]:.0%}** of the model.")


def view_picker(key: str = "view") -> str:
    return st.sidebar.radio(
        "View", ["Fixture", "Season"], horizontal=True, key=f"{key}_view",
        help="**Fixture**: pick a gameweek fixture; opponent, venue and the team's expected "
             "shots come from the match model.  \n**Season**: the player's typical match "
             "against a chosen opponent (or an average one).")


def fixture_picker(ctx: dict, seasons, recency, key: str = "fx") -> dict:
    """Gameweek → fixture → player from both squads."""
    fx = cached_fpl_fixtures()
    if fx.empty:
        st.sidebar.error("The FPL fixture list did not load."); st.stop()
    fx = fx.dropna(subset=["gw"]).copy()
    fx["gw"] = fx["gw"].astype(int)
    cur, nxt = D.fpl_current_gw()
    gws = sorted(fx["gw"].unique().tolist())
    default = nxt if nxt in gws else cur if cur in gws else gws[0]
    gw = st.sidebar.selectbox("Gameweek", gws, index=gws.index(default), key=f"{key}_gw")
    wg = fx[fx["gw"] == gw].sort_values("kickoff").reset_index(drop=True)
    labels = [fixture_label(r) for _, r in wg.iterrows()]
    unplayed = [i for i, r in wg.iterrows() if not bool(r["finished"])]
    pick = st.sidebar.selectbox("Fixture", labels, index=unplayed[0] if unplayed else 0, key=f"{key}_game")
    row = wg.iloc[labels.index(pick)]
    return dict(gw=int(gw), home=row["home"], away=row["away"], row=row)


def fixture_label(r) -> str:
    ko = pd.to_datetime(r["kickoff"])
    s = f"{r['home']} v {r['away']} · {ko:%a %d %b %H:%M}" if pd.notna(ko) else f"{r['home']} v {r['away']}"
    if bool(r.get("finished")) and pd.notna(r.get("home_goals")):
        s += f" (FT {int(r['home_goals'])}–{int(r['away_goals'])})"
    return s


def player_picker(rosters: pd.DataFrame, groups: list[str] | None, key: str, label: str = "Player",
                  sort_col: str = "exp_shots") -> pd.Series:
    pool = rosters if groups is None else rosters[rosters["group"].isin(groups)]
    side = pool["_side"] if "_side" in pool.columns else pd.Series(0, index=pool.index)
    pool = pool.assign(_side=side).sort_values(["_side", sort_col], ascending=[True, False])
    if pool.empty:
        st.sidebar.error("No eligible players."); st.stop()
    pick = st.sidebar.selectbox(label, pool["label"].tolist(), key=f"{key}_player",
                                help="Tags: INJ / SUSP / UNAV / DOUBT (chance) from the FPL "
                                     "availability feed; 'no history' = position prior only; "
                                     "(new club) = history is from another club.")
    return pool[pool["label"] == pick].iloc[0]


def start_control(row: pd.Series, key: str) -> bool | None:
    choice = st.sidebar.radio(
        "Lineup", ["Use his start probability", "Assume he starts", "Assume he is a sub"],
        key=f"{key}_start",
        help=f"Modelled: starts {row['p_start']:.0%}, comes off the bench "
             f"{row['p_subapp'] * (1 - row['p_start']):.0%}, unused "
             f"{(1 - row['p_start']) * (1 - row['p_subapp']):.0%}. A prop is usually void if he "
             "does not play; 'assume he starts' is the number to compare with a line.")
    return {"Use his start probability": None, "Assume he starts": True, "Assume he is a sub": False}[choice]


# ---------------------------------------------------------------------------
# Main-panel pieces
# ---------------------------------------------------------------------------

def role_caption(row: pd.Series, share_col: str, what: str) -> str:
    prior = RO.GROUP_PRIOR[row["group"]]["shot" if share_col == "shot_share" else "kp"]
    bits = [f"**{row['team']} {row['group']}** ({row['position']} in FPL)",
            f"starts {row['p_start']:.0%}, expected {row['exp_minutes']:.0f} minutes"]
    if row["group"] == "GK":
        bits.append(f"{int(row['games'])} appearances in the priors window" if row["games"] else "no Understat history")
    elif row["games"]:
        raw = row.get(f"raw_{share_col}", np.nan)
        bits.append(f"share of team {what} per 90 on the pitch **{row[share_col]:.1%}** "
                    f"(own history {raw:.1%} over {int(row['games'])} appearances, regressed toward the "
                    f"{row['group']} prior {prior:.1%})")
    else:
        bits.append(f"no Understat history — {row['group']} prior {prior:.1%} of team {what}")
    if row.get("moved"):
        bits.append("history is from another club; volume uses the new club's")
    return " · ".join(bits)


def status_warning(row: pd.Series) -> None:
    s = str(row.get("status", "a"))
    if s in ("i", "s", "u", "n"):
        st.warning(f"**{row['name']} is {row['status_text'].lower()}** — {row['news'] or 'no detail'}. "
                   "His start probability is zero; use *Assume he starts* to see an 'if he plays' number.")
    elif s == "d":
        st.info(f"{row['name']} is **doubtful** ({row['chance']:.0f}% chance) — {row['news'] or 'no detail'}.")


def count_chart(x: np.ndarray, line: float | None, title: str, max_k: int | None = None) -> go.Figure:
    k = np.arange(0, int(max(x.max(), 1)) + 1) if max_k is None else np.arange(0, max_k + 1)
    p = np.array([(x == i).mean() for i in k])
    if max_k is not None:
        p[-1] = (x >= k[-1]).mean()
    colors = [ACCENT if (line is None or i > line) else ACCENT2 for i in k]
    fig = go.Figure()
    fig.add_bar(x=[str(i) if (max_k is None or i < k[-1]) else f"{i}+" for i in k], y=p, marker_color=colors,
                hovertemplate="%{x}: %{y:.1%}<extra></extra>")
    fig.update_layout(height=320, bargap=0.15, showlegend=False, xaxis_title=title,
                      yaxis_title="Share of simulations", yaxis_tickformat=".0%",
                      margin=dict(t=20, b=45, l=60, r=20))
    return fig


def line_metrics(x: np.ndarray, line: float, label: str) -> None:
    s = P.summarize(x, line)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Mean {label}", f"{s['mean']:.2f}")
    c2.metric("Median", f"{s['median']:.0f}", help=f"80% of simulations: {s['p10']:.0f}–{s['p90']:.0f}")
    c3.metric(f"P(over {line:g})", f"{s['p_over']:.1%}", help=f"Fair decimal odds {s['fair_over']}")
    c4.metric(f"P(under {line:g})", f"{s['p_under']:.1%}",
              help=f"Fair {s['fair_under']}" + (f"; push {s['p_push']:.1%}" if s["p_push"] > 0 else ""))


def threshold_table(x: np.ndarray, label: str, ks=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    return pd.DataFrame({f"{label} ≥ k": [f"{k}+" for k in ks],
                         "Probability": [(x >= k).mean() for k in ks],
                         "Fair odds": [D.decimal_odds((x >= k).mean()) for k in ks]})


# ---------------------------------------------------------------------------
# The generic single-stat page
# ---------------------------------------------------------------------------

STAT_CONFIG = {
    "shots": dict(title="Shots", icon="🎯", array="shots", label="shots", groups=["CB", "FB", "CM", "AM", "W", "ST"], share="shot_share",
                  what="shots", default_line=1.5, sort="exp_shots"),
    "sot": dict(title="Shots on Target", icon="🥅", array="sot", label="shots on target", groups=["CB", "FB", "CM", "AM", "W", "ST"],
                share="shot_share", what="shots", default_line=0.5, sort="exp_shots"),
    "goals": dict(title="Goals", icon="⚽", array="goals", label="goals", groups=["CB", "FB", "CM", "AM", "W", "ST"], share="shot_share",
                  what="shots", default_line=0.5, sort="exp_shots"),
    "assists": dict(title="Assists", icon="🅰️", array="assists", label="assists", groups=["CB", "FB", "CM", "AM", "W", "ST"], share="kp_share",
                    what="key passes", default_line=0.5, sort="exp_kp"),
    "saves": dict(title="Goalkeeper Saves", icon="🧤", array="saves", label="saves", groups=["GK"],
                  share="shot_share", what="shots", default_line=2.5, sort="exp_minutes"),
    "cards": dict(title="Cards", icon="🟨", array="yellow", label="yellow cards", groups=["CB", "FB", "CM", "AM", "W", "ST"], share="shot_share",
                  what="shots", default_line=0.5, sort="exp_minutes"),
}


def stat_page(stat: str) -> None:
    cfg = STAT_CONFIG[stat]
    st.set_page_config(page_title=cfg["title"], page_icon=cfg["icon"], layout="wide")
    st.sidebar.header("Setup")
    seasons, recency = priors_picker()
    ctx = cached_context(tuple(seasons), recency)
    recency_caption(ctx)
    st.sidebar.divider()
    view = view_picker(key=stat)

    if view == "Fixture":
        f = fixture_picker(ctx, seasons, recency, key=stat)
        frames = []
        for t in (f["home"], f["away"]):
            try:
                frames.append(cached_roster(tuple(seasons), recency, t))
            except ValueError:
                continue
        if not frames:
            st.error("No squads for this fixture."); st.stop()
        pool = pd.concat(frames, ignore_index=True)
        pool["_side"] = (pool["team"] == f["away"]).astype(int)
        row = player_picker(pool, cfg["groups"], key=stat, sort_col=cfg["sort"])
        team = row["team"]; opp = f["away"] if team == f["home"] else f["home"]
        home = team == f["home"]
        fixture_txt = f"{f['home']} v {f['away']} (GW{f['gw']})"
    else:
        r = ctx["ratings"]
        teams = list(r["teams"])
        squads = ctx["live"]["squads"]
        teams_now = sorted(squads["team"].unique()) if squads is not None else teams
        team = st.sidebar.selectbox("Team", teams_now, index=teams_now.index("MCI") if "MCI" in teams_now else 0,
                                    key=f"{stat}_team")
        ros = cached_roster(tuple(seasons), recency, team)
        row = player_picker(ros, cfg["groups"], key=stat, sort_col=cfg["sort"])
        opp_choice = st.sidebar.selectbox("Opponent", ["League average"] + [t for t in teams if t != team],
                                          key=f"{stat}_opp")
        opp = None if opp_choice == "League average" else opp_choice
        venue = st.sidebar.radio("Venue", ["Home", "Away", "Neutral"], horizontal=True, key=f"{stat}_venue")
        home = {"Home": True, "Away": False, "Neutral": None}[venue]
        fixture_txt = f"{team} vs {opp_choice}" + (f" ({venue.lower()})" if home is not None else "")

    force = start_control(row, key=stat)
    use_avail = availability_toggle(stat)
    n_sims = st.sidebar.select_slider("Simulations", [5000, 20000, 50000], value=20000, key=f"{stat}_n")
    line = st.sidebar.number_input(f"Line ({cfg['label']})", 0.0, 30.0, float(cfg["default_line"]), 0.5,
                                   key=f"{stat}_line")

    st.title(f"{cfg['icon']} {cfg['title']} — {row['name']}")
    st.caption(f"{fixture_txt} · priors from {', '.join(str(s) for s in ctx['seasons'])} · {n_sims:,} simulations")
    status_warning(row)
    st.markdown(role_caption(row, cfg["share"], cfg["what"]))

    av = cached_availability(tuple(seasons), recency, team) if use_avail else None
    av_opp = cached_availability(tuple(seasons), recency, opp) if (use_avail and opp) else None
    mc = P.match_context(ctx, team, opp, home, avail=av, avail_opp=av_opp)
    if use_avail:
        st.caption(availability_caption(team, av) + (("  \n" + availability_caption(opp, av_opp)) if av_opp else ""))
    sim = P.simulate(row, mc, n_sims=n_sims, seed=7, force_start=force)
    x = sim[cfg["array"]]
    if stat == "saves":
        st.caption(f"Opponent expected to take **{mc['b']['shots']:.1f} shots**, "
                   f"{mc['b']['sot']:.1f} on target, scoring {mc['b']['goals']:.2f}; saves are on-target minus goals.")
    else:
        st.caption(f"Team expected to take **{mc['a']['shots']:.1f} shots** in this match "
                   f"(typical {mc['typical_shots']:.1f}); {P.opponent_label(mc)}.")

    line_metrics(x, line, cfg["label"])
    st.plotly_chart(count_chart(x, line, f"{cfg['label']} in the match", max_k=8 if stat in ("shots", "saves") else 5),
                    width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.dataframe(threshold_table(x, cfg["title"]).style.format({"Probability": "{:.1%}"}),
                     hide_index=True, width="stretch")
    with right:
        rt = sim["rates"]
        st.markdown("**Where the number comes from**")
        lines = [f"- Minutes: {rt['exp_minutes']:.0f} expected (plays {sim['played'].mean():.0%} of simulations)"]
        if stat in ("shots", "sot", "goals"):
            lines.append(f"- Shots: team {mc['a']['shots']:.1f} × minutes/90 × share {row['shot_share']:.1%} "
                         f"→ **{sim['np_shots'].mean():.2f}** non-penalty shots")
        if stat in ("sot",):
            lines.append(f"- On target: own rate {row['sot_rate']:.1%} × opponent/venue ×{rt['sot_mult']:.2f} "
                         f"→ **{rt['p_sot']:.1%}** per shot")
        if stat == "goals":
            lines.append(f"- Conversion: xG/shot {row['xgps']:.3f} × finishing {row['finishing']:.2f} × "
                         f"opponent/venue ×{rt['q_mult'] * rt['fin_mult']:.2f} → **{rt['p_goal']:.1%}** per shot"
                         + (f"; penalties {sim['pens'].mean():.2f} (designated taker)" if rt["pen_taker"] else ""))
        if stat == "assists":
            lines.append(f"- Key passes: team shots × minutes/90 × share {row['kp_share']:.1%} → "
                         f"**{sim['key_passes'].mean():.2f}**; assist per key pass {rt['p_assist']:.1%} "
                         f"(xA per key pass {row['xa_kp']:.3f})")
        if stat == "cards":
            lines.append(f"- Yellows per 90: {row['y90']:.2f} (regressed toward the {row['group']} rate)")
        if stat == "saves":
            lines.append(f"- Saves = opponent on-target shots − goals conceded; clean sheet {sim['clean_sheet'].mean():.0%}")
        st.markdown("\n".join(lines))

    with st.expander("Full simulated line"):
        cols = {"minutes": "Min", "shots": "Shots", "sot": "SoT", "goals": "Goals", "key_passes": "KP",
                "assists": "Assists", "yellow": "YC", "saves": "Saves"}
        st.dataframe(pd.DataFrame({v: [float(sim[k].mean())] for k, v in cols.items()}).style.format("{:.2f}"),
                     hide_index=True, width="stretch")
    st.caption("Self-contained: team strength is solved from shots, never from a bookmaker line. "
               "For research and entertainment.")
