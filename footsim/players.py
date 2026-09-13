"""
footsim/players.py — the single-player Monte Carlo (pages 1–6): shots, shots
on target, goals, assists, goalkeeper saves and cards for ONE player in ONE
match, simulated from the same chain every page shares:

    minutes → team shots in the match → the player's shots
            → on target → goals              (+ penalties, if he takes them)
    minutes → the player's key passes → assists
    (goalkeeper) opponent shots → on target → goals conceded → saves

so a player's on-target shots can never exceed his shots on any page, and
his goals never his shots. Every step is drawn from a distribution fitted to
his own match-to-match record (`roster.py`), and the team-level volume and
quality come from the team-strength layer (`teams.py`) for the fixture
("Game view") or against a chosen opponent ("Season view").

The full-match engine (`game.py`) does the same thing for every player at
once, with the score feeding back into the shot rates and every count
reconciling to the team totals. The two paths share their priors; this one
is faster and lets a page ask "what if he starts".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import roster as RO
from . import teams as T

# Team shots per match, conditional on the matchup, are overdispersed: the
# spike measured var/mean 1.95 given the team-season mean. A Poisson mixed
# with a log-normal multiplier of this sd reproduces it.
TEAM_SHOT_LOGSD = 0.27
# How strongly the opponent's rates move a player's own. 1 = the matchup
# ratio applied in full; the shares are already opponent-neutral estimates.
DEF_SHRINK = 0.7
# A key pass produces an assist only if it is converted: assists per xA at
# league level (assists per key pass 0.106 over xA per key pass 0.121).
LG_ASSIST_PER_XA = 0.87
SUB_MINUTES_CV = 0.65
# Yellow cards: the opponent's fouls-drawn effect is not modelled (v1).


# ---------------------------------------------------------------------------
# Team context for the match: where the player's volume and quality come from
# ---------------------------------------------------------------------------

def match_context(ctx: dict, team: str, opp: str | None, home: bool | None,
                  avail: dict | None = None, avail_opp: dict | None = None) -> dict:
    """Expected team shots / on-target rate / goal-per-shot for `team`, and
    the same for the opponent (for a goalkeeper), from the ratings. With
    `opp=None` (Season view, no opponent) the team faces a league-average
    defence at a neutral venue."""
    r = ctx["ratings"]
    lg_team = "_LEAGUE_"
    if opp is None or opp not in r["vol_att"].index:
        a = _neutral_matchup(r, team, attack=True, home=home)
        b = _neutral_matchup(r, team, attack=False, home=(None if home is None else not home))
    else:
        a = T.matchup(r, team, opp, home, avail)
        b = T.matchup(r, opp, team, (None if home is None else not home), avail_opp)
    a, b = T.apply_context(a, b, r)
    typical = ctx["live"]["team_vol"].get(team, ctx["live"]["team_vol"][lg_team])
    return dict(team=team, opp=opp, home=home, a=a, b=b, ratings=r,
                team_shots=(a["shots"], TEAM_SHOT_LOGSD),
                typical_shots=float(typical["shots"]),
                typical_goals=float(typical["goals"]),
                goal_factor=float(np.clip(a["goals"] / max(typical["goals"], 1e-6), 0.5, 2.0)),
                shot_factor=float(np.clip(a["shots"] / max(typical["shots"], 1e-6), 0.5, 2.0)))


def _neutral_matchup(r: dict, team: str, attack: bool, home: bool | None) -> dict:
    """`teams.matchup` against an average side: every opponent rating zero."""
    h = 0.0 if home is None else (1.0 if home else -1.0)
    if attack:
        shots = r["lg_shots"] * np.exp(r["vol_att"][team] + r["home_vol"] * h)
        xgps = r["lg_xgps"] * np.exp(r["q_att"][team] + r["home_q"] * h)
        sot = r["lg_sot"] * np.exp(r["sot_att"][team] + r["home_sot"] * h)
        fin = r["lg_fin"] * np.exp(r["fin"][team])
        pens = r["lg_pen"] * np.exp(r["pen_att"][team])
    else:
        shots = r["lg_shots"] * np.exp(r["vol_def"][team] + r["home_vol"] * h)
        xgps = r["lg_xgps"] * np.exp(r["q_def"][team] + r["home_q"] * h)
        sot = r["lg_sot"] * np.exp(r["sot_def"][team] + r["home_sot"] * h)
        fin = r["lg_fin"] * np.exp(r["gk"][team])
        pens = r["lg_pen"] * np.exp(r["pen_def"][team])
    shots = float(np.clip(shots, *T.SHOTS_CLIP)); xgps = float(np.clip(xgps, *T.XGPS_CLIP))
    sot = float(np.clip(sot, *T.SOT_CLIP))
    np_goals = shots * xgps * fin
    goals = float(np.clip(np_goals + pens * T.PEN_CONVERSION + r["lg_og"], *T.GOALS_CLIP))
    return dict(attack=team if attack else "avg", defence="avg" if attack else team, home=home,
                shots=shots, xgps=xgps, npxg=shots * xgps, sot_rate=sot, sot=shots * sot,
                finishing=float(fin), pens=float(pens), og=float(r["lg_og"]),
                np_goals=float(np_goals), goals=goals,
                p_goal=float(np.clip(np_goals / max(shots, 1e-6), 0.01, 0.5)),
                vol_att=0.0, vol_def=0.0, q_att=0.0, q_def=0.0,
                opp_vol_mult=float(np.exp(r["home_vol"] * h)), opp_q_mult=float(np.exp(r["home_q"] * h)),
                opp_sot_mult=float(np.exp(r["home_sot"] * h)), opp_fin_mult=1.0)


# ---------------------------------------------------------------------------
# Minutes
# ---------------------------------------------------------------------------

def draw_minutes(rng, row: pd.Series, n: int, force_start: bool | None = None) -> np.ndarray:
    """Minutes played in each simulation: start (subbed off around his usual
    minute) / substitute appearance / unused. `force_start=True` conditions
    on starting, False on coming off the bench (he plays, as a sub), None
    uses his probabilities."""
    if force_start is True:
        start = np.ones(n, bool)
    elif force_start is False:
        start = np.zeros(n, bool)
    else:
        start = rng.random(n) < float(row["p_start"])
    off = rng.random(n) < float(row["p_suboff"])
    off_min = np.clip(rng.normal(float(row["suboff_min"]), RO.LG_SUBOFF_SD, n), 30, 90)
    start_min = np.where(off, off_min, 90.0)
    sub_app = (rng.random(n) < float(row["p_subapp"])) if force_start is None else np.ones(n, bool)
    k, th = D.gamma_params(float(row["sub_minutes"]), SUB_MINUTES_CV * float(row["sub_minutes"]))
    sub_min = np.clip(rng.gamma(k, th, n), 1, 45)
    return np.where(start, start_min, np.where(sub_app, sub_min, 0.0))


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------

def simulate(row: pd.Series, mc: dict, n_sims: int = 20000, seed: int | None = None,
             force_start: bool | None = None, def_shrink: float = DEF_SHRINK,
             pen_taker: bool | None = None) -> dict:
    """One player's full line in `n_sims` matches. `row` is a roster row
    (`roster.build_roster`), `mc` a `match_context`."""
    rng = np.random.default_rng(seed)
    n = int(n_sims)
    a, b = mc["a"], mc["b"]

    minutes = draw_minutes(rng, row, n, force_start)
    on = minutes / 90.0

    # --- team shots in this match (shared with the goalkeeper path) --------
    mu_s, lsd = mc["team_shots"]
    team_shots = rng.poisson(mu_s * rng.lognormal(-0.5 * lsd ** 2, lsd, n))
    opp_shots = rng.poisson(b["shots"] * rng.lognormal(-0.5 * lsd ** 2, lsd, n))

    # --- the player's shots: his share of the team's, for his minutes -------
    exp_shots = team_shots * on * float(row["shot_share"])
    shots = np.minimum(rng.poisson(np.clip(exp_shots, 0, None)), team_shots)

    # --- on target and goals: his own rates, moved by the matchup ----------
    sot_mult = D.shrink(a["opp_sot_mult"], def_shrink)
    q_mult = D.shrink(a["opp_q_mult"], def_shrink)
    fin_mult = D.shrink(a["opp_fin_mult"], def_shrink) * a.get("context_mult", 1.0)
    p_sot = float(np.clip(row["sot_rate"] * sot_mult, 0.03, 0.9))
    p_goal = float(np.clip(row["xgps"] * q_mult * row["finishing"] * fin_mult, 0.005, 0.6))
    sot = rng.binomial(shots, p_sot)
    # a goal is (nearly always) on target: draw goals among the on-target shots
    p_goal_given_sot = float(np.clip(p_goal / p_sot, 0.01, 0.95))
    np_goals = rng.binomial(sot, p_goal_given_sot)

    # --- penalties: the taker, if on the pitch when one is won --------------
    takes = _is_pen_taker(row) if pen_taker is None else bool(pen_taker)
    pens = rng.poisson(a["pens"] * on) if takes else np.zeros(n, int)
    pen_goals = rng.binomial(pens, T.PEN_CONVERSION)
    goals = np_goals + pen_goals
    shots_all = shots + pens
    sot_all = sot + pens

    # --- key passes and assists --------------------------------------------
    exp_kp = team_shots * on * float(row["kp_share"])
    kp = rng.poisson(np.clip(exp_kp, 0, None))
    p_assist = float(np.clip(row["xa_kp"] * LG_ASSIST_PER_XA * q_mult * fin_mult, 0.01, 0.6))
    assists = rng.binomial(kp, p_assist)

    # --- cards ---------------------------------------------------------------
    yellow = rng.poisson(float(row["y90"]) * on)

    # --- goalkeeper: saves and clean sheet from the opponent's shots --------
    opp_sot = rng.binomial(opp_shots, float(np.clip(b["sot_rate"], 0.05, 0.9)))
    p_conc = float(np.clip(b["p_goal"] / max(b["sot_rate"], 1e-6), 0.02, 0.95))
    conceded = rng.binomial(opp_sot, p_conc) + rng.poisson(b["pens"] * T.PEN_CONVERSION + b["og"], n)
    saves = np.clip(opp_sot - np.minimum(conceded, opp_sot), 0, None)
    played = minutes > 0
    saves = np.where(played, saves, 0)
    clean_sheet = played & (minutes >= 60) & (conceded == 0)

    return dict(
        minutes=minutes, played=played, started=minutes >= 45 if force_start is None else np.full(n, bool(force_start)),
        team_shots=team_shots, opp_shots=opp_shots,
        shots=shots_all, np_shots=shots, sot=sot_all, goals=goals, np_goals=np_goals, pens=pens,
        key_passes=kp, assists=assists, yellow=yellow,
        saves=saves, conceded=conceded, clean_sheet=clean_sheet,
        rates=dict(p_sot=p_sot, p_goal=p_goal, p_assist=p_assist, sot_mult=sot_mult,
                   q_mult=q_mult, fin_mult=fin_mult, pen_taker=takes,
                   exp_minutes=float(minutes.mean())),
    )


def _is_pen_taker(row: pd.Series) -> bool:
    po = row.get("pen_order")
    try:
        return po is not None and np.isfinite(float(po)) and int(po) == 1
    except (TypeError, ValueError):
        return False


def summarize(x: np.ndarray, line: float | None = None, count: bool = True) -> dict:
    out = dict(mean=float(x.mean()), median=float(np.median(x)), std=float(x.std()),
               p10=float(np.percentile(x, 10)), p25=float(np.percentile(x, 25)),
               p75=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)))
    if count:
        out["dist"] = {int(k): float(v) for k, v in zip(*np.unique(x, return_counts=True))}
        for k in range(0, 6):
            out[f"p_ge{k}"] = float((x >= k).mean())
    if line is not None:
        p_over = float((x > line).mean())
        p_push = float((x == line).mean())
        out.update(line=float(line), p_over=p_over, p_under=1 - p_over - p_push, p_push=p_push,
                   fair_over=D.decimal_odds(p_over), fair_under=D.decimal_odds(1 - p_over - p_push))
    return out


def opponent_label(mc: dict) -> str:
    a, b = mc["a"], mc["b"]
    r = mc["ratings"]
    vol = a["shots"] / r["lg_shots"]
    q = a["xgps"] / r["lg_xgps"]
    def word(x, lo, hi, low, high, mid):
        return low if x < lo else high if x > hi else mid
    return (f"this opponent lets an average attack take {word(vol, 0.92, 1.08, 'fewer', 'more', 'about average')} "
            f"shots ({a['shots']:.1f} vs {r['lg_shots']:.1f}) of "
            f"{word(q, 0.95, 1.05, 'poorer', 'better', 'average')} quality "
            f"({a['xgps']:.3f} vs {r['lg_xgps']:.3f} xG per shot)")


# ---------------------------------------------------------------------------
# Context: everything the pages need, built once per (seasons, recency)
# ---------------------------------------------------------------------------

def prepare(seasons: tuple[int, ...], recency: D.Recency = D.RECENCY_DEFAULT,
            use_fpl: bool = True) -> dict:
    """Load every feed once and build the shared pieces: live rosters, the
    team ratings, league rates."""
    live = RO.load_live(tuple(sorted(seasons)), recency, use_fpl=use_fpl)
    ratings = T.team_ratings(live["tm"])
    return dict(live=live, ratings=ratings, seasons=live["seasons"], recency=live["recency"],
                lg=RO.league_rates(live))


def roster_for(ctx: dict, team: str, use_fpl: bool = True) -> pd.DataFrame:
    return RO.build_roster(ctx["live"], team, use_fpl, ctx["lg"])


if __name__ == "__main__":
    ctx = prepare((2024, 2025, 2026))
    for team, opp, name in (("MCI", "ARS", "Haaland"), ("ARS", "MCI", "Saka"), ("MCI", "ARS", "Donnarumma")):
        r = roster_for(ctx, team)
        row = r[r["name"].str.contains(name)].iloc[0]
        mc = match_context(ctx, team, opp, home=(team == "MCI"))
        sim = simulate(row, mc, n_sims=20000, seed=1)
        s = {k: summarize(sim[k]) for k in ("shots", "sot", "goals", "assists", "key_passes", "saves", "minutes")}
        print(f"\n{row['name']} ({team} {row['group']}, {row['p_start']:.0%} start, "
              f"{sim['rates']['exp_minutes']:.0f} min) vs {opp}: "
              f"shots {s['shots']['mean']:.2f}, SoT {s['sot']['mean']:.2f}, goals {s['goals']['mean']:.2f} "
              f"(anytime {sim['goals'].__ge__(1).mean():.1%}), KP {s['key_passes']['mean']:.2f}, "
              f"assists {s['assists']['mean']:.2f}, saves {s['saves']['mean']:.2f}, "
              f"CS {sim['clean_sheet'].mean():.1%} | " + opponent_label(mc))
