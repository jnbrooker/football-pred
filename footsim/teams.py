"""
footsim/teams.py — the team-strength layer (roadmap §3.7).

The model is SELF-CONTAINED: no bookmaker line anywhere, so it has to form
its own opinion of how good each team is. The atom is a SHOT, and a
team-match decomposes into three multiplicative pieces:

    shots            = league × attack_volume(i)  × defence_volume(j)  × home
    xG per shot      = league × attack_quality(i) × defence_quality(j) × home
    goals per xG     = league × finishing(i)      × keeping(j)

Each set of ratings is solved jointly by iterative proportional fitting — a
team's attack-volume rating is what its shots were, relative to what the
defences it faced would concede to an average side, and vice versa — which
is the SRS idea on the log scale. Every rating is shrunk toward league by
pseudo-observations at league average, so a rating on eight matches is
mostly prior and one on eighty is mostly data, and recent matches are
weighted more (`w` from `data.game_weights`).

What the spike found (2023–26 EPL, recorded in ROADMAP §6) and this layer
therefore encodes:

  * Home advantage is VOLUME: home sides take 20% more shots at the same
    xG per shot. So it is fitted on the shot rate, and separately (small)
    on quality, never as a flat goals bonus.
  * Finishing (goals / xG) has no year-to-year signal for players and very
    little for teams, so it is shrunk toward league over `FIN_PRIOR_XG`
    expected goals-worth. Understat's xG runs ~12% above actual goals in
    this window; the league finishing level is measured, not assumed.
  * Penalties (0.12 per team-match, 85% converted) and own goals (0.055)
    are separate small events with their own rates.

Outputs per matchup: expected shots, on-target shots, xG per shot and goals
for each side, plus an analytic 1X2 read (independent Poissons — the residual
correlation between the two sides' goals, once strength is known, is −0.015).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import data as D

# Shrinkage, in matches-worth of league-average pseudo-observations.
VOL_PRIOR_N = 1.0          # attack / defence shot volume (grid on the 2024–26 backtest; slope 1.04 / 0.99)
QUAL_PRIOR_N = 2.0         # attack / defence xG per shot (in matches of shots)
SOT_PRIOR_N = 10.0         # on-target rate for / against
FIN_PRIOR_XG = 80.0        # finishing and keeping, in xG-worth (heavy: mostly noise; lighter was worse)
PEN_PRIOR_N = 30.0         # penalties won / conceded per match

SOLVE_ITERS = 40
SOLVE_DAMPING = 0.6

# Promoted sides (roadmap §6.15). A team with no top-flight matches in the
# previous season is not an average side. Every promoted class since 2023
# has lacked ATTACKING quality — 0.65–0.83× league xG, from fewer shots
# (log −0.20) of lower quality (−0.11), with a spread of only 0.05–0.07
# across teams — so the attack prior is worth PROMOTED_PRIOR_N matches
# rather than the usual one. Their DEFENCE is not predictable from promotion:
# the 2023–24 classes conceded 1.2–1.4× league, Sunderland / Leeds 2025 and
# Coventry / Hull 2026 conceded at or below average, and the backtest flips
# sign with it — so the defence prior is zero (shown in the log, not priced).
# The prior is also what a team with NO data at all is rated on (round one),
# instead of being unrated.
PROMOTED_PRIOR = dict(vol_att=-0.204, q_att=-0.114, sot_att=-0.041,
                      vol_def=0.0, q_def=0.0, sot_def=0.0, fin=0.0, gk=0.0, pen_att=0.0, pen_def=0.0)
PROMOTED_PRIOR_N = 8.0

# Guard rails on a single matchup.
SHOTS_CLIP = (4.0, 30.0)
XGPS_CLIP = (0.05, 0.25)
GOALS_CLIP = (0.15, 5.0)
SOT_CLIP = (0.20, 0.55)

PEN_CONVERSION = 0.855
MAX_GOALS = 10

# Attacks (roadmap §6.19). Understat's `deep` completions — passes completed
# within ~20 yards of goal — are the closest observed thing to an attack,
# and they predict a team's FUTURE xG better than its past xG does (odd →
# even rounds within a season, 140 team-seasons: R² 0.68 vs 0.64; together
# deep t = 6.3, xG t = 4.5; year-to-year r 0.75 vs 0.70). Attack counts are
# a less noisy read on attacking quality than shots, which carry shot-level
# luck. So the volume rating CAN be a blend, in log space, of the shot-volume
# rating and an attack-volume rating (same IPF on `deep`), weighted by
# DEEP_WEIGHT. Fitted on the 2022–25 backtest the weight is zero: blending
# widens the spread of the probabilities toward the market's (0.77 → 0.95 at
# 0.5) but worsens log-loss in three seasons of four and drops the
# calibration slope to 0.71 — it stretches the extremes without earning
# them; 0.15 is within noise of zero. The attack ratings are still computed
# (`deep_att` / `deep_def`) and shown on the Team Strength page.
DEEP_WEIGHT = 0.0
DEEP_PRIOR_N = 1.0

# Match context and conversion (roadmap §6.17). Out of sample over 2021–25,
# expected xG is properly spread (actual on expected, slope 0.90) but goals
# are not (1.06 per side, 1.13 on margins): a side expected to concede a lot
# converts its own chances at 0.89 of its xG-based expectation, a side
# expected to concede little at 1.03. Fitted by Poisson regression of goals
# on the offset of expected goals: coefficient −0.171 (t = −4.2, n = 3 800)
# on log(opponent's expected xG / league). Applied to both sides of a
# matchup by `apply_context`.
FIN_CONTEXT_BETA = -0.12     # pooled fit −0.171; −0.12 keeps the margin slope at 0.95 (see ROADMAP §6.17)


def _wsum_by(keys, vals, w):
    return (vals * w).groupby(keys).sum()


def _ipf(tm: pd.DataFrame, y: str, expo: str, teams: list, prior_n: float,
         lg_rate: float, home_col: str = "home", iters: int = SOLVE_ITERS,
         fit_home: bool = True, prior_expo: float | None = None,
         prior_att: pd.Series | None = None, prior_def: pd.Series | None = None,
         prior_n_team: pd.Series | None = None) -> tuple[pd.Series, pd.Series, float]:
    """Iterative proportional fitting of  y ~ expo × exp(att_i + def_j + home).

    `y` is the count (or sum) column, `expo` the exposure column (1 for
    per-match rates, shots for per-shot rates). Returns (att, def, home) on
    the log scale; att and def are each centred on zero. `prior_n` matches-
    worth of pseudo-data shrinks every rating toward `prior_att` / `prior_def`
    (log scale, default 0 = league average); `prior_n_team` overrides the
    strength per team (promoted sides). `prior_expo` is the exposure of one
    pseudo-match (defaults to the mean exposure).
    """
    w = tm["w"].astype(float)
    yv = tm[y].astype(float)
    ex = tm[expo].astype(float) if expo != "_one_" else pd.Series(1.0, index=tm.index)
    pe = float(prior_expo if prior_expo is not None else D.wmean(ex, w))
    pn = (prior_n_team.reindex(teams).fillna(prior_n) if prior_n_team is not None
          else pd.Series(prior_n, index=teams))
    pseudo = pn * pe * lg_rate                 # pseudo-count of y per team
    pa = prior_att.reindex(teams).fillna(0.0) if prior_att is not None else pd.Series(0.0, index=teams)
    pdf = prior_def.reindex(teams).fillna(0.0) if prior_def is not None else pd.Series(0.0, index=teams)
    att = pd.Series(0.0, index=teams)
    dfn = pd.Series(0.0, index=teams)
    home = 0.0
    hv = tm[home_col].astype(float)
    for _ in range(int(iters)):
        base = ex * lg_rate * np.exp(dfn.reindex(tm["opp"]).values + home * hv)
        num = _wsum_by(tm["team"], yv, w).reindex(teams).fillna(0.0) + pseudo * np.exp(pa)
        den = _wsum_by(tm["team"], base, w).reindex(teams).fillna(0.0) + pseudo
        new_att = np.log(num / den)
        new_att -= new_att.mean()
        att = SOLVE_DAMPING * new_att + (1 - SOLVE_DAMPING) * att

        base = ex * lg_rate * np.exp(att.reindex(tm["team"]).values + home * hv)
        num = _wsum_by(tm["opp"], yv, w).reindex(teams).fillna(0.0) + pseudo * np.exp(pdf)
        den = _wsum_by(tm["opp"], base, w).reindex(teams).fillna(0.0) + pseudo
        new_def = np.log(num / den)
        new_def -= new_def.mean()
        dfn = SOLVE_DAMPING * new_def + (1 - SOLVE_DAMPING) * dfn

        if fit_home:
            base = ex * lg_rate * np.exp(att.reindex(tm["team"]).values + dfn.reindex(tm["opp"]).values)
            hm = hv > 0.5
            ratio_h = float((yv[hm] * w[hm]).sum() / (base[hm] * w[hm]).sum())
            ratio_a = float((yv[~hm] * w[~hm]).sum() / (base[~hm] * w[~hm]).sum())
            # home advantage is the log ratio; the league rate absorbs the mean
            home = 0.5 * np.log(ratio_h / ratio_a)
    return att, dfn, float(home)


def promoted_teams(tm: pd.DataFrame, season: int, extra_teams=()) -> set:
    """Teams playing `season` (in the data or in `extra_teams`) with no
    matches in the season before it — promoted, or back after an absence."""
    prev = set(tm.loc[tm["season"] == season - 1, "team"])
    cur = set(tm.loc[tm["season"] == season, "team"]) | set(extra_teams)
    return {t for t in cur if t not in prev}


def team_ratings(tm: pd.DataFrame, iters: int = SOLVE_ITERS, season: int | None = None,
                 extra_teams=()) -> dict:
    """Opponent-adjusted ratings from the team-match table (`data.load_team_matches`).

    `season` is the season being predicted (default: the latest in the data)
    and `extra_teams` any teams in its fixtures that have no matches yet —
    both get the promoted prior if they have no matches the season before."""
    if tm is None or tm.empty:
        raise ValueError("No team-match data — the Understat cache is empty.")
    d = tm.copy()
    season = int(season) if season is not None else int(d["season"].max())
    promoted = promoted_teams(d, season, extra_teams)
    d["w"] = d["w"].astype(float) if "w" in d.columns else 1.0
    d["np_shots"] = d["shots"] - d["pens"]
    d["np_goals"] = (d["goals_from_shots"] - d["pen_goals"]).clip(lower=0)
    d["np_sot"] = d["sot"] - d["pens"]                    # penalties are (nearly) always on target
    d["np_sot"] = d["np_sot"].clip(lower=0)
    d["one"] = 1.0
    teams = sorted(set(d["team"]) | set(d["opp"]) | set(extra_teams))
    W = d["w"]
    pn = pd.Series({t: PROMOTED_PRIOR_N for t in promoted}, dtype=float)

    def prior(key):
        return pd.Series({t: PROMOTED_PRIOR[key] for t in promoted}, dtype=float)

    lg_shots = D.wmean(d["np_shots"], W)                   # non-penalty shots per team-match
    lg_xgps = float((d["npxg"] * W).sum() / (d["np_shots"] * W).sum())
    lg_fin = float((d["np_goals"] * W).sum() / (d["npxg"] * W).sum())
    lg_sot = float((d["np_sot"] * W).sum() / (d["np_shots"] * W).sum())
    lg_pen = D.wmean(d["pens"], W)
    lg_og = D.wmean(d["og_scored"], W)

    vol_att, vol_def, home_vol = _ipf(d, "np_shots", "one", teams, VOL_PRIOR_N, lg_shots, iters=iters,
                                      prior_att=prior("vol_att"), prior_def=prior("vol_def"), prior_n_team=pn)
    q_att, q_def, home_q = _ipf(d, "npxg", "np_shots", teams, QUAL_PRIOR_N, lg_xgps, iters=iters,
                                prior_expo=lg_shots,
                                prior_att=prior("q_att"), prior_def=prior("q_def"), prior_n_team=pn)
    fin, gk, _ = _ipf(d, "np_goals", "npxg", teams, FIN_PRIOR_XG / max(lg_xgps * lg_shots, 1e-6),
                      lg_fin, iters=iters, fit_home=False, prior_expo=lg_xgps * lg_shots)
    sot_att, sot_def, home_sot = _ipf(d, "np_sot", "np_shots", teams, SOT_PRIOR_N, lg_sot, iters=iters,
                                      prior_expo=lg_shots,
                                      prior_att=prior("sot_att"), prior_def=prior("sot_def"), prior_n_team=pn)
    # attack volume from deep completions, blended into the shot-volume rating
    deep_att = pd.Series(0.0, index=teams); deep_def = pd.Series(0.0, index=teams); lg_deep = np.nan
    if "deep" in d.columns and d["deep"].notna().mean() > 0.9:
        dd = d[d["deep"].notna()].copy()
        lg_deep = D.wmean(dd["deep"], dd["w"])
        deep_att, deep_def, _ = _ipf(dd, "deep", "one", teams, DEEP_PRIOR_N, lg_deep, iters=iters,
                                     prior_att=prior("vol_att"), prior_def=prior("vol_def"), prior_n_team=pn)
        if DEEP_WEIGHT > 0:
            vol_att = (1 - DEEP_WEIGHT) * vol_att + DEEP_WEIGHT * deep_att
            vol_def = (1 - DEEP_WEIGHT) * vol_def + DEEP_WEIGHT * deep_def
    pen_att, pen_def, _ = _ipf(d, "pens", "one", teams, PEN_PRIOR_N, max(lg_pen, 1e-4), iters=iters,
                               fit_home=False)

    raw = _raw(d, teams)
    matches = d.groupby("team")["w"].sum().reindex(teams).fillna(0.0)
    return dict(
        teams=teams,
        vol_att=vol_att, vol_def=vol_def, home_vol=home_vol,
        q_att=q_att, q_def=q_def, home_q=home_q,
        fin=fin, gk=gk,
        sot_att=sot_att, sot_def=sot_def, home_sot=home_sot,
        pen_att=pen_att, pen_def=pen_def,
        deep_att=deep_att, deep_def=deep_def, lg_deep=lg_deep,
        lg_shots=lg_shots, lg_xgps=lg_xgps, lg_fin=lg_fin, lg_sot=lg_sot,
        lg_pen=lg_pen, lg_og=lg_og,
        lg_goals=float(D.wmean(d["goals"], W)),
        raw=raw, matches_w=matches, n_matches=int(len(d) // 2),
        seasons=sorted(d["season"].unique().tolist()), season=season, promoted=sorted(promoted),
    )


def _raw(d: pd.DataFrame, teams: list) -> pd.DataFrame:
    w = d["w"]
    g = d.assign(ws=d["np_shots"] * w, wsa=d["shots_against"] * w, wx=d["npxg"] * w,
                 wxa=d["npxg_against"] * w, wg=d["goals"] * w, wga=d["goals_against"] * w)
    s = g.groupby("team").agg(ws=("ws", "sum"), wsa=("wsa", "sum"), wx=("wx", "sum"),
                              wxa=("wxa", "sum"), wg=("wg", "sum"), wga=("wga", "sum"), w=("w", "sum"))
    return pd.DataFrame({
        "shots_for": s["ws"] / s["w"], "shots_against": s["wsa"] / s["w"],
        "xg_for": s["wx"] / s["w"], "xg_against": s["wxa"] / s["w"],
        "goals_for": s["wg"] / s["w"], "goals_against": s["wga"] / s["w"],
    }).reindex(teams).fillna(np.nan)


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------

def matchup(r: dict, attack: str, defence: str, home: bool | None,
            avail: dict | None = None) -> dict:
    """Expected production of one attack against one defence: shots, xG per
    shot, on-target rate, finishing multiplier, penalties, goals.
    `home` True = the attacking side is at home, False = away, None = neutral.
    `avail` is `roster.availability_factor` for the ATTACKING side: volume and
    quality multipliers for the squad actually available."""
    for t in (attack, defence):
        if t not in r["vol_att"].index:
            raise ValueError(f"No rating for {t!r}.")
    h = 0.0 if home is None else (1.0 if home else -1.0)
    av_v = float(avail.get("volume", 1.0)) if avail else 1.0
    av_q = float(avail.get("quality", 1.0)) if avail else 1.0
    shots = r["lg_shots"] * np.exp(r["vol_att"][attack] + r["vol_def"][defence] + r["home_vol"] * h) * av_v
    xgps = r["lg_xgps"] * np.exp(r["q_att"][attack] + r["q_def"][defence] + r["home_q"] * h) * av_q
    sot = r["lg_sot"] * np.exp(r["sot_att"][attack] + r["sot_def"][defence] + r["home_sot"] * h)
    fin = r["lg_fin"] * np.exp(r["fin"][attack] + r["gk"][defence])
    pens = r["lg_pen"] * np.exp(r["pen_att"][attack] + r["pen_def"][defence])
    shots = float(np.clip(shots, *SHOTS_CLIP))
    xgps = float(np.clip(xgps, *XGPS_CLIP))
    sot = float(np.clip(sot, *SOT_CLIP))
    npxg = shots * xgps
    np_goals = npxg * fin
    goals = float(np.clip(np_goals + pens * PEN_CONVERSION + r["lg_og"], *GOALS_CLIP))
    return dict(
        attack=attack, defence=defence, home=home, avail_volume=av_v, avail_quality=av_q,
        shots=shots, xgps=xgps, npxg=float(npxg), sot_rate=sot, sot=float(shots * sot),
        finishing=float(fin), pens=float(pens), og=float(r["lg_og"]),
        np_goals=float(np_goals), goals=goals,
        # per-shot goal probability the engine uses (non-penalty)
        p_goal=float(np.clip(np_goals / max(shots, 1e-6), 0.01, 0.5)),
        vol_att=float(r["vol_att"][attack]), vol_def=float(r["vol_def"][defence]),
        q_att=float(r["q_att"][attack]), q_def=float(r["q_def"][defence]),
        # what the DEFENCE and the venue do to a player's own rates (his share
        # and quality already carry his team's context): multipliers on shot
        # volume, xG per shot, on-target rate and finishing
        opp_vol_mult=float(np.exp(r["vol_def"][defence] + r["home_vol"] * h)),
        opp_q_mult=float(np.exp(r["q_def"][defence] + r["home_q"] * h)),
        opp_sot_mult=float(np.exp(r["sot_def"][defence] + r["home_sot"] * h)),
        opp_fin_mult=float(np.exp(r["gk"][defence])),
    )


def apply_context(a: dict, b: dict, r: dict, beta: float | None = None) -> tuple[dict, dict]:
    """Scale each side's conversion by the match context: the more the
    opponent is expected to create, the fewer goals a side scores per xG of
    its own (see FIN_CONTEXT_BETA). Returns adjusted copies of both matchups."""
    lg = r["lg_shots"] * r["lg_xgps"]
    beta = FIN_CONTEXT_BETA if beta is None else beta
    out = []
    for me, other in ((a, b), (b, a)):
        m = dict(me)
        k = float(np.clip((other["npxg"] / max(lg, 1e-6)) ** beta, 0.7, 1.4))
        m["context_mult"] = k
        m["finishing"] = me["finishing"] * k
        m["np_goals"] = me["np_goals"] * k
        m["goals"] = float(np.clip(m["np_goals"] + me["pens"] * PEN_CONVERSION + me["og"], *GOALS_CLIP))
        m["p_goal"] = float(np.clip(m["np_goals"] / max(me["shots"], 1e-6), 0.01, 0.5))
        out.append(m)
    return out[0], out[1]


def score_matrix(mu_a: float, mu_b: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(score = i–j) under independent Poissons."""
    pa = poisson.pmf(np.arange(max_goals + 1), mu_a)
    pb = poisson.pmf(np.arange(max_goals + 1), mu_b)
    return np.outer(pa, pb)


def outcome_probs(mu_a: float, mu_b: float) -> dict:
    m = score_matrix(mu_a, mu_b)
    i, j = np.indices(m.shape)
    return dict(win_a=float(m[i > j].sum()), draw=float(m[i == j].sum()),
                win_b=float(m[i < j].sum()),
                over25=float(m[(i + j) > 2.5].sum()), btts=float(m[(i > 0) & (j > 0)].sum()))


def expected_goals(r: dict, team_a: str, team_b: str, home: str | None = "a",
                   avail_a: dict | None = None, avail_b: dict | None = None) -> dict:
    """The deterministic read on a fixture: expected shots and goals each way
    and the analytic 1X2. `home` = "a", "b" or None for a neutral venue."""
    ha = True if home in ("a", team_a) else False if home in ("b", team_b) else None
    a = matchup(r, team_a, team_b, ha, avail_a)
    b = matchup(r, team_b, team_a, (None if ha is None else not ha), avail_b)
    a, b = apply_context(a, b, r)
    p = outcome_probs(a["goals"], b["goals"])
    return dict(team_a=team_a, team_b=team_b, home=home,
                goals_a=a["goals"], goals_b=b["goals"],
                shots_a=a["shots"], shots_b=b["shots"],
                margin=a["goals"] - b["goals"], total=a["goals"] + b["goals"],
                a=a, b=b, **p)


def strength_table(r: dict) -> pd.DataFrame:
    """Every team's ratings, sorted by net expected goal difference vs an
    average side at a neutral venue — the layer's headline view."""
    rows = []
    for t in r["teams"]:
        att = r["lg_shots"] * np.exp(r["vol_att"][t]) * r["lg_xgps"] * np.exp(r["q_att"][t]) \
            * r["lg_fin"] * np.exp(r["fin"][t]) + r["lg_pen"] * np.exp(r["pen_att"][t]) * PEN_CONVERSION + r["lg_og"]
        dfn = r["lg_shots"] * np.exp(r["vol_def"][t]) * r["lg_xgps"] * np.exp(r["q_def"][t]) \
            * r["lg_fin"] * np.exp(r["gk"][t]) + r["lg_pen"] * np.exp(r["pen_def"][t]) * PEN_CONVERSION + r["lg_og"]
        rows.append(dict(
            team=t, xg_for=att, xg_against=dfn, net=att - dfn,
            shots_for=r["lg_shots"] * np.exp(r["vol_att"][t]),
            shots_against=r["lg_shots"] * np.exp(r["vol_def"][t]),
            attacks_for=(r["lg_deep"] * np.exp(r["deep_att"][t]) if np.isfinite(r.get("lg_deep", np.nan)) else np.nan),
            attacks_against=(r["lg_deep"] * np.exp(r["deep_def"][t]) if np.isfinite(r.get("lg_deep", np.nan)) else np.nan),
            xgps_for=r["lg_xgps"] * np.exp(r["q_att"][t]),
            xgps_against=r["lg_xgps"] * np.exp(r["q_def"][t]),
            finishing=np.exp(r["fin"][t]), keeping=np.exp(-r["gk"][t]),
            raw_shots_for=r["raw"].loc[t, "shots_for"], raw_shots_against=r["raw"].loc[t, "shots_against"],
            raw_goals_for=r["raw"].loc[t, "goals_for"], raw_goals_against=r["raw"].loc[t, "goals_against"],
            matches_w=r["matches_w"][t],
        ))
    return pd.DataFrame(rows).set_index("team").sort_values("net", ascending=False)


if __name__ == "__main__":
    seasons = (2024, 2025, 2026)
    tm = D.load_team_matches(seasons)
    r = team_ratings(tm)
    print(f"{r['n_matches']} matches, seasons {r['seasons']} | league: {r['lg_shots']:.2f} np shots, "
          f"{r['lg_xgps']:.4f} xG/shot, finishing {r['lg_fin']:.3f}, SoT {r['lg_sot']:.3f}, "
          f"pens {r['lg_pen']:.3f}, OG {r['lg_og']:.3f} | goals/team-match {r['lg_goals']:.3f}")
    print(f"home advantage: shots x{np.exp(2 * r['home_vol']):.3f} (home vs away), "
          f"xG/shot x{np.exp(2 * r['home_q']):.3f}, SoT rate x{np.exp(2 * r['home_sot']):.3f}")
    t = strength_table(r)
    cols = ["xg_for", "xg_against", "net", "shots_for", "shots_against", "xgps_for", "finishing", "keeping"]
    print("\nTop 6:"); print(t.head(6)[cols].to_string(float_format=lambda x: f"{x:6.3f}"))
    print("\nBottom 4:"); print(t.tail(4)[cols].to_string(float_format=lambda x: f"{x:6.3f}"))
    for a, b in (("MCI", "ARS"), ("LIV", "SUN")):
        e = expected_goals(r, a, b, home="a")
        print(f"\n{a} (home) vs {b}: {e['goals_a']:.2f} - {e['goals_b']:.2f} | "
              f"1X2 {e['win_a']:.1%} / {e['draw']:.1%} / {e['win_b']:.1%} | O2.5 {e['over25']:.1%} | "
              f"shots {e['shots_a']:.1f} - {e['shots_b']:.1f}")
