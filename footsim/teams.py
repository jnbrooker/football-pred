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

# Guard rails on a single matchup.
SHOTS_CLIP = (4.0, 30.0)
XGPS_CLIP = (0.05, 0.25)
GOALS_CLIP = (0.15, 5.0)
SOT_CLIP = (0.20, 0.55)

PEN_CONVERSION = 0.855
MAX_GOALS = 10


def _wsum_by(keys, vals, w):
    return (vals * w).groupby(keys).sum()


def _ipf(tm: pd.DataFrame, y: str, expo: str, teams: list, prior_n: float,
         lg_rate: float, home_col: str = "home", iters: int = SOLVE_ITERS,
         fit_home: bool = True, prior_expo: float | None = None) -> tuple[pd.Series, pd.Series, float]:
    """Iterative proportional fitting of  y ~ expo × exp(att_i + def_j + home).

    `y` is the count (or sum) column, `expo` the exposure column (1 for
    per-match rates, shots for per-shot rates). Returns (att, def, home) on
    the log scale; att and def are each centred on zero. `prior_n` matches-
    worth of league-average pseudo-data shrinks every rating; `prior_expo` is
    the exposure of one such match (defaults to the mean exposure).
    """
    w = tm["w"].astype(float)
    yv = tm[y].astype(float)
    ex = tm[expo].astype(float) if expo != "_one_" else pd.Series(1.0, index=tm.index)
    pe = float(prior_expo if prior_expo is not None else D.wmean(ex, w))
    pseudo = prior_n * pe * lg_rate            # pseudo-count of y at league rate
    att = pd.Series(0.0, index=teams)
    dfn = pd.Series(0.0, index=teams)
    home = 0.0
    hv = tm[home_col].astype(float)
    for _ in range(int(iters)):
        base = ex * lg_rate * np.exp(dfn.reindex(tm["opp"]).values + home * hv)
        num = _wsum_by(tm["team"], yv, w).reindex(teams).fillna(0.0) + pseudo
        den = _wsum_by(tm["team"], base, w).reindex(teams).fillna(0.0) + pseudo
        new_att = np.log(num / den)
        new_att -= new_att.mean()
        att = SOLVE_DAMPING * new_att + (1 - SOLVE_DAMPING) * att

        base = ex * lg_rate * np.exp(att.reindex(tm["team"]).values + home * hv)
        num = _wsum_by(tm["opp"], yv, w).reindex(teams).fillna(0.0) + pseudo
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


def team_ratings(tm: pd.DataFrame, iters: int = SOLVE_ITERS) -> dict:
    """Opponent-adjusted ratings from the team-match table (`data.load_team_matches`)."""
    if tm is None or tm.empty:
        raise ValueError("No team-match data — the Understat cache is empty.")
    d = tm.copy()
    d["w"] = d["w"].astype(float) if "w" in d.columns else 1.0
    d["np_shots"] = d["shots"] - d["pens"]
    d["np_goals"] = (d["goals_from_shots"] - d["pen_goals"]).clip(lower=0)
    d["np_sot"] = d["sot"] - d["pens"]                    # penalties are (nearly) always on target
    d["np_sot"] = d["np_sot"].clip(lower=0)
    d["one"] = 1.0
    teams = sorted(set(d["team"]) | set(d["opp"]))
    W = d["w"]

    lg_shots = D.wmean(d["np_shots"], W)                   # non-penalty shots per team-match
    lg_xgps = float((d["npxg"] * W).sum() / (d["np_shots"] * W).sum())
    lg_fin = float((d["np_goals"] * W).sum() / (d["npxg"] * W).sum())
    lg_sot = float((d["np_sot"] * W).sum() / (d["np_shots"] * W).sum())
    lg_pen = D.wmean(d["pens"], W)
    lg_og = D.wmean(d["og_scored"], W)

    vol_att, vol_def, home_vol = _ipf(d, "np_shots", "one", teams, VOL_PRIOR_N, lg_shots, iters=iters)
    q_att, q_def, home_q = _ipf(d, "npxg", "np_shots", teams, QUAL_PRIOR_N, lg_xgps, iters=iters,
                                prior_expo=lg_shots)
    fin, gk, _ = _ipf(d, "np_goals", "npxg", teams, FIN_PRIOR_XG / max(lg_xgps * lg_shots, 1e-6),
                      lg_fin, iters=iters, fit_home=False, prior_expo=lg_xgps * lg_shots)
    sot_att, sot_def, home_sot = _ipf(d, "np_sot", "np_shots", teams, SOT_PRIOR_N, lg_sot, iters=iters,
                                      prior_expo=lg_shots)
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
        lg_shots=lg_shots, lg_xgps=lg_xgps, lg_fin=lg_fin, lg_sot=lg_sot,
        lg_pen=lg_pen, lg_og=lg_og,
        lg_goals=float(D.wmean(d["goals"], W)),
        raw=raw, matches_w=matches, n_matches=int(len(d) // 2),
        seasons=sorted(d["season"].unique().tolist()),
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
    }).reindex(teams)


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------

def matchup(r: dict, attack: str, defence: str, home: bool | None) -> dict:
    """Expected production of one attack against one defence: shots, xG per
    shot, on-target rate, finishing multiplier, penalties, goals.
    `home` True = the attacking side is at home, False = away, None = neutral."""
    for t in (attack, defence):
        if t not in r["vol_att"].index:
            raise ValueError(f"No rating for {t!r}.")
    h = 0.0 if home is None else (1.0 if home else -1.0)
    shots = r["lg_shots"] * np.exp(r["vol_att"][attack] + r["vol_def"][defence] + r["home_vol"] * h)
    xgps = r["lg_xgps"] * np.exp(r["q_att"][attack] + r["q_def"][defence] + r["home_q"] * h)
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
        attack=attack, defence=defence, home=home,
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


def expected_goals(r: dict, team_a: str, team_b: str, home: str | None = "a") -> dict:
    """The deterministic read on a fixture: expected shots and goals each way
    and the analytic 1X2. `home` = "a", "b" or None for a neutral venue."""
    ha = True if home in ("a", team_a) else False if home in ("b", team_b) else None
    a = matchup(r, team_a, team_b, ha)
    b = matchup(r, team_b, team_a, (None if ha is None else not ha))
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
