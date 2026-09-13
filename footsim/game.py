"""
footsim/game.py — the match engine (roadmap §3.8).

Two squads in, a simulated box score out. Each simulation runs top-down so
the box score can never disagree with the scoreline:

  1. **Lineups.** Every player starts / comes on / stays unused with his own
     probabilities and minutes; exactly one goalkeeper starts.
  2. **Segments.** The match is six 15-minute segments (the last one carries
     stoppage time — 23% of all shots). A side's shots in a segment are
     Poisson at its matchup rate × the segment's share × a latent DOMINANCE
     draw (one side's gain is the other's loss — the two sides' shot counts
     are negatively correlated in real matches) × the game-state factor
     exp(−LEAD_BETA_VOL × (lead − expected lead)). Each shot is on target and
     a goal at the matchup rates, both RAISED when leading (a side sitting on
     a lead shoots less but on the counter): the spike measured trailing /
     level / leading xG per shot at 0.110 / 0.121 / 0.147 and shot rate at
     ×1.13 / ×1.00 / ×0.87, so goals per minute barely move with the state
     while shots do — the football twin of "points are pace-invariant".
     Penalties and own goals are separate small Poisson events.
  3. **Allocation.** The side's shots are split across the players on the
     pitch by a multinomial over minutes × shot share; on-target shots are
     handed out among those shots by each player's on-target rate; goals
     among the on-target shots by each player's conversion — capped, so a
     player's goals ≤ his on-target shots ≤ his shots in every simulation.
     Each non-penalty goal is assisted with P = ASSIST_RATE, the assister
     drawn from teammates on the pitch by key-pass weight (never the scorer);
     the rest of the side's key passes are split the same way, so assists ⊂
     key passes. The goalkeeper's saves are the opponent's on-target shots
     minus the goals he conceded; clean sheets and goals conceded per player
     use the segment he was on the pitch for.
  4. **Reporting.** Score distribution, 1X2, goal lines, both box scores.

Centring the game-state factor on the EXPECTED lead (the ratings' running
margin) rather than zero is essential, as it was in the NFL engine: the
ratings are fitted to actual matches, which already contain favourites
sitting on leads, so centring on zero would double-count it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import players as P
from . import roster as RO
from . import teams as T

# Segment shares of shots (measured 2023–26; the sixth carries stoppage time).
SEG_SHARE = np.array([0.130, 0.145, 0.149, 0.193, 0.155, 0.227])
SEG_END = np.array([15, 30, 45, 60, 75, 90], float)
SEG_START = np.array([0, 15, 30, 45, 60, 75], float)

# Game state (roadmap §6): per goal of lead relative to the expected lead,
# clipped at ±2. Volume down, on-target and conversion up when leading.
# Calibrated by `calibrate.engine_moments` on 2024–26 (grid over these four
# constants) to reproduce the real draw rate, goal-difference sd, team-goals
# dispersion, shot dispersion, the two sides' shot correlation and the
# within-matchup elasticity of goals to shots (0.33). The volume effect is
# stronger than the raw state table suggests (×0.87 when leading ≈ −0.14):
# that table is confounded — good sides both lead more and shoot more.
LEAD_BETA_VOL = -0.22
LEAD_BETA_SOT = 0.06
LEAD_BETA_GOAL = 0.10
LEAD_CLIP = 2.0

# Within-match randomness in team shot volume beyond Poisson: a shared
# dominance draw (added to one side, subtracted from the other) and an
# independent wobble per side. Most of the extra-Poisson spread in real shot
# counts turns out to be the game state, not exogenous wobble.
DOMINANCE_SD = 0.16
TEAM_WOBBLE_SD = 0.10

# The state feedback is convex (a side behind expectation shoots more than a
# side ahead shoots less), so without correction the engine's mean shots run
# ~6% and mean goals ~2% above the ratings that drive it. These bring the
# simulated means back onto the analytic layer (checked in __main__).
SHOT_MEAN_CORR = 0.952
GOAL_MEAN_CORR = 1.034

ASSIST_RATE = 0.77           # non-penalty goals with an assist
KP_PER_SHOT = 0.78           # key passes per non-penalty shot (league)
PEN_ON_TARGET = 0.97

SUB_MINUTES_CV = P.SUB_MINUTES_CV


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------

def draw_lineups(rng, roster: pd.DataFrame, n: int) -> dict:
    """Per simulation, per player: started (bool), on-pitch window [t0, t1]
    in minutes (t1 = t0 for unused) and minutes."""
    k = len(roster)
    p_start = roster["p_start"].values.astype(float)
    gk = (roster["group"].values == "GK")
    start = rng.random((n, k)) < p_start[None, :]
    # exactly one goalkeeper: a categorical draw over the keepers
    if gk.any():
        pg = np.clip(p_start[gk], 1e-6, None)
        pg = pg / pg.sum()
        pick = rng.choice(np.where(gk)[0], size=n, p=pg)
        start[:, gk] = False
        start[np.arange(n), pick] = True
    off = rng.random((n, k)) < roster["p_suboff"].values[None, :]
    off_min = np.clip(rng.normal(roster["suboff_min"].values[None, :], RO.LG_SUBOFF_SD, (n, k)), 30, 90)
    t1_start = np.where(off, off_min, 90.0)
    sub_app = (~start) & (rng.random((n, k)) < roster["p_subapp"].values[None, :])
    sm = roster["sub_minutes"].values.astype(float)
    shape = 1.0 / SUB_MINUTES_CV ** 2
    sub_min = np.clip(rng.gamma(shape, (sm / shape)[None, :], (n, k)), 1, 45)
    t0 = np.where(start, 0.0, np.where(sub_app, 90.0 - sub_min, 90.0))
    t1 = np.where(start, t1_start, 90.0)
    t1 = np.where(start | sub_app, t1, 90.0)
    minutes = np.where(start | sub_app, t1 - t0, 0.0)
    return dict(start=start, t0=t0, t1=t1, minutes=minutes, sub_app=sub_app)


def _on_pitch_share(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """[n, k, 6]: fraction of each segment the player was on the pitch."""
    lo = np.maximum(t0[:, :, None], SEG_START[None, None, :])
    hi = np.minimum(t1[:, :, None], SEG_END[None, None, :])
    return np.clip(hi - lo, 0, None) / (SEG_END - SEG_START)[None, None, :]


# ---------------------------------------------------------------------------
# The segment engine: team shots, on-target, goals, with the score feeding back
# ---------------------------------------------------------------------------

def _score_sides(rng, mix_a: dict, mix_b: dict, n: int) -> dict:
    dom = rng.normal(0.0, DOMINANCE_SD, n)
    wob_a = rng.normal(0.0, TEAM_WOBBLE_SD, n)
    wob_b = rng.normal(0.0, TEAM_WOBBLE_SD, n)
    adj = SHOT_MEAN_CORR * np.exp(-0.5 * (DOMINANCE_SD ** 2 + TEAM_WOBBLE_SD ** 2))
    rate_a = mix_a["shots"] * adj * np.exp(dom + wob_a)
    rate_b = mix_b["shots"] * adj * np.exp(-dom + wob_b)

    out = {}
    for key, mix, rate in (("a", mix_a, rate_a), ("b", mix_b, rate_b)):
        out[key] = dict(shots=np.zeros((n, 6), int), sot=np.zeros((n, 6), int),
                        goals=np.zeros((n, 6), int), pens=np.zeros((n, 6), int),
                        pen_goals=np.zeros((n, 6), int), og=np.zeros((n, 6), int),
                        rate=rate, mix=mix)
    exp_margin = mix_a["goals"] - mix_b["goals"]
    pts = {"a": np.zeros(n, int), "b": np.zeros(n, int)}
    cum = 0.0
    for k in range(6):
        cum += SEG_SHARE[k]
        exp_lead_a = exp_margin * cum
        for key, other in (("a", "b"), ("b", "a")):
            s = out[key]
            mix = s["mix"]
            lead = pts[key] - pts[other] - (exp_lead_a if key == "a" else -exp_lead_a)
            lead = np.clip(lead, -LEAD_CLIP, LEAD_CLIP)
            mu = s["rate"] * SEG_SHARE[k] * np.exp(LEAD_BETA_VOL * lead)
            shots = rng.poisson(mu)
            p_sot = np.clip(mix["sot_rate"] * np.exp(LEAD_BETA_SOT * lead), 0.05, 0.9)
            p_goal = np.clip(mix["p_goal"] * GOAL_MEAN_CORR * np.exp(LEAD_BETA_GOAL * lead), 0.005, 0.6)
            sot = rng.binomial(shots, p_sot)
            goals = rng.binomial(sot, np.clip(p_goal / p_sot, 0.01, 0.95))
            pens = rng.poisson(mix["pens"] * SEG_SHARE[k], n)
            pen_goals = rng.binomial(pens, T.PEN_CONVERSION)
            og = rng.poisson(mix["og"] * SEG_SHARE[k], n)
            s["shots"][:, k] = shots; s["sot"][:, k] = sot; s["goals"][:, k] = goals
            s["pens"][:, k] = pens; s["pen_goals"][:, k] = pen_goals; s["og"][:, k] = og
            pts[key] = pts[key] + goals + pen_goals + og
    for key in ("a", "b"):
        s = out[key]
        s["np_shots_total"] = s["shots"].sum(axis=1)
        s["np_sot_total"] = s["sot"].sum(axis=1)
        s["np_goals_total"] = s["goals"].sum(axis=1)
        s["pens_total"] = s["pens"].sum(axis=1)
        s["pen_goals_total"] = s["pen_goals"].sum(axis=1)
        s["og_total"] = s["og"].sum(axis=1)
        s["goals_by_seg"] = s["goals"] + s["pen_goals"] + s["og"]
        s["points"] = pts[key]
        s["shots_total"] = s["np_shots_total"] + s["pens_total"]
        s["sot_total"] = s["np_sot_total"] + rng.binomial(s["pens_total"], PEN_ON_TARGET)
    return out


# ---------------------------------------------------------------------------
# Allocation to players
# ---------------------------------------------------------------------------

def _multinomial_rows(rng, counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Split `counts[i]` across columns with row-wise probabilities."""
    w = np.clip(weights, 0, None)
    tot = w.sum(axis=1, keepdims=True)
    p = np.where(tot > 0, w / np.clip(tot, 1e-12, None), 1.0 / w.shape[1])
    return rng.multinomial(counts, p)


def _capped_alloc(rng, counts: np.ndarray, weights: np.ndarray, caps: np.ndarray,
                  kp_weights: np.ndarray | None = None, p_assist: float = 0.0):
    """Hand out `counts` items one at a time by weight, never giving a player
    more than his cap. Optionally, for each item also draw an assister from
    the other players by `kp_weights` with probability `p_assist`. Returns
    (alloc, assists)."""
    n, k = weights.shape
    alloc = np.zeros((n, k), int)
    assists = np.zeros((n, k), int)
    if counts.max() <= 0:
        return alloc, assists
    for slot in range(int(counts.max())):
        live = counts > slot
        if not live.any():
            break
        w = np.where(alloc < caps, weights, 0.0)
        tot = w.sum(axis=1)
        ok = live & (tot > 0)
        if not ok.any():
            break
        cum = np.cumsum(w, axis=1) / np.clip(tot[:, None], 1e-12, None)
        u = rng.random(n)
        idx = (cum < u[:, None]).sum(axis=1).clip(0, k - 1)
        rows = np.where(ok)[0]
        alloc[rows, idx[rows]] += 1
        if kp_weights is not None and p_assist > 0:
            a_rows = rows[rng.random(len(rows)) < p_assist]
            if len(a_rows):
                kw = kp_weights[a_rows].copy()
                kw[np.arange(len(a_rows)), idx[a_rows]] = 0.0
                kt = kw.sum(axis=1)
                good = kt > 0
                if good.any():
                    kcum = np.cumsum(kw[good], axis=1) / kt[good][:, None]
                    ku = rng.random(good.sum())
                    kidx = (kcum < ku[:, None]).sum(axis=1).clip(0, k - 1)
                    assists[a_rows[good], kidx] += 1
    return alloc, assists


def _side_box(rng, roster: pd.DataFrame, side: dict, opp_side: dict, lineup: dict, n: int) -> dict:
    k = len(roster)
    minutes = lineup["minutes"]
    on = minutes / 90.0
    # shot / key-pass weights: minutes × share
    w_shot = on * roster["shot_share"].values[None, :]
    w_kp = on * roster["kp_share"].values[None, :]
    shots = _multinomial_rows(rng, side["np_shots_total"], w_shot)
    # on target among the shots, goals among the on-target shots
    sot, _ = _capped_alloc(rng, side["np_sot_total"], shots * roster["sot_rate"].values[None, :], shots)
    conv = roster["xgps"].values * roster["finishing"].values
    goals, assists = _capped_alloc(rng, side["np_goals_total"], sot * conv[None, :], sot,
                                   kp_weights=w_kp, p_assist=ASSIST_RATE)
    # penalties: the taker on the pitch (FPL order 1; else most penalties taken;
    # else the best conversion among the starters)
    pens = np.zeros((n, k), int); pen_goals = np.zeros((n, k), int)
    if side["pens_total"].max() > 0:
        taker_w = _taker_weights(roster) * (minutes > 0)
        pen_alloc = _multinomial_rows(rng, side["pens_total"], taker_w)
        pens = pen_alloc
        pen_goals = np.minimum(_multinomial_rows(rng, side["pen_goals_total"], pen_alloc), pen_alloc)
    # key passes: assists plus the rest of the side's key passes
    n_kp = rng.binomial(side["np_shots_total"], KP_PER_SHOT)
    n_kp = np.maximum(n_kp, assists.sum(axis=1))
    extra = _multinomial_rows(rng, n_kp - assists.sum(axis=1), w_kp)
    key_passes = assists + extra
    yellow = rng.poisson(roster["y90"].values[None, :] * on)
    # goalkeeper / defence: goals conceded while on the pitch, saves, clean sheet
    seg_share = _on_pitch_share(lineup["t0"], lineup["t1"])
    conceded_on = np.einsum("nks,ns->nk", seg_share, opp_side["goals_by_seg"].astype(float))
    conceded_on = np.round(conceded_on).astype(int)
    played = minutes > 0
    clean_sheet = played & (minutes >= 60) & (conceded_on == 0)
    gk = (roster["group"].values == "GK")[None, :] & lineup["start"]
    opp_sot_goals = opp_side["np_goals_total"] + opp_side["pen_goals_total"]
    saves_team = np.clip(opp_side["sot_total"] - opp_sot_goals, 0, None)
    saves = np.where(gk, saves_team[:, None], 0)
    defcon = np.zeros((n, k))
    if "defcon90" in roster.columns:
        dc = np.nan_to_num(roster["defcon90"].values.astype(float), nan=0.0)
        defcon = rng.poisson(np.clip(dc[None, :] * on * 1.0, 0, None))
    return dict(
        roster=roster, minutes=minutes, started=lineup["start"], played=played,
        shots=shots + pens, np_shots=shots, sot=sot + pens, goals=goals + pen_goals, np_goals=goals,
        pens=pens, pen_goals=pen_goals, assists=assists, key_passes=key_passes,
        yellow=yellow, conceded=conceded_on, clean_sheet=clean_sheet, saves=saves, defcon=defcon,
        team_shots=side["shots_total"], team_sot=side["sot_total"], team_goals=side["points"],
        team_np_goals=side["np_goals_total"], team_og=side["og_total"], team_pens=side["pens_total"],
    )


def _taker_weights(roster: pd.DataFrame) -> np.ndarray:
    w = np.zeros(len(roster))
    po = pd.to_numeric(roster.get("pen_order"), errors="coerce") if "pen_order" in roster.columns else None
    if po is not None and po.notna().any():
        w = np.where(po.values == 1, 1.0, np.where(po.values == 2, 0.15, np.where(po.values == 3, 0.05, 0.0)))
    if w.sum() <= 0:
        pt = roster["pens_taken"].values.astype(float)
        w = pt if pt.sum() > 0 else roster["xgps"].values * roster["p_start"].values
    return np.clip(w, 0, None) + 1e-9


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

def simulate_match(ctx: dict, roster_a: pd.DataFrame, roster_b: pd.DataFrame,
                   team_a: str, team_b: str, home: str | None = "a",
                   n_sims: int = 20000, seed: int | None = None,
                   avail_a: dict | None = None, avail_b: dict | None = None) -> dict:
    """Simulate the fixture `n_sims` times; both box scores per simulation.
    `avail_*` are `roster.availability_factor` dicts (attack multipliers for
    the squad actually available); None = the ratings' typical squad."""
    rng = np.random.default_rng(seed)
    n = int(n_sims)
    r = ctx["ratings"]
    ha = True if home in ("a", team_a) else False if home in ("b", team_b) else None
    mix_a = T.matchup(r, team_a, team_b, ha, avail_a)
    mix_b = T.matchup(r, team_b, team_a, None if ha is None else not ha, avail_b)
    mix_a, mix_b = T.apply_context(mix_a, mix_b, r)
    sides = _score_sides(rng, mix_a, mix_b, n)
    lu_a = draw_lineups(rng, roster_a, n)
    lu_b = draw_lineups(rng, roster_b, n)
    box_a = _side_box(rng, roster_a, sides["a"], sides["b"], lu_a, n)
    box_b = _side_box(rng, roster_b, sides["b"], sides["a"], lu_b, n)
    return dict(team_a=team_a, team_b=team_b, home=home, n_sims=n,
                goals_a=sides["a"]["points"], goals_b=sides["b"]["points"],
                shots_a=sides["a"]["shots_total"], shots_b=sides["b"]["shots_total"],
                sot_a=sides["a"]["sot_total"], sot_b=sides["b"]["sot_total"],
                box_a=box_a, box_b=box_b, mix_a=mix_a, mix_b=mix_b,
                avail_a=avail_a, avail_b=avail_b)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(sim: dict) -> dict:
    a, b = sim["goals_a"], sim["goals_b"]
    tot = a + b
    m = a - b
    out = dict(
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        win_a=float((m > 0).mean()), draw=float((m == 0).mean()), win_b=float((m < 0).mean()),
        mean_total=float(tot.mean()), total_sd=float(tot.std()), margin_sd=float(m.std()),
        over15=float((tot > 1.5).mean()), over25=float((tot > 2.5).mean()), over35=float((tot > 3.5).mean()),
        btts=float(((a > 0) & (b > 0)).mean()),
        cs_a=float((b == 0).mean()), cs_b=float((a == 0).mean()),
        shots_a=float(sim["shots_a"].mean()), shots_b=float(sim["shots_b"].mean()),
        sot_a=float(sim["sot_a"].mean()), sot_b=float(sim["sot_b"].mean()),
        shots_sd_a=float(sim["shots_a"].std()),
    )
    # most likely scorelines
    sc = pd.Series(list(zip(a, b))).value_counts(normalize=True).head(8)
    out["scorelines"] = [(f"{i}-{j}", float(p)) for (i, j), p in sc.items()]
    return out


def box_score(side: dict) -> pd.DataFrame:
    r = side["roster"].reset_index(drop=True)
    played = side["played"]
    n_played = np.clip(played.sum(axis=0), 1, None)
    out = pd.DataFrame({
        "Player": r["name"], "Pos": r["group"], "Start%": side["started"].mean(axis=0),
        "Min": side["minutes"].mean(axis=0),
        "Shots": side["shots"].mean(axis=0), "SoT": side["sot"].mean(axis=0),
        "Goals": side["goals"].mean(axis=0), "Anytime": (side["goals"] >= 1).mean(axis=0),
        "KP": side["key_passes"].mean(axis=0), "Assists": side["assists"].mean(axis=0),
        "Saves": side["saves"].mean(axis=0), "CS": side["clean_sheet"].mean(axis=0),
        "YC": side["yellow"].mean(axis=0),
        "Status": r["status_text"].where(r["status"] != "a", ""),
    })
    out["G+A"] = out["Goals"] + out["Assists"]
    return out.sort_values(["Min", "Shots"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def prepare(seasons, recency: D.Recency = D.RECENCY_DEFAULT, use_fpl: bool = True) -> dict:
    return P.prepare(tuple(sorted(seasons)), recency, use_fpl)


def roster_for(ctx: dict, team: str, use_fpl: bool = True) -> pd.DataFrame:
    return P.roster_for(ctx, team, use_fpl)


if __name__ == "__main__":
    ctx = prepare((2024, 2025, 2026))
    team_a, team_b = "MCI", "ARS"
    ra, rb = roster_for(ctx, team_a), roster_for(ctx, team_b)
    sim = simulate_match(ctx, ra, rb, team_a, team_b, home="a", n_sims=20000, seed=3)
    s = summarize(sim)
    e = T.expected_goals(ctx["ratings"], team_a, team_b, home="a")
    print(f"{team_a} (home) {s['mean_a']:.2f} - {s['mean_b']:.2f} {team_b} | analytic {e['goals_a']:.2f} - {e['goals_b']:.2f}")
    print(f"1X2 {s['win_a']:.1%} / {s['draw']:.1%} / {s['win_b']:.1%} (analytic {e['win_a']:.1%} / {e['draw']:.1%} / {e['win_b']:.1%}) | "
          f"O2.5 {s['over25']:.1%} | BTTS {s['btts']:.1%} | shots {s['shots_a']:.1f} - {s['shots_b']:.1f} (sd {s['shots_sd_a']:.1f}) | "
          f"SoT {s['sot_a']:.1f} - {s['sot_b']:.1f}")
    print("scorelines:", ", ".join(f"{k} {p:.1%}" for k, p in s["scorelines"][:6]))
    for team, side in ((team_a, sim["box_a"]), (team_b, sim["box_b"])):
        bs = box_score(side)
        print(f"\n{team}: team goals {side['team_goals'].mean():.2f} = player goals {side['goals'].sum(axis=1).mean():.2f} + OG {side['team_og'].mean():.2f}; "
              f"shots {side['team_shots'].mean():.2f} = {side['shots'].sum(axis=1).mean():.2f}; "
              f"identities: goals<=sot {(side['goals'] <= side['sot']).all()}, sot<=shots {(side['sot'] <= side['shots']).all()}, "
              f"assists<=kp {(side['assists'] <= side['key_passes']).all()}, assists<=goals {(side['assists'].sum(axis=1) <= side['np_goals'].sum(axis=1)).all()}")
        print(bs[bs["Min"] > 10][["Player", "Pos", "Start%", "Min", "Shots", "SoT", "Goals", "Anytime", "KP", "Assists", "Saves", "CS", "YC"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
