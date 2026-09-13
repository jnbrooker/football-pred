"""
footsim/fantasy.py — FPL points from the match engine, per simulation.

The engine already produces every player's line in every simulation, so
fantasy points are a scoring rule applied to those arrays: minutes, goals by
position, assists, clean sheets (60+ minutes and no goal conceded while on
the pitch), saves, goals conceded, penalties saved / missed, cards, own
goals, defensive contributions and bonus. Done per simulation rather than on
the means, so every player has a floor and a ceiling and the projection is
decomposed into where the points come from.

Bonus is the one component the engine does not simulate directly: it goes
to the three highest BPS in the match. BPS is rebuilt here from the
simulated line with the published weights (goal 24/18/12, assist 9, clean
sheet 12, save 2, ...) plus an approximation for the unmodelled parts
(pass completion, tackles, recoveries — a position baseline, the defensive-
contribution rate and noise), then ranked within the match per simulation.
Treat bonus as the least certain component.

Not modelled: red cards (0.1 per match, priced as nothing), own goals per
player (a team-level rate in the engine, not attributed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import game as G

# FPL scoring, 2025-26 rules (defensive contribution included). Editable on
# the page; the keys are the breakdown components.
PRESETS = {
    "FPL 2025-26": dict(
        play=1, play60=2, goal_gk=6, goal_def=6, goal_mid=5, goal_fwd=4, assist=3,
        cs_gk=4, cs_def=4, cs_mid=1, cs_fwd=0, saves_per=3, pen_save=5, pen_miss=-2,
        gc_per2=-1, yellow=-1, red=-3, own_goal=-2, defcon=2,
        defcon_thresh_def=10, defcon_thresh_other=12, bonus=True),
    "FPL classic (no defcon)": dict(
        play=1, play60=2, goal_gk=6, goal_def=6, goal_mid=5, goal_fwd=4, assist=3,
        cs_gk=4, cs_def=4, cs_mid=1, cs_fwd=0, saves_per=3, pen_save=5, pen_miss=-2,
        gc_per2=-1, yellow=-1, red=-3, own_goal=-2, defcon=0,
        defcon_thresh_def=10, defcon_thresh_other=12, bonus=True),
}

COMPONENT_LABELS = {
    "mins": "Minutes", "goal": "Goals", "assist": "Assists", "cs": "Clean sheet",
    "saves": "Saves", "gc": "Goals conceded", "pen": "Penalties", "cards": "Cards",
    "defcon": "Def. contribution", "bonus": "Bonus",
}

# BPS weights (published) and the approximation for the rest.
BPS_GOAL = {"GK": 12, "DEF": 12, "MID": 18, "FWD": 24}
BPS_BASE = {"GK": 4.0, "DEF": 5.0, "MID": 5.0, "FWD": 2.0}   # passes / touches / etc, per 90
BPS_NOISE_SD = 5.0
PEN_SAVED_GIVEN_MISS = 0.7


def side_points(side: dict, rules: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """Per-simulation points [n, k] for every player on one side, the
    components, and the BPS array used for bonus (bonus itself is added by
    `match_points`, which sees both sides)."""
    r = side["roster"].reset_index(drop=True)
    pos = r["position"].values
    n, k = side["minutes"].shape
    minutes = side["minutes"]
    played = minutes > 0
    sixty = minutes >= 60
    is_gk = (pos == "GK")[None, :]; is_def = (pos == "DEF")[None, :]
    is_mid = (pos == "MID")[None, :]; is_fwd = (pos == "FWD")[None, :]
    goal_pts = np.select([is_gk, is_def, is_mid, is_fwd],
                         [rules["goal_gk"], rules["goal_def"], rules["goal_mid"], rules["goal_fwd"]], 0)
    cs_pts = np.select([is_gk, is_def, is_mid, is_fwd],
                       [rules["cs_gk"], rules["cs_def"], rules["cs_mid"], rules["cs_fwd"]], 0)
    conceded = side["conceded"]
    pen_missed = side["pens"] - side["pen_goals"]
    comp = {
        "mins": np.where(played, np.where(sixty, rules["play60"], rules["play"]), 0),
        "goal": goal_pts * side["goals"],
        "assist": rules["assist"] * side["assists"],
        "cs": cs_pts * side["clean_sheet"],
        "saves": np.where(is_gk, side["saves"] // max(rules["saves_per"], 1), 0),
        "gc": np.where(is_gk | is_def, rules["gc_per2"] * (conceded // 2), 0),
        "pen": rules["pen_miss"] * pen_missed,
        "cards": rules["yellow"] * side["yellow"],
        "defcon": np.zeros((n, k)),
        "bonus": np.zeros((n, k)),
    }
    if rules.get("defcon", 0):
        thresh = np.where(is_def, rules["defcon_thresh_def"], rules["defcon_thresh_other"])
        comp["defcon"] = np.where(played & (side["defcon"] >= thresh), rules["defcon"], 0)
    # penalty saves for the keeper: the opponent's missed penalties, some saved
    if "opp_pen_missed" in side:
        saved = side["opp_pen_missed"][:, None]
        comp["pen"] = comp["pen"] + np.where(is_gk & side["started"], rules["pen_save"] * saved, 0)
    # BPS for the bonus ranking
    bps = (np.where(played, np.where(sixty, 6, 3), 0)
           + np.select([is_gk, is_def, is_mid, is_fwd], [12, 12, 18, 24], 0) * side["goals"]
           + 9 * side["assists"] + 12 * side["clean_sheet"] * (is_gk | is_def)
           + 2 * side["saves"] - 4 * (conceded // 2) * (is_gk | is_def)
           - 3 * side["yellow"] - 6 * pen_missed
           + np.select([is_gk, is_def, is_mid, is_fwd], [BPS_BASE[p] for p in ("GK", "DEF", "MID", "FWD")], 0) * minutes / 90.0
           + np.nan_to_num(side["defcon"]) * 1.0)
    return sum(comp.values()), comp, bps


def match_points(sim: dict, rules: dict, rng=None) -> dict:
    """Points for both sides of one simulated match, bonus included."""
    rng = rng or np.random.default_rng(0)
    for side, other in ((sim["box_a"], sim["box_b"]), (sim["box_b"], sim["box_a"])):
        missed = other["team_pens"] - other["pen_goals"].sum(axis=1)
        side["opp_pen_missed"] = rng.binomial(np.clip(missed, 0, None), PEN_SAVED_GIVEN_MISS)
    pa, ca, ba = side_points(sim["box_a"], rules)
    pb, cb, bb = side_points(sim["box_b"], rules)
    if rules.get("bonus", True):
        ka = ba.shape[1]
        bps = np.concatenate([ba, bb], axis=1) + rng.normal(0, BPS_NOISE_SD, (ba.shape[0], ka + bb.shape[1]))
        played = np.concatenate([sim["box_a"]["played"], sim["box_b"]["played"]], axis=1)
        bps = np.where(played, bps, -1e9)
        order = np.argsort(-bps, axis=1)
        bonus = np.zeros_like(bps)
        rows = np.arange(bps.shape[0])
        for rank, pts in enumerate((3, 2, 1)):
            bonus[rows, order[:, rank]] = pts
        ca["bonus"] = bonus[:, :ka]; cb["bonus"] = bonus[:, ka:]
        pa = pa + ca["bonus"]; pb = pb + cb["bonus"]
    return dict(points_a=pa, comp_a=ca, points_b=pb, comp_b=cb)


def game_projections(sim: dict, rules: dict, gw: int | None = None, fixture: str = "") -> tuple[pd.DataFrame, dict]:
    """Projection table for both teams of one simulated match (mean, floor
    p10, median, ceiling p90, P(≥ 6 / 10), components) plus per-player
    samples (float32) so a page can draw the full distribution."""
    pts = match_points(sim, rules)
    rows, samples = [], {}
    for team, opp, side, p, comp in ((sim["team_a"], sim["team_b"], sim["box_a"], pts["points_a"], pts["comp_a"]),
                                     (sim["team_b"], sim["team_a"], sim["box_b"], pts["points_b"], pts["comp_b"])):
        r = side["roster"].reset_index(drop=True)
        for j, pl in r.iterrows():
            x = p[:, j]
            row = dict(
                player_id=pl["player_id"], fpl_id=pl.get("fpl_id"), Player=pl["name"], Team=team, Opp=opp,
                Pos=pl["position"], Role=pl["group"], Price=pl.get("price", np.nan),
                Status=(pl["status_text"] if pl["status"] != "a" else ""),
                Start=float(side["started"][:, j].mean()), Min=float(side["minutes"][:, j].mean()),
                Proj=float(x.mean()), Floor=float(np.percentile(x, 10)), Median=float(np.median(x)),
                Ceiling=float(np.percentile(x, 90)), P6=float((x >= 6).mean()), P10=float((x >= 10).mean()),
                Goals=float(side["goals"][:, j].mean()), Assists=float(side["assists"][:, j].mean()),
                CS=float(side["clean_sheet"][:, j].mean()), Saves=float(side["saves"][:, j].mean()),
                Shots=float(side["shots"][:, j].mean()), SoT=float(side["sot"][:, j].mean()),
                gw=gw, fixture=fixture,
            )
            for key, arr in comp.items():
                row[f"pts_{key}"] = float(arr[:, j].mean())
            rows.append(row)
            samples[pl["player_id"]] = x.astype(np.float32)
    return pd.DataFrame(rows), samples


def gameweek_projections(ctx: dict, fixtures: pd.DataFrame, rules: dict, n_sims: int = 8000,
                         seed: int = 7, progress=None) -> tuple[pd.DataFrame, dict, dict]:
    """Every player on the gameweek's slate. `fixtures` needs home / away
    (team codes) and optionally `gw`. Returns (table, samples, sims)."""
    tables, samples, sims = [], {}, {}
    for i, (_, f) in enumerate(fixtures.iterrows()):
        try:
            ra = G.roster_for(ctx, f["home"]); rb = G.roster_for(ctx, f["away"])
        except ValueError:
            continue
        sim = G.simulate_match(ctx, ra, rb, f["home"], f["away"], home="a", n_sims=n_sims, seed=seed + i)
        label = f"{f['home']} v {f['away']}"
        t, s = game_projections(sim, rules, gw=f.get("gw"), fixture=label)
        tables.append(t); samples.update(s); sims[label] = sim
        if progress:
            progress(i + 1, len(fixtures))
    if not tables:
        return pd.DataFrame(), {}, {}
    return pd.concat(tables, ignore_index=True), samples, sims


if __name__ == "__main__":
    ctx = G.prepare((2024, 2025, 2026))
    ra, rb = G.roster_for(ctx, "MCI"), G.roster_for(ctx, "ARS")
    sim = G.simulate_match(ctx, ra, rb, "MCI", "ARS", home="a", n_sims=20000, seed=3)
    t, _ = game_projections(sim, PRESETS["FPL 2025-26"])
    cols = ["Player", "Team", "Pos", "Price", "Start", "Min", "Proj", "Floor", "Ceiling", "P6",
            "pts_goal", "pts_assist", "pts_cs", "pts_saves", "pts_defcon", "pts_bonus"]
    print(t.sort_values("Proj", ascending=False).head(20)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
