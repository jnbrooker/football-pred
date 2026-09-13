"""
footsim/roster.py — squads → who plays, for how long, and what each player is
worth per minute (roadmap §3, the depth-chart layer of this suite).

Football has no depth chart. What it has is the FPL squad list — every player
registered with each club, with a live availability status (injured,
suspended, doubtful with a percentage, not in squad) — and each player's
history of starts, substitute appearances and minutes. This module turns
those into, per player:

  * `p_start`   — probability he starts, from his recency-weighted start rate
                  in his club's matches (a match he did not appear in counts
                  as a non-start), blended with a prior for his position,
                  gated by the FPL status, and then fitted so each team's
                  expected starters sum to eleven (exactly one goalkeeper),
                  taking the adjustment from the least-evidenced players first
  * minutes     — when starting: subbed off with `p_suboff` around minute
                  `suboff_min`; as a sub: `p_subapp` to appear, for
                  `sub_minutes` on average
  * shares      — `shot_share` and `kp_share`: his share of the team's shots /
                  key passes PER 90 MINUTES ON THE PITCH, so a match's volume
                  is team shots × minutes/90 × share. Measured in his own
                  team's context and regressed toward his position group's
                  share; a player who changed clubs keeps his share and gets
                  his new club's volume.
  * quality     — xG per shot (relative to league, regressed), on-target rate,
                  finishing (regressed almost to league — it is noise), xA per
                  key pass, cards per 90, and the FPL defensive-contribution
                  rate.

Everything is a recency-weighted estimate with pseudo-count regression, the
strengths set from the year-to-year correlations the spike measured
(shots/90 r = 0.92, key passes/90 0.87, xG/shot 0.64, on-target 0.41,
goals/xG 0.11, assists/key pass 0.00 — ROADMAP §6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

# ---------------------------------------------------------------------------
# Position groups (Understat's detailed positions → six roles) and their priors
# ---------------------------------------------------------------------------

POS_GROUP = {"GK": "GK", "DC": "CB", "DR": "FB", "DL": "FB", "DMR": "FB", "DML": "FB",
             "DMC": "CM", "MC": "CM", "MR": "W", "ML": "W", "AMC": "AM", "AMR": "W",
             "AML": "W", "FWR": "W", "FWL": "W", "FW": "ST"}
FPL_TO_GROUP = {"GK": "GK", "DEF": "CB", "MID": "CM", "FWD": "ST"}
GROUP_ORDER = {"GK": 0, "CB": 1, "FB": 2, "CM": 3, "AM": 4, "W": 5, "ST": 6}

# Measured 2023–26 EPL (spike): share of team shots / key passes per 90 on the
# pitch, xG per shot, on-target rate, xA per key pass, yellows per 90, and
# the substitution pattern, by group.
GROUP_PRIOR = {
    #        shot_share kp_share  xgps    sot     xa_kp   y90     p_suboff suboff_min
    "GK": dict(shot=0.0002, kp=0.003, xgps=0.06, sot=0.10, xa_kp=0.12, y90=0.07, suboff=0.02, smin=60.0),
    "CB": dict(shot=0.041, kp=0.022, xgps=0.117, sot=0.27, xa_kp=0.115, y90=0.18, suboff=0.13, smin=70.5),
    "FB": dict(shot=0.051, kp=0.069, xgps=0.088, sot=0.28, xa_kp=0.116, y90=0.20, suboff=0.32, smin=74.4),
    "CM": dict(shot=0.086, kp=0.081, xgps=0.088, sot=0.29, xa_kp=0.103, y90=0.26, suboff=0.45, smin=74.8),
    "AM": dict(shot=0.162, kp=0.127, xgps=0.117, sot=0.35, xa_kp=0.130, y90=0.19, suboff=0.60, smin=75.8),
    "W": dict(shot=0.157, kp=0.114, xgps=0.122, sot=0.35, xa_kp=0.135, y90=0.15, suboff=0.62, smin=74.8),
    "ST": dict(shot=0.199, kp=0.070, xgps=0.187, sot=0.41, xa_kp=0.132, y90=0.16, suboff=0.60, smin=76.0),
}
LG_SUBOFF_SD = 14.0          # minute of substitution, sd
LG_SUB_MINUTES = 15.3        # minutes for a substitute
LG_SUB_APP = 0.31            # P(appears as a sub | did not start)
LG_ASSIST_PER_KP = 0.106     # league assists per key pass
LG_FINISHING = 0.88          # goals per xG (league level; refined by teams layer)

# Regression pseudo-counts.
START_PRIOR_N = 3.0          # matches-worth on the start rate (toward the
                             # player's own long-run rate, else his group's)
LONG_RUN_MIN_MATCHES = 5     # matches before the long-run rate is the prior
SUB_PRIOR_N = 6.0            # matches-worth on sub-appearance / sub-off rates
SHARE_PRIOR_MIN = 600.0      # minutes-worth on shot share (r = 0.92; 300–900 within noise on the harness)
KP_PRIOR_MIN = 600.0         # minutes-worth on key-pass share (r = 0.87)
XGPS_PRIOR_SHOTS = 30.0      # shots-worth on xG per shot (r = 0.64)
SOT_PRIOR_SHOTS = 90.0       # shots-worth on on-target rate (r = 0.41)
FIN_PRIOR_XG = 60.0          # xG-worth on finishing (r = 0.11)
XA_PRIOR_KP = 80.0           # key-passes-worth on xA per key pass
CARD_PRIOR_MIN = 900.0       # minutes-worth on yellows per 90
DEFCON_PRIOR_MIN = 450.0     # minutes-worth on defensive contributions per 90

START_PRIOR = {"GK": 0.33, "CB": 0.45, "FB": 0.40, "CM": 0.40, "AM": 0.40, "W": 0.38, "ST": 0.40}
NO_HISTORY_START = 0.25      # a squad player with no Understat history at all

ACTIVE_STATUSES = ("a", "d")            # FPL: available, doubtful
INJURY_TAG = {"i": "INJ", "s": "SUSP", "u": "UNAV", "n": "N/A", "d": "DOUBT"}

# How many players a page lists per team (the rest are deep reserves).
MAX_LISTED = 24


# ---------------------------------------------------------------------------
# Loading: everything the roster layer needs, once
# ---------------------------------------------------------------------------

def load_live(seasons: tuple[int, ...], recency: D.Recency = D.RECENCY_DEFAULT,
              use_fpl: bool = True) -> dict:
    """Rosters, shots, team-matches for the priors window; FPL squads and the
    crosswalk for the season being played. `use_fpl=False` builds squads from
    Understat appearances only (the backtest's history-only path)."""
    seasons = tuple(sorted(int(s) for s in seasons))
    rosters = D.load_rosters(seasons, recency=recency)
    shots = D.load_shots(seasons, recency=recency)
    tm = D.load_team_matches(seasons, recency=recency)
    squads = xw = None
    fpl_hist = pd.DataFrame()
    if use_fpl:
        try:
            squads = D.load_squads()
            xw = D.crosswalk(rosters, squads)
            fpl_hist = fpl_history(seasons)
        except Exception:
            squads = xw = None
    return dict(seasons=seasons, recency=D.Recency(*recency), rosters=rosters, shots=shots,
                tm=tm, squads=squads, xw=xw, fpl_hist=fpl_hist,
                groups=player_groups(rosters), team_vol=team_volumes(tm),
                shot_quality=player_shot_quality(shots))


def fpl_history(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Per-player per-GW FPL lines for the seasons (defensive contribution,
    saves, bonus), keyed on the FPL player CODE (stable across seasons)."""
    frames = []
    for s in seasons:
        try:
            d = D.load_fpl_gws(int(s))
        except Exception:
            continue
        if d is None or d.empty:
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def player_groups(rosters: pd.DataFrame) -> pd.Series:
    """Understat player_id → position group (modal starting position)."""
    st = rosters[rosters["started"] == 1]
    if st.empty:
        return pd.Series(dtype=object)
    g = st.assign(grp=st["position"].map(POS_GROUP)).dropna(subset=["grp"])
    # weight recent matches: the modal group over the weighted appearances
    agg = g.groupby(["player_id", "grp"])["w"].sum().reset_index()
    return agg.sort_values("w", ascending=False).drop_duplicates("player_id").set_index("player_id")["grp"]


def team_volumes(tm: pd.DataFrame) -> dict:
    """Per team: recency-weighted shots, on-target, key passes-ish per match."""
    out = {}
    for t, g in tm.groupby("team"):
        out[t] = dict(shots=D.wmean(g["shots"], g["w"]), sot=D.wmean(g["sot"], g["w"]),
                      xg=D.wmean(g["xg"], g["w"]), goals=D.wmean(g["goals"], g["w"]))
    out["_LEAGUE_"] = dict(shots=D.wmean(tm["shots"], tm["w"]), sot=D.wmean(tm["sot"], tm["w"]),
                           xg=D.wmean(tm["xg"], tm["w"]), goals=D.wmean(tm["goals"], tm["w"]))
    return out


def player_shot_quality(shots: pd.DataFrame) -> pd.DataFrame:
    """Per player: weighted non-penalty shots, xG, on-target, goals (for the
    quality / on-target / finishing priors) and penalties taken."""
    if shots.empty:
        return pd.DataFrame(columns=["player_id"]).set_index("player_id")
    s = shots.assign(ws=shots["w"], wx=shots["xg"] * shots["w"], wo=shots["on_target"] * shots["w"],
                     wg=shots["goal"] * shots["w"])
    np_ = s[s["penalty"] == 0].groupby("player_id").agg(n_shots=("ws", "sum"), xg=("wx", "sum"),
                                                          sot=("wo", "sum"), goals=("wg", "sum"),
                                                          raw_shots=("shot_id", "size"))
    pens = s[s["penalty"] == 1].groupby("player_id").agg(pens=("ws", "sum"), pen_goals=("wg", "sum"))
    return np_.join(pens, how="left").fillna(0.0)


# ---------------------------------------------------------------------------
# Squad construction
# ---------------------------------------------------------------------------

def _team_matches(tm: pd.DataFrame, team: str) -> pd.DataFrame:
    return tm[tm["team"] == team][["match_id", "date", "season", "w", "shots", "sot"]].sort_values("date")


def _stint_span(h: pd.DataFrame, team_matches: pd.DataFrame) -> pd.DataFrame:
    """The team's matches from the player's first to last appearance with it."""
    if h.empty or team_matches.empty:
        return team_matches.iloc[0:0]
    lo, hi = h["date"].min(), h["date"].max()
    return team_matches[(team_matches["date"] >= lo) & (team_matches["date"] <= hi)]


def _rate(num: float, den: float, prior: float, prior_n: float) -> float:
    return (num + prior * prior_n) / (den + prior_n) if den + prior_n > 0 else prior


def _player_row(pid, name, team, grp, fpl_pos, h: pd.DataFrame, tmatches: pd.DataFrame,
                q: pd.Series | None, lg: dict, status: str = "a", chance: float = np.nan,
                fpl_row: pd.Series | None = None, fpl_h: pd.DataFrame | None = None) -> dict:
    """One player's roster row from his history `h` (all clubs, weighted)."""
    P = GROUP_PRIOR[grp]
    games = int(len(h))
    w_games = float(h["w"].sum()) if games else 0.0

    # --- starts and minutes: measured against the CURRENT club's matches ----
    own = h[h["team"] == team]
    span = _stint_span(own, tmatches)
    if len(span):
        w_span = float(span["w"].sum())
        starts_w = float((own["started"] * own["w"]).sum())
        p_start_raw = starts_w / w_span if w_span > 0 else 0.0
        # the prior is the player's own long-run (unweighted) start rate over
        # the stint when there is enough of it — recent form vs role — else
        # the group's typical squad member
        long_run = (float(own["started"].sum()) / len(span)
                    if len(span) >= LONG_RUN_MIN_MATCHES else START_PRIOR[grp])
        p_start = _rate(starts_w, w_span, long_run, START_PRIOR_N)
        nonstart_w = max(w_span - starts_w, 0.0)
        subapp_w = float(((1 - own["started"]) * own["w"]).sum())
        p_subapp = _rate(subapp_w, nonstart_w, LG_SUB_APP, SUB_PRIOR_N)
        evidence = w_span / (w_span + START_PRIOR_N)
        moved = False
    elif games:
        # history with another club: his start rate there, discounted
        other_matches = h.drop_duplicates("match_id")
        w_h = float(other_matches["w"].sum())
        starts_w = float((h["started"] * h["w"]).sum())
        # appearances only (we do not know the other club's fixture list here),
        # so this overstates; halve the evidence
        p_start_raw = starts_w / w_h if w_h > 0 else 0.0
        p_start = _rate(starts_w, w_h, START_PRIOR[grp], START_PRIOR_N * 2)
        p_subapp = LG_SUB_APP
        evidence = 0.5 * w_h / (w_h + START_PRIOR_N)
        moved = True
    else:
        p_start_raw, p_start, p_subapp, evidence, moved = 0.0, NO_HISTORY_START, LG_SUB_APP, 0.0, False
        if grp == "GK":
            p_start = 0.05
    # FPL's own season-to-date starts and minutes: the one role signal for a
    # player Understat has never seen (a new signing from abroad, a youth
    # debutant), and a cross-check for everyone else once a few rounds are in
    if fpl_row is not None and np.isfinite(fpl_row.get("starts", np.nan)):
        n_team = float(fpl_row.get("team_played", 0.0) or 0.0)
        if n_team >= 2:
            fpl_rate = float(fpl_row["starts"]) / n_team
            if games == 0:
                p_start = (fpl_rate * n_team + p_start * 1.0) / (n_team + 1.0)
                evidence = n_team / (n_team + START_PRIOR_N)
                if float(fpl_row.get("minutes", 0)) == 0 and n_team >= 3:
                    p_subapp = 0.15

    st = h[h["started"] == 1]
    so_w = float((st["sub_off"] * st["w"]).sum()) if len(st) else 0.0
    st_w = float(st["w"].sum()) if len(st) else 0.0
    p_suboff = _rate(so_w, st_w, P["suboff"], SUB_PRIOR_N)
    off = st[st["sub_off"] == 1]
    suboff_min = D.wmean(off["minutes"], off["w"]) if len(off) >= 3 else P["smin"]
    suboff_min = (suboff_min * len(off) + P["smin"] * 3) / (len(off) + 3)
    subs = h[(h["started"] == 0) & (h["minutes"] > 0)]
    sub_minutes = D.wmean(subs["minutes"], subs["w"]) if len(subs) >= 3 else LG_SUB_MINUTES
    sub_minutes = (sub_minutes * len(subs) + LG_SUB_MINUTES * 3) / (len(subs) + 3)

    # --- shares: shots and key passes per 90 on the pitch, as a share of the
    #     team's shots in those matches -------------------------------------
    mins_w = float((h["minutes"] * h["w"]).sum())
    expo = float((h["minutes"] / 90.0 * h["team_shots"] * h["w"]).sum())    # team shots-worth
    shots_w = float((h["shots"] * h["w"]).sum())
    kp_w = float((h["key_passes"] * h["w"]).sum())
    lg_shots = lg["shots"]
    shot_share = _rate(shots_w, expo, P["shot"], SHARE_PRIOR_MIN / 90.0 * lg_shots)
    kp_share = _rate(kp_w, expo, P["kp"], KP_PRIOR_MIN / 90.0 * lg_shots)
    raw_share = shots_w / expo if expo > 0 else np.nan
    raw_kp = kp_w / expo if expo > 0 else np.nan

    # --- quality: xG per shot, on-target, finishing, xA per key pass --------
    if q is not None and q.get("n_shots", 0) > 0:
        n = float(q["n_shots"])
        xgps = _rate(float(q["xg"]), n, P["xgps"], XGPS_PRIOR_SHOTS)
        sot = _rate(float(q["sot"]), n, P["sot"], SOT_PRIOR_SHOTS)
        fin = _rate(float(q["goals"]), float(q["xg"]), lg["fin"], FIN_PRIOR_XG)
        pens_taken = float(q.get("pens", 0.0))
    else:
        xgps, sot, fin, pens_taken = P["xgps"], P["sot"], lg["fin"], 0.0
    xa_w = float((h["xa"] * h["w"]).sum())
    xa_kp = _rate(xa_w, kp_w, P["xa_kp"], XA_PRIOR_KP)
    y_w = float((h["yellow"] * h["w"]).sum())
    y90 = _rate(y_w, mins_w / 90.0, P["y90"], CARD_PRIOR_MIN / 90.0)

    # --- FPL-only rates: defensive contributions, saves ---------------------
    defcon90, saves90, bonus_pg = lg["defcon"].get(fpl_pos, 4.0), np.nan, np.nan
    if fpl_h is not None and len(fpl_h):
        fm = float(fpl_h["minutes"].sum())
        if fm > 0:
            # defensive contributions exist from 2025-26 only; rate them on
            # the matches that recorded them
            if "defensive_contribution" in fpl_h:
                dcr = fpl_h[fpl_h["defensive_contribution"].notna()]
                dm = float(dcr["minutes"].sum())
                defcon90 = _rate(float(dcr["defensive_contribution"].sum()), dm / 90.0,
                                 lg["defcon"].get(fpl_pos, 4.0), DEFCON_PRIOR_MIN / 90.0)
            if fpl_pos == "GK":
                saves90 = float(fpl_h["saves"].sum()) / fm * 90.0
            bonus_pg = float(fpl_h["bonus"].sum()) / max(len(fpl_h), 1)

    active = status in ACTIVE_STATUSES
    gate = 1.0
    if status == "d" and np.isfinite(chance):
        gate = float(np.clip(chance / 100.0, 0.0, 1.0))
    elif not active:
        gate = 0.0

    return dict(
        player_id=pid, name=name, team=team, group=grp, position=fpl_pos,
        status=status, status_text=D.FPL_STATUS.get(status, ""), chance=chance,
        news=(str(fpl_row["news"]) if fpl_row is not None else ""),
        fpl_id=(int(fpl_row["fpl_id"]) if fpl_row is not None else np.nan),
        price=(float(fpl_row["price"]) if fpl_row is not None else np.nan),
        pen_order=(fpl_row["pen_order"] if fpl_row is not None else None),
        games=games, games_w=w_games, evidence=float(evidence), moved=moved,
        active=active, gate=gate,
        p_start_raw=float(p_start_raw), p_start=float(np.clip(p_start * gate, 0.0, 1.0)),
        p_subapp=float(np.clip(p_subapp * gate, 0.0, 1.0)),
        p_suboff=float(np.clip(p_suboff, 0.0, 0.95)), suboff_min=float(suboff_min),
        sub_minutes=float(np.clip(sub_minutes, 3.0, 45.0)),
        shot_share=float(shot_share), kp_share=float(kp_share),
        raw_shot_share=float(raw_share) if np.isfinite(raw_share) else np.nan,
        raw_kp_share=float(raw_kp) if np.isfinite(raw_kp) else np.nan,
        xgps=float(xgps), sot_rate=float(np.clip(sot, 0.05, 0.7)), finishing=float(fin),
        xa_kp=float(xa_kp), y90=float(y90), pens_taken=float(pens_taken),
        defcon90=defcon90, saves90=saves90, bonus_pg=bonus_pg,
        minutes_w=mins_w,
    )


def _fit_group(vals: np.ndarray, evidence: np.ndarray, want: float, cap) -> np.ndarray:
    """Scale a group's probabilities to sum to `want`, taking the adjustment
    from the least-evidenced first (as the NFL roster layer does), without
    pushing anyone past his `cap` (the availability gate: 0 for a player
    ruled out, his chance for a doubt, 1 otherwise)."""
    vals = vals.copy()
    cap = np.asarray(cap, float)
    for _ in range(8):
        tot = float(vals.sum())
        gap = want - tot
        if abs(gap) < 1e-6:
            break
        give = (vals if gap < 0 else (cap - vals)) * (1.0 - np.clip(evidence, 0.0, 0.95))
        if give.sum() <= 1e-9:
            give = vals if gap < 0 else (cap - vals)
        if give.sum() <= 1e-9:
            break
        vals = np.clip(vals + give * (gap / give.sum()), 0.0, cap)
    return vals


def build_roster(live: dict, team: str, use_fpl: bool = True, lg: dict | None = None,
                 ruled_out=None) -> pd.DataFrame:
    """One team's roster: every squad player (FPL) or every player who
    appeared for the club in the priors window (history-only path), with the
    start / minutes / share / quality columns described in the module doc,
    fitted so expected starters sum to eleven."""
    rosters, tm = live["rosters"], live["tm"]
    lg = lg or league_rates(live)
    tmatches = _team_matches(tm, team)
    hist = rosters.merge(tm[["match_id", "team", "shots"]].rename(columns={"shots": "team_shots"}),
                         on=["match_id", "team"], how="left")
    hist["team_shots"] = hist["team_shots"].fillna(lg["shots"])
    groups = live["groups"]
    q = live["shot_quality"]
    fpl_hist = live.get("fpl_hist", pd.DataFrame())

    rows = []
    if use_fpl and live.get("squads") is not None:
        sq = live["squads"][live["squads"]["team"] == team]
        xw = live["xw"]
        us_by_fpl = xw.dropna(subset=["fpl_id"]).drop_duplicates("fpl_id").set_index("fpl_id")["player_id"]
        for _, s in sq.iterrows():
            pid = us_by_fpl.get(s["fpl_id"], np.nan)
            h = hist[hist["player_id"] == pid] if np.isfinite(pid) else hist.iloc[0:0]
            grp = groups.get(pid, FPL_TO_GROUP[s["position"]]) if np.isfinite(pid) else FPL_TO_GROUP[s["position"]]
            if grp == "GK" and s["position"] != "GK":
                grp = FPL_TO_GROUP[s["position"]]
            fh = fpl_hist[fpl_hist["fpl_code"] == s["fpl_code"]] if len(fpl_hist) and "fpl_code" in fpl_hist else None
            row = _player_row(int(pid) if np.isfinite(pid) else -int(s["fpl_id"]), s["web_name"], team, grp,
                              s["position"], h, tmatches, q.loc[pid] if np.isfinite(pid) and pid in q.index else None,
                              lg, status=s["status"], chance=s["chance"], fpl_row=s, fpl_h=fh)
            row["full_name"] = s["name"]
            rows.append(row)
    else:
        # history-only: everyone who appeared for the club in its last 8 matches
        recent = tmatches.tail(8)["match_id"]
        pids = hist[(hist["team"] == team) & hist["match_id"].isin(recent)]["player_id"].unique()
        for pid in pids:
            h = hist[hist["player_id"] == pid]
            grp = groups.get(pid, "CM")
            fpl_pos = {"GK": "GK", "CB": "DEF", "FB": "DEF", "CM": "MID", "AM": "MID", "W": "MID", "ST": "FWD"}[grp]
            row = _player_row(int(pid), str(h["player"].iloc[-1]), team, grp, fpl_pos, h, tmatches,
                              q.loc[pid] if pid in q.index else None, lg)
            row["full_name"] = row["name"]
            rows.append(row)
    if not rows:
        raise ValueError(f"No squad for {team}.")
    r = pd.DataFrame(rows)
    if ruled_out:
        out = r["player_id"].isin({int(x) for x in ruled_out})
        r.loc[out, ["gate", "p_start", "p_subapp"]] = 0.0
        r.loc[out, "active"] = False
        r.loc[out, "status"] = "i"
    r = fit_eleven(r, team_group_starters(live, team))
    r["exp_minutes"] = expected_minutes(r)
    tv = live["team_vol"].get(team, live["team_vol"]["_LEAGUE_"])["shots"]
    r["exp_shots"] = r["exp_minutes"] / 90.0 * r["shot_share"] * tv
    r["exp_kp"] = r["exp_minutes"] / 90.0 * r["kp_share"] * tv
    r["pos_order"] = r["group"].map(GROUP_ORDER).fillna(9)
    r = r.sort_values(["pos_order", "p_start", "exp_minutes"], ascending=[True, False, False]).drop(columns="pos_order")
    r["label"] = [player_label(x) for _, x in r.iterrows()]
    return r.reset_index(drop=True)


LEAGUE_GROUP_STARTERS = {"CB": 2.6, "FB": 2.0, "CM": 2.4, "AM": 0.9, "W": 1.4, "ST": 0.9}
GROUP_PRIOR_MATCHES = 6.0    # matches-worth of league prior on the team's formation


def team_group_starters(live: dict, team: str) -> dict:
    """The team's typical number of starters per position group (recency-
    weighted, shrunk toward league) — its formation, so a missing centre-back
    is replaced by a centre-back rather than by a fifth attacker."""
    ros = live["rosters"]
    st = ros[(ros["team"] == team) & (ros["started"] == 1)]
    if st.empty:
        return dict(LEAGUE_GROUP_STARTERS)
    grp = st["player_id"].map(live["groups"]).fillna("CM")
    per_match = st.assign(grp=grp).groupby(["match_id", "grp"])["w"].agg(["size", "first"]).reset_index()
    tot_w = float(st.drop_duplicates("match_id")["w"].sum())
    k = tot_w / (tot_w + GROUP_PRIOR_MATCHES)
    out = {}
    for g, prior in LEAGUE_GROUP_STARTERS.items():
        pm = per_match[per_match["grp"] == g]
        own = float((pm["size"] * pm["first"]).sum() / tot_w) if tot_w > 0 else prior
        out[g] = k * own + (1 - k) * prior
    z = sum(out.values())
    return {g: v * 10.0 / z for g, v in out.items()}


def fit_eleven(r: pd.DataFrame, group_targets: dict | None = None) -> pd.DataFrame:
    """Fit start probabilities so the goalkeepers sum to one and each outfield
    group to the team's typical count for it (then the outfield to ten);
    also keep sub-appearance probabilities consistent (about four used subs
    per match)."""
    r = r.copy()
    gk = (r["group"] == "GK").values
    targets = group_targets or LEAGUE_GROUP_STARTERS
    if gk.any():
        r.loc[gk, "p_start"] = _fit_group(r.loc[gk, "p_start"].values.astype(float),
                                          r.loc[gk, "evidence"].values.astype(float), 1.0,
                                          r.loc[gk, "gate"].values.astype(float))
    present = {g: (r["group"] == g).values for g in targets if ((r["group"] == g) & (r["gate"] > 0)).any()}
    z = sum(targets[g] for g in present) or 1.0
    for g, mask in present.items():
        want = targets[g] * 10.0 / z
        r.loc[mask, "p_start"] = _fit_group(r.loc[mask, "p_start"].values.astype(float),
                                            r.loc[mask, "evidence"].values.astype(float), want,
                                            r.loc[mask, "gate"].values.astype(float))
    if (~gk).any():
        r.loc[~gk, "p_start"] = _fit_group(r.loc[~gk, "p_start"].values.astype(float),
                                           r.loc[~gk, "evidence"].values.astype(float), 10.0,
                                           r.loc[~gk, "gate"].values.astype(float))
    # subs: P(sub app) × P(not start) should sum to ~4.1 for the outfield
    p_sub = (r["p_subapp"] * (1 - r["p_start"])).values
    tot = float(p_sub[~gk].sum())
    if tot > 0:
        scale = float(np.clip(4.1 / tot, 0.5, 2.0))
        r["p_subapp"] = np.where(~gk, np.clip(r["p_subapp"] * scale, 0, 1), r["p_subapp"] * 0.1)
    return r


def expected_minutes(r: pd.DataFrame) -> pd.Series:
    """E[minutes] = P(start) × (90 − P(subbed) × (90 − sub-off minute)) +
    P(not start) × P(sub app) × sub minutes."""
    start_min = 90.0 - r["p_suboff"] * (90.0 - r["suboff_min"])
    return r["p_start"] * start_min + (1 - r["p_start"]) * r["p_subapp"] * r["sub_minutes"]


def league_rates(live: dict) -> dict:
    tm = live["tm"]
    shots = live["shots"]
    np_ = shots[shots["penalty"] == 0]
    fin = float((np_["goal"] * np_["w"]).sum() / max((np_["xg"] * np_["w"]).sum(), 1e-6)) if len(np_) else LG_FINISHING
    defcon = {"GK": 1.0, "DEF": 9.0, "MID": 7.0, "FWD": 3.0}
    fh = live.get("fpl_hist", pd.DataFrame())
    if len(fh) and "defensive_contribution" in fh.columns:
        g = fh.groupby("position").agg(dc=("defensive_contribution", "sum"), m=("minutes", "sum"))
        for pos, row in g.iterrows():
            if row["m"] > 0:
                defcon[pos] = float(row["dc"] / row["m"] * 90.0)
    return dict(shots=D.wmean(tm["shots"], tm["w"]), fin=fin, defcon=defcon)


def player_label(row) -> str:
    """'Haaland — MCI ST · 92% start · DOUBT (75%)'"""
    tag = INJURY_TAG.get(str(row.get("status", "a")), "")
    s = f"{row['name']} — {row['team']} {row['group']} · {row['p_start']:.0%} start"
    if tag == "DOUBT" and np.isfinite(row.get("chance", np.nan)):
        s += f" · {tag} ({row['chance']:.0f}%)"
    elif tag:
        s += f" · {tag}"
    if row.get("moved"):
        s += " (new club)"
    if not row.get("games", 0):
        s += " · no history"
    return s


def league_rosters(live: dict, use_fpl: bool = True) -> pd.DataFrame:
    """Every team's roster in one table."""
    lg = league_rates(live)
    teams = (sorted(live["squads"]["team"].dropna().unique()) if use_fpl and live.get("squads") is not None
             else sorted(live["tm"]["team"].unique()))
    frames = []
    for t in teams:
        try:
            frames.append(build_roster(live, t, use_fpl, lg))
        except ValueError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    live = load_live((2024, 2025, 2026))
    xw = live["xw"]
    print(f"crosswalk: {xw['fpl_id'].notna().mean():.1%} of {len(xw)} Understat players matched "
          f"({xw['how'].value_counts().to_dict()})")
    recent = live["rosters"][live["rosters"]["season"] == max(live["seasons"])]
    unmatched = xw[xw["fpl_id"].isna() & xw["player_id"].isin(recent["player_id"])]
    print(f"unmatched with minutes this season: {len(unmatched)}")
    print(unmatched.sort_values("team").head(20)[["player", "team"]].to_string(index=False))
    for team in ("MCI", "ARS"):
        r = build_roster(live, team)
        print(f"\n{team}: {len(r)} listed, expected starters {r['p_start'].sum():.2f}, "
              f"GK {r.loc[r['group'] == 'GK', 'p_start'].sum():.2f}, expected minutes {r['exp_minutes'].sum():.0f}")
        cols = ["name", "group", "status", "p_start", "exp_minutes", "shot_share", "kp_share", "xgps", "sot_rate", "exp_shots", "games"]
        print(r[cols].head(18).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ---------------------------------------------------------------------------
# Availability: what the squad on hand produces relative to the typical squad
# ---------------------------------------------------------------------------
# The team ratings describe a unit averaged over the window; the squad layer
# knows who is out. The factor is a physical quantity, not a regression
# coefficient: expected non-penalty xG per match of the available squad
# (expected minutes × shot share × xG per shot, summed) over the same sum for
# the ungated squad. Split into a volume part and a quality part so the
# engine can move shots and xG per shot separately. AVAIL_COEF scales the
# log-factor. Fitted on the 2024–26 backtest with HINDSIGHT lineups
# (`backtest.availability_fit`): the best coefficient is 0 in both seasons
# (1X2 log-loss 1.0271 → 1.0326 at 1.0; Poisson likelihood flat-to-worse), so
# the pages show the factor but do not price it unless asked. It is kept at
# 1.0 here so the toggle shows the factor's full effect.

AVAIL_COEF = 1.0
AVAIL_CLIP = (0.7, 1.3)


def _production(r: pd.DataFrame) -> tuple[float, float]:
    on = r["exp_minutes"].values / 90.0
    vol = float((on * r["shot_share"].values).sum())
    xg = float((on * r["shot_share"].values * r["xgps"].values).sum())
    return vol, xg


def availability_factor(live: dict, team: str, lg: dict | None = None,
                        roster: pd.DataFrame | None = None, ruled_out=None,
                        ungated: pd.DataFrame | None = None) -> dict:
    """Volume and quality multipliers for `team`'s attack given who is
    available, relative to the ungated squad. `roster` is the gated roster
    if already built. `ruled_out` (player ids) gates a history-only squad
    instead of the FPL statuses — the backtest's hindsight mode."""
    lg = lg or league_rates(live)
    use_fpl = live.get("squads") is not None and ruled_out is None
    if use_fpl:
        gated = roster if roster is not None else build_roster(live, team, True, lg)
        full_live = dict(live)
        sq = live["squads"].copy()
        sq.loc[sq["team"] == team, "status"] = "a"
        sq.loc[sq["team"] == team, "chance"] = np.nan
        full_live["squads"] = sq
        ungated = ungated if ungated is not None else build_roster(full_live, team, True, lg)
    else:
        ungated = ungated if ungated is not None else build_roster(live, team, False, lg)
        gated = build_roster(live, team, False, lg, ruled_out=ruled_out)
    v1, x1 = _production(gated); v0, x0 = _production(ungated)
    if v0 <= 0 or x0 <= 0 or v1 <= 0:
        return dict(volume=1.0, quality=1.0, xg=1.0, missing=[])
    vol = float(np.clip((v1 / v0) ** AVAIL_COEF, *AVAIL_CLIP))
    qual = float(np.clip(((x1 / v1) / (x0 / v0)) ** AVAIL_COEF, *AVAIL_CLIP))
    out_ids = set(ungated.loc[ungated["p_start"] > 0.25, "player_id"]) - set(gated.loc[gated["p_start"] > 0.25, "player_id"])
    missing = ungated[ungated["player_id"].isin(out_ids)].sort_values("p_start", ascending=False)["name"].tolist()
    return dict(volume=vol, quality=qual, xg=float(vol * qual), missing=missing,
                xg_available=float(x1), xg_typical=float(x0))
