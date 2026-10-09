"""
footsim/season.py — the season layer: the table this season, and the league
five years out.

Everything above this file is about ONE match. This one runs the whole
competition: every fixture left this season (and, further out, whole seasons
that have not been scheduled yet), thousands of times, and counts how often
each club finishes where. It answers the two questions a supporter actually
asks — *where do we finish?* and *what are the odds we go down, or reach
Europe, in the next few years?*

Three layers, each fitted and each scored out of sample by `backtest()`:

1. **The rest of this season.** The points already on the board are facts;
   the remaining fixtures are simulated from the team ratings
   (`teams.expected_goals`, vectorised here so 380 fixtures x 20 000
   simulations is a second's work). Two corrections the single-match layer
   does not need:

   * **Drift** (`DRIFT_SD`). Today's rating is not the strength a club will
     carry to May: managers go, players get injured, form is partly real.
     Split-sample on 2014-2026 — rate a club on its first k matches, rate it
     again on the rest, subtract the sampling noise — leaves a residual of
     0.19-0.28 in log expected-goal difference that no rating fitted before
     the cut could have known (ROADMAP §6.20). So each simulation gives every
     club a strength offset for the rest of the season, drawn once and used
     for all of its remaining fixtures. Without it the middle 80% of the
     simulations holds the outcome only 67% of the time; with it, 82%.
   * **Draws** (`DRAW_TAU`). The analytic layer's independent Poissons give
     slightly too few draws — in `game.py` it is the game-state feedback that
     produces them, and re-running the full engine 380 times per simulation
     is not affordable — so the score comes from a score matrix with a fitted
     Dixon-Coles adjustment on the four low scores.

2. **Season to season.** A club's season strength decomposes, on twelve
   completed seasons of Understat (2014/15-2025/26, 240 club-seasons), into

       net(season) = club LEVEL + one-season FORM + measurement noise

   with the level a near-random walk (sd 0.362 across clubs, psi = 0.951, so
   it moves 0.112 a season) and form pure one-season noise (sd 0.181, psi
   fitted at 0). That is the whole finding: four fifths of what separates
   clubs, once the rating's own noise is out, is a slow-moving level — call
   it resources, stature, whatever a club's owner is buying — and the rest is
   a season that does not repeat. A naive AR(1) on the season rating (phi =
   0.69) badly underestimates how much of a good season is still there five
   years out: it implies a correlation of 0.16, and the data say 0.55. The
   level is estimated per club by a Kalman pass over its own seasons, and it
   is the level — not the rating — that **investment moves**: `spend` shifts
   it in units of the fitted level innovation, so one step is a standard
   year's worth of what clubs actually do to themselves.

3. **Promotion and relegation.** Three down; three up at the fitted promoted
   strength (net -0.452, sd 0.265 — the same finding as
   `teams.PROMOTED_PRIOR`, measured here on the season rating). Which clubs
   come up is drawn from the observed distance distribution: a third of
   promoted sides had missed exactly one season, an eighth two, and about
   half had not been in the league for four years or more. A club the user is
   tracking is therefore followed out of the league and back.

Caveats, stated once: no cup or European fixtures, so no congestion and no
cup-winner European places — the European places here are the league ones
only (top four / five / six, selectable). Ownership, takeovers, points
deductions and the financial rules are not modelled; the `spend` dial is the
handle for all of them. Five seasons out the honest width of these numbers is
wide, and the backtest is what says how wide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import teams as T

# ---------------------------------------------------------------------------
# Fitted constants (ROADMAP §6.20; refit with `python -m footsim.season fit`)
# ---------------------------------------------------------------------------

# Within-season drift: how far a club's strength over the REST of the season
# sits from a rating fitted on what it has played so far, in log expected-goal
# difference, beyond the sampling noise of the rating itself. Split-sample on
# 2014-2026 at cuts of 4 / 8 / 12 / 19 / 26 matches: 0.282 / 0.228 / 0.197 /
# 0.181 / 0.185 on net — a floor of ~0.18 plus an early-season term. The live
# ratings carry several seasons, so their early-season uncertainty is smaller
# than a one-season rating's; both constants are then checked on the
# out-of-sample coverage of final points (`backtest`).
DRIFT_SD = 0.18            # net (attack - defence), per club, rest of season
DRIFT_EARLY = 0.45         # extra, as DRIFT_EARLY / sqrt(matches played + 4)
DRIFT_STYLE_SD = 0.10      # attack + defence, drawn with it

# Draws. Independent Poissons give slightly too few; a Dixon-Coles tweak on
# the four low scores, fitted on results against the model's own expected
# goals.
DRAW_TAU = 0.0

# Season-to-season strength process, on the season net rating (ROADMAP §6.20).
LEVEL_SD = 0.362           # spread of the club level across clubs
LEVEL_PSI = 0.951          # the level's own mean reversion, per season
LEVEL_INNOV_SD = 0.112     # how far a club's level moves in a season
FORM_SD = 0.181            # one season's form; psi fitted at 0, so it does not carry
MEAS_SD = 0.110            # sampling noise in a 38-match season rating
STYLE_PSI = 0.389          # attack + defence balance, year to year
STYLE_SD = 0.156

# Promoted sides, on the season rating: the fitted class average and spread.
# PROMOTED_NET_SD is the spread of what promoted clubs are OBSERVED to do, so
# the spread of the level they arrive with is what is left of it once a
# season's form and the rating's noise are taken out.
PROMOTED_NET = -0.452      # recency-weighted (flat mean -0.413; classes have worsened)
PROMOTED_NET_SD = 0.265
PROMOTED_STYLE = -0.047    # attack + defence: promoted sides are below on both

# The divisions below. A club can only be promoted into the Premier League
# from the division immediately below it, so the projection has to know which
# division each club is actually in — a club relegated twice is not a
# promotion candidate, however recently it was in the top flight.
# `data.load_division_teams` reads that from football-data.co.uk; where the
# feed is unavailable the tier can be set by hand, and anything still unknown
# is assumed to be in the second tier, which is the old behaviour.
#
# Movement below the top flight is NOT modelled from strength — this suite has
# no rating for those divisions (ROADMAP §7.9). It is the mechanical rate: of
# 24 clubs, three go up, and three (Championship), four (League One) or two
# (League Two, out of the League) go down, drawn independently per club, so
# the rate is right and which club it is, is not. A club in League One
# therefore reaches the Premier League at about the rate a random League One
# club does — which for a fallen giant is too low, and is exactly what a
# second-tier rating would fix.
TIER_MAX = 4
TIER_UP = 3 / 24
TIER_DOWN = {2: 3 / 24, 3: 4 / 24, 4: 2 / 24}

# Where a promoted club comes from: seasons since it was last in the league.
# Of 33 promoted sides 2015/16-2026/27, 11 had missed exactly one season, 4 two,
# 2 three, and 16 had not been in the league for four years or more.
PROMOTED_FROM = {1: 0.333, 2: 0.121, 3: 0.061}   # the rest is the long-absent pool
LONG_ABSENT_POOL = 21      # second-tier clubs four or more seasons out of the top flight

# Attack / defence split of a strength shift. Strength moves mostly as shot
# VOLUME (the same finding as home advantage); the promoted prior's own split
# is 0.204 volume to 0.114 quality.
VOL_SHARE = 0.64

# De-attenuation of the season rating (`fit_spread`). Hand each past season
# its OWN ratings and play it out: the middle of the table comes back right
# (4th 68 against 68, 17th 38 against 38, the draw rate to within a point) but
# the ends are pulled in — the champion 86 where champions average 90 and the
# bottom club 26 where they average 21. A season rating is a shrunk estimate,
# so the spread it reports is not the spread clubs really have; this is the
# multiplier that makes a simulated table the shape of a real one. It applies
# only to seasons built from the season-to-season process, never to the live
# ratings, whose own spread is already scored match by match (ROADMAP §6.12).
SPREAD_MULT = 1.20

# Scoring and the shape of the league.
# League points per unit of net rating, on 240 club-seasons: points =
# 52.5 + 38.9 x net. What the spend dial is quoted in.
POINTS_PER_NET = 38.9

WIN, DRAW = 3, 1
RELEGATED = 3
MAX_GOALS = 12


# ---------------------------------------------------------------------------
# The table as it stands
# ---------------------------------------------------------------------------

def season_fixtures(season: int | None = None, league: str = D.LEAGUE) -> pd.DataFrame:
    """Every fixture of a season, played or not."""
    if season is None:
        season = D.fpl_season()
    fx = D.load_fixtures((int(season),), league)
    if fx.empty:
        raise ValueError(f"No fixture list for {season}.")
    return fx


def table_from(fx: pd.DataFrame, teams: list | None = None) -> pd.DataFrame:
    """The league table from the played fixtures: P W D L GF GA GD Pts."""
    played = fx[fx["played"]]
    teams = sorted(set(fx["home"]) | set(fx["away"])) if teams is None else list(teams)
    rows = {t: dict(team=t, P=0, W=0, D=0, L=0, GF=0, GA=0) for t in teams}
    for _, m in played.iterrows():
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        for t, gf, ga in ((m["home"], hg, ag), (m["away"], ag, hg)):
            r = rows.setdefault(t, dict(team=t, P=0, W=0, D=0, L=0, GF=0, GA=0))
            r["P"] += 1
            r["GF"] += gf
            r["GA"] += ga
            r["W"] += int(gf > ga)
            r["D"] += int(gf == ga)
            r["L"] += int(gf < ga)
    t = pd.DataFrame(list(rows.values()))
    t["GD"] = t["GF"] - t["GA"]
    t["Pts"] = WIN * t["W"] + DRAW * t["D"]
    return (t.sort_values(["Pts", "GD", "GF"], ascending=False)
             .reset_index(drop=True).set_index("team"))


# ---------------------------------------------------------------------------
# The vectorised matchup: the analytic layer, many fixtures at a time
# ---------------------------------------------------------------------------

def team_terms(r: dict, teams: list) -> dict:
    """The per-team pieces of `teams.matchup`, as arrays in `teams` order.

    A club in the fixture list with no rating at all — a promoted side before
    it has played — falls back to the promoted prior, the same one the rating
    solver would give it (`teams.PROMOTED_PRIOR`)."""
    def ix(s, key=None):
        prior = T.PROMOTED_PRIOR.get(key, 0.0) if key else 0.0
        return np.array([float(s[t]) if t in s.index else prior for t in teams])
    return dict(
        teams=list(teams),
        vol_att=ix(r["vol_att"], "vol_att"), vol_def=ix(r["vol_def"], "vol_def"),
        q_att=ix(r["q_att"], "q_att"), q_def=ix(r["q_def"], "q_def"),
        fin=ix(r["fin"], "fin"), gk=ix(r["gk"], "gk"),
        pen_att=ix(r["pen_att"], "pen_att"), pen_def=ix(r["pen_def"], "pen_def"),
        lg_shots=r["lg_shots"], lg_xgps=r["lg_xgps"], lg_fin=r["lg_fin"],
        lg_pen=r["lg_pen"], lg_og=r["lg_og"],
        home_vol=r["home_vol"], home_q=r["home_q"],
    )


def _combine(npxg_h, npxg_a, fin_h, fin_a, pen_h, pen_a, og, lg):
    """Both sides' expected goals from their expected non-penalty xG, with the
    conversion-context coupling of `teams.apply_context` and the same clips."""
    k_h = np.clip((npxg_a / max(lg, 1e-6)) ** T.FIN_CONTEXT_BETA, 0.7, 1.4)
    k_a = np.clip((npxg_h / max(lg, 1e-6)) ** T.FIN_CONTEXT_BETA, 0.7, 1.4)
    mu_h = np.clip(npxg_h * fin_h * k_h + pen_h * T.PEN_CONVERSION + og, *T.GOALS_CLIP)
    mu_a = np.clip(npxg_a * fin_a * k_a + pen_a * T.PEN_CONVERSION + og, *T.GOALS_CLIP)
    return mu_h, mu_a


def future_mu(lg: dict, att_h, dfn_h, att_a, dfn_a) -> tuple[np.ndarray, np.ndarray]:
    """Expected goals for a fixture between two clubs described only by their
    strength — attack and defence in log expected goals against an average
    side. This is how a season nobody has scheduled yet is played: finishing,
    penalties and own goals are at league level, because those are the pieces
    the season-to-season layer has no opinion about."""
    shots_h = np.clip(lg["shots"] * np.exp(VOL_SHARE * (att_h + dfn_a) + lg["home_vol"]), *T.SHOTS_CLIP)
    shots_a = np.clip(lg["shots"] * np.exp(VOL_SHARE * (att_a + dfn_h) - lg["home_vol"]), *T.SHOTS_CLIP)
    xgps_h = np.clip(lg["xgps"] * np.exp((1 - VOL_SHARE) * (att_h + dfn_a) + lg["home_q"]), *T.XGPS_CLIP)
    xgps_a = np.clip(lg["xgps"] * np.exp((1 - VOL_SHARE) * (att_a + dfn_h) - lg["home_q"]), *T.XGPS_CLIP)
    return _combine(shots_h * xgps_h, shots_a * xgps_a, lg["fin"], lg["fin"],
                    lg["pen"], lg["pen"], lg["og"], lg["shots"] * lg["xgps"])


def league_level(r: dict) -> dict:
    """The league's own rates, the base every future season is built on."""
    return dict(shots=r["lg_shots"], xgps=r["lg_xgps"], fin=r["lg_fin"],
                pen=r["lg_pen"], og=r["lg_og"], home_vol=r["home_vol"], home_q=r["home_q"])


def fixture_mu(tt: dict, hi: np.ndarray, ai: np.ndarray,
               d_att: np.ndarray | None = None, d_def: np.ndarray | None = None,
               neutral: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Expected goals for both sides of a set of fixtures, exactly as
    `teams.expected_goals` computes them (clips, penalties, own goals and the
    conversion-context coupling included), but vectorised.

    `hi` / `ai` index into `tt["teams"]`; they broadcast against `d_att` /
    `d_def`, which are per-simulation strength offsets in log expected goals
    shaped (sims, teams) — so the result is (sims, fixtures)."""
    da = np.zeros_like(tt["vol_att"]) if d_att is None else d_att
    dd = np.zeros_like(tt["vol_def"]) if d_def is None else d_def
    h = 0.0 if neutral else 1.0

    def side(att_i, def_i, home_sign):
        # a strength shift lands mostly on shot volume (VOL_SHARE), the rest
        # on quality — the same split as the promoted prior
        dv = VOL_SHARE * np.take(da, att_i, axis=-1) + VOL_SHARE * np.take(dd, def_i, axis=-1)
        dq = (1 - VOL_SHARE) * np.take(da, att_i, axis=-1) + (1 - VOL_SHARE) * np.take(dd, def_i, axis=-1)
        shots = tt["lg_shots"] * np.exp(tt["vol_att"][att_i] + tt["vol_def"][def_i]
                                        + tt["home_vol"] * home_sign * h + dv)
        xgps = tt["lg_xgps"] * np.exp(tt["q_att"][att_i] + tt["q_def"][def_i]
                                      + tt["home_q"] * home_sign * h + dq)
        shots = np.clip(shots, *T.SHOTS_CLIP)
        xgps = np.clip(xgps, *T.XGPS_CLIP)
        fin = tt["lg_fin"] * np.exp(tt["fin"][att_i] + tt["gk"][def_i])
        pens = tt["lg_pen"] * np.exp(tt["pen_att"][att_i] + tt["pen_def"][def_i])
        return shots * xgps, fin, pens

    npxg_h, fin_h, pen_h = side(hi, ai, +1.0)
    npxg_a, fin_a, pen_a = side(ai, hi, -1.0)
    return _combine(npxg_h, npxg_a, fin_h, fin_a, pen_h, pen_a, tt["lg_og"],
                    tt["lg_shots"] * tt["lg_xgps"])


def draw_scores(rng, mu_h: np.ndarray, mu_a: np.ndarray, tau: float = None) -> tuple[np.ndarray, np.ndarray]:
    """Goals for both sides: independent Poissons, then the Dixon-Coles
    correction on the four low scores when `tau` is non-zero. 0-0 and 1-1 gain
    and 1-0 and 0-1 lose (tau < 0), which is where the analytic layer loses
    its draws; the mass of the low block itself is left alone, so the expected
    goals of the layer above are untouched."""
    tau = DRAW_TAU if tau is None else tau
    gh = rng.poisson(mu_h)
    ga = rng.poisson(mu_a)
    if abs(tau) < 1e-9:
        return gh, ga
    low = (gh <= 1) & (ga <= 1)
    if not low.any():
        return gh, ga
    mh = np.broadcast_to(mu_h, gh.shape)[low]
    ma = np.broadcast_to(mu_a, ga.shape)[low]
    # the independent-Poisson weights of the four cells, times Dixon-Coles tau
    w = np.stack([
        np.exp(-mh - ma) * (1.0 - mh * ma * tau),          # 0-0
        np.exp(-mh - ma) * ma * (1.0 + mh * tau),          # 0-1
        np.exp(-mh - ma) * mh * (1.0 + ma * tau),          # 1-0
        np.exp(-mh - ma) * mh * ma * (1.0 - tau),          # 1-1
    ], axis=-1)
    w = np.clip(w, 1e-12, None)
    c = np.cumsum(w / w.sum(axis=-1, keepdims=True), axis=-1)
    pick = (rng.random((len(mh), 1)) > c).sum(axis=-1)      # 0..3
    gh[low] = (pick >= 2).astype(gh.dtype)
    ga[low] = (pick % 2).astype(ga.dtype)
    return gh, ga


# ---------------------------------------------------------------------------
# Simulating the rest of a season
# ---------------------------------------------------------------------------

def _rank(points: np.ndarray, gd: np.ndarray, gf: np.ndarray) -> np.ndarray:
    """Finishing position (1 = champions) per simulation, on points, then goal
    difference, then goals scored, ties split at random (as a play-off would)."""
    n_sims, n_teams = points.shape
    key = (points.astype(np.float64) * 1e6 + (gd.astype(np.float64) + 200.0) * 1e3
           + gf.astype(np.float64))
    jitter = np.random.default_rng(0).random(key.shape) * 1e-3
    order = np.argsort(-(key + jitter), axis=1, kind="stable")
    pos = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    pos[rows, order] = np.arange(1, n_teams + 1)[None, :]
    return pos


def simulate_table(r: dict, fx: pd.DataFrame, n_sims: int = 10000, seed: int = 7,
                   drift: bool | float = True, shifts: dict | None = None,
                   chunk: int = 2500) -> dict:
    """Play out every unplayed fixture `n_sims` times.

    `drift` True uses the fitted within-season drift, a float overrides its
    size, False switches it off. `shifts` is a per-team hand adjustment to the
    net rating (log expected-goal difference), for asking what-if questions."""
    teams = sorted(set(fx["home"]) | set(fx["away"]))
    ti = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    start = table_from(fx, teams)
    base_pts = start["Pts"].reindex(teams).to_numpy(float)
    base_gd = start["GD"].reindex(teams).to_numpy(float)
    base_gf = start["GF"].reindex(teams).to_numpy(float)
    played_n = start["P"].reindex(teams).to_numpy(float)

    rest = fx[~fx["played"]]
    hi = rest["home"].map(ti).to_numpy()
    ai = rest["away"].map(ti).to_numpy()
    tt = team_terms(r, teams)
    hand = np.zeros(n_teams)
    if shifts:
        for t, v in shifts.items():
            if t in ti:
                hand[ti[t]] = float(v)

    sd_net = 0.0
    if drift is not False:
        floor = DRIFT_SD if drift is True else float(drift)
        sd_net = float(np.hypot(floor, DRIFT_EARLY / np.sqrt(played_n.mean() + 4.0)))
    sd_sty = DRIFT_STYLE_SD if drift is not False else 0.0

    rng = np.random.default_rng(seed)
    pts = np.empty((n_sims, n_teams), np.int32)
    gds = np.empty((n_sims, n_teams), np.int32)
    gfs = np.empty((n_sims, n_teams), np.int32)
    MH = _incidence(hi, n_teams) if len(hi) else None
    MA = _incidence(ai, n_teams) if len(ai) else None
    for lo in range(0, n_sims, chunk):
        m = min(chunk, n_sims - lo)
        d_net = rng.normal(0.0, sd_net, (m, n_teams)) + hand[None, :]
        d_sty = rng.normal(0.0, sd_sty, (m, n_teams))
        d_att = 0.5 * (d_net + d_sty)
        d_def = 0.5 * (d_sty - d_net)
        p = np.repeat(base_pts[None, :], m, axis=0)
        gd = np.repeat(base_gd[None, :], m, axis=0)
        gf = np.repeat(base_gf[None, :], m, axis=0)
        if len(hi):
            mu_h, mu_a = fixture_mu(tt, hi, ai, d_att, d_def)
            gh, ga = draw_scores(rng, mu_h, mu_a)
            _accumulate(p, gd, gf, MH, MA, gh, ga)
        pts[lo:lo + m] = p
        gds[lo:lo + m] = gd
        gfs[lo:lo + m] = gf
    pos = _rank(pts, gds, gfs)
    return dict(teams=teams, points=pts, gd=gds, gf=gfs, position=pos,
                start=start, n_remaining=int(len(rest)), drift_sd=sd_net)


def _incidence(idx: np.ndarray, n_teams: int) -> np.ndarray:
    """Fixture -> team indicator, so `values @ M` sums a (sims x fixtures)
    block into (sims x teams) in one matrix multiply (np.add.at is far slower)."""
    m = np.zeros((len(idx), n_teams))
    m[np.arange(len(idx)), idx] = 1.0
    return m


def _accumulate(p: np.ndarray, gd: np.ndarray, gf: np.ndarray, mh: np.ndarray, ma: np.ndarray,
                gh: np.ndarray, ga: np.ndarray) -> None:
    """Add a block of simulated fixtures (sims x fixtures) into the table.
    `mh` / `ma` are the home / away incidence matrices from `_incidence`."""
    gh = gh.astype(np.float64, copy=False)
    ga = ga.astype(np.float64, copy=False)
    home_pts = np.where(gh > ga, WIN, np.where(gh == ga, DRAW, 0)).astype(np.float64)
    away_pts = np.where(ga > gh, WIN, np.where(gh == ga, DRAW, 0)).astype(np.float64)
    p += home_pts @ mh + away_pts @ ma
    gd += (gh - ga) @ mh + (ga - gh) @ ma
    gf += gh @ mh + ga @ ma


def projected_table(sim: dict, europe: int = 5, ucl: int = 4) -> pd.DataFrame:
    """One row per club: expected points and the probability of each outcome."""
    pts, pos = sim["points"], sim["position"]
    teams = sim["teams"]
    n = len(pos)
    rows = []
    for i, t in enumerate(teams):
        pp = pts[:, i]
        po = pos[:, i]
        rows.append(dict(
            team=t, Pts=pp.mean(), P10=np.percentile(pp, 10), P90=np.percentile(pp, 90),
            Pos=po.mean(), Title=(po == 1).mean(), UCL=(po <= ucl).mean(),
            Europe=(po <= europe).mean(), Top10=(po <= 10).mean(),
            Bottom3=(po > len(teams) - RELEGATED).mean(),
            GD=sim["gd"][:, i].mean(),
        ))
    t = pd.DataFrame(rows).set_index("team").sort_values(["Pts", "GD"], ascending=False)
    t.insert(0, "Now", sim["start"]["Pts"].reindex(t.index))
    t.insert(1, "P", sim["start"]["P"].reindex(t.index))
    return t


def position_matrix(sim: dict) -> pd.DataFrame:
    """P(finish in position j) for every club — the heat map."""
    pos = sim["position"]
    n_teams = pos.shape[1]
    m = np.zeros((n_teams, n_teams))
    for j in range(n_teams):
        m[:, j] = (pos == j + 1).mean(axis=0)
    out = pd.DataFrame(m, index=sim["teams"], columns=range(1, n_teams + 1))
    return out.loc[out.mul(np.arange(1, n_teams + 1), axis=1).sum(axis=1).sort_values().index]


# ---------------------------------------------------------------------------
# Season ratings, and the club level behind them
# ---------------------------------------------------------------------------

def season_ratings(tm: pd.DataFrame, seasons=None, min_matches: int = 300) -> pd.DataFrame:
    """Each club's strength in each COMPLETED season, rated on that season's
    matches alone by the ordinary solver — net (attack minus defence, in log
    expected goals) and style (attack plus defence: where a club puts its
    quality). Flat weights and no promoted prior: this is the historical
    record the season-to-season process is fitted on, not a forecast."""
    rows = []
    for s in sorted(tm["season"].unique() if seasons is None else seasons):
        d = tm[tm["season"] == int(s)].copy()
        if len(d) < min_matches:
            continue
        d["w"] = 1.0
        r = T.team_ratings(d, season=int(s), promoted=set())
        att = r["vol_att"] + r["q_att"]
        dfn = r["vol_def"] + r["q_def"]
        for t in r["teams"]:
            rows.append(dict(season=int(s), team=t, att=float(att[t]), dfn=float(dfn[t]),
                             net=float(att[t] - dfn[t]), style=float(att[t] + dfn[t]),
                             matches=int((d["team"] == t).sum())))
    return pd.DataFrame(rows)


def live_net(r: dict) -> pd.DataFrame:
    """The live ratings on the same scale: net and style per club."""
    att = r["vol_att"] + r["q_att"]
    dfn = r["vol_def"] + r["q_def"]
    return pd.DataFrame(dict(net=att - dfn, style=att + dfn))


def club_levels(hist: pd.DataFrame, live: pd.DataFrame | None = None,
                played: float = 0.0) -> pd.DataFrame:
    """Every club's LEVEL — the slow-moving part of its strength — filtered
    from its season ratings.

    The state is the level; it moves by LEVEL_PSI and LEVEL_INNOV_SD a season,
    and each season's rating sees it through one season of form (FORM_SD) and
    the rating's own sampling noise (MEAS_SD). Seasons a club spent outside
    the league are propagated without an observation, which is why a club
    comes back with the level it left, widened. If `live` is given it is the
    current season's observation, its noise scaled by how much of the season
    has been played (`played` matches)."""
    seasons = sorted(hist["season"].unique())
    last = int(seasons[-1]) if seasons else None
    obs_var = FORM_SD ** 2 + MEAS_SD ** 2
    rows = []
    for team, g in hist.groupby("team"):
        y = g.set_index("season")["net"]
        m, v = 0.0, LEVEL_SD ** 2
        first = int(y.index.min())
        for s in range(first, (last if last is not None else first) + 1):
            if s > first:
                m *= LEVEL_PSI
                v = min(LEVEL_PSI ** 2 * v + LEVEL_INNOV_SD ** 2, LEVEL_SD ** 2)
            if s in y.index:
                k = v / (v + obs_var)
                m += k * (float(y[s]) - m)
                v *= (1 - k)
        rows.append(dict(team=team, level=m, level_sd=float(np.sqrt(v)),
                         style=float(g.sort_values("season")["style"].iloc[-1]),
                         seasons=int(len(y)), last_season=int(y.index.max())))
    out = pd.DataFrame(rows).set_index("team") if rows else \
        pd.DataFrame(columns=["level", "level_sd", "style", "seasons", "last_season"])
    if live is None or not len(live):
        return out
    share = min(max(float(played) / 38.0, 0.05), 1.0)
    var_now = FORM_SD ** 2 + MEAS_SD ** 2 / share
    for team, row in live.iterrows():
        if team in out.index:
            m, v = float(out.at[team, "level"]), float(out.at[team, "level_sd"]) ** 2
        else:
            m, v = 0.0, LEVEL_SD ** 2
        m *= LEVEL_PSI
        v = min(LEVEL_PSI ** 2 * v + LEVEL_INNOV_SD ** 2, LEVEL_SD ** 2)
        k = v / (v + var_now)
        out.loc[team, ["level", "level_sd", "style"]] = [m + k * (float(row["net"]) - m),
                                                         float(np.sqrt(v * (1 - k))),
                                                         float(row["style"])]
    out["seasons"] = out["seasons"].fillna(0).astype(int)
    out["last_season"] = out["last_season"].fillna(last if last is not None else 0).astype(int)
    return out


# ---------------------------------------------------------------------------
# Seasons nobody has scheduled yet
# ---------------------------------------------------------------------------

GENERIC = 40        # unnamed clubs, so the promotion pool is never empty
AWAY_MAX = 9        # seasons out of the league, capped


def club_tiers(season: int | None = None, current=(), overrides: dict | None = None) -> tuple[dict, bool]:
    """Which division each club is in this season, and whether the feed
    answered. Top flight from the fixture list, the rest from
    `data.load_division_teams`, then any hand corrections.

    The feed is the only thing in this suite that knows a club has been
    relegated twice, and it is a third-party CSV: when it cannot be reached
    the tiers fall back to "everyone outside the top flight is in the
    Championship", which is what the model assumed before it asked, and the
    caller should say so rather than quietly pretend."""
    season = int(D.fpl_season()) if season is None else int(season)
    tiers, live = {}, False
    try:
        dt = D.load_division_teams((season,))
    except Exception:
        dt = pd.DataFrame()
    if not dt.empty:
        live = True
        for _, row in dt.iterrows():
            tiers[row["team"]] = int(row["tier"])
    for t in current:
        tiers[t] = 1
    for t, v in (overrides or {}).items():
        tiers[t] = int(v)
    return tiers, live


def club_universe(hist: pd.DataFrame, current) -> list:
    """Every club that could appear: the ones with a top-flight history, plus
    unnamed clubs standing for the rest of the second tier."""
    known = sorted(set(hist["team"]) | set(current))
    return known + [f"club {i + 1}" for i in range(GENERIC)]


def promoted_level_sd() -> float:
    """The spread of the LEVEL a promoted club arrives with. What is observed
    of a promoted class (PROMOTED_NET_SD) is its level, its form and the
    rating's own noise together, and form is drawn again every season."""
    return float(np.sqrt(max(PROMOTED_NET_SD ** 2 - FORM_SD ** 2 - MEAS_SD ** 2, 0.02)))


def _round_robin(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Every ordered pair of slots — the fixture list of a season nobody has
    drawn yet. The order of the fixtures does not change a final table."""
    hi, ai = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keep = hi != ai
    return hi[keep], ai[keep]


def _scatter(values: np.ndarray, slots: np.ndarray, U: int, fill=0) -> np.ndarray:
    """A (sims x 20) quantity spread over the (sims x universe) board."""
    out = np.full((values.shape[0], U), fill, dtype=values.dtype)
    np.put_along_axis(out, slots, values, axis=1)
    return out


def _next_league(rng, slots, level, style, out_since, tier, pos_prev, n_teams, n_known):
    """Bottom three down, three up, every club's level a season older — and
    every club below the top flight a division up or down at the mechanical
    rate. The promotion places are filled from the clubs that were in the
    SECOND tier during the season just finished, before the divisions below
    move, so nobody climbs two divisions in a year."""
    n_sims, U = level.shape
    # who stays: everything but the bottom RELEGATED of the previous table
    stay = np.take_along_axis(slots, np.argsort(pos_prev, axis=1)[:, :n_teams - RELEGATED], axis=1)
    out_since = np.minimum(out_since + 1, AWAY_MAX)
    # levels age for every club in the universe, in the league or not
    level = LEVEL_PSI * level + rng.normal(0.0, LEVEL_INNOV_SD, (n_sims, U))
    style = STYLE_PSI * style + rng.normal(0.0, STYLE_SD, (n_sims, U))
    up = np.empty((n_sims, RELEGATED), np.int64)
    bucket = _bucket_draw(rng, n_sims * RELEGATED).reshape(n_sims, RELEGATED)
    for s in range(n_sims):
        up[s] = _promote(rng, out_since[s], stay[s], bucket[s], n_known, tier[s])
    slots = np.concatenate([stay, up], axis=1)
    rows = np.arange(n_sims)[:, None]
    out_since[rows, slots] = 0
    # the divisions below, after the promotion places are settled
    tier = _move_tiers(rng, tier, n_known)
    tier[rows, slots] = 1                       # in the top flight next season
    relegated = np.take_along_axis(slots, np.argsort(pos_prev, axis=1)[:, n_teams - RELEGATED:], axis=1)
    tier[rows, relegated] = 2
    # a promoted club arrives at the promoted class's strength, whoever it is
    level[rows, up] = PROMOTED_NET + rng.normal(0.0, promoted_level_sd(), up.shape)
    style[rows, up] = PROMOTED_STYLE
    return slots, level, style, out_since, tier


def _move_tiers(rng, tier: np.ndarray, n_known: int) -> np.ndarray:
    """Clubs below the top flight climb and fall one division at a time, at
    the rate the promotion and relegation places set (TIER_UP / TIER_DOWN).
    Only named clubs move: the unnamed ones stand for "the rest of the second
    tier" and stay there, so the promotion pool is always a division's worth."""
    out = tier.copy()
    named = np.arange(tier.shape[1]) < n_known
    low = (tier >= 2) & named[None, :]
    if not low.any():
        return out
    u = rng.random(tier.shape)
    down_rate = np.zeros_like(tier, float)
    for t, p in TIER_DOWN.items():
        down_rate[tier == t] = p
    moved = np.where(u < TIER_UP, -1, np.where(u > 1.0 - down_rate, 1, 0))
    out = np.where(low, np.clip(tier + moved, 2, TIER_MAX), tier)
    return out.astype(tier.dtype)


def _bucket_draw(rng, n: int) -> np.ndarray:
    """Which distance bucket each promotion slot comes from (PROMOTED_FROM)."""
    keys = sorted(PROMOTED_FROM)
    p = np.array([PROMOTED_FROM[k] for k in keys] + [max(1.0 - sum(PROMOTED_FROM.values()), 0.0)])
    return rng.choice(len(p), size=n, p=p / p.sum())


def _promote(rng, out_since: np.ndarray, staying: np.ndarray, buckets: np.ndarray,
             n_known: int, tier: np.ndarray) -> np.ndarray:
    """The three clubs coming up in one simulation. Only clubs in the SECOND
    tier are eligible; among them, how long a club has been away is the only
    thing the record says about which it will be.

    The long-absent bucket is a second tier of LONG_ABSENT_POOL clubs: the
    ones with an old top-flight record, topped up with unnamed clubs so the
    pool is always the size of a division rather than of the universe."""
    keys = sorted(PROMOTED_FROM)
    taken = set(staying.tolist())
    eligible = tier == 2

    def pool(sel):
        return [c for c in np.flatnonzero(sel & eligible).tolist() if c not in taken]

    far = out_since >= (max(keys) + 1)
    known_far = pool(far & (np.arange(len(out_since)) < n_known))
    generic = pool(far & (np.arange(len(out_since)) >= n_known))
    long_absent = known_far + generic[:max(LONG_ABSENT_POOL - len(known_far), 1)]
    picks = []
    for b in buckets:
        for cand in ([pool(out_since == keys[b])] if b < len(keys) else []) + \
                    [[c for c in long_absent if c not in taken], pool(out_since >= 1)]:
            if cand:
                pick = int(cand[rng.integers(len(cand))])
                picks.append(pick)
                taken.add(pick)
                break
    return np.array(picks, np.int64)


def _play_season(rng, lg: dict, slots, level, style, hi, ai, MH, MA, chunk: int = 1000):
    """One unscheduled season: every club's form drawn on top of its level,
    then the whole round robin."""
    n_sims, n_teams = slots.shape
    pts = np.empty((n_sims, n_teams), np.float64)
    gds = np.empty((n_sims, n_teams), np.float64)
    gfs = np.empty((n_sims, n_teams), np.float64)
    nets = np.empty((n_sims, n_teams), np.float64)
    for lo in range(0, n_sims, chunk):
        sl = slice(lo, min(lo + chunk, n_sims))
        s = slots[sl]
        net = np.take_along_axis(level[sl], s, axis=1) + rng.normal(0.0, FORM_SD, s.shape)
        sty = np.take_along_axis(style[sl], s, axis=1)
        wide = net * SPREAD_MULT        # the rating's spread, de-attenuated
        att, dfn = 0.5 * (wide + sty), 0.5 * (sty - wide)
        mu_h, mu_a = future_mu(lg, att[:, hi], dfn[:, hi], att[:, ai], dfn[:, ai])
        gh, ga = draw_scores(rng, mu_h, mu_a)
        p = np.zeros((s.shape[0], n_teams))
        gd = np.zeros_like(p)
        gf = np.zeros_like(p)
        _accumulate(p, gd, gf, MH, MA, gh, ga)
        pts[sl], gds[sl], gfs[sl], nets[sl] = p, gd, gf, net
    pos = _rank(pts, gds, gfs)
    return pos, pts, nets


def simulate_future(r: dict, fx: pd.DataFrame, hist: pd.DataFrame, years: int = 5,
                    n_sims: int = 3000, seed: int = 11, spend: dict | None = None,
                    levels: pd.DataFrame | None = None, tiers: dict | None = None,
                    progress=None) -> dict:
    """This season played out, then `years` more.

    Between seasons every club's level moves (LEVEL_PSI, LEVEL_INNOV_SD) and
    it picks up a fresh season of form (FORM_SD); the bottom three go down and
    three come up at the promoted strength. `spend` shifts a club's LEVEL, in
    level innovations per season — +1 is a club doing to itself, every summer,
    as much as a typical club does in a year. It is a standing shift, so it
    compounds over the horizon, which is what sustained investment (or
    sustained selling) looks like.

    `tiers` is which division each club is in right now (1 = Premier League,
    2 = Championship, …), from `club_tiers`. It matters: only a second-tier
    club can be promoted next May, so a club that has been relegated twice
    has to climb back a division at a time."""
    now = simulate_table(r, fx, n_sims=n_sims, seed=seed)
    teams_now = now["teams"]
    uni = club_universe(hist, teams_now)
    ui = {t: i for i, t in enumerate(uni)}
    U, n_teams = len(uni), len(teams_now)

    if levels is None:
        lv = live_net(r).reindex(teams_now).dropna()
        levels = club_levels(hist, lv, played=float(now["start"]["P"].mean()))
    lvl0 = np.array([float(levels["level"].get(t, PROMOTED_NET)) for t in uni])
    lvl_sd0 = np.array([float(levels["level_sd"].get(t, LEVEL_SD)) for t in uni])
    sty0 = np.array([float(levels["style"].get(t, 0.0)) for t in uni])
    push = np.zeros(U)
    if spend:
        for t, v in spend.items():
            if t in ui:
                push[ui[t]] = float(v) * LEVEL_INNOV_SD

    rng = np.random.default_rng(seed + 1)
    level = lvl0[None, :] + rng.normal(0.0, 1.0, (n_sims, U)) * lvl_sd0[None, :]
    style = np.repeat(sty0[None, :], n_sims, axis=0)
    slots = np.repeat(np.array([ui[t] for t in teams_now])[None, :], n_sims, axis=0)
    # how long each club has been out of the league, which is what says how
    # likely it is to come up: a club relegated last May is the likeliest of all
    season0 = int(fx["season"].iloc[0])
    away = np.full(U, AWAY_MAX, np.int16)
    if "last_season" in levels.columns:
        for t, last in levels["last_season"].items():
            if t in ui and last == last:
                # seasons missed before the one now being played
                away[ui[t]] = int(np.clip(season0 - int(last) - 1, 0, AWAY_MAX))
    out_since = np.repeat(away[None, :], n_sims, axis=0)
    np.put_along_axis(out_since, slots, np.zeros_like(slots, np.int16), axis=1)

    n_known = sum(1 for t in uni if not t.startswith("club "))
    # which division every club is in now: the top flight from the fixture
    # list, the rest from the feed / by hand, anything unknown assumed second
    t0 = np.full(U, 2, np.int8)
    for t, v in (tiers or {}).items():
        if t in ui:
            t0[ui[t]] = int(np.clip(int(v), 1, TIER_MAX))
    for t in teams_now:
        t0[ui[t]] = 1
    tier = np.repeat(t0[None, :], n_sims, axis=0)

    lg = league_level(r)
    hi, ai = _round_robin(n_teams)
    MH, MA = _incidence(hi, n_teams), _incidence(ai, n_teams)

    pos_all = [_scatter(now["position"].astype(np.int16), slots, U)]
    pts_all = [_scatter(now["points"].astype(np.float64), slots, U, fill=np.nan)]
    in_all = [_scatter(np.ones_like(slots, np.int8), slots, U)]
    net_all = [_scatter(np.take_along_axis(level, slots, axis=1), slots, U, fill=np.nan)]

    pos_prev = now["position"]
    for y in range(years):
        slots, level, style, out_since, tier = _next_league(
            rng, slots, level, style, out_since, tier, pos_prev, n_teams, n_known)
        level = level + push[None, :]        # investment, every season, compounding
        pos_prev, pts, net = _play_season(rng, lg, slots, level, style, hi, ai, MH, MA)
        pos_all.append(_scatter(pos_prev.astype(np.int16), slots, U))
        pts_all.append(_scatter(pts, slots, U, fill=np.nan))
        in_all.append(_scatter(np.ones_like(slots, np.int8), slots, U))
        net_all.append(_scatter(net, slots, U, fill=np.nan))
        if progress:
            progress(y + 1, years)
    return dict(universe=uni, index={t: i for i, t in enumerate(uni)}, now=now,
                position=pos_all, points=pts_all, in_league=in_all, net=net_all,
                years=years, n_sims=n_sims, levels=levels, teams_now=teams_now,
                season0=season0, tier0={t: int(t0[i]) for i, t in enumerate(uni)})


# ---------------------------------------------------------------------------
# Reading the future back
# ---------------------------------------------------------------------------

def club_outlook(fut: dict, team: str, europe: int = 5, ucl: int = 4) -> pd.DataFrame:
    """One row per season for one club: where it is, and the odds of each
    outcome. Probabilities are unconditional — out of the league counts as
    neither European nor relegated, and `In league` says how often it is
    there at all."""
    i = fut["index"][team]
    rows = []
    for y in range(fut["years"] + 1):
        pos = fut["position"][y][:, i]
        inl = fut["in_league"][y][:, i].astype(bool)
        pts = fut["points"][y][:, i]
        n = len(pos)
        rows.append(dict(
            Season=season_label(fut["season0"] + y),
            **{"In league": inl.mean()},
            Title=((pos == 1) & inl).mean(),
            UCL=((pos <= ucl) & inl).mean(),
            Europe=((pos <= europe) & inl).mean(),
            Relegated=((pos > len(fut["teams_now"]) - RELEGATED) & inl).mean(),
            Pts=np.nanmean(np.where(inl, pts, np.nan)) if inl.any() else np.nan,
            Pos=np.nanmean(np.where(inl, pos, np.nan)) if inl.any() else np.nan,
        ))
    t = pd.DataFrame(rows).set_index("Season")
    ever = {}
    for col in ("Title", "UCL", "Europe", "Relegated"):
        hit = np.zeros(fut["n_sims"], bool)
        for y in range(fut["years"] + 1):
            pos = fut["position"][y][:, i]
            inl = fut["in_league"][y][:, i].astype(bool)
            if col == "Title":
                hit |= (pos == 1) & inl
            elif col == "UCL":
                hit |= (pos <= ucl) & inl
            elif col == "Europe":
                hit |= (pos <= europe) & inl
            else:
                hit |= (pos > len(fut["teams_now"]) - RELEGATED) & inl
        ever[col] = hit.mean()
    t.attrs["ever"] = ever
    return t


def season_label(s: int) -> str:
    return f"{s}/{str(s + 1)[-2:]}"


def future_table(fut: dict, year: int, europe: int = 5, ucl: int = 4,
                 min_p: float = 0.005) -> pd.DataFrame:
    """Every club with a real chance of being in the league in that season."""
    pos = fut["position"][year]
    inl = fut["in_league"][year].astype(bool)
    pts = fut["points"][year]
    n_teams = len(fut["teams_now"])
    rows = []
    for i, t in enumerate(fut["universe"]):
        p_in = inl[:, i].mean()
        if p_in < min_p:
            continue
        rows.append(dict(team=t, **{"In league": p_in},
                         Pts=np.nanmean(np.where(inl[:, i], pts[:, i], np.nan)),
                         Title=((pos[:, i] == 1) & inl[:, i]).mean(),
                         UCL=((pos[:, i] <= ucl) & inl[:, i]).mean(),
                         Europe=((pos[:, i] <= europe) & inl[:, i]).mean(),
                         Relegated=((pos[:, i] > n_teams - RELEGATED) & inl[:, i]).mean(),
                         Pos=np.nanmean(np.where(inl[:, i], pos[:, i], np.nan))))
    return (pd.DataFrame(rows).set_index("team")
            .sort_values(["Europe", "Pts"], ascending=False))


def group_generic(t: pd.DataFrame) -> pd.DataFrame:
    """Fold the unnamed second-tier clubs into one row — individually they are
    interchangeable, and the model has no opinion about which is which."""
    mask = t.index.str.startswith("club ")
    if not mask.any():
        return t
    rest = t[~mask]
    gen = t[mask]
    row = pd.DataFrame([{**{c: np.nan for c in t.columns},
                         "In league": gen["In league"].sum()}], index=["(second tier)"])
    for c in ("Title", "UCL", "Europe", "Relegated"):
        if c in t.columns:
            row[c] = gen[c].sum()
    return pd.concat([rest, row])


# ---------------------------------------------------------------------------
# Fitting the constants (ROADMAP §6.20)
# ---------------------------------------------------------------------------

def fit_process(hist: pd.DataFrame, hmax: int = 7, verbose: bool = True) -> dict:
    """The season-to-season process, fitted to the autocovariance of the season
    net rating: a slow club LEVEL plus a one-season FORM, with the rating's own
    sampling noise held at its split-half estimate.

    Fitting the autocovariance rather than a one-step regression is the whole
    point: an AR(1) matches h = 1 by construction and then decays far too fast,
    because most of what separates clubs is permanent."""
    from scipy.optimize import minimize
    ix = hist.set_index(["season", "team"]).sort_index()
    v0 = float(hist["net"].var())
    cov, n = {}, {}
    for h in range(1, hmax + 1):
        x, y = [], []
        for (s, t) in ix.index:
            if (s + h, t) in ix.index:
                x.append(ix.at[(s, t), "net"]); y.append(ix.at[(s + h, t), "net"])
        if len(x) > 20:
            cov[h] = float(np.cov(x, y)[0, 1]); n[h] = len(x)
    var_m = MEAS_SD ** 2

    def unpack(p):
        return np.exp(p[0]), np.exp(p[1]), 1 / (1 + np.exp(-p[2])), 1 / (1 + np.exp(-p[3]))

    def loss(p):
        sL, su, psiu, psiL = unpack(p)
        e = 4 * (v0 - (sL ** 2 + su ** 2 + var_m)) ** 2
        for h, c in cov.items():
            e += (1.0 / h) * (c - (psiL ** h * sL ** 2 + psiu ** h * su ** 2)) ** 2
        return e * 1e4

    best = None
    for a in (-1.5, -1.0, -0.8):
        for b in (-2.0, -1.5):
            res = minimize(loss, [a, b, -1.0, 3.0], method="Nelder-Mead",
                           options=dict(maxiter=8000, xatol=1e-7, fatol=1e-12))
            if best is None or res.fun < best.fun:
                best = res
    sL, su, psiu, psiL = unpack(best.x)
    out = dict(level_sd=sL, level_psi=psiL, level_innov=sL * np.sqrt(max(1 - psiL ** 2, 0.0)),
               form_sd=su * np.sqrt(max(1 - psiu ** 2, 0.0)) if psiu > 0.01 else su,
               form_psi=psiu, var_net=v0, cov=cov, n=n)
    if verbose:
        print(f"  net: var {v0:.4f} | level sd {sL:.3f} psi {psiL:.3f} (innovation {out['level_innov']:.3f}/season)"
              f" | form sd {su:.3f} psi {psiu:.3f}")
        print("    autocovariance  observed: " + " ".join(f"h{h}:{c:.4f}" for h, c in cov.items()))
        print("    autocovariance   fitted : " + " ".join(
            f"h{h}:{psiL ** h * sL ** 2 + psiu ** h * su ** 2:.4f}" for h in cov))
        phi = cov[1] / v0
        print(f"    for comparison, a plain AR(1) would be phi {phi:.3f}: "
              f"h5 correlation {phi ** 5:.2f} against the observed {cov.get(5, np.nan) / v0:.2f}")
    return out


def fit_measurement(tm: pd.DataFrame, verbose: bool = True) -> float:
    """Sampling noise in a season rating: rate each club on its odd-numbered
    matches and again on its even ones. var(odd - even) / 4 is the variance of
    a full-season rating."""
    diffs = []
    for s in sorted(tm["season"].unique()):
        d = tm[tm["season"] == int(s)].copy()
        if len(d) < 700:
            continue
        d["w"] = 1.0
        d = d.sort_values("date")
        d["k"] = d.groupby("team").cumcount()
        parts = []
        for half in (0, 1):
            h = d[d["k"] % 2 == half]
            r = T.team_ratings(h, season=int(s), promoted=set())
            parts.append((r["vol_att"] + r["q_att"]) - (r["vol_def"] + r["q_def"]))
        diffs += (parts[0] - parts[1]).tolist()
    sd = float(np.std(diffs) / 2.0)
    if verbose:
        print(f"  measurement: odd vs even halves, n={len(diffs)} club-seasons -> season rating sd {sd:.3f}")
    return sd


def fit_drift(tm: pd.DataFrame, cuts=(4, 8, 12, 19, 26), verbose: bool = True) -> pd.DataFrame:
    """How far a club's strength over the rest of a season sits from a rating
    fitted on the matches before the cut, beyond the sampling noise of the
    rating. The residual is what no rating could have known: DRIFT_SD."""
    rows = []
    for k in cuts:
        pre, post, oe = [], [], []
        for s in sorted(tm["season"].unique()):
            d = tm[tm["season"] == int(s)].copy()
            if len(d) < 700:
                continue
            d["w"] = 1.0
            d = d.sort_values("date")
            d["k"] = d.groupby("team").cumcount() + 1
            a = _net_of(d[d["k"] <= k], s)
            b = _net_of(d[d["k"] > k], s)
            o = _net_of(d[(d["k"] > k) & (d["k"] % 2 == 1)], s)
            e = _net_of(d[(d["k"] > k) & (d["k"] % 2 == 0)], s)
            teams = sorted(set(a.index) & set(b.index))
            pre += a[teams].tolist(); post += b[teams].tolist()
            oe += (o[teams] - e[teams]).tolist()
        pre, post, oe = np.array(pre), np.array(post), np.array(oe)
        X = np.column_stack([np.ones(len(pre)), pre])
        beta = np.linalg.lstsq(X, post, rcond=None)[0]
        resid = float(np.std(post - X @ beta))
        samp = float(np.var(oe) / 4.0)
        rows.append(dict(cut=k, n=len(pre), slope=beta[1], resid_sd=resid,
                         sampling_sd=np.sqrt(samp), drift_sd=np.sqrt(max(resid ** 2 - samp, 0.0))))
    out = pd.DataFrame(rows)
    if verbose:
        print(out.round(3).to_string(index=False))
    return out


def _net_of(d: pd.DataFrame, season: int) -> pd.Series:
    r = T.team_ratings(d, season=int(season), promoted=set())
    return (r["vol_att"] + r["q_att"]) - (r["vol_def"] + r["q_def"])


def fit_promoted(hist: pd.DataFrame, decay: float = 0.85, verbose: bool = True) -> dict:
    """The promoted class: its average season rating, its spread, and how long
    the clubs that come up have been away."""
    seasons = sorted(hist["season"].unique())
    ix = hist.set_index(["season", "team"])
    present = {s: set(hist.loc[hist["season"] == s, "team"]) for s in seasons}
    rows = []
    for s in seasons[1:]:
        for t in present[s] - present[s - 1]:
            prior = [q for q in seasons if q < s and t in present[q]]
            rows.append(dict(season=s, team=t, net=ix.at[(s, t), "net"], style=ix.at[(s, t), "style"],
                             away=(s - max(prior) if prior else np.nan)))
    P = pd.DataFrame(rows)
    w = decay ** (max(seasons) - P["season"])
    out = dict(n=len(P), net=float((P["net"] * w).sum() / w.sum()), net_flat=float(P["net"].mean()),
               net_sd=float(P["net"].std()), style=float(P["style"].mean()),
               away=P["away"].value_counts(dropna=False).to_dict())
    if verbose:
        print(f"  promoted sides n={out['n']}: net {out['net']:.3f} recency-weighted "
              f"({out['net_flat']:.3f} flat), sd {out['net_sd']:.3f}, style {out['style']:.3f}")
        share = {int(k) - 1: round(v / len(P), 3) for k, v in sorted(out["away"].items())
                 if k == k and k <= 4}
        print(f"    seasons MISSED before coming up: {share} "
              "(the rest had been gone four or more) -> PROMOTED_FROM")
    return out


def fit_draw_tau(tm: pd.DataFrame, r: dict, taus=(0.0, -0.05, -0.10, -0.15), verbose: bool = True):
    """The Dixon-Coles low-score correction, scored on how close the simulated
    home / draw / away rates come to the real ones at the model's own expected
    goals. In sample on the rating, which is why only the SHAPE is taken from
    it — the rates themselves are what the rating already reproduces."""
    d = tm[tm["home"]].dropna(subset=["goals", "goals_against"])
    teams = sorted(r["vol_att"].index)
    ti = {t: i for i, t in enumerate(teams)}
    keep = d["team"].isin(ti) & d["opp"].isin(ti)
    d = d[keep]
    tt = team_terms(r, teams)
    mu_h, mu_a = fixture_mu(tt, d["team"].map(ti).to_numpy(), d["opp"].map(ti).to_numpy())
    actual = np.array([(d["goals"] > d["goals_against"]).mean(), (d["goals"] == d["goals_against"]).mean(),
                       (d["goals"] < d["goals_against"]).mean()])
    best, rows = None, []
    for tau in taus:
        rng = np.random.default_rng(5)
        gh, ga = draw_scores(rng, np.repeat(mu_h[None, :], 400, 0), np.repeat(mu_a[None, :], 400, 0), tau=tau)
        sim = np.array([(gh > ga).mean(), (gh == ga).mean(), (gh < ga).mean()])
        err = float(np.abs(sim - actual).sum())
        rows.append(dict(tau=tau, home=sim[0], draw=sim[1], away=sim[2], abs_err=err))
        if best is None or err < best[1]:
            best = (tau, err)
    out = pd.DataFrame(rows)
    if verbose:
        print(f"  actual  home {actual[0]:.3f} draw {actual[1]:.3f} away {actual[2]:.3f}  (n={len(d)})")
        print(out.round(4).to_string(index=False))
        print(f"  -> DRAW_TAU {best[0]}")
    return best[0], out


def fit_spread(tm: pd.DataFrame, hist: pd.DataFrame, fx_all: pd.DataFrame | None = None,
               grid=(1.0, 1.05, 1.10, 1.15, 1.20, 1.25), n_sims: int = 1500,
               verbose: bool = True) -> pd.DataFrame:
    """Hand each completed season its own ratings, play it out, and compare the
    table with the one that really happened — position by position. The rating
    is a shrunk estimate, so the spread it reports needs de-attenuating before
    a simulated table has the shape of a real one (SPREAD_MULT)."""
    seasons = sorted(hist["season"].unique())
    if fx_all is None:
        fx_all = D.load_fixtures(tuple(seasons))
    actual, setups = [], []
    for s in seasons:
        fx = fx_all[(fx_all["season"] == s) & fx_all["played"]]
        g = hist[hist["season"] == s]
        if len(fx) < 370 or len(g) < 18:
            continue
        r = T.team_ratings(D.game_weights(tm[tm["season"] <= s], "team", D.RECENCY_DEFAULT), season=int(s))
        setups.append((league_level(r), g["net"].to_numpy(), g["style"].to_numpy()))
        actual.append(np.sort(table_from(fx)["Pts"].to_numpy())[::-1])
    A = np.array(actual)
    rows = []
    for m in grid:
        sims = []
        rng = np.random.default_rng(9)
        for lg, net, sty in setups:
            n = len(net)
            hi, ai = _round_robin(n)
            MH, MA = _incidence(hi, n), _incidence(ai, n)
            wide = np.repeat((net * m)[None, :], n_sims, 0)
            styr = np.repeat(sty[None, :], n_sims, 0)
            att, dfn = 0.5 * (wide + styr), 0.5 * (styr - wide)
            mu_h, mu_a = future_mu(lg, att[:, hi], dfn[:, hi], att[:, ai], dfn[:, ai])
            gh, ga = draw_scores(rng, mu_h, mu_a)
            p = np.zeros((n_sims, n)); gd = np.zeros_like(p); gf = np.zeros_like(p)
            _accumulate(p, gd, gf, MH, MA, gh, ga)
            sims.append(np.sort(p, axis=1)[:, ::-1].mean(axis=0))
        S = np.array(sims)
        rows.append(dict(mult=m, rmse=float(np.sqrt(((S - A) ** 2).mean())),
                         champion=float(S[:, 0].mean()), fourth=float(S[:, 3].mean()),
                         seventeenth=float(S[:, 16].mean()), last=float(S[:, -1].mean()),
                         spread_sd=float(S.std(axis=1).mean()),
                         draws=float(1140 - S.sum(axis=1).mean())))
    out = pd.DataFrame(rows)
    if verbose:
        print(f"  actual ({len(A)} seasons): champion {A[:, 0].mean():.1f}, 4th {A[:, 3].mean():.1f}, "
              f"17th {A[:, 16].mean():.1f}, last {A[:, -1].mean():.1f}, spread sd {A.std(axis=1).mean():.1f}, "
              f"draws {1140 - A.sum(axis=1).mean():.0f}")
        print(out.round(3).to_string(index=False))
        print(f"  -> SPREAD_MULT {out.loc[out['rmse'].idxmin(), 'mult']}")
    return out


def points_per_net(hist: pd.DataFrame, tm: pd.DataFrame, verbose: bool = True) -> float:
    """League points per unit of net rating — the translation the dial on the
    page needs, so 'one level innovation' can be quoted in points."""
    pts = {}
    for s in sorted(tm["season"].unique()):
        d = tm[(tm["season"] == int(s))]
        if len(d) < 700:
            continue
        w = np.where(d["goals"] > d["goals_against"], WIN, np.where(d["goals"] == d["goals_against"], DRAW, 0))
        pts.update({(int(s), t): v for t, v in pd.Series(w, index=d["team"].values).groupby(level=0).sum().items()})
    h = hist[[(int(s), t) in pts for s, t in zip(hist["season"], hist["team"])]]
    y = np.array([pts[(int(s), t)] for s, t in zip(h["season"], h["team"])], float)
    X = np.column_stack([np.ones(len(h)), h["net"].to_numpy()])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    if verbose:
        print(f"  points = {b[0]:.1f} + {b[1]:.1f} x net  (n={len(h)}); "
              f"one level innovation ({LEVEL_INNOV_SD:.3f}) is {b[1] * LEVEL_INNOV_SD:.1f} points a season")
    return float(b[1])


# ---------------------------------------------------------------------------
# Out of sample: project past seasons from part-way through and score them
# ---------------------------------------------------------------------------

def _cut_fixtures(fx: pd.DataFrame, season: int, cut: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    """The season as it looked after `cut` rounds: later results hidden."""
    f = fx[fx["season"] == int(season)].copy()
    hide = f["round"] > cut
    f.loc[hide, ["home_goals", "away_goals", "home_xg", "away_xg"]] = np.nan
    f.loc[hide, "played"] = False
    date = f.loc[~hide, "date"].max()
    return f, date


def backtest(score_seasons, cuts=(5, 10, 19, 28), n_prior: int = 2,
             recency: D.Recency = D.RECENCY_DEFAULT, n_sims: int = 4000,
             drift: bool | float = True, drifts=None, progress=None) -> pd.DataFrame:
    """Project each season from part-way through, with everything fitted on
    matches before the cut, and compare with how it actually finished. One row
    per club per season per cut.

    `drifts` projects the same ratings at several drift sizes — the rating fit
    is the expensive part and it does not depend on the drift — which is how
    `calibrate_drift` finds the size that makes the intervals honest."""
    grid = [drift] if drifts is None else list(drifts)
    seasons = tuple(int(s) for s in score_seasons)
    tm_all = D.load_team_matches(backtest_window(seasons, n_prior))
    fx_all = D.load_fixtures(seasons)
    rows = []
    jobs = [(s, k) for s in seasons for k in cuts]
    for j, (s, k) in enumerate(jobs):
        fx = fx_all[fx_all["season"] == s]
        if not fx["played"].all():
            continue                                   # only completed seasons
        cut_fx, cut_date = _cut_fixtures(fx_all, s, k)
        prior = tm_all[tm_all["date"] <= cut_date]
        if prior["match_id"].nunique() < 60:
            continue
        prior = D.game_weights(prior, "team", recency)
        teams = sorted(set(fx["home"]) | set(fx["away"]))
        try:
            r = T.team_ratings(prior, season=s, extra_teams=teams)
        except ValueError:
            continue
        final = table_from(fx, teams)
        n_teams = len(teams)
        for dr in grid:
            sim = simulate_table(r, cut_fx, n_sims=n_sims, seed=1000 + 10 * s + k, drift=dr)
            pace = sim["start"]["Pts"] / sim["start"]["P"].clip(lower=1) * (n_teams - 1) * 2
            for t in teams:
                pp = sim["points"][:, sim["teams"].index(t)]
                po = sim["position"][:, sim["teams"].index(t)]
                actual = float(final.at[t, "Pts"])
                rows.append(dict(
                    season=s, cut=k, drift=(0.0 if dr is False else DRIFT_SD if dr is True else float(dr)),
                    drift_sd=float(sim["drift_sd"]), team=t,
                    played=int(sim["start"].at[t, "P"]),
                    proj=float(pp.mean()), p10=float(np.percentile(pp, 10)), p90=float(np.percentile(pp, 90)),
                    pace=float(pace[t]), actual=actual,
                    pos=float(final.index.get_loc(t) + 1),
                    p_title=float((po == 1).mean()), p_ucl=float((po <= 4).mean()),
                    p_rel=float((po > n_teams - RELEGATED).mean()),
                    title=float(final.index[0] == t),
                    ucl=float(final.index.get_loc(t) < 4),
                    rel=float(final.index.get_loc(t) >= n_teams - RELEGATED),
                    pit=float((pp < actual).mean() + 0.5 * (pp == actual).mean()),
                ))
        if progress:
            progress(j + 1, len(jobs))
    return pd.DataFrame(rows)


def backtest_window(score_seasons, n_prior: int) -> tuple[int, ...]:
    lo = min(score_seasons) - n_prior
    return tuple(range(lo, max(score_seasons) + 1))


def backtest_metrics(bt: pd.DataFrame, by=("cut",)) -> pd.DataFrame:
    """Per cut: how close the projected points were, whether the interval
    covered the outcome, and how well the outcome probabilities scored."""
    rows = []
    for key, g in bt.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        cover = ((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()
        rows.append(dict(
            **{c: (round(float(v), 3) if c == "drift" else int(v)) for c, v in zip(by, key)}, n=len(g),
            mae=float((g["proj"] - g["actual"]).abs().mean()),
            rmse=float(np.sqrt(((g["proj"] - g["actual"]) ** 2).mean())),
            mae_pace=float((g["pace"] - g["actual"]).abs().mean()),
            bias=float((g["proj"] - g["actual"]).mean()),
            cover_80=float(cover),
            pit_sd=float(g["pit"].std()),
            brier_rel=float(((g["p_rel"] - g["rel"]) ** 2).mean()),
            brier_rel_base=float(((RELEGATED / 20 - g["rel"]) ** 2).mean()),
            brier_ucl=float(((g["p_ucl"] - g["ucl"]) ** 2).mean()),
            brier_ucl_base=float(((4 / 20 - g["ucl"]) ** 2).mean()),
            logloss_title=float(-np.log(np.clip(np.where(g["title"] > 0, g["p_title"], 1 - g["p_title"]),
                                                1e-6, 1)).mean()),
        ))
    return pd.DataFrame(rows)


def calibrate_drift(score_seasons, grid=(0.0, 0.10, 0.18, 0.26, 0.34), cuts=(5, 19),
                    n_sims: int = 3000, **kw) -> pd.DataFrame:
    """The drift that makes the intervals honest. Too little and the 10-90
    band covers far less than 80% of the outcomes; too much and it covers more
    than it should, and the relegation odds go soft. One rating fit per season
    and cut serves the whole grid."""
    bt = backtest(score_seasons, cuts=cuts, n_sims=n_sims,
                  drifts=[(False if sd == 0 else sd) for sd in grid], **kw)
    m = backtest_metrics(bt, by=("cut", "drift")).sort_values(["cut", "drift"])
    eff = bt.groupby(["cut", "drift"])["drift_sd"].mean().rename("effective_sd").reset_index()
    return m.merge(eff, on=["cut", "drift"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cached_seasons(lo: int = 2014, hi: int = 2027) -> tuple[int, ...]:
    return tuple(s for s in range(lo, hi) if (D.CACHE / f"shots_{D.LEAGUE}_{s}.parquet").exists())


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    args = [a for a in sys.argv[2:]]

    if cmd == "fit":
        seasons = tuple(int(a) for a in args) or _cached_seasons()
        print(f"seasons in the cache: {seasons}")
        tm = D.load_team_matches(seasons)
        hist = season_ratings(tm)
        print(f"\n{len(hist)} club-seasons rated, {hist['season'].nunique()} seasons\n")
        print("MEASUREMENT NOISE")
        meas = fit_measurement(tm)
        print("\nSEASON TO SEASON")
        globals()["MEAS_SD"] = meas
        p = fit_process(hist)
        print("\nSTYLE (attack + defence)")
        ixs = hist.set_index(["season", "team"]).sort_index()
        x = [ixs.at[(s, t), "style"] for (s, t) in ixs.index if (s + 1, t) in ixs.index]
        y = [ixs.at[(s + 1, t), "style"] for (s, t) in ixs.index if (s + 1, t) in ixs.index]
        b = np.polyfit(x, y, 1)
        print(f"  psi {b[0]:.3f}, residual sd {np.std(np.array(y) - np.polyval(b, x)):.3f} (n={len(x)})")
        print("\nPROMOTED")
        fit_promoted(hist)
        print("\nWITHIN-SEASON DRIFT")
        fit_drift(tm)
        print("\nDRAWS")
        fit_draw_tau(tm, T.team_ratings(D.game_weights(tm, "team", D.RECENCY_DEFAULT)))
        print("\nTABLE SHAPE (each season played from its own ratings)")
        fit_spread(tm, hist)
        print("\nTRANSLATION")
        points_per_net(hist, tm)
        print("\nCurrent constants: "
              f"LEVEL_SD {LEVEL_SD} PSI {LEVEL_PSI} INNOV {LEVEL_INNOV_SD} FORM {FORM_SD} "
              f"MEAS {MEAS_SD} PROMOTED {PROMOTED_NET} DRIFT {DRIFT_SD}/{DRIFT_EARLY} TAU {DRAW_TAU}")

    elif cmd == "backtest":
        seasons = tuple(int(a) for a in args) or _cached_seasons(2018, 2026)
        print(f"scoring seasons {seasons} out of sample…")
        bt = backtest(seasons, progress=lambda i, n: print(f"\r  {i}/{n}", end=""))
        print()
        m = backtest_metrics(bt)
        print(m.round(3).to_string(index=False))
        print("\nmae = mean error in a club's final points; mae_pace = the same for "
              "'carry on at the current rate'; cover_80 should be 0.80; brier vs its base rate.")

    elif cmd == "drift":
        seasons = tuple(int(a) for a in args) or _cached_seasons(2018, 2026)
        print(calibrate_drift(seasons).round(3).to_string(index=False))

    elif cmd == "future":
        seasons = _cached_seasons()
        tm = D.load_team_matches(seasons)
        hist = season_ratings(tm)
        r = T.team_ratings(D.game_weights(tm, "team", D.RECENCY_DEFAULT))
        fx = season_fixtures()
        tiers, live = club_tiers(int(fx["season"].iloc[0]), sorted(set(fx["home"])))
        print("division feed: " + ("read" if live else
              "UNREACHABLE — every club outside the top flight assumed to be in the second tier"))
        fut = simulate_future(r, fx, hist, years=5, n_sims=4000, tiers=tiers)
        print(f"\n--- {season_label(fut['season0'] + 5)} ---")
        print(group_generic(future_table(fut, 5)).head(12).round(3).to_string())
        for club in (args or ["MCI", "SUN"]):
            o = club_outlook(fut, club)
            print(f"\n=== {club} ===")
            print(o.round(3).to_string())
            print("within five seasons: " + ", ".join(f"{k} {v:.1%}" for k, v in o.attrs["ever"].items()))

    else:                                            # table
        seasons = _cached_seasons()
        tm = D.load_team_matches(seasons)
        r = T.team_ratings(D.game_weights(tm, "team", D.RECENCY_DEFAULT))
        fx = season_fixtures()
        sim = simulate_table(r, fx, n_sims=20000)
        t = projected_table(sim)
        print(f"{season_label(int(fx['season'].iloc[0]))}: {sim['n_remaining']} fixtures left, "
              f"drift sd {sim['drift_sd']:.3f}")
        print(t.round(3).to_string())
