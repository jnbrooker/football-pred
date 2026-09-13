"""
footsim/data.py — shared football data layer for the whole model suite.

Every page and the match simulator import from here so they all speak the
same data. Responsibilities:

  * pull & cache Understat shot-level and roster-level data per match
    (the play-by-play of this suite), and the league fixture / team tables
  * pull & cache the FPL API: squads, availability, positions, per-gameweek
    scoring components (saves, bonus, defensive contribution)
  * pull football-data.co.uk results with closing odds (backtest comparator)
  * one canonical team code (FPL short name) across all three sources
  * recency weighting (recent seasons AND recent matches weighted more)
  * small shared stats helpers (weighted mean/std, Beta matching, shrinkage)

Seasons are named by their START year everywhere: 2025 means 2025/26.

Understat trap (2025): the site no longer embeds `shotsData` / `rostersData`
in the HTML. Data is served by JSON endpoints — `getLeagueData/{league}/{season}`
and `getMatchData/{match_id}` — gzip-compressed and only with an
`X-Requested-With: XMLHttpRequest` header. Every scraping library that parses
the page source is dead; this module talks to the endpoints directly.
"""

from __future__ import annotations

import datetime as _dt
import functools
import gzip
import json
import re
import time as _time
import unicodedata
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

LEAGUE = "EPL"                  # Understat league code; the FPL layers are EPL-only
REFRESH_HOURS = 6.0
FETCH_PAUSE = 0.12              # seconds between Understat match requests


# ---------------------------------------------------------------------------
# Caching: a time-bucketed lru_cache, so a long-running app picks up a new
# matchweek without a restart.
# ---------------------------------------------------------------------------

def ttl_cache(maxsize: int = 8):
    def deco(fn):
        @functools.lru_cache(maxsize=maxsize)
        def inner(_bucket, *args, **kwargs):
            return fn(*args, **kwargs)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bucket = int(_time.time() // (REFRESH_HOURS * 3600))
            return inner(bucket, *args, **kwargs)

        wrapper.cache_clear = inner.cache_clear
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Recency: how much a match counts, by how long ago it was played
# ---------------------------------------------------------------------------
# season_w — season_decay ** (seasons ago)
# w        — season_w × 0.5 ** (matches_ago / half_life), matches_ago counted
#            PER TEAM over the matches it actually played, running continuously
#            back through earlier seasons.
#
# Current-season share of weight with two prior seasons in the window
# (38-match seasons):
#                    after match:  1     5    10    19    30
#   Long memory  (0.70, inf)        2%   10%   19%   31%   42%
#   Balanced     (0.85, 15)         5%   22%   39%   58%   72%
#   Recent form  (0.70, 8)         10%   40%   62%   80%   90%

class Recency(NamedTuple):
    season_decay: float = 0.85
    half_life: float = 15.0      # matches; inf = seasons as flat blocks


RECENCY_DEFAULT = Recency()
RECENCY_PRESETS = {
    "Long memory": Recency(season_decay=0.7, half_life=float("inf")),
    "Balanced": RECENCY_DEFAULT,
    "Recent form": Recency(season_decay=0.7, half_life=8.0),
}


def game_weights(df: pd.DataFrame, team_col: str,
                 recency: Recency = RECENCY_DEFAULT,
                 season_col: str = "season", order_col: str = "date") -> pd.DataFrame:
    """Return `df` with `season_w` and `w` set. Works on any per-match frame
    with a season, a date (or match ordinal) and a team column."""
    r = Recency(*recency)
    out = df.copy()
    if season_col not in out.columns or out.empty:
        out["season_w"] = 1.0
        out["w"] = 1.0
        return out
    season = pd.to_numeric(out[season_col], errors="coerce")
    latest = int(season.max())
    sw = np.power(float(r.season_decay), (latest - season).clip(lower=0).astype(float))
    w = sw.to_numpy(dtype=float).copy()
    if (np.isfinite(r.half_life) and r.half_life > 0
            and order_col in out.columns and team_col in out.columns):
        stamp = pd.to_datetime(out[order_col], errors="coerce")
        ago = stamp.groupby(out[team_col]).rank(method="dense", ascending=False) - 1.0
        w *= np.power(0.5, ago.to_numpy(dtype=float) / float(r.half_life))
    out["season_w"] = sw.to_numpy(dtype=float)
    out["w"] = w
    return out


def weight_shares(df: pd.DataFrame, team_col: str = "team") -> dict:
    """Share of total weight each season carries, over team-matches."""
    if df is None or df.empty or "w" not in df.columns:
        return {}
    keys = [c for c in (team_col, "season", "match_id") if c in df.columns]
    tg = df.drop_duplicates(keys)
    tot = float(tg["w"].sum())
    if tot <= 0:
        return {}
    return {int(s): float(v) for s, v in (tg.groupby("season")["w"].sum() / tot).items()}


# ---------------------------------------------------------------------------
# Team codes: one canonical short code (the FPL one) across every source
# ---------------------------------------------------------------------------

TEAM_ALIASES = {
    "ARS": ("Arsenal",),
    "AVL": ("Aston Villa",),
    "BOU": ("Bournemouth", "AFC Bournemouth"),
    "BRE": ("Brentford",),
    "BHA": ("Brighton", "Brighton and Hove Albion", "Brighton & Hove Albion"),
    "BUR": ("Burnley",),
    "CHE": ("Chelsea",),
    "COV": ("Coventry", "Coventry City"),
    "CRY": ("Crystal Palace",),
    "EVE": ("Everton",),
    "FUL": ("Fulham",),
    "HUL": ("Hull", "Hull City"),
    "IPS": ("Ipswich", "Ipswich Town"),
    "LEE": ("Leeds", "Leeds United"),
    "LEI": ("Leicester", "Leicester City"),
    "LIV": ("Liverpool",),
    "LUT": ("Luton", "Luton Town"),
    "MCI": ("Man City", "Manchester City"),
    "MUN": ("Man Utd", "Man United", "Manchester United"),
    "NEW": ("Newcastle", "Newcastle United"),
    "NFO": ("Nott'm Forest", "Nottingham Forest", "Nottm Forest"),
    "SHU": ("Sheffield United", "Sheffield Utd"),
    "SOU": ("Southampton",),
    "SUN": ("Sunderland",),
    "TOT": ("Spurs", "Tottenham", "Tottenham Hotspur"),
    "WHU": ("West Ham", "West Ham United"),
    "WOL": ("Wolves", "Wolverhampton Wanderers"),
}
_ALIAS_LOOKUP = {a.lower(): code for code, names in TEAM_ALIASES.items() for a in names}
_ALIAS_LOOKUP.update({code.lower(): code for code in TEAM_ALIASES})
TEAM_NAMES = {code: names[0] for code, names in TEAM_ALIASES.items()}


def team_code(name: str) -> str:
    """Canonical code for any source's team name; the name itself if unknown."""
    if name is None:
        return ""
    s = str(name).strip()
    return _ALIAS_LOOKUP.get(s.lower(), _ALIAS_LOOKUP.get(s.replace("_", " ").lower(), s))


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_UA = "Mozilla/5.0 (footsim research; contact via github)"

# The python.org framework build on macOS ships without root certificates, so
# every download fails with CERTIFICATE_VERIFY_FAILED unless certifi's bundle is
# used. Build one SSL context here and use it for every request.
try:
    import ssl as _ssl
    import certifi as _certifi
    _SSL_CTX = _ssl.create_default_context(cafile=_certifi.where())
except Exception:                      # certifi missing: fall back to the system store
    _SSL_CTX = None


def _get_json(url: str, referer: str = "https://understat.com/", timeout: float = 25.0):
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA, "X-Requested-With": "XMLHttpRequest",
        "Referer": referer, "Accept-Encoding": "gzip", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _get_text(url: str, timeout: float = 25.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        return r.read().decode("utf-8-sig", errors="replace")


# ---------------------------------------------------------------------------
# Understat: league table (fixtures, team-match history)
# ---------------------------------------------------------------------------

_US = "https://understat.com"


@ttl_cache(maxsize=16)
def understat_league(league: str, season: int) -> dict:
    """The raw `getLeagueData` payload: dates (fixtures), teams (match
    history per team), players (season totals)."""
    return _get_json(f"{_US}/getLeagueData/{league}/{int(season)}",
                     referer=f"{_US}/league/{league}/{int(season)}")


def load_fixtures(seasons: tuple[int, ...], league: str = LEAGUE) -> pd.DataFrame:
    """One row per fixture (played or not): match_id, season, date, home,
    away, home_goals, away_goals, played, plus Understat's pre-match xG-based
    forecast (for curiosity only — never read by a model)."""
    rows = []
    for s in seasons:
        try:
            L = understat_league(league, int(s))
        except Exception:
            continue
        for d in L["dates"]:
            rows.append(dict(
                match_id=int(d["id"]), season=int(s),
                date=pd.to_datetime(d["datetime"]),
                home=team_code(d["h"]["title"]), away=team_code(d["a"]["title"]),
                home_id=int(d["h"]["id"]), away_id=int(d["a"]["id"]),
                home_goals=(int(d["goals"]["h"]) if d["isResult"] else np.nan),
                away_goals=(int(d["goals"]["a"]) if d["isResult"] else np.nan),
                home_xg=(float(d["xG"]["h"]) if d["isResult"] else np.nan),
                away_xg=(float(d["xG"]["a"]) if d["isResult"] else np.nan),
                played=bool(d["isResult"]),
                fc_home=float(d["forecast"]["w"]) if d.get("forecast") else np.nan,
                fc_draw=float(d["forecast"]["d"]) if d.get("forecast") else np.nan,
                fc_away=float(d["forecast"]["l"]) if d.get("forecast") else np.nan,
            ))
    if not rows:
        return pd.DataFrame()
    f = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    return add_rounds(f)


def add_rounds(f: pd.DataFrame) -> pd.DataFrame:
    """A `round` column: the matchweek. Football has no fixed week; each
    team's k-th fixture of the season is its round k (postponed matches keep
    their ordinal, which is what a rolling backtest needs)."""
    f = f.copy()
    f["round"] = 0
    for s, g in f.groupby("season"):
        long = pd.concat([
            g[["match_id", "date", "home"]].rename(columns={"home": "team"}),
            g[["match_id", "date", "away"]].rename(columns={"away": "team"})])
        long = long.sort_values("date")
        long["k"] = long.groupby("team").cumcount() + 1
        # a match's round is the LATER of the two teams' ordinals
        r = long.groupby("match_id")["k"].max()
        f.loc[g.index, "round"] = g["match_id"].map(r).values
    return f


# ---------------------------------------------------------------------------
# Understat: per-match shots and rosters, cached to parquet incrementally
# ---------------------------------------------------------------------------

SHOT_COLS = ["match_id", "season", "date", "shot_id", "minute", "team", "opp", "home",
             "player_id", "player", "result", "xg", "x", "y", "situation",
             "shot_type", "last_action", "assist_player", "goal", "on_target",
             "blocked", "penalty", "own_goal"]
ROSTER_COLS = ["match_id", "season", "date", "team", "opp", "home", "player_id", "player",
               "position", "position_order", "started", "minutes", "sub_on", "sub_off",
               "goals", "own_goals", "shots", "xg", "key_passes", "assists", "xa",
               "xg_chain", "xg_buildup", "yellow", "red"]

ON_TARGET = {"Goal", "SavedShot"}


def _cache_paths(league: str, season: int) -> tuple[Path, Path]:
    return (CACHE / f"shots_{league}_{season}.parquet",
            CACHE / f"rosters_{league}_{season}.parquet")


def _parse_match(mid: int, season: int, date, home: str, away: str, payload: dict):
    shots, rosters = [], []
    for side in ("h", "a"):
        team, opp = (home, away) if side == "h" else (away, home)
        for s in payload["shots"].get(side, []):
            res = s["result"]
            assisted = s.get("player_assisted")
            shots.append(dict(
                match_id=mid, season=season, date=date, shot_id=int(s["id"]),
                minute=int(float(s["minute"])), team=team, opp=opp, home=(side == "h"),
                player_id=int(s["player_id"]), player=s["player"], result=res,
                xg=float(s["xG"]), x=float(s["X"]), y=float(s["Y"]),
                situation=s.get("situation", ""), shot_type=s.get("shotType", ""),
                last_action=s.get("lastAction") or "", assist_player=assisted or "",
                goal=int(res == "Goal"), on_target=int(res in ON_TARGET),
                blocked=int(res == "BlockedShot"),
                penalty=int(s.get("situation") == "Penalty"), own_goal=int(res == "OwnGoal"),
            ))
        for r in payload["rosters"].get(side, {}).values():
            pos = r.get("position", "")
            rosters.append(dict(
                match_id=mid, season=season, date=date, team=team, opp=opp, home=(side == "h"),
                player_id=int(r["player_id"]), player=r["player"], position=pos,
                position_order=int(r.get("positionOrder") or 0),
                started=int(pos != "Sub"), minutes=int(float(r.get("time") or 0)),
                # Understat: a STARTER's `roster_in` is the id of the sub who
                # replaced him (so he was subbed off); a SUB's `roster_out` is
                # the starter he replaced (so he came on).
                sub_off=int(int(r.get("roster_in") or 0) != 0),
                sub_on=int(int(r.get("roster_out") or 0) != 0),
                goals=int(r.get("goals") or 0), own_goals=int(r.get("own_goals") or 0),
                shots=int(r.get("shots") or 0), xg=float(r.get("xG") or 0),
                key_passes=int(r.get("key_passes") or 0), assists=int(r.get("assists") or 0),
                xa=float(r.get("xA") or 0), xg_chain=float(r.get("xGChain") or 0),
                xg_buildup=float(r.get("xGBuildup") or 0),
                yellow=int(r.get("yellow_card") or 0), red=int(r.get("red_card") or 0),
            ))
    return shots, rosters


def sync_season(season: int, league: str = LEAGUE, progress=None, max_fetch: int | None = None) -> int:
    """Fetch every played match of a season not yet in the cache. Returns the
    number fetched. `progress(done, total)` is called as it goes."""
    fx = load_fixtures((int(season),), league)
    if fx.empty:
        return 0
    sp, rp = _cache_paths(league, int(season))
    have = set()
    old_shots = old_rosters = None
    if rp.exists():
        old_rosters = pd.read_parquet(rp)
        old_shots = pd.read_parquet(sp) if sp.exists() else pd.DataFrame(columns=SHOT_COLS)
        have = set(old_rosters["match_id"].unique())
    todo = fx[fx["played"] & ~fx["match_id"].isin(have)]
    if max_fetch is not None:
        todo = todo.head(int(max_fetch))
    if todo.empty:
        return 0
    shots, rosters = [], []
    n = len(todo)
    for i, (_, m) in enumerate(todo.iterrows()):
        mid = int(m["match_id"])
        try:
            payload = _get_json(f"{_US}/getMatchData/{mid}", referer=f"{_US}/match/{mid}")
        except Exception:
            continue
        s, r = _parse_match(mid, int(season), m["date"], m["home"], m["away"], payload)
        shots += s
        rosters += r
        if progress:
            progress(i + 1, n)
        _time.sleep(FETCH_PAUSE)
    if not rosters:
        return 0
    new_s = pd.DataFrame(shots, columns=SHOT_COLS)
    new_r = pd.DataFrame(rosters, columns=ROSTER_COLS)
    if old_rosters is not None:
        new_s = pd.concat([old_shots, new_s], ignore_index=True)
        new_r = pd.concat([old_rosters, new_r], ignore_index=True)
    new_s.to_parquet(sp, index=False)
    new_r.to_parquet(rp, index=False)
    return len(set(new_r["match_id"]) - have)


def _read_cached(seasons, league, which: str) -> pd.DataFrame:
    frames = []
    for s in seasons:
        sp, rp = _cache_paths(league, int(s))
        p = sp if which == "shots" else rp
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=SHOT_COLS if which == "shots" else ROSTER_COLS)
    return pd.concat(frames, ignore_index=True)


@ttl_cache(maxsize=8)
def _synced(seasons: tuple[int, ...], league: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shots and rosters for the seasons, syncing the cache first (once per
    REFRESH_HOURS)."""
    for s in seasons:
        try:
            sync_season(int(s), league)
        except Exception:
            continue
    return _read_cached(seasons, league, "shots"), _read_cached(seasons, league, "rosters")


def load_shots(seasons: tuple[int, ...], league: str = LEAGUE,
               recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """Every shot in the seasons, with recency weights keyed on the shooting
    team's schedule."""
    shots, _ = _synced(tuple(sorted(int(s) for s in seasons)), league)
    return game_weights(shots, "team", recency)


def load_rosters(seasons: tuple[int, ...], league: str = LEAGUE,
                 recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """Every player-match line (minutes, shots, key passes, ...) with weights."""
    _, rosters = _synced(tuple(sorted(int(s) for s in seasons)), league)
    r = game_weights(rosters, "team", recency)
    r["player_id"] = r["player_id"].astype(int)
    return r


def load_team_matches(seasons: tuple[int, ...], league: str = LEAGUE,
                      recency: Recency = RECENCY_DEFAULT) -> pd.DataFrame:
    """One row per team-match: shots, on target, xG, goals (for and against),
    penalties, own goals, from the shot table. The drive table of this suite."""
    shots, rosters = _synced(tuple(sorted(int(s) for s in seasons)), league)
    fx = load_fixtures(tuple(sorted(int(s) for s in seasons)), league)
    fx = fx[fx["played"]]
    if shots.empty or fx.empty:
        return pd.DataFrame()
    agg = (shots.groupby(["match_id", "team"], as_index=False)
                .agg(shots=("shot_id", "size"), sot=("on_target", "sum"),
                     xg=("xg", "sum"), goals_from_shots=("goal", "sum"),
                     pens=("penalty", "sum"), pen_xg=("xg", lambda s: 0.0),
                     og_conceded=("own_goal", "sum")))
    # penalty xG separately (so non-penalty quality can be rated)
    pen = (shots[shots["penalty"] == 1].groupby(["match_id", "team"])
                .agg(pen_xg=("xg", "sum"), pen_goals=("goal", "sum")).reset_index())
    agg = agg.drop(columns="pen_xg").merge(pen, on=["match_id", "team"], how="left")
    agg["pen_xg"] = agg["pen_xg"].fillna(0.0)
    agg["pen_goals"] = agg["pen_goals"].fillna(0.0)
    rows = []
    for _, m in fx.iterrows():
        for team, opp, home, gf, ga in ((m["home"], m["away"], True, m["home_goals"], m["away_goals"]),
                                        (m["away"], m["home"], False, m["away_goals"], m["home_goals"])):
            rows.append(dict(match_id=int(m["match_id"]), season=int(m["season"]), date=m["date"],
                             round=int(m["round"]), team=team, opp=opp, home=home,
                             goals=float(gf), goals_against=float(ga)))
    tm = pd.DataFrame(rows)
    tm = tm.merge(agg, on=["match_id", "team"], how="left")
    # An own goal is listed under the team of the player who put it in (with
    # goal = 0), so the opponent's own goals are this team's `og_scored` and
    # goals == goals_from_shots + og_scored in every match.
    against = agg.rename(columns={"team": "opp", "shots": "shots_against", "sot": "sot_against",
                                  "xg": "xg_against", "goals_from_shots": "gfs_against",
                                  "pens": "pens_against", "pen_xg": "pen_xg_against",
                                  "pen_goals": "pen_goals_against", "og_conceded": "og_scored"})
    tm = tm.merge(against, on=["match_id", "opp"], how="left")
    for c in ("shots", "sot", "xg", "goals_from_shots", "pens", "pen_xg", "pen_goals", "og_conceded",
              "shots_against", "sot_against", "xg_against", "gfs_against", "pens_against",
              "pen_xg_against", "pen_goals_against", "og_scored"):
        tm[c] = pd.to_numeric(tm[c], errors="coerce").fillna(0.0)
    # roster-derived: were there red cards, total minutes
    if not rosters.empty:
        rc = rosters.groupby(["match_id", "team"])["red"].sum().rename("reds").reset_index()
        tm = tm.merge(rc, on=["match_id", "team"], how="left")
        tm["reds"] = tm["reds"].fillna(0).astype(int)
    tm["npxg"] = tm["xg"] - tm["pen_xg"]
    tm["npxg_against"] = tm["xg_against"] - tm["pen_xg_against"]
    return game_weights(tm.sort_values(["date", "match_id"]).reset_index(drop=True), "team", recency)


# ---------------------------------------------------------------------------
# FPL API: squads, availability, per-gameweek scoring components
# ---------------------------------------------------------------------------

_FPL = "https://fantasy.premierleague.com/api"
FPL_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
FPL_STATUS = {"a": "Available", "d": "Doubtful", "i": "Injured", "s": "Suspended",
              "u": "Unavailable", "n": "Not in squad"}


@ttl_cache(maxsize=2)
def fpl_bootstrap() -> dict:
    return _get_json(f"{_FPL}/bootstrap-static/", referer="https://fantasy.premierleague.com/")


@ttl_cache(maxsize=2)
def fpl_fixtures() -> pd.DataFrame:
    fx = _get_json(f"{_FPL}/fixtures/", referer="https://fantasy.premierleague.com/")
    teams = {t["id"]: t["short_name"] for t in fpl_bootstrap()["teams"]}
    rows = [dict(fpl_fixture=f["id"], gw=f.get("event"), kickoff=pd.to_datetime(f.get("kickoff_time")),
                 home=teams.get(f["team_h"]), away=teams.get(f["team_a"]),
                 home_goals=f.get("team_h_score"), away_goals=f.get("team_a_score"),
                 finished=bool(f.get("finished")), started=bool(f.get("started")))
            for f in fx]
    return pd.DataFrame(rows)


def fpl_season() -> int:
    """Start year of the season the FPL API is currently serving."""
    ev = fpl_bootstrap()["events"]
    first = pd.to_datetime(ev[0]["deadline_time"])
    return int(first.year)


def fpl_current_gw() -> tuple[int | None, int | None]:
    """(current gameweek, next gameweek) ids."""
    ev = fpl_bootstrap()["events"]
    cur = next((e["id"] for e in ev if e.get("is_current")), None)
    nxt = next((e["id"] for e in ev if e.get("is_next")), None)
    return cur, nxt


def load_squads() -> pd.DataFrame:
    """Every player in the FPL game right now: team, position, status, chance
    of playing, news, price, penalty / set-piece order, season totals."""
    b = fpl_bootstrap()
    teams = {t["id"]: t["short_name"] for t in b["teams"]}
    rows = []
    for e in b["elements"]:
        rows.append(dict(
            fpl_id=int(e["id"]), fpl_code=int(e["code"]), team=teams.get(e["team"]),
            name=f"{e['first_name']} {e['second_name']}", web_name=e["web_name"],
            position=FPL_POS.get(e["element_type"], "?"),
            status=str(e.get("status", "a")), status_text=FPL_STATUS.get(e.get("status", "a"), ""),
            chance=(float(e["chance_of_playing_next_round"])
                    if e.get("chance_of_playing_next_round") is not None else np.nan),
            news=e.get("news") or "", price=float(e.get("now_cost", 0)) / 10.0,
            minutes=int(e.get("minutes", 0)), starts=int(e.get("starts", 0)),
            pen_order=e.get("penalties_order"), fk_order=e.get("direct_freekicks_order"),
            corner_order=e.get("corners_and_indirect_freekicks_order"),
            fpl_points=int(e.get("total_points", 0)),
            saves=int(e.get("saves", 0)), bonus=int(e.get("bonus", 0)),
            defcon=int(e.get("defensive_contribution", 0)),
        ))
    sq = pd.DataFrame(rows)
    # matches each club has played this season, for a season-to-date start rate
    try:
        fx = fpl_fixtures()
        fin = fx[fx["finished"]]
        played = pd.concat([fin["home"], fin["away"]]).value_counts()
        sq["team_played"] = sq["team"].map(played).fillna(0).astype(int)
    except Exception:
        sq["team_played"] = 0
    return sq


@ttl_cache(maxsize=4)
def fpl_gw_live(gw: int) -> pd.DataFrame:
    """Every player's FPL stats for one gameweek (current season)."""
    d = _get_json(f"{_FPL}/event/{int(gw)}/live/", referer="https://fantasy.premierleague.com/")
    rows = []
    for e in d["elements"]:
        s = dict(e["stats"])
        s["fpl_id"] = int(e["id"])
        s["gw"] = int(gw)
        s["fpl_fixture"] = e["explain"][0]["fixture"] if e.get("explain") else None
        rows.append(s)
    return pd.DataFrame(rows)


_FPL_NUM = ["minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
            "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus",
            "bps", "starts", "expected_goals", "expected_assists", "expected_goals_conceded",
            "clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution",
            "total_points"]

_VAASTAV = ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
            "{season}/gws/merged_gw.csv")
_VAASTAV_PLAYERS = ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
                    "{season}/players_raw.csv")


def _read_csv_url(url: str) -> pd.DataFrame:
    import io
    return pd.read_csv(io.StringIO(_get_text(url)))


def _season_str(season: int) -> str:
    return f"{int(season)}-{str(int(season) + 1)[-2:]}"


@ttl_cache(maxsize=6)
def load_fpl_gws(season: int) -> pd.DataFrame:
    """Per-player per-gameweek FPL lines for a season: current season from the
    live endpoint (one call per finished gameweek), past seasons from the
    vaastav archive. Columns are the FPL identifiers plus fpl_id, gw, team,
    position, name, kickoff."""
    b = fpl_bootstrap()
    if int(season) == fpl_season():
        gws = [e["id"] for e in b["events"] if e.get("finished") or e.get("is_current")]
        sq = load_squads()[["fpl_id", "team", "position", "name", "web_name", "fpl_code"]]
        frames = []
        for gw in gws:
            try:
                frames.append(fpl_gw_live(gw))
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        d = pd.concat(frames, ignore_index=True).merge(sq, on="fpl_id", how="left")
        fx = fpl_fixtures()[["fpl_fixture", "kickoff", "home", "away"]]
        d = d.merge(fx, on="fpl_fixture", how="left")
        d["opp"] = np.where(d["team"] == d["home"], d["away"], d["home"])
        d["was_home"] = d["team"] == d["home"]
    else:
        try:
            d = _read_csv_url(_VAASTAV.format(season=_season_str(season)))
            pr = _read_csv_url(_VAASTAV_PLAYERS.format(season=_season_str(season)))
        except Exception:
            return pd.DataFrame()
        d = d.rename(columns={"element": "fpl_id", "GW": "gw", "kickoff_time": "kickoff"})
        # the archive's ids are season-specific; `code` is the stable player key
        d["fpl_code"] = d["fpl_id"].map(pr.set_index("id")["code"])
        d["team"] = d["team"].map(team_code)
        d["position"] = d["position"].replace({"GKP": "GK"})
        d["was_home"] = d["was_home"].astype(bool)
        d["kickoff"] = pd.to_datetime(d["kickoff"], errors="coerce")
        d["opp"] = d.get("opponent_team")
    for c in _FPL_NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["season"] = int(season)
    d = d[d["minutes"] > 0].copy() if "minutes" in d.columns else d
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# football-data.co.uk: results with closing odds (the backtest comparator)
# ---------------------------------------------------------------------------

_FD_URL = "https://www.football-data.co.uk/mmz4281/{yy}/E0.csv"


@ttl_cache(maxsize=6)
def load_odds(seasons: tuple[int, ...]) -> pd.DataFrame:
    """One row per match: date, home, away, goals, shots, and the Pinnacle /
    market-average closing 1X2 and over-2.5 prices where published."""
    frames = []
    for s in seasons:
        yy = f"{str(int(s))[-2:]}{str(int(s) + 1)[-2:]}"
        try:
            txt = _get_text(_FD_URL.format(yy=yy))
            d = pd.read_csv(pd.io.common.StringIO(txt))
        except Exception:
            continue
        d = d.dropna(subset=["HomeTeam", "AwayTeam"])
        out = pd.DataFrame({
            "season": int(s),
            "date": pd.to_datetime(d["Date"], dayfirst=True, errors="coerce"),
            "home": d["HomeTeam"].map(team_code), "away": d["AwayTeam"].map(team_code),
            "home_goals": pd.to_numeric(d["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(d["FTAG"], errors="coerce"),
            "home_shots": pd.to_numeric(d.get("HS"), errors="coerce"),
            "away_shots": pd.to_numeric(d.get("AS"), errors="coerce"),
            "home_sot": pd.to_numeric(d.get("HST"), errors="coerce"),
            "away_sot": pd.to_numeric(d.get("AST"), errors="coerce"),
        })
        for k, cols in (("ps", ("PSCH", "PSCD", "PSCA")), ("avg", ("AvgCH", "AvgCD", "AvgCA")),
                        ("b365", ("B365CH", "B365CD", "B365CA"))):
            if all(c in d.columns for c in cols):
                for name, c in zip(("h", "d", "a"), cols):
                    out[f"{k}_{name}"] = pd.to_numeric(d[c], errors="coerce")
        for name, c in (("over25", "PC>2.5"), ("under25", "PC<2.5"),
                        ("avg_over25", "AvgC>2.5"), ("avg_under25", "AvgC<2.5")):
            if c in d.columns:
                out[name] = pd.to_numeric(d[c], errors="coerce")
        # Pinnacle closing is missing for part of some seasons; the market
        # average closing price stands in where it is
        for k in ("h", "d", "a"):
            if f"ps_{k}" in out.columns and f"avg_{k}" in out.columns:
                out[f"ps_{k}"] = out[f"ps_{k}"].fillna(out[f"avg_{k}"])
        if "over25" in out.columns and "avg_over25" in out.columns:
            out["over25"] = out["over25"].fillna(out["avg_over25"])
            out["under25"] = out["under25"].fillna(out["avg_under25"])
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def vig_free(prices: pd.DataFrame, cols=("ps_h", "ps_d", "ps_a")) -> pd.DataFrame:
    """Decimal prices → probabilities with the overround removed (proportional)."""
    inv = pd.DataFrame({c: 1.0 / prices[c] for c in cols if c in prices.columns})
    if inv.empty:
        return inv
    tot = inv.sum(axis=1)
    return inv.div(tot, axis=0)


# ---------------------------------------------------------------------------
# Player crosswalk: Understat ids ↔ FPL ids
# ---------------------------------------------------------------------------

def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower())
    return " ".join(s.split())


# Names that no normalisation reconciles (Understat name → FPL full name).
NAME_OVERRIDES = {
    "Emerson Palmieri": "Emerson Palmieri dos Santos",
}


def crosswalk(rosters: pd.DataFrame, squads: pd.DataFrame) -> pd.DataFrame:
    """Match Understat players to FPL players: exact normalised full name,
    then last-name-within-team, then a similarity fallback within team.
    Returns one row per Understat player with fpl_id (NaN if unmatched) and
    how it matched."""
    import difflib
    us = (rosters.sort_values("date").groupby("player_id")
                 .agg(player=("player", "last"), team=("team", "last"), minutes=("minutes", "sum"))
                 .reset_index())
    us["n"] = us["player"].map(lambda s: norm_name(NAME_OVERRIDES.get(s, s)))
    sq = squads.copy()
    sq["n_full"] = sq["name"].map(norm_name)
    sq["n_web"] = sq["web_name"].map(norm_name)
    by_full = {}
    for _, r in sq.iterrows():
        by_full.setdefault(r["n_full"], []).append(r)
    out = []
    for _, u in us.iterrows():
        cands = by_full.get(u["n"], [])
        how, pick = "", None
        if len(cands) == 1:
            pick, how = cands[0], "full"
        elif len(cands) > 1:
            same = [c for c in cands if c["team"] == u["team"]]
            pick, how = (same[0], "full+team") if same else (cands[0], "full")
        if pick is None:
            tm = sq[sq["team"] == u["team"]]
            toks = u["n"].split()
            last = toks[-1] if toks else ""
            # every token of the Understat name inside the FPL full name, or
            # the web name equal to the Understat name / its last token
            tok_hit = tm["n_full"].map(lambda f: all(t in f.split() for t in toks)) if toks else tm.iloc[0:0]
            hit = tm[(tm["n_web"] == last) | (tm["n_web"] == u["n"]) | tok_hit] if last else tm.iloc[0:0]
            if len(hit) == 1:
                pick, how = hit.iloc[0], "web/last+team"
            elif len(hit) > 1 and (hit["n_web"] == u["n"]).sum() == 1:
                pick, how = hit[hit["n_web"] == u["n"]].iloc[0], "web/last+team"
            elif len(tm):
                m = difflib.get_close_matches(u["n"], tm["n_full"].tolist(), n=1, cutoff=0.82)
                if m:
                    pick, how = tm[tm["n_full"] == m[0]].iloc[0], "fuzzy+team"
        out.append(dict(player_id=int(u["player_id"]), player=u["player"], team=u["team"],
                        fpl_id=(int(pick["fpl_id"]) if pick is not None else np.nan),
                        fpl_name=(pick["name"] if pick is not None else ""),
                        fpl_position=(pick["position"] if pick is not None else ""),
                        how=how))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

def season_choices(n_options: int = 5, n_default: int = 3,
                   today: _dt.date | None = None) -> tuple[list[int], list[int]]:
    """(options, defaults) for a season picker. A season starts in August;
    before August the current calendar year's season has not begun."""
    today = today or _dt.date.today()
    cur = today.year if today.month >= 8 else today.year - 1
    options = list(range(cur, cur - n_options, -1))
    return options, options[:n_default]


# ---------------------------------------------------------------------------
# Shared stats helpers
# ---------------------------------------------------------------------------

def wmean(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def wstd(x, w, fallback):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return fallback
    mu = np.average(x[m], weights=w[m])
    var = np.average((x[m] - mu) ** 2, weights=w[m]) * (m.sum() / (m.sum() - 1))
    sd = float(np.sqrt(var))
    return sd if sd > 1e-6 else fallback


def between_sd(x, w, sampling_var, floor: float, fallback: float | None = None) -> float:
    """Match-to-match spread of a per-match rate net of its sampling noise
    (method of moments), floored. The simulator draws the sampling noise
    itself; feeding the raw SD in would count it twice."""
    x = np.asarray(x, float); w = np.asarray(w, float); sv = np.asarray(sampling_var, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0) & np.isfinite(sv)
    if m.sum() < 3:
        return float(fallback if fallback is not None else floor)
    obs = wstd(x[m], w[m], 0.0) ** 2
    noise = float(np.average(sv[m], weights=w[m]))
    return float(max(np.sqrt(max(obs - noise, 0.0)), floor))


def beta_params(mean, sd):
    mean = float(np.clip(mean, 1e-3, 1 - 1e-3))
    max_sd = np.sqrt(mean * (1 - mean)) * 0.99
    sd = float(np.clip(sd, 1e-4, max_sd))
    k = mean * (1 - mean) / (sd ** 2) - 1
    return max(mean * k, 0.1), max((1 - mean) * k, 0.1)


def gamma_params(mean, sd):
    """Gamma(shape, scale) matched to a mean and sd."""
    mean = max(float(mean), 1e-6); sd = max(float(sd), 1e-6)
    shape = (mean / sd) ** 2
    return shape, mean / shape


def nb_draw(rng, mean, var, size=None):
    """Negative-binomial draw with the given mean and variance (Poisson if
    var <= mean). `mean`/`var` may be arrays."""
    mean = np.asarray(mean, float); var = np.asarray(var, float)
    mean = np.clip(mean, 1e-9, None)
    over = var > mean * 1.001
    out = rng.poisson(mean, size=size if size is not None else mean.shape)
    if np.any(over):
        m = np.where(over, mean, 1.0); v = np.where(over, var, 2.0)
        r = m ** 2 / (v - m)
        p = r / (r + m)
        nb = rng.negative_binomial(np.clip(r, 1e-6, None), np.clip(p, 1e-9, 1 - 1e-9))
        out = np.where(over, nb, out)
    return out


def shrink(ratio, strength):
    """Pull a ratio toward 1 (no effect) by (1 - strength)."""
    return 1.0 + strength * (ratio - 1.0)


def decimal_odds(p: float) -> str:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return f"{1.0 / p:.2f}"


def american(p: float) -> str:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    dec = 1.0 / p
    return f"+{round((dec - 1) * 100)}" if dec >= 2.0 else f"-{round(100 / (dec - 1))}"


if __name__ == "__main__":
    import sys
    seasons = tuple(int(s) for s in sys.argv[1:]) or (2024, 2025, 2026)
    for s in seasons:
        n = sync_season(s, progress=lambda i, n: print(f"\r  {s}: {i}/{n}", end=""))
        print(f"\n{s}: fetched {n} new matches")
    tm = load_team_matches(seasons)
    print(f"{len(tm)} team-matches; shots/team-match {tm['shots'].mean():.2f}, "
          f"xG {tm['xg'].mean():.3f}, goals {tm['goals'].mean():.3f}")
