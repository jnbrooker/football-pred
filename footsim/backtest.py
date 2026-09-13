"""
footsim/backtest.py — out-of-sample scoring of every model (roadmap §3.10).

For each match date of a scored season everything is refitted on matches
played BEFORE that date (the prior seasons plus the earlier part of the
season), then that date's fixtures are predicted. Nothing the model sees
was known after the fact.

Team layer (`team_backtest`): expected goals each way → 1X2 probabilities
(independent Poissons), expected total and goal difference, expected shots.
Scored on 1X2 log-loss, ranked probability score and Brier, goal-difference
and total RMSE, over-2.5 log-loss, shots RMSE — and, on the same matches,
against the Pinnacle closing prices from football-data.co.uk with the
overround removed. The closing line is the benchmark any self-contained
model should expect to trail; the question is by how much.

Player layer (`player_backtest`): shots, shots on target, goals, assists,
key passes and (for keepers) saves for every player who appeared with at
least MIN_PRIOR_APPS prior appearances, simulated from history-only rosters
(the FPL squad list cannot be replayed for past dates) CONDITIONAL ON HIS
ACTUAL ROLE — started or came off the bench — so the volume and quality
models are scored apart from the lineup model. The lineup model is scored
separately: P(start) against who actually started (Brier, log-loss). Each
stat is a full simulated distribution scored on MAE / bias / RMSE, CRPS,
10–90 coverage and, for counts, the Brier of P(≥ 1). The naive baseline is
the player's trailing recency-weighted average of the stat in the same role.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data as D
from . import players as P
from . import roster as RO
from . import teams as T

MIN_PRIOR_APPS = 5
MIN_EXP_SHOTS = 0.6          # prop-worthy volume for the shot stats
PLAYER_STATS = ("shots", "sot", "goals", "assists", "key_passes", "saves")
DEFAULT_N_SIMS = 3000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def window(score_seasons, n_prior: int) -> tuple[int, ...]:
    lo = min(score_seasons) - n_prior
    return tuple(range(lo, max(score_seasons) + 1))


def _before(df: pd.DataFrame, date) -> pd.DataFrame:
    return df[df["date"] < date]


def _crps(samples: np.ndarray, y: float) -> float:
    s = np.sort(samples.astype(float))
    n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :])) if n <= 400
                 else np.mean(np.abs(s - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / (n * n))


def _rps(p: np.ndarray, outcome: int) -> float:
    """Ranked probability score for a 3-way outcome (0 = home, 1 = draw, 2 = away)."""
    o = np.zeros(3); o[outcome] = 1.0
    return float(np.sum((np.cumsum(p) - np.cumsum(o)) ** 2) / 2.0)


# ---------------------------------------------------------------------------
# Team layer
# ---------------------------------------------------------------------------

def team_backtest(score_seasons, n_prior: int = 2, recency: D.Recency = D.RECENCY_DEFAULT,
                  progress=None) -> pd.DataFrame:
    seasons = window(score_seasons, n_prior)
    tm_all = D.load_team_matches(seasons)
    fx = D.load_fixtures(tuple(score_seasons))
    fx = fx[fx["played"]].sort_values("date")
    odds = D.load_odds(tuple(score_seasons))
    rows = []
    dates = sorted(fx["date"].dt.normalize().unique())
    for i, d in enumerate(dates):
        prior = tm_all[tm_all["date"] < d]
        if prior["match_id"].nunique() < 60:
            continue
        prior = D.game_weights(prior, "team", recency)
        day = fx[fx["date"].dt.normalize() == d]
        try:
            r = T.team_ratings(prior, season=int(day["season"].iloc[0]),
                               extra_teams=set(day["home"]) | set(day["away"]))
        except ValueError:
            continue
        for _, m in day.iterrows():
            if m["home"] not in r["teams"] or m["away"] not in r["teams"]:
                continue
            e = T.expected_goals(r, m["home"], m["away"], home="a")
            rows.append(dict(
                season=int(m["season"]), round=int(m["round"]), date=m["date"], match_id=int(m["match_id"]),
                home=m["home"], away=m["away"], hg=float(m["home_goals"]), ag=float(m["away_goals"]),
                exp_hg=e["goals_a"], exp_ag=e["goals_b"], exp_shots_h=e["shots_a"], exp_shots_a=e["shots_b"],
                exp_xg_h=e["a"]["npxg"], exp_xg_a=e["b"]["npxg"],
                p_home=e["win_a"], p_draw=e["draw"], p_away=e["win_b"], p_over25=e["over25"],
            ))
        if progress:
            progress(i + 1, len(dates))
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    bt["margin"] = bt["hg"] - bt["ag"]; bt["exp_margin"] = bt["exp_hg"] - bt["exp_ag"]
    bt["total"] = bt["hg"] + bt["ag"]; bt["exp_total"] = bt["exp_hg"] + bt["exp_ag"]
    bt["outcome"] = np.where(bt["margin"] > 0, 0, np.where(bt["margin"] == 0, 1, 2))
    # actual shots
    tmr = tm_all.set_index(["match_id", "team"])["shots"]
    bt["shots_h"] = [tmr.get((mid, h), np.nan) for mid, h in zip(bt["match_id"], bt["home"])]
    bt["shots_a"] = [tmr.get((mid, a), np.nan) for mid, a in zip(bt["match_id"], bt["away"])]
    tmx = tm_all.set_index(["match_id", "team"])["npxg"]
    bt["xg_h"] = [tmx.get((mid, h), np.nan) for mid, h in zip(bt["match_id"], bt["home"])]
    bt["xg_a"] = [tmx.get((mid, a), np.nan) for mid, a in zip(bt["match_id"], bt["away"])]
    # closing odds on the same matches
    if not odds.empty and "ps_h" in odds.columns:
        o = odds.copy()
        o["key"] = o["date"].dt.normalize().astype(str) + o["home"] + o["away"]
        vf = D.vig_free(o)
        o["mk_home"], o["mk_draw"], o["mk_away"] = vf["ps_h"], vf["ps_d"], vf["ps_a"]
        if "over25" in o.columns:
            inv = 1.0 / o["over25"] + 1.0 / o["under25"]
            o["mk_over25"] = (1.0 / o["over25"]) / inv
        bt["key"] = bt["date"].dt.normalize().astype(str) + bt["home"] + bt["away"]
        keep = ["key", "mk_home", "mk_draw", "mk_away"] + (["mk_over25"] if "mk_over25" in o.columns else [])
        bt = bt.merge(o[keep].drop_duplicates("key"), on="key", how="left").drop(columns="key")
    return bt


def team_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """Model vs closing market vs base rates, on the matches where all exist."""
    out = {}
    base = np.array([(bt["outcome"] == k).mean() for k in range(3)])
    def score(ph, pd_, pa, label, d):
        p = np.clip(np.stack([ph, pd_, pa], axis=1), 1e-6, 1)
        p = p / p.sum(axis=1, keepdims=True)
        o = d["outcome"].values
        ll = -np.log(p[np.arange(len(o)), o]).mean()
        rps = np.mean([_rps(p[i], o[i]) for i in range(len(o))])
        onehot = np.eye(3)[o]
        brier = np.mean(np.sum((p - onehot) ** 2, axis=1))
        pick = p.argmax(axis=1)
        return dict(n=len(d), log_loss=ll, rps=rps, brier=brier, pick_rate=float((pick == o).mean()))
    have_mk = bt["mk_home"].notna() if "mk_home" in bt.columns else pd.Series(False, index=bt.index)
    d = bt[have_mk] if have_mk.any() else bt
    out["model"] = score(d["p_home"], d["p_draw"], d["p_away"], "model", d)
    out["model"].update(gd_rmse=float(np.sqrt(((d["margin"] - d["exp_margin"]) ** 2).mean())),
                        total_rmse=float(np.sqrt(((d["total"] - d["exp_total"]) ** 2).mean())),
                        gd_corr=float(np.corrcoef(d["margin"], d["exp_margin"])[0, 1]),
                        shots_rmse=float(np.sqrt(np.nanmean((d["shots_h"] - d["exp_shots_h"]) ** 2
                                                            + (d["shots_a"] - d["exp_shots_a"]) ** 2) / 2)),
                        over25_ll=float(-np.mean(np.where(d["total"] > 2.5, np.log(np.clip(d["p_over25"], 1e-6, 1)),
                                                          np.log(np.clip(1 - d["p_over25"], 1e-6, 1))))))
    if have_mk.any():
        out["closing"] = score(d["mk_home"], d["mk_draw"], d["mk_away"], "closing", d)
        if "mk_over25" in d.columns and d["mk_over25"].notna().any():
            dd = d[d["mk_over25"].notna()]
            out["closing"]["over25_ll"] = float(-np.mean(np.where(dd["total"] > 2.5, np.log(dd["mk_over25"]),
                                                                  np.log(1 - dd["mk_over25"]))))
    out["base rates"] = score(np.full(len(d), base[0]), np.full(len(d), base[1]), np.full(len(d), base[2]), "base", d)
    return pd.DataFrame(out).T


def calibration(bt: pd.DataFrame, col: str = "p_home", outcome: int = 0, bins: int = 8) -> pd.DataFrame:
    d = bt.copy()
    d["bin"] = pd.qcut(d[col], bins, duplicates="drop")
    return (d.groupby("bin", observed=True)
              .agg(n=(col, "size"), predicted=(col, "mean"), actual=("outcome", lambda s: (s == outcome).mean()))
              .reset_index())


# ---------------------------------------------------------------------------
# Player layer
# ---------------------------------------------------------------------------

def _ctx_before(live_all: dict, date, recency: D.Recency) -> dict:
    """A context (live + ratings + league rates) from data before `date`."""
    rosters = D.game_weights(_before(live_all["rosters"], date), "team", recency)
    shots = D.game_weights(_before(live_all["shots"], date), "team", recency)
    tm = D.game_weights(_before(live_all["tm"], date), "team", recency)
    live = dict(seasons=live_all["seasons"], recency=recency, rosters=rosters, shots=shots, tm=tm,
                squads=None, xw=None, fpl_hist=pd.DataFrame(),
                groups=RO.player_groups(rosters), team_vol=RO.team_volumes(tm),
                shot_quality=RO.player_shot_quality(shots))
    ratings = T.team_ratings(tm)
    return dict(live=live, ratings=ratings, seasons=live_all["seasons"], recency=recency,
                lg=RO.league_rates(live))


def _naive(h: pd.DataFrame, stat: str, started: bool) -> float:
    """Trailing weighted average of the stat in the same role."""
    col = {"shots": "shots", "sot": "sot", "goals": "goals", "assists": "assists",
           "key_passes": "key_passes", "saves": "saves"}[stat]
    same = h[h["started"] == int(started)]
    if len(same) < 2:
        same = h
    if col not in same.columns:
        return np.nan
    return D.wmean(same[col], same["w"])


def player_backtest(score_seasons, stats=PLAYER_STATS, n_prior: int = 2,
                    recency: D.Recency = D.RECENCY_DEFAULT, n_sims: int = DEFAULT_N_SIMS,
                    rounds: tuple[int, ...] | None = None, max_matches: int | None = None,
                    seed: int = 1, progress=None) -> pd.DataFrame:
    seasons = window(score_seasons, n_prior)
    live_all = RO.load_live(seasons, recency, use_fpl=False)
    fx = D.load_fixtures(tuple(score_seasons))
    fx = fx[fx["played"]].sort_values("date")
    if rounds is not None:
        fx = fx[fx["round"].isin(rounds)]
    if max_matches:
        fx = fx.head(int(max_matches))
    rosters_all = live_all["rosters"]
    shots_all = live_all["shots"]
    # per-player per-match on-target and saves (actuals)
    sot_pm = shots_all.groupby(["match_id", "player_id"])["on_target"].sum()
    rng = np.random.default_rng(seed)
    rows, lineup_rows = [], []
    dates = sorted(fx["date"].dt.normalize().unique())
    for i, d in enumerate(dates):
        try:
            ctx = _ctx_before(live_all, d, recency)
        except ValueError:
            continue
        day = fx[fx["date"].dt.normalize() == d]
        r = ctx["ratings"]
        roster_cache = {}
        for _, m in day.iterrows():
            if m["home"] not in r["teams"] or m["away"] not in r["teams"]:
                continue
            actual = rosters_all[rosters_all["match_id"] == m["match_id"]]
            for team, opp, home in ((m["home"], m["away"], True), (m["away"], m["home"], False)):
                if team not in roster_cache:
                    try:
                        roster_cache[team] = RO.build_roster(ctx["live"], team, use_fpl=False, lg=ctx["lg"])
                    except ValueError:
                        roster_cache[team] = None
                ros = roster_cache[team]
                if ros is None:
                    continue
                mc = P.match_context(ctx, team, opp, home)
                opp_goals = float(m["home_goals"] if not home else m["away_goals"])
                # the lineup model, over the WHOLE history-only squad
                played_ids = actual[actual["team"] == team].set_index("player_id")["started"]
                for _, rr in ros.iterrows():
                    lineup_rows.append(dict(match_id=int(m["match_id"]), team=team, player_id=int(rr["player_id"]),
                                            p_start=float(rr["p_start"]),
                                            started=int(played_ids.get(int(rr["player_id"]), 0)),
                                            played=int(int(rr["player_id"]) in played_ids.index)))
                for _, a in actual[actual["team"] == team].iterrows():
                    pid = int(a["player_id"])
                    row = ros[ros["player_id"] == pid]
                    if row.empty or int(row.iloc[0]["games"]) < MIN_PRIOR_APPS:
                        continue
                    row = row.iloc[0]
                    started = bool(a["started"])
                    sim = P.simulate(row, mc, n_sims=n_sims, seed=int(rng.integers(1 << 31)),
                                     force_start=started)
                    h = ctx["live"]["rosters"][ctx["live"]["rosters"]["player_id"] == pid]
                    is_gk = row["group"] == "GK"
                    for stat in stats:
                        if stat == "saves" and not is_gk:
                            continue
                        if stat != "saves" and is_gk:
                            continue
                        if stat in ("shots", "sot", "goals") and row["exp_shots"] < MIN_EXP_SHOTS * (1 if started else 0.3):
                            continue
                        y = (float(a[stat]) if stat in a and stat != "sot" else
                             float(sot_pm.get((int(m["match_id"]), pid), 0.0)) if stat == "sot" else np.nan)
                        if stat == "saves":
                            opp_sot = float(sot_pm[sot_pm.index.get_level_values(0) == int(m["match_id"])]
                                            .loc[int(m["match_id"])].reindex(
                                                actual[actual["team"] == opp]["player_id"]).fillna(0).sum())
                            y = max(opp_sot - opp_goals, 0.0)
                        s = sim[stat]
                        rows.append(dict(
                            season=int(m["season"]), round=int(m["round"]), date=m["date"],
                            match_id=int(m["match_id"]), team=team, opp=opp, home=home,
                            player_id=pid, player=row["name"], group=row["group"], started=started,
                            minutes=int(a["minutes"]), stat=stat, actual=y,
                            mean=float(s.mean()), median=float(np.median(s)),
                            p10=float(np.percentile(s, 10)), p90=float(np.percentile(s, 90)),
                            p_ge1=float((s >= 1).mean()), crps=_crps(s[:400], y),
                            naive=_naive(h, stat, started), exp_minutes=float(sim["minutes"].mean()),
                            p_start=float(row["p_start"]),
                        ))
        if progress:
            progress(i + 1, len(dates))
    pb = pd.DataFrame(rows)
    pb.attrs["lineup"] = pd.DataFrame(lineup_rows)
    return pb


def player_metrics(pb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stat, g in pb.groupby("stat"):
        err = g["actual"] - g["mean"]
        nv = g["actual"] - g["naive"]
        rows.append(dict(
            stat=stat, n=len(g), mae=float(err.abs().mean()), bias=float(err.mean()),
            rmse=float(np.sqrt((err ** 2).mean())), naive_mae=float(nv.abs().mean()),
            median_mae=float((g["actual"] - g["median"]).abs().mean()),
            crps=float(g["crps"].mean()),
            cover_10_90=float(((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()),
            brier_ge1=float(((g["actual"] >= 1).astype(float) - g["p_ge1"]).pow(2).mean()),
            base_ge1=float((g["actual"] >= 1).mean()),
            corr=float(np.corrcoef(g["actual"], g["mean"])[0, 1]) if len(g) > 2 else np.nan,
        ))
    return pd.DataFrame(rows).set_index("stat")


def lineup_metrics(pb: pd.DataFrame) -> dict:
    """The lineup model: P(start) vs who started, over every player on the
    history-only squad (starters, subs and absentees alike)."""
    d = pb.attrs.get("lineup") if isinstance(pb.attrs.get("lineup"), pd.DataFrame) else None
    if d is None or d.empty:
        d = pb.drop_duplicates(["match_id", "player_id"])
    y = d["started"].astype(float); p = np.clip(d["p_start"], 1e-4, 1 - 1e-4)
    return dict(n=len(d), brier=float(((y - p) ** 2).mean()),
                log_loss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                base_brier=float(((y - y.mean()) ** 2).mean()),
                start_rate=float(y.mean()), mean_p=float(p.mean()))


def quintile_bias(pb: pd.DataFrame, stat: str) -> pd.DataFrame:
    g = pb[pb["stat"] == stat].copy()
    g["q"] = pd.qcut(g["mean"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    return g.groupby("q", observed=True).agg(n=("actual", "size"), projected=("mean", "mean"),
                                             actual=("actual", "mean")).assign(bias=lambda x: x["actual"] - x["projected"])


if __name__ == "__main__":
    seasons = [int(s) for s in sys.argv[1:]] or [2025]
    print(f"Team layer, out of sample, {seasons}:")
    bt = team_backtest(seasons, progress=lambda i, n: print(f"\r  {i}/{n} dates", end=""))
    print()
    print(team_metrics(bt).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nPlayer layer (rounds 5-38, 3000 sims):")
    pb = player_backtest(seasons, rounds=tuple(range(5, 39)), progress=lambda i, n: print(f"\r  {i}/{n} dates", end=""))
    print()
    print(player_metrics(pb).to_string(float_format=lambda x: f"{x:.3f}"))
    print("lineup:", lineup_metrics(pb))


# ---------------------------------------------------------------------------
# Availability: fit the coefficient on hindsight lineups
# ---------------------------------------------------------------------------

def availability_fit(score_seasons, n_prior: int = 2, recency: D.Recency = D.RECENCY_DEFAULT,
                     coefs=(0.0, 0.5, 1.0, 1.5, 2.0), progress=None) -> dict:
    """For every match, the attack availability factor of each side given
    who ACTUALLY played (hindsight — the FPL status feed cannot be replayed),
    relative to the history-only squad; then the Poisson log-likelihood of
    the goals and the 1X2 log-loss for each candidate coefficient on the
    log-factor. An upper bound on what a pre-match availability feed can
    add, since real lineups are only partly known before kickoff."""
    seasons = window(score_seasons, n_prior)
    live_all = RO.load_live(seasons, recency, use_fpl=False)
    fx = D.load_fixtures(tuple(score_seasons))
    fx = fx[fx["played"]].sort_values("date")
    rosters_all = live_all["rosters"]
    rows = []
    dates = sorted(fx["date"].dt.normalize().unique())
    for i, d in enumerate(dates):
        try:
            ctx = _ctx_before(live_all, d, recency)
        except ValueError:
            continue
        r = ctx["ratings"]
        day = fx[fx["date"].dt.normalize() == d]
        ung = {}
        for _, m in day.iterrows():
            if m["home"] not in r["teams"] or m["away"] not in r["teams"]:
                continue
            played = rosters_all[rosters_all["match_id"] == m["match_id"]]
            fac = {}
            for team in (m["home"], m["away"]):
                try:
                    if team not in ung:
                        ung[team] = RO.build_roster(ctx["live"], team, use_fpl=False, lg=ctx["lg"])
                    u = ung[team]
                    out_ids = set(u["player_id"]) - set(played[played["team"] == team]["player_id"])
                    fac[team] = RO.availability_factor(ctx["live"], team, ctx["lg"], ruled_out=out_ids, ungated=u)
                except ValueError:
                    fac[team] = dict(volume=1.0, quality=1.0, xg=1.0)
            a = T.matchup(r, m["home"], m["away"], True); b = T.matchup(r, m["away"], m["home"], False)
            rows.append(dict(match_id=int(m["match_id"]), home=m["home"], away=m["away"],
                             hg=float(m["home_goals"]), ag=float(m["away_goals"]),
                             exp_hg=a["goals"], exp_ag=b["goals"], np_hg=a["np_goals"], np_ag=b["np_goals"],
                             fac_h=fac[m["home"]]["xg"], fac_a=fac[m["away"]]["xg"]))
        if progress:
            progress(i + 1, len(dates))
    bt = pd.DataFrame(rows)
    out = dict(n=len(bt), coefs={})
    for c in coefs:
        mh = (bt["exp_hg"] - bt["np_hg"]) + bt["np_hg"] * bt["fac_h"] ** c
        ma = (bt["exp_ag"] - bt["np_ag"]) + bt["np_ag"] * bt["fac_a"] ** c
        ll_pois = float((bt["hg"] * np.log(mh) - mh + bt["ag"] * np.log(ma) - ma).mean())
        p = np.array([list(T.outcome_probs(x, y).values())[:3] for x, y in zip(mh, ma)])
        o = np.where(bt["hg"] > bt["ag"], 0, np.where(bt["hg"] == bt["ag"], 1, 2))
        ll_1x2 = float(-np.log(np.clip(p[np.arange(len(o)), o], 1e-6, 1)).mean())
        out["coefs"][c] = dict(poisson_ll=ll_pois, log_loss_1x2=ll_1x2)
    out["factor_sd"] = float(np.log(pd.concat([bt["fac_h"], bt["fac_a"]])).std())
    out["table"] = bt
    return out
