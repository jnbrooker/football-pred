"""
footsim/calibrate.py — does the engine reproduce real matches?

Two checks, both run from the command line:

  python -m footsim.calibrate moments 2025
      Simulate every played fixture of a season at team level with ratings
      fitted on the window, and compare the simulated distribution of
      results, goals and shots with the real one: home / draw / away rates,
      goal-difference and total sd, team-goals var/mean and the correlation
      between the two sides' goals and shots, the shot-count dispersion, and
      the within-matchup elasticity of goals to shots (0.46 in the data —
      the engine must produce it from the game-state feedback, not assume
      it). These are the targets DOMINANCE_SD, TEAM_WOBBLE_SD and the
      LEAD_BETA_* constants in `game.py` are set against.

  python -m footsim.calibrate refit 2024 2025
      Out-of-sample refit report for the constants the backtest can score:
      the ratings' calibration slope (adjust the *_PRIOR_N shrinkage toward
      slope 1) and home advantage. Prints current beside refitted.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data as D
from . import game as G
from . import teams as T


def engine_moments(season: int, n_sims: int = 400, seed: int = 1,
                   window: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Simulated vs real moments over one season's fixtures."""
    window = window or (season - 1, season)
    tm = D.load_team_matches(tuple(window))
    r = T.team_ratings(tm[tm["season"] <= season])
    fx = D.load_fixtures((season,))
    fx = fx[fx["played"] & fx["home"].isin(r["teams"]) & fx["away"].isin(r["teams"])]
    rng = np.random.default_rng(seed)
    rows = []
    for _, m in fx.iterrows():
        mix_a = T.matchup(r, m["home"], m["away"], True)
        mix_b = T.matchup(r, m["away"], m["home"], False)
        s = G._score_sides(rng, mix_a, mix_b, n_sims)
        rows.append(pd.DataFrame(dict(
            match_id=m["match_id"], sim=np.arange(n_sims),
            home=m["home"], away=m["away"],
            ga=s["a"]["points"], gb=s["b"]["points"],
            sa=s["a"]["shots_total"], sb=s["b"]["shots_total"],
            xa=mix_a["goals"], xb=mix_b["goals"], xsa=mix_a["shots"], xsb=mix_b["shots"])))
    sim = pd.concat(rows, ignore_index=True)
    real = fx.copy()
    tmr = tm[tm["season"] == season]
    real_shots = tmr.set_index(["match_id", "team"])["shots"]
    real["sa"] = [real_shots.get((mid, h), np.nan) for mid, h in zip(real["match_id"], real["home"])]
    real["sb"] = [real_shots.get((mid, a), np.nan) for mid, a in zip(real["match_id"], real["away"])]
    real = real.rename(columns={"home_goals": "ga", "away_goals": "gb"})
    # expected values for the real matches (for residual moments)
    exp = sim.groupby("match_id")[["xa", "xb", "xsa", "xsb"]].first()
    real = real.join(exp, on="match_id")

    def moments(d: pd.DataFrame) -> dict:
        gd = d["ga"] - d["gb"]
        tot = d["ga"] + d["gb"]
        ra, rb = d["ga"] - d["xa"], d["gb"] - d["xb"]
        sra, srb = d["sa"] - d["xsa"], d["sb"] - d["xsb"]
        # elasticity: within-matchup log goals on log shots (pooled residuals)
        ok = (d["sa"] > 0) & (d["sb"] > 0)
        ls = np.log(pd.concat([d.loc[ok, "sa"] / d.loc[ok, "xsa"], d.loc[ok, "sb"] / d.loc[ok, "xsb"]]))
        gg = pd.concat([d.loc[ok, "ga"] - d.loc[ok, "xa"], d.loc[ok, "gb"] - d.loc[ok, "xb"]])
        slope = np.polyfit(ls, gg, 1)[0] / float(pd.concat([d["xa"], d["xb"]]).mean())
        return dict(
            home=float((gd > 0).mean()), draw=float((gd == 0).mean()), away=float((gd < 0).mean()),
            goals_mean=float(pd.concat([d["ga"], d["gb"]]).mean()),
            gd_sd=float(gd.std()), total_sd=float(tot.std()),
            team_goals_var_mean=float(pd.concat([ra, rb]).var() / pd.concat([d["xa"], d["xb"]]).mean()),
            goals_corr=float(np.corrcoef(ra, rb)[0, 1]),
            shots_mean=float(pd.concat([d["sa"], d["sb"]]).mean()),
            shots_resid_sd=float(pd.concat([sra, srb]).std()),
            shots_var_mean=float(pd.concat([sra, srb]).var() / pd.concat([d["xsa"], d["xsb"]]).mean()),
            shots_corr=float(np.corrcoef(sra.fillna(0), srb.fillna(0))[0, 1]),
            elasticity=float(slope),
            over25=float((tot > 2.5).mean()), btts=float(((d["ga"] > 0) & (d["gb"] > 0)).mean()),
        )
    return pd.DataFrame({"engine": moments(sim), "real": moments(real.dropna(subset=["sa", "sb"]))})


def refit(score_seasons, n_prior: int = 2) -> dict:
    """Calibration slope of the out-of-sample expected goal difference and
    the home advantage implied by the residuals (see `backtest`)."""
    from . import backtest as B
    bt = B.team_backtest(tuple(score_seasons), n_prior=n_prior)
    x = bt["exp_margin"]; y = bt["margin"]
    slope = float(np.polyfit(x, y, 1)[0])
    resid_home = float((y - x).mean())
    return dict(n=len(bt), slope=slope, resid_home_goals=resid_home,
                gd_rmse=float(np.sqrt(((y - x) ** 2).mean())),
                current=dict(VOL_PRIOR_N=T.VOL_PRIOR_N, QUAL_PRIOR_N=T.QUAL_PRIOR_N,
                             FIN_PRIOR_XG=T.FIN_PRIOR_XG))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "moments"
    seasons = [int(s) for s in sys.argv[2:]] or [2025]
    if what == "moments":
        for s in seasons:
            print(f"\nEngine vs real, {s}/{str(s + 1)[-2:]}:")
            print(engine_moments(s).to_string(float_format=lambda x: f"{x:7.3f}"))
    else:
        r = refit(seasons)
        print(f"team layer out of sample, {seasons}: n={r['n']}, GD RMSE {r['gd_rmse']:.3f}, "
              f"calibration slope {r['slope']:.3f} (1 = right amount of shrinkage; > 1 = too much), "
              f"unexplained home margin {r['resid_home_goals']:+.3f} goals")
