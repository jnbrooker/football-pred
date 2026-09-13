# Football Monte Carlo Model Suite

A self-contained Premier League match and player-prop model, as a multi-page
Streamlit app — the football port of the NFL suite in `recyards`. Pick a
fixture and it simulates it thousands of times from distributions fitted to
real shot-by-shot data — who is on the pitch and for how long, how many
shots each side takes, who takes them, whether they are on target, whether
they go in, who assisted — and reports the score distribution, 1X2 and goal
lines, a projected box score (shots, shots on target, goals, assists, key
passes, saves, cards) and FPL points, all from the same simulations. Nothing
anchors to a bookmaker; the closing market appears only in the backtest, as
the benchmark.

Every model is scored **out of sample** by its own backtest harness, and
every constant that could be fitted was fitted on those residuals.

## Pages

| Page | What it does |
|---|---|
| Shots | minutes → team shots → the player's share; the reference chain |
| Shots on Target | each shot on target at the player's rate, moved by the opponent |
| Goals | shots × conversion (xG per shot × finishing), penalties by taker; anytime-scorer odds |
| Assists | key passes → converted, scaled by the team's expected goals |
| Goalkeeper Saves | the opponent's on-target shots minus goals conceded; clean-sheet odds |
| Cards | yellows per 90, regressed toward the position |
| Team Strength | attack and defence solved from shots, xG per shot and finishing; home advantage |
| Match Simulation | a fixture: score distribution, 1X2, goal lines, both box scores |
| Fantasy Projections | the gameweek's slate scored on FPL rules per simulation; floor, ceiling, breakdown |
| Backtest | every model scored date by date out of sample, vs the closing market and a trailing average |

Pages 1–6 pick players from the **live FPL squad** with availability tags
(injured / suspended / doubtful with a percentage). Each has a **Fixture /
Season** view: *Fixture* picks a gameweek fixture and takes the opponent,
venue and the team's expected shots from the match model; *Season* is the
player's typical match against any opponent. A *Lineup* control switches
between his modelled start probability and "assume he starts" — the number
to compare with a prop line, which is usually void if he does not play.

## How the match model works

1. **Team strength** (`footsim/teams.py`) — three multiplicative ratings per
   team, solved jointly by iterative proportional fitting so a schedule of
   tough defences is not held against an attack: **shot volume**, **xG per
   shot** and **finishing** (goals per xG), for and against, each shrunk
   toward league by the matches behind it, recent matches weighted more.
   Home advantage is fitted on the shot rate — home sides take 20% more
   shots at the same quality — never as a flat goals bonus.
2. **Squads** (`footsim/roster.py`) — the FPL squad list with its
   availability feed is the depth chart. Each player's start probability
   comes from his start rate in his club's matches (a match he missed counts
   as a non-start), gated by availability, fitted so each team's expected
   starters sum to eleven; his minutes from his own substitution pattern;
   his **share of the team's shots and key passes per 90 on the pitch** from
   his history, regressed toward his position group.
3. **Match engine** (`footsim/game.py`) — six 15-minute segments. Each
   side's shots are Poisson at the matchup rate, with a shared *dominance*
   draw (the two sides' shot counts are negatively correlated) and a
   game-state factor: a side ahead of where the ratings expected it shoots
   less but better. Each shot is on target and a goal at the matchup rates.
   Penalties and own goals are separate small events.
4. **Allocation** — shots are split across the players on the pitch by
   minutes × share; on-target shots and goals are handed out among them by
   each player's rates, capped so goals ≤ on target ≤ shots; assists are
   drawn from teammates by key-pass weight; the keeper's saves are the
   opponent's on-target shots minus goals. Every identity holds in every
   simulation.

Recency: every feed carries one weight column — a season curve times a
per-team match decay — and every rate, share and volume is a weighted
estimate. The sidebar's *How much to trust this season* control sets both.

## What the data said (2023–26 EPL, `ROADMAP.md` §6)

- Home advantage is volume: shots ×1.21, xG per shot ×1.00.
- Trailing sides shoot more (×1.13) and worse (0.110 xG/shot); leading sides
  shoot less (×0.87) and better (0.147). Goals per minute barely move.
- Conditional on strength, the two sides' goals are independent Poissons
  (corr −0.015, var/mean 0.92) while shots are overdispersed (var/mean 1.95,
  corr −0.16) — the engine is calibrated to both.
- Goals rise only 0.3–0.5× as fast as shots within a matchup; the engine
  reproduces it from the state feedback rather than assuming it.
- Player shots/90 (year-to-year r = 0.92) and key passes/90 (0.87) are
  stable; xG per shot (0.64) and on-target rate (0.41) moderately; finishing
  (0.11) and assists per key pass (0.00) are noise. The regression strengths
  follow.
- A player's match-to-match shot share is pure Poisson noise around his rate.

## Out-of-sample performance (fitted only on matches before each date)

**Team layer**, 1X2 log-loss (lower is better), Pinnacle / market-average
closing prices with the overround removed on the same matches:

| season | model | closing market | base rates |
|---|---|---|---|
| 2024/25 | 0.974 | 0.968 | 1.07 |
| 2025/26 | 1.027 | 1.014 | 1.07 |

The model captures ~85–90% of the market's information over base rates and
its win probabilities are calibrated (slope 1.04 / 0.99). Player-layer results
are on the Backtest page (`python -m footsim.backtest 2025` prints them).

## Setup & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m footsim.data 2024 2025 2026     # first sync: ~2 minutes a season
streamlit run Home.py
```

Data: Understat (`getLeagueData` / `getMatchData` — every shot and every
player-match line; cached to `cache/*.parquet`, only new matches refetched),
the FPL API (squads, availability, fixtures, per-gameweek scoring
components; vaastav's archive for past seasons) and football-data.co.uk
(results with closing odds, for the backtest). Caches expire every six
hours, so a long-running app picks up a new gameweek on its own.

Command-line checks, no dashboard:

```bash
python -m footsim.game                 # simulate one fixture and print the box scores
python -m footsim.backtest 2025        # score the season out of sample
python -m footsim.calibrate moments 2025   # engine vs real distributions
python -m footsim.calibrate refit 2024 2025 # calibration slope, home advantage
```

## Maintaining it

The engine constants (`game.LEAD_BETA_*`, `DOMINANCE_SD`, `TEAM_WOBBLE_SD`)
were set by `calibrate moments` on 2024–26; the rating shrinkage
(`teams.*_PRIOR_N`) by the backtest's calibration slope. Refit each
off-season and re-run the harness. `ROADMAP.md` records every design
decision and every finding, including the ones that did not pan out.

## Caveats

- The FPL squad feed cannot be replayed, so the player backtest uses
  history-only squads and scores the stat models conditional on each
  player's actual role; the lineup model is scored separately.
- Bonus points are rebuilt from a BPS approximation; treat them as the
  least certain fantasy component.
- No red cards, referee effects or in-match injuries.
- Prop-line evaluation against a bookmaker is the last item on the roadmap
  and is not built yet.
- For research and entertainment.
