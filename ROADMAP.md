# Football Match Model — Build Roadmap

A port of the NFL Monte Carlo suite (`recyards`) to association football:
a **self-contained** match simulator that forms its own view of team strength
from shot data, simulates a fixture thousands of times, and reports the score
distribution, 1X2 / goal-line probabilities, a projected player box score
(shots, shots on target, goals, assists, key passes, saves, cards) and
fantasy (FPL) points — all from the same simulations, so no two numbers can
contradict each other.

Nothing anchors to a bookmaker. Closing odds appear only as a comparator in
the backtest, and the prop-line evaluation is the *last* item on the list.

---

## 1. Design principles (inherited, unchanged)

1. **Fit priors with variance** — every step of a match is drawn from a
   distribution fitted to real match-to-match data, not a point estimate.
2. **Right distribution per step** — Poisson / Negative Binomial for counts,
   Beta for rates, Dirichlet for shares, Gamma where a quantity is skewed.
3. **Opponent adjustment with shrinkage** — compare what a defence allows
   with league average and shrink toward league; a season of splits is noisy.
4. **Recency** — one weight column on every feed: a season curve × a
   per-team, match-based decay (`data.game_weights`). Every rate, share and
   volume in the suite is a weighted estimate.
5. **Consistency by construction** — player shots sum to team shots, player
   goals to team goals, assists to goals-with-an-assist, saves to the
   opponent's on-target shots minus goals. Largest-remainder apportionment on
   Dirichlet shares keeps the counts exact in *every* simulation.
6. **Shared game state** — the running score in a simulation feeds back into
   both sides' shot rates (a trailing side presses), so volume, outcome and
   score come from the same draw.
7. **Every fitted constant is fitted out of sample**, on the backtest's
   residuals, and a constant the harness cannot distinguish from zero is shown
   but not priced.

---

## 2. The mapping: NFL model → football model

| NFL (recyards) | Football (football-pred) | Atom / data |
|---|---|---|
| Drive | **Shot** | Understat per-shot: player, xG, result, situation, assister |
| Points per drive (team strength) | **Shots per match × xG per shot × finishing**, attack and defence solved jointly | Understat team-match table |
| Pace (drives per game) | **Shot tempo** — total shots in the fixture, a property of the pairing | same |
| Home field (+1.97 pts) | Home advantage on shot volume and quality (fitted) | same |
| Lead-dependent drive scoring (`LEAD_BETA`) | **Game state**: shot rate scales with (lead − expected lead), resolved in 15-minute segments | shot minutes + running score |
| Depth chart + injuries | **FPL squad + availability status/news**, start rate, minutes-when-starting, sub minutes | FPL API `bootstrap-static` |
| Target share / carry share (Dirichlet) | **Shot share** and **key-pass share** ∝ expected minutes × per-90 rate (Dirichlet) | Understat rosters |
| Receiving yards (targets→catches→yards) | **Shots → on target → goals** per player | shots table |
| Touchdowns: team total then multinomial split by role | **Goals**: engine decides team goals, multinomial split across the players who *took shots* in that sim, weighted by shots × conversion | same |
| Passing TDs = allocated receiving TDs | **Assists** = goals × P(assisted), assister drawn by key-pass weight among teammates on the pitch | `player_assisted` |
| QB sacks (a defensive-side stat viewed from the offence) | **Goalkeeper saves** = opponent's on-target shots − goals conceded | derived |
| Interceptions (rare, heavily regressed) | **Cards** — yellow per 90, heavily regressed toward position; red rare | rosters |
| Fantasy (PPR/half/standard) | **FPL scoring** (minutes, goals by position, assists, clean sheets, saves, GC, cards, bonus, defensive contribution) — editable | FPL rules |
| Pick'em card | (not ported — no equivalent product) | — |
| Backtest vs closing line | Backtest vs **Pinnacle closing 1X2 / O2.5** from football-data.co.uk | free CSV |
| Prop evaluation (Odds API) | Same, `soccer_epl` player markets — **last** | Odds API |

**What football adds that the NFL had no analogue for**

- **Draws.** A three-way result; the engine reports 1X2, and the backtest
  scores 1X2 log-loss and ranked probability score, not just winners.
- **Minutes.** A footballer's volume is minutes × rate; the NFL had snaps as
  a *prior* only. Minutes are simulated explicitly (start / sub / unused).
- **Penalties.** ~0.1 per team-match, xG 0.76, taken by a designated taker —
  modelled as a separate team-level event allocated to the FPL penalty-order
  taker on the pitch.
- **Own goals** — a small team-level rate, not attributed to a player's shots.

**What is deliberately not modelled (locked)**

- No possession / passing-network model; shots are the atom.
- No red-card branch in the engine (rate ≈ 0.1 per match, priced as noise).
- No referee effects on cards in v1 (roadmap §8).
- Bonus points are approximated from the simulated line (§3.9).

---

## 3. The stat modules

### 3.1 Shots (`footsim/players.py`, page 1) — the reference implementation
*Built: pages 1–6 all run one shared simulation chain in `players.py` (minutes → team shots → player shots → on target → goals; key passes → assists; opponent shots → saves), so no two pages can disagree.*
Team shots in the match (NB) → player's share of them (Beta, ∝ expected
minutes × shots per 90) → integer shots (Poisson-binomial via the share).
Player priors: shots per 90 on recency-weighted minutes, regressed toward the
positional mean over `SHOT_PRIOR_MIN` minutes-worth; between-match SD net of
Poisson noise (`data.between_sd`). Opponent: shots allowed vs league, shrunk.

### 3.2 Shots on target (page 2)
Shots (3.1) → each on target with the player's regressed on-target rate ×
the opponent's block/on-target ratio, shrunk. On-target rate is far more
stable by *situation* than by player, so the prior is the player's situation
mix (open play / set piece / penalty) × league rates.

### 3.3 Goals — anytime scorer (page 3)
Expected goals = shots × conversion. Conversion = regressed xG per shot
(the player's *chance quality*, stable) × finishing (goals / xG, shrunk hard
toward 1 — mostly noise), scaled by the match's expected team goals over the
team's typical (the game-view factor). Penalties added by taker. Poisson /
NB draw on the single-stat page; in the engine goals are allocated top-down.

### 3.4 Assists (page 4)
Key passes per 90 → assists per key pass (regressed toward league; xA per
key pass is the stable part) × the team's expected goals in the match over
typical. In the engine: each team goal is assisted with P ≈ 0.72 (measured),
assister drawn by key-pass weight among teammates on the pitch.

### 3.5 Goalkeeper saves (page 5)
Opponent's shots (3.1 at team level) → on target (team rate vs this defence)
→ saved = on target − goals. Saves are therefore emergent from the match
model, like the NFL QB line: the GK never has a separate volume model.

### 3.6 Cards (page 6) — DONE (v1: no referee / opponent effect)
Yellow cards per 90, regressed toward position over `CARD_PRIOR_MIN`,
Poisson on minutes. Opponent adjustment: fouls drawn per match, shrunk.

### 3.7 Team strength (`footsim/teams.py`, page 7)
Log-additive ratings solved by iteration (the SRS idea on the log scale):

    log(shots)      = μ_s + att_s(i) + def_s(j) + home_s
    log(xG / shot)  = μ_q + att_q(i) + def_q(j) + home_q
    log(goals / xG) = fin(i) + gk(j)            (shrunk hard)

Each rating shrunk toward zero by the matches behind it; recent matches
weighted more. Per matchup: expected shots each way, xG per shot each way,
expected goals each way; the score level is calibrated against actual goals
(the residual — own goals and finishing drift — is measured, not assumed).

### 3.8 Match engine (`footsim/game.py`, page 8)
Per simulation:
1. **Lineups** — for each player: start / sub / unused with his minutes
   distribution; exactly one goalkeeper.
2. **Segments** — six 15-minute segments. Each side's shot count in a segment
   is Poisson at its matchup rate × the segment's share of shots (late
   segments carry more) × exp(−LEAD_BETA × (lead − expected lead)). Each shot
   is on target with the matchup rate and a goal with the matchup conversion.
   Penalties and own goals are separate small Poisson events. This yields the
   score, both shot totals, on-target totals and the running state.
3. **Allocation** — Dirichlet over players on the pitch (weights = minutes
   this sim × shots per 90) splits the team's shots; per-player on-target
   rates split the on-target shots; team goals are allocated across the
   players who took shots, weighted by shots × conversion, capped at their
   shots; assists as in 3.4; the opposing GK's saves are on-target − goals.
4. **Reporting** — score distribution, 1X2, goal lines, both box scores.

The engine's calibration targets, all measured from 2023–26 EPL: home win /
draw / away win frequencies, goal-difference SD, team-goals SD, correlation of
the two teams' shots and goals, and the shot-total distribution.

### 3.9 Fantasy points (`footsim/fantasy.py`, page 9)
FPL rules applied per simulation to every player on the gameweek's slate:
minutes points, goals by position, assists, clean sheets (≥ 60 min), saves
(per 3), goals conceded (per 2, GK/DEF), cards, own goals, penalties,
defensive contribution (per-90 rate × minutes, NB; FPL 2025–26 rule), and
bonus approximated from a fitted BPS-to-bonus map on the simulated line.
Floor / ceiling / P(≥ N) per player, breakdown by component.

### 3.10 Backtest (`footsim/backtest.py`, page 10)
Rolling out-of-sample: for each gameweek, fit everything on matches before it.
Team layer: 1X2 log-loss, RPS, goal-difference RMSE, total-goals RMSE, vs
Pinnacle closing odds on the same matches. Player layer: shots, shots on
target, goals, assists, saves on every player with ≥ 5 prior appearances and
prop-worthy expected volume — MAE / bias / CRPS / 10–90 coverage / Brier of
P(≥ 1) — vs a trailing weighted average.

### 3.11 Calibration (`footsim/calibrate.py`)
Refit report for home advantage, rating shrinkage (calibration slope),
`LEAD_BETA` (score spread), assist rate, penalty rate; `--write` updates.

### 3.12 Prop evaluation — LAST
Odds API `soccer_epl` player markets (shots, shots on target, anytime scorer,
assists, cards) beside the frozen projection; settled from Understat.

---

## 4. Data

| Feed | Source | What it gives | Cache |
|---|---|---|---|
| Fixtures + team match table | Understat `getLeagueData/{league}/{season}` | match ids, dates, xG/xGA, scored/missed, deep, ppda | 1 request / season |
| Shots + rosters | Understat `getMatchData/{id}` | every shot; every player's minutes, shots, xG, key passes, xA, cards, sub in/out | 1 request / match, ~0.3 s; incremental |
| Squads, availability, positions, penalty order | FPL `bootstrap-static` | status, chance of playing, news, element_type, price | live |
| Per-GW FPL stats (saves, bonus, BPS, defensive contribution) | FPL `event/{gw}/live/` (current), vaastav archive (past seasons) | scoring components the shot data lacks | per GW |
| Results, shots, closing odds | football-data.co.uk `E0.csv` | Pinnacle closing 1X2 / O2.5 for the backtest | 1 CSV / season |

Understat's site moved its data behind JSON endpoints in 2025
(`getLeagueData`, `getMatchData`; gzip; needs `X-Requested-With`). The old
"parse JSON out of the HTML" approach every library uses is dead — this is
the first schema trap, recorded here.

Canonical team code = FPL short name (ARS, MCI, …); Understat titles and
football-data names map onto it (`data.TEAM_ALIASES`). Player crosswalk
Understat ↔ FPL by normalised name within team, with a manual override table.

Local cache: `cache/*.parquet` (gitignored). A full season is ~380 match
fetches ≈ 2 minutes on first load; afterwards only unplayed matches refetch.

---

## 5. Build order

| Phase | Deliverable |
|---|---|
| 1 | `data.py` ✅ Understat + FPL + football-data loaders, incremental parquet cache, recency, helpers; data spike (§6) |
| 2 | `teams.py` + page 7 ✅ shot / quality / finishing ratings by IPF, home advantage on volume, expected goals, analytic 1X2 |
| 3 | `roster.py` ✅ FPL squad → start probability fitted to eleven, minutes model, shares per 90, availability gate, Understat↔FPL crosswalk |
| 4–5 | `players.py` + pages 1–6 ✅ one simulation chain for shots, on target, goals, assists, saves, cards; Fixture / Season view; "assume he starts" |
| 6 | `game.py` + page 8 ✅ six-segment engine with game state and dominance, capped allocation, box score; calibrated by `calibrate moments` (§6.5) |
| 7 | `fantasy.py` + page 9 ✅ FPL rules per simulation, bonus via BPS ranking, gameweek slate |
| 8 | `backtest.py` + page 10 ✅ team layer vs closing market; player layer conditional on role; lineup model; `calibrate.py` |
| 9 | prop evaluation (Odds API) — not started, last by design |

---

## 6. Findings log (2023/24–2026/27 EPL, 1 177 matches, 31 000 shots)

What the spike and the harness said, and what each finding changed. The
build is dated 2026-09-13; the current season had four rounds played.

### 6.1 Home advantage is volume
Home sides take **14.4 shots to the away side's 11.9 (×1.21)** at an
identical xG per shot (0.125 vs 0.124) and a slightly *lower* on-target
rate (×0.96). Goals 1.61 vs 1.37. So the ratings layer fits home advantage
on the shot rate (jointly with the ratings: ×1.207) and separately, tiny,
on quality (×1.003) — never as a flat goals bonus, which is what most
Poisson models do.

### 6.2 The game state cancels out on goals — but not on shots
Within a team-match, relative to its own average: a **trailing** side takes
shots at ×1.13 with 0.110 xG per shot; **level** ×1.00 at 0.121; **leading**
×0.87 at 0.147. On-target 0.315 / 0.331 / 0.368. Goals per minute are
therefore nearly flat across states while shot counts swing — the football
twin of the NFL finding that points are pace-invariant while plays are not.
The engine moves volume down and quality *up* with the lead
(`LEAD_BETA_VOL = −0.22`, `LEAD_BETA_GOAL = +0.10`, `LEAD_BETA_SOT = +0.06`).
The fitted volume effect is stronger than the raw table's −0.14 because the
table is confounded: good sides both lead more and shoot more.

### 6.3 Shots are near zero-sum; goals are independent
Raw: corr(home shots, away shots) −0.39; conditional on team strength
−0.16, with the shot count overdispersed (var/mean 1.95, log-sd 0.42).
Goals conditional on strength: corr −0.015, var/mean 0.92 — independent,
slightly *under*-Poisson. An NFL-style shared tempo draw would give a
positive shot correlation, so the engine has a latent **dominance** draw
(one side's gain is the other's loss, sd 0.16) and only a small independent
wobble (0.10) — most of the extra-Poisson spread turned out to be the game
state, not exogenous noise.

### 6.4 Goals rise 0.3–0.5× as fast as shots
Within team-season, elasticity of goals to shots is **0.46** (0.29–0.36
against in-sample expectations); xG per shot does not fall with volume
(elasticity 0.02). The mechanism is scoring itself: a goal makes the scorer
sit back, so high-shot matches are the ones where nobody scored early. The
engine reproduces 0.33 from the state feedback alone — it is a validation
target (`calibrate moments`), not an input. Independent Poissons would give
1.0.

### 6.5 Engine calibration (grid on 2024–26)
| | engine | real |
|---|---|---|
| home / draw / away | 0.422 / 0.261 / 0.317 | 0.426 / 0.274 / 0.300 |
| goal-difference sd | 1.63 | 1.62 |
| team goals var/mean | 0.96 | 0.83–0.87 |
| shots var/mean · corr | 1.59 · −0.18 | 1.32 · −0.20 |
| elasticity | 0.33 | 0.29–0.36 |

The engine produces *more* draws than independent Poissons (28.9% vs 26.7%
for a level fixture) purely from the state feedback — the Dixon–Coles
correction, emergent rather than imposed. The convex feedback inflates mean
shots 6% and goals 2% over the ratings; `SHOT_MEAN_CORR` / `GOAL_MEAN_CORR`
bring the simulated means back onto the analytic layer (checked to 1%).

### 6.6 Understat xG runs above actual goals
Goals / xG = 0.877 over the window (0.81 recency-weighted). The league
finishing level is measured against actual goals in `teams.team_ratings`,
never assumed to be 1.

### 6.7 What is stable year to year (players with 900+ minutes both seasons)
| rate | r | regression |
|---|---|---|
| shots per 90 | **0.92** | 300 minutes-worth |
| key passes per 90 | **0.87** | 400 minutes |
| xG per shot | 0.64 | 30 shots |
| on-target rate (40+ shots) | 0.41 | 90 shots |
| goals per xG (finishing) | 0.11 | 60 xG — effectively league |
| assists per key pass | 0.00 | league × position |
| xA per key pass | 0.14 | 80 key passes |

So a player's *volume* is his own within a few matches; his finishing never
is. A player's match-to-match shot share, net of Poisson noise, has a
between-match sd of ~0 — the allocation is a plain multinomial, no
Dirichlet wobble needed (the NFL needed concentration 30–45).

### 6.8 Small rates
Assist on 71% of goals (77% of non-penalty goals). Penalties 0.123 per
team-match, 85.5% scored, 97% on target. Own goals 0.055 per team-match,
listed by Understat under the *conceding* player's team with `goal = 0`, so
score = goal-shots + opponent's own goals (identity holds in 100% of
matches). Key passes 9.7 per team-match = 0.78 per shot. On-target by
situation: open play 35.5%, corner 24%, set piece 28%, direct free kick 30%.

### 6.9 Minutes
Starters average 84 minutes; 37% are subbed off, at minute 75 (sd 14) —
60% of forwards and wingers, 13% of centre-backs. 4.1 subs used per match,
15 minutes each (median 13). Understat's roster only lists players who
played, and its sub flags are the reverse of their names (a starter's
`roster_in` is the sub who replaced him).

### 6.10 Position-group priors (share of team shots / key passes per 90 on the pitch)
ST 0.199 / 0.070 · AM 0.162 / 0.127 · W 0.157 / 0.114 · CM 0.086 / 0.081 ·
FB 0.051 / 0.069 · CB 0.041 / 0.022. xG per shot: ST 0.187, W 0.122, AM
0.117, CB 0.117 (headers), CM/FB 0.088.

### 6.11 Team layer, out of sample (fitted on matches before each date)
1X2 log-loss, closing market with the overround removed on the same matches:

| shrinkage (vol, quality) | 2024/25 | 2025/26 | slope |
|---|---|---|---|
| 6, 10 (first draft) | 0.988 | 1.031 | 1.25 |
| 2, 4 | 0.979 | 1.028 | 1.06 |
| **1, 2** (set) | **0.974** | **1.027** | 1.04 / 0.99 |
| closing market | 0.968 | 1.014 | — |
| base rates | 1.07 | 1.07 | — |

As in the NFL layer, the first-draft ratings were over-shrunk (actual goal
difference moved 1.25× the predicted). Heavier shrinkage on *finishing*
(80 xG-worth) helps a little; lighter hurts. The model sits ~0.012 of
log-loss behind the market, capturing ~85–90% of its information over base
rates.

### 6.12 The model's spread vs the market's (the "every match is 1–2 goals each" question)
Against the closing market on all 378 matches of 2025/26 the model's 1X2
probabilities correlate 0.93 with the market's but have **0.68× its spread
on the logit scale** (P(home) sd 0.131 vs 0.172; Arsenal v Burnley 80% vs
87%). Where the two disagree by more than five points the market is right
(model 0.50 / market 0.60 / actual 0.61). Yet the model's own calibration
slope is 1.0: it is calibrated *for what it knows* and knows less than the
market. The levers tested:

| lever | effect |
|---|---|
| rating shrinkage 6/10 → 1/2 | log-loss 0.988/1.031 → 0.974/1.027; spread 0.79/0.68 — most of what was fixable |
| recency: long memory / flat | spread 0.79/0.75 but log-loss worse and slope < 1 (over-confident) |
| recency: recent form | no better |
| squad availability (§6.13) | nothing, even with hindsight lineups |

The rest of the gap is information the market has and a shot model does
not (lineups pre-announcement, motivation, other markets). A self-contained
model should expect exactly this; the fix is not to stretch the spread.

### 6.13 Squad availability as a team effect — built, fitted at zero
`roster.availability_factor`: the expected shot production (minutes ×
share × xG per shot, summed, split into volume and quality) of the
available squad over the ungated squad, with starts fitted by position
group to the team's own formation so a missing centre-back is replaced by a
centre-back. It moves attacks by ±8% (log sd 0.08). Fitted with hindsight
lineups on 2024–26, the coefficient that maximises the Poisson likelihood
of goals is 0 (2025) / 0.5 (2024, flat), and the 1X2 log-loss worsens
monotonically with it. Shown on the pages, off by default. Two reasons it
may fail that are worth revisiting with a real pre-match feed: the team
ratings already average over the periods without the player, and shot
share is not the same as chance creation (a missing striker's shots go to
teammates at lower quality than the quality ratio captures).

### 6.14 Where the extremes go — a closer look
Binned out of sample (2024–26), the tails are *asymmetric*, not uniformly
compressed: away favourites do better than predicted (predicted margin
−1.35 → actual −1.82; −0.70 → −0.93, ~1.3σ each) while big home favourites
do slightly worse (+1.39 → +1.09). That is the signature of a home advantage
that is too generous to weak home sides, but the in-sample test of home
advantage by strength gap is confounded by regression to the mean, and 757
matches cannot fit a gap-dependent home term. Left open. The solver is
converged (40 vs 400 iterations agree to 1e-12) and the in-sample fit of
actual xG on expected xG has slope 0.96, so the compression is not numerical.

### 6.15 Promoted sides — attack prior priced, defence not
A team with no top-flight matches the season before started at league
average (and with no data at all was *unrated*, so round-one fixtures were
skipped by the backtest). Every promoted class since 2023 has lacked
attacking quality — xG for 0.65–0.83× league, from fewer shots (log −0.20)
of lower quality (−0.11), spread across teams only 0.05–0.07 — so the
attack prior is now worth 8 matches. Scored on the promoted side's own xG
in rounds 1–8 it helps in all three classes (bias −0.01 → +0.01 in 2024,
−0.17 → −0.09 in 2025, −0.11 → +0.11 in 2026 on four matches).

The defence half does not survive: the 2023–24 classes conceded 1.2–1.4×
league, but Sunderland / Leeds 2025 conceded 1.04× / 0.97× and Coventry /
Hull 2026 0.89× / 0.87× so far, and the 1X2 log-loss flips with it (2024
−0.004, 2025 +0.004). Not priced.

Sunderland 2025/26 is the cautionary tale for scoring an xG model on
results: first half xGD −0.57 per match but GD +0.11 and 29 points (the 7th
luckiest of 80 team-seasons), second half GD −0.42 — right on their xGD.
The prior called them a typical promoted side on xG, which they were.

### 6.16 Squad production across transfers — tested, not priced
At each season start (2023–25, 60 team-seasons), the XI's production at
its players' *previous-season* per-90 rates (whichever club) was regressed
alongside the previous-season unit rating against the team's xG over rounds
5–38. The unit rating carries the signal (t = 3.5); squad production adds
nothing (t = 1.0, R² 0.482 → 0.491, correlation between the two 0.81). For
promoted sides most of the XI has no top-flight history anyway (Sunderland
2025: 15% known), which is what §6.15 covers. Rest days between league
matches were also tested on the residuals: nothing on margins (t < 1), and
the "long rest" bin is really the European-midweek proxy — not usable
without cup and European fixtures.

### 6.17 Where the extremes really go: conversion follows the match context
Pooled over 1 900 out-of-sample matches (2021–25), expected **xG** is
properly spread (actual on expected: slope 0.90 per side, 0.94 on the xG
margin) but **goals** are not (1.06 per side, 1.13 on the goal margin).
Binning by the opponent's expected xG: a side expected to concede 0.9 xG
converts its own chances at 1.03 of its xG-based expectation, a side
expected to concede 2.2 at 0.89. In a mismatch the stronger side
over-converts and the weaker under-converts relative to xG — elite
finishing and keeping that Understat's xG does not carry, and trailing
sides' chances being worse than their xG. Poisson regression of goals on
the offset of expected goals gives −0.171 (t = −4.2) on log(opponent's
expected xG / league); the dominance form is positive in every season.

`teams.apply_context` scales each side's conversion by (opponent xG /
league)^β and is applied to both sides of every matchup (analytic, engine,
single-player pages). Grid on 2022–25:

| β | pooled log-loss | spread vs market (2025) | margin slope |
|---|---|---|---|
| 0 | 0.9780 | 0.69 | 1.06 |
| −0.08 | 0.9761 | 0.75 | 0.99 |
| **−0.12** (set) | 0.9755 | 0.77 | 0.95 |
| −0.171 | 0.9750 | 0.81 | 0.91 |

This is the answer to "it underweights extremes": it did, by ~13% on goal
margins, and the reason was conversion, not the ratings. The 2025/26 spread
against the market went 0.68 → 0.77 (2024/25: 0.76 → 0.85) with log-loss
improving in every season.

### 6.18 Data traps, for the next person
- Understat serves data from `getLeagueData/{league}/{season}` and
  `getMatchData/{id}` (gzip, needs `X-Requested-With`), not the page HTML.
- Pinnacle closing prices are missing for ~45% of 2025/26 rows on
  football-data.co.uk; the market-average closing price stands in.
- vaastav's FPL archive keys players by season-specific `element`; the
  stable key is `code` from `players_raw.csv`.
- The python.org framework Python on macOS has no root certificates:
  `data.py` builds its SSL context from `certifi`.

## 7. Next (ranked)

1. ~~Player-level team strength across transfers~~ — tested, nothing
   beyond the unit rating (§6.16).
2. ~~Home advantage by strength gap~~ — pooled 2021–25: the tail asymmetry
   was conversion, not home advantage (§6.17); priced.
3. ~~Record the FPL availability feed~~ — `data.record_availability` logs
   every squad's status once a day whenever the squads load
   (`cache/availability.parquet`; `python -m footsim.data availability`).
   Fit a pre-match availability index once a season of snapshots exists.
4. ~~Rest days~~ — tested, confounded without cup / European fixtures
   (§6.16). Needs a fixture source for those competitions.
5. ~~Elite finishing / keeping as a team trait~~ — tested on the pooled
   2021–25 residuals with three-season finishing and keeping ratings
   (shrunk over 60 / 150 / 300 xG): finishing t = 1.6–1.8, keeping 1.2–1.3,
   effect size ±2% at most on top of §6.17. Right sign, not priced.
6. **A fixture source for cups and European matches**, which would unlock
   rest days / congestion (§6.16) and is the one variable the market
   visibly prices that this suite cannot see.
7. **Prop evaluation** (Odds API) — last, by design.
3. **Referee and opponent effects on cards**; fouls drawn.
4. **Prop evaluation** — Odds API `soccer_epl` markets, frozen projection,
   settled from Understat. Last, by design.
