"""
footsim/planner.py — FPL over a horizon: the best squad for the weeks ahead,
and the best transfer path from the team you actually have.

Both are exact integer programmes (scipy's HiGHS) over per-player,
per-gameweek expected points from the match engine (`horizon_projections`).

Rules encoded (FPL 2025-26): 15 players (2 GK / 5 DEF / 5 MID / 3 FWD), at
most three from a club, a legal starting XI each week (1 GK, 3–5 DEF, 2–5
MID, 1–3 FWD), one captain counted twice, one free transfer a gameweek
banked to a maximum of five, −4 points for every transfer beyond the free
ones, and a budget of squad value plus bank. Prices are FPL's current
prices for buying and selling (the half-profit selling rule is ignored).
Chips (wildcard, free hit, bench boost, triple captain) are not modelled.

Later gameweeks are discounted (`discount` per week) because the engine's
view of a fixture five weeks out is less certain than of the next one:
lineups change, prices move, the ratings will have moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

from . import data as D
from . import fantasy as F
from . import game as G

MAX_FREE_TRANSFERS = 5
HIT_COST = 4.0
POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


# ---------------------------------------------------------------------------
# Projections over the horizon
# ---------------------------------------------------------------------------

def horizon_projections(ctx: dict, gws: list[int], rules: dict, n_sims: int = 2000,
                        progress=None) -> pd.DataFrame:
    """One row per player per gameweek: Proj, Floor, Ceiling, Price, Team,
    Pos, Player, Start, plus a blank row (0 points) for a player without a
    fixture that week. Double gameweeks are summed by `fantasy.best_squad`'s
    aggregation, here kept per fixture."""
    fx = D.fpl_fixtures()
    frames = []
    for j, gw in enumerate(gws):
        f = fx[fx["gw"] == gw].sort_values("kickoff")
        t, _, _ = F.gameweek_projections(ctx, f, rules, n_sims=n_sims, seed=100 + gw)
        if t.empty:
            continue
        t["gw"] = gw
        frames.append(t)
        if progress:
            progress(j + 1, len(gws))
    if not frames:
        return pd.DataFrame()
    h = pd.concat(frames, ignore_index=True)
    agg = (h.groupby(["player_id", "gw"])
             .agg(Proj=("Proj", "sum"), Floor=("Floor", "sum"), Ceiling=("Ceiling", "sum"),
                  Start=("Start", "mean"), Price=("Price", "first"), Team=("Team", "first"),
                  Pos=("Pos", "first"), Player=("Player", "first"), Status=("Status", "first"),
                  fpl_id=("fpl_id", "first"), Opp=("Opp", lambda s: " + ".join(s)))
             .reset_index())
    return agg


def _panel(h: pd.DataFrame, gws: list[int], objective: str = "Proj") -> tuple[pd.DataFrame, np.ndarray]:
    """Players × gameweeks matrix of expected points (0 for a blank)."""
    players = (h.sort_values("gw").groupby("player_id")
                 .agg(Player=("Player", "last"), Team=("Team", "last"), Pos=("Pos", "last"),
                      Price=("Price", "last"), fpl_id=("fpl_id", "last"), Status=("Status", "last"))
                 .reset_index())
    players = players[players["Price"] > 0].reset_index(drop=True)
    pts = np.zeros((len(players), len(gws)))
    idx = {pid: i for i, pid in enumerate(players["player_id"])}
    for _, r in h.iterrows():
        if r["player_id"] in idx and r["gw"] in gws:
            pts[idx[r["player_id"]], gws.index(r["gw"])] += float(r[objective])
    return players, pts


def _prune(players: pd.DataFrame, pts: np.ndarray, keep_ids=(), per_pos: int = 28, cheap: int = 6):
    """Keep the strongest candidates per position by horizon points, the
    cheapest playable options (bench fodder), and any ids that must stay."""
    total = pts.sum(axis=1)
    keep = set(keep_ids)
    for pos in POS_ORDER:
        m = np.where(players["Pos"].values == pos)[0]
        top = m[np.argsort(-total[m])[:per_pos]]
        keep.update(players["player_id"].values[top])
        playable = m[total[m] > 0.5 * len(pts[0])]
        cheap_ids = playable[np.argsort(players["Price"].values[playable])[:cheap]]
        keep.update(players["player_id"].values[cheap_ids])
    mask = players["player_id"].isin(keep).values
    return players[mask].reset_index(drop=True), pts[mask]


# ---------------------------------------------------------------------------
# The horizon MILP (shared by both planners)
# ---------------------------------------------------------------------------

def _solve(players: pd.DataFrame, pts: np.ndarray, gws: list[int], budget: float,
           discount: float, bench_weight: float, x0: np.ndarray | None = None,
           free0: int = 1, allow_transfers: bool = True, hit_cost: float = HIT_COST,
           max_transfers_per_gw: int | None = None, time_limit: float = 90.0,
           lock_mask: np.ndarray | None = None) -> dict:
    n, H = pts.shape
    # variable blocks: x, y, c (n×H each), tin, tout (n×H), paid (H), ft (H),
    # z (H: 1 if this week's transfers use up the free ones)
    nb = n * H
    ix = lambda i, t: i * H + t
    off = dict(x=0, y=nb, c=2 * nb, tin=3 * nb, tout=4 * nb, paid=5 * nb, ft=5 * nb + H, z=5 * nb + 2 * H)
    nv = 5 * nb + 3 * H
    disc = np.array([discount ** t for t in range(H)])
    cost = np.zeros(nv)
    for i in range(n):
        for t in range(H):
            cost[off["x"] + ix(i, t)] = -bench_weight * pts[i, t] * disc[t]
            cost[off["y"] + ix(i, t)] = -(1 - bench_weight) * pts[i, t] * disc[t]
            cost[off["c"] + ix(i, t)] = -pts[i, t] * disc[t]
    for t in range(H):
        cost[off["paid"] + t] = hit_cost * disc[t]
    rows, lo, hi = [], [], []

    def add(coefs: dict, l, h):
        r = np.zeros(nv)
        for k, v in coefs.items():
            r[k] = v
        rows.append(r); lo.append(l); hi.append(h)

    pos = players["Pos"].values; club = players["Team"].values; price = players["Price"].values.astype(float)
    for t in range(H):
        add({off["x"] + ix(i, t): 1 for i in range(n)}, 15, 15)
        add({off["y"] + ix(i, t): 1 for i in range(n)}, 11, 11)
        add({off["c"] + ix(i, t): 1 for i in range(n)}, 1, 1)
        add({off["x"] + ix(i, t): price[i] for i in range(n)}, 0, budget)
        for p, k in F.SQUAD_RULES["squad"].items():
            m = [i for i in range(n) if pos[i] == p]
            add({off["x"] + ix(i, t): 1 for i in m}, k, k)
            add({off["y"] + ix(i, t): 1 for i in m}, F.SQUAD_RULES["xi_min"][p], F.SQUAD_RULES["xi_max"][p])
        for cl in np.unique(club):
            m = [i for i in range(n) if club[i] == cl]
            add({off["x"] + ix(i, t): 1 for i in m}, 0, F.SQUAD_RULES["max_per_club"])
        for i in range(n):
            add({off["y"] + ix(i, t): 1, off["x"] + ix(i, t): -1}, -np.inf, 0)
            add({off["c"] + ix(i, t): 1, off["y"] + ix(i, t): -1}, -np.inf, 0)
        if allow_transfers:
            for i in range(n):
                prev = {off["x"] + ix(i, t - 1): -1} if t > 0 else {}
                # x_t - x_{t-1} - tin + tout = x0 (t = 0) / 0
                coefs = {off["x"] + ix(i, t): 1, off["tin"] + ix(i, t): -1, off["tout"] + ix(i, t): 1}
                coefs.update(prev)
                rhs = float(x0[i]) if t == 0 else 0.0
                add(coefs, rhs, rhs)
                add({off["tin"] + ix(i, t): 1, off["tout"] + ix(i, t): 1}, 0, 1)
            # transfers this gw = sum tin; paid >= transfers - ft, and free
            # transfers are used FIRST (z = 1 iff transfers >= ft):
            #   transfers >= ft - 15(1-z); transfers <= ft + 15z
            #   paid <= transfers - ft + 15(1-z); paid <= 15z
            tin_t = {off["tin"] + ix(i, t): 1 for i in range(n)}
            add({**tin_t, off["paid"] + t: -1, off["ft"] + t: -1}, -np.inf, 0)
            add({**tin_t, off["ft"] + t: -1, off["z"] + t: -15}, -15, np.inf)
            add({**tin_t, off["ft"] + t: -1, off["z"] + t: -15}, -np.inf, 0)
            add({off["paid"] + t: 1, **{k: -v for k, v in tin_t.items()}, off["ft"] + t: 1, off["z"] + t: 15}, -np.inf, 15)
            add({off["paid"] + t: 1, off["z"] + t: -15}, -np.inf, 0)
            if max_transfers_per_gw is not None:
                add({off["tin"] + ix(i, t): 1 for i in range(n)}, 0, max_transfers_per_gw)
            # free transfers: ft_0 = free0; ft_{t+1} <= ft_t - transfers_t + paid_t + 1, <= MAX
            if t == 0:
                add({off["ft"] + 0: 1}, free0, free0)
            if t + 1 < H:
                add({off["ft"] + t + 1: 1, off["ft"] + t: -1, off["paid"] + t: -1,
                     **{off["tin"] + ix(i, t): 1 for i in range(n)}}, -np.inf, 1)
    lb = np.zeros(nv); ub = np.ones(nv)
    ub[off["paid"]:off["paid"] + H] = 15; ub[off["ft"]:off["ft"] + H] = MAX_FREE_TRANSFERS
    if lock_mask is not None:
        for i in np.where(lock_mask)[0]:
            for t in range(H):
                lb[off["x"] + ix(i, t)] = 1
    if not allow_transfers:
        # a fixed squad: x identical across gameweeks, no transfer variables
        for i in range(n):
            for t in range(1, H):
                add({off["x"] + ix(i, t): 1, off["x"] + ix(i, 0): -1}, 0, 0)
        ub[off["tin"]:off["z"] + H] = 0
    integrality = np.ones(nv)
    res = milp(c=cost, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit})
    if res.x is None:
        raise ValueError("No feasible plan (budget, locks or rules cannot be met).")
    x = np.round(res.x[off["x"]:off["x"] + nb]).reshape(n, H).astype(bool)
    y = np.round(res.x[off["y"]:off["y"] + nb]).reshape(n, H).astype(bool)
    c = np.round(res.x[off["c"]:off["c"] + nb]).reshape(n, H).astype(bool)
    tin = np.round(res.x[off["tin"]:off["tin"] + nb]).reshape(n, H).astype(bool)
    tout = np.round(res.x[off["tout"]:off["tout"] + nb]).reshape(n, H).astype(bool)
    paid = np.round(res.x[off["paid"]:off["paid"] + H]).astype(int)
    ft = np.round(res.x[off["ft"]:off["ft"] + H]).astype(int)
    weeks = []
    for t, gw in enumerate(gws):
        xi = np.where(y[:, t])[0]; cap = int(np.where(c[:, t])[0][0]) if c[:, t].any() else None
        weeks.append(dict(
            gw=gw, xi_points=float(pts[y[:, t], t].sum() + (pts[cap, t] if cap is not None else 0.0)),
            captain=(players.loc[cap, "Player"] if cap is not None else ""),
            transfers_in=players.loc[tin[:, t], "Player"].tolist(), transfers_out=players.loc[tout[:, t], "Player"].tolist(),
            hits=int(paid[t]), free_before=int(ft[t]), cost=float(price[x[:, t]].sum()),
            squad=players.loc[x[:, t], "player_id"].tolist(), xi=players.loc[y[:, t], "player_id"].tolist(),
        ))
    total = sum(w["xi_points"] - HIT_COST * w["hits"] for w in weeks)
    return dict(weeks=weeks, players=players, x=x, y=y, c=c, pts=pts, total=float(total),
                status=res.message, optimal=bool(res.success))


# ---------------------------------------------------------------------------
# Public planners
# ---------------------------------------------------------------------------

def best_season_squad(h: pd.DataFrame, gws: list[int], budget: float = 100.0, discount: float = 0.9,
                      bench_weight: float = 0.1, objective: str = "Proj", locked=(), excluded=()) -> dict:
    """One fixed 15 for the whole horizon, with the best XI and captain each
    week. Returns the solve plus a squad table with per-week points."""
    players, pts = _panel(h, gws, objective)
    players, pts = _prune(players, pts, keep_ids=locked)
    if excluded:
        m = ~players["player_id"].isin(excluded).values
        players, pts = players[m].reset_index(drop=True), pts[m]
    lock = players["player_id"].isin(locked).values if locked else None
    sol = _solve(players, pts, gws, budget, discount, bench_weight, allow_transfers=False, lock_mask=lock)
    return _with_table(sol, gws)


def plan_transfers(h: pd.DataFrame, gws: list[int], current_ids: list, bank: float, free_transfers: int = 1,
                   discount: float = 0.9, bench_weight: float = 0.1, objective: str = "Proj",
                   max_transfers_per_gw: int | None = None, hit_cost: float = HIT_COST) -> dict:
    """The best transfer path from `current_ids` (15 player ids) over the
    horizon under the transfer rules. Budget = current squad value at today's
    prices + bank."""
    players, pts = _panel(h, gws, objective)
    missing = [pid for pid in current_ids if pid not in set(players["player_id"])]
    if missing:
        raise ValueError(f"{len(missing)} of your players have no projection (no fixture data): {missing[:5]}")
    players, pts = _prune(players, pts, keep_ids=current_ids)
    x0 = players["player_id"].isin(current_ids).values.astype(float)
    if x0.sum() != 15:
        raise ValueError(f"Current squad must have 15 players ({int(x0.sum())} matched).")
    budget = float(players.loc[x0 > 0, "Price"].sum() + bank)
    sol = _solve(players, pts, gws, budget, discount, bench_weight, x0=x0, free0=int(free_transfers),
                 allow_transfers=True, hit_cost=hit_cost, max_transfers_per_gw=max_transfers_per_gw)
    sol["budget"] = budget
    # the do-nothing baseline: keep the squad, best XI each week
    base = _solve(players, pts, gws, budget, discount, bench_weight, x0=x0, free0=int(free_transfers),
                  allow_transfers=True, hit_cost=hit_cost, max_transfers_per_gw=0)
    sol["baseline_total"] = base["total"]
    return _with_table(sol, gws)


def _with_table(sol: dict, gws: list[int]) -> dict:
    p = sol["players"]; x, y, c, pts = sol["x"], sol["y"], sol["c"], sol["pts"]
    ever = x.any(axis=1)
    rows = []
    for i in np.where(ever)[0]:
        row = dict(Player=p.loc[i, "Player"], Team=p.loc[i, "Team"], Pos=p.loc[i, "Pos"], Price=p.loc[i, "Price"],
                   Status=p.loc[i, "Status"], Total=float(pts[i][x[i]].sum()))
        for t, gw in enumerate(gws):
            if not x[i, t]:
                row[f"GW{gw}"] = "—"
            else:
                row[f"GW{gw}"] = f"{pts[i, t]:.1f}" + (" (C)" if c[i, t] else "" if y[i, t] else " b")
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["_o"] = tab["Pos"].map(POS_ORDER)
    sol["table"] = tab.sort_values(["_o", "Total"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    return sol


# ---------------------------------------------------------------------------
# Importing a real FPL team
# ---------------------------------------------------------------------------

def fpl_entry(entry_id: int) -> dict:
    """A manager's current 15 (last deadline's picks), bank and free
    transfers, from the public FPL endpoints."""
    base = "https://fantasy.premierleague.com/api"
    ref = "https://fantasy.premierleague.com/"
    e = D._get_json(f"{base}/entry/{int(entry_id)}/", referer=ref)
    hist = D._get_json(f"{base}/entry/{int(entry_id)}/history/", referer=ref)
    cur = hist.get("current", [])
    if not cur:
        raise ValueError("This team has no gameweek history yet.")
    last = cur[-1]
    picks = D._get_json(f"{base}/entry/{int(entry_id)}/event/{last['event']}/picks/", referer=ref)
    ids = [p["element"] for p in picks["picks"]]
    # free transfers: 1 after GW1, +1 a week, banked to 5, minus the free ones used
    ft = 1
    for gw in cur[1:]:
        paid = int(gw.get("event_transfers_cost", 0)) // int(HIT_COST)
        used_free = max(int(gw.get("event_transfers", 0)) - paid, 0)
        ft = min(MAX_FREE_TRANSFERS, max(ft - used_free, 0) + 1)
    return dict(name=f"{e.get('player_first_name', '')} {e.get('player_last_name', '')}".strip(),
                team_name=e.get("name", ""), fpl_ids=ids, bank=float(last.get("bank", 0)) / 10.0,
                value=float(last.get("value", 0)) / 10.0, free_transfers=int(ft), last_gw=int(last["event"]),
                overall_rank=e.get("summary_overall_rank"))
