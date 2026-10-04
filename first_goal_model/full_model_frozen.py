"""
Variant of full_model.py that uses FROZEN constants instead of a live
per-match refit:

  - header: lambda_header(i) = lambda_total(i) * c * a_i * b_j * calib,
    where c, a_i, b_j, calib all come from header_constants_2016_2018.json
    (fit ONCE on seasons 2016-2018 by train_frozen_constants.py) instead of
    being refit for every match. Teams not seen in that training window
    (promoted/relegated sides, teams that didn't exist in 2016-2018) fall
    back to a=b=1 (average).

  - penalty (model B): lambda_penalty(i) = P_i(awarded >= 1 penalty) *
    conversion_rate, where
        P_i(awarded) = sigmoid(b0 + b1*lambda_own + b2*lambda_opp + b3*draw_rate_i)
    beta=(b0,b1,b2,b3) and conversion_rate come from
    penalty_constants_2016_2018.json (also frozen on 2016-2018). draw_rate_i
    is the one quantity that still updates live: team i's own decayed,
    shrunk penalty-award rate using ALL history strictly before this
    match's kickoff (leak-safe), shrunk toward the FROZEN league_rate from
    the same training window (so train and test use a consistent prior).

    This P(>=1 award) -> lambda conversion relies on the standard small-
    probability Poisson approximation P(X>=1) = 1-exp(-lambda) ~= lambda:
    penalties are rare enough per team per match that "expected penalties
    awarded" and "probability of at least one" are nearly interchangeable
    (penalty_model.py's own docstring notes multiple-penalty matches are
    rare enough to treat identically to one anyway).

  - free_kick, own_goal, no_goal, shot (residual): unchanged from
    full_model.py - these were never part of this freeze request, and
    still use the live as-of-date league lookups / Stage 1 Dixon-Coles fit
    cached in dc_lambda_home/away/dc_rho.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from db import get_connection
from dixon_coles import tau
from method_shares import league_free_kick_share, league_own_goal_share
from penalty_model import team_penalty_match_history, compute_decayed_series

_DIR = Path(__file__).resolve().parent
MAX_NON_SHOT_SHARE = 0.95
METHODS = ("header", "penalty", "free_kick", "own_goal")


def load_constants():
    with open(_DIR / "header_constants_2016_2018.json") as f:
        header_constants = json.load(f)
    with open(_DIR / "penalty_constants_2016_2018.json") as f:
        penalty_constants = json.load(f)
    return header_constants, penalty_constants


def _header_strength(header_constants: dict, league: str, team: str) -> dict:
    return header_constants[league]["strengths"].get(team, {"a": 1.0, "b": 1.0})


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class DrawRateCache:
    """
    Precomputes each team's decayed penalty-award rate at every one of
    their own historical match_ids, ONCE per team (O(n) via
    compute_decayed_series), then serves O(1) lookups by match_id during a
    backtest loop - avoids an O(n) recompute per match, which otherwise
    makes a multi-season backtest very slow.

    Uses the FROZEN league_rate and xi from penalty_constants_2016_2018.json
    as the shrinkage prior, so train and test are consistent.
    """

    def __init__(self, conn, penalty_constants: dict, xi: float):
        self.conn = conn
        self.penalty_constants = penalty_constants
        self.xi = xi
        self._series_by_team = {}

    def _build(self, team: str, league: str, up_to_date: str):
        history = team_penalty_match_history(self.conn, team, league, up_to_date)
        league_rate = self.penalty_constants[league]["league_rate"]
        series = compute_decayed_series(history, self.xi, league_rate, key="pens_for")
        self._series_by_team[(team, league)] = series
        return series

    def rate(self, team: str, league: str, match_id: int, up_to_date: str) -> float:
        key = (team, league)
        series = self._series_by_team.get(key)
        if series is None or match_id not in series:
            series = self._build(team, league, up_to_date)
        league_rate = self.penalty_constants[league]["league_rate"]
        return series.get(match_id, league_rate)


def predict_full_frozen(conn, match_row: dict, league: str, header_constants: dict,
                         penalty_constants: dict, draw_rate_cache: "DrawRateCache") -> dict:
    lam_home, lam_away, rho = match_row["dc_lambda_home"], match_row["dc_lambda_away"], match_row["dc_rho"]
    as_of_date = match_row["datetime"]

    # ---- header (frozen a, b, c, calib) ----
    hc = header_constants[league]
    home_h = _header_strength(header_constants, league, match_row["home_team"])
    away_h = _header_strength(header_constants, league, match_row["away_team"])
    c, calib = hc["scale"], hc["calib"]

    lam = {
        "header": {
            "home": lam_home * c * home_h["a"] * away_h["b"] * calib,
            "away": lam_away * c * away_h["a"] * home_h["b"] * calib,
        },
    }

    # ---- penalty (model B: logistic P(awarded) * conversion_rate) ----
    # Most leagues use the full 4-feature model (intercept, lambda_own,
    # lambda_opp, draw_rate); leagues in SIMPLE_MODEL_LEAGUES (see
    # train_frozen_constants.py - currently La_liga, whose draw_rate
    # coefficient was unstable on the 2016-2018 window) use the simpler
    # 3-feature model with draw_rate dropped entirely.
    pc = penalty_constants[league]
    coeffs = pc["coefficients"]
    is_simple = len(coeffs) == 3

    if is_simple:
        b0, b1, b2 = coeffs
        p_pen_home = sigmoid(b0 + b1 * lam_home + b2 * lam_away)
        p_pen_away = sigmoid(b0 + b1 * lam_away + b2 * lam_home)
    else:
        b0, b1, b2, b3 = coeffs
        draw_home = draw_rate_cache.rate(match_row["home_team"], league, match_row["match_id"], as_of_date)
        draw_away = draw_rate_cache.rate(match_row["away_team"], league, match_row["match_id"], as_of_date)
        p_pen_home = sigmoid(b0 + b1 * lam_home + b2 * lam_away + b3 * draw_home)
        p_pen_away = sigmoid(b0 + b1 * lam_away + b2 * lam_home + b3 * draw_away)

    conversion_rate = pc["conversion_rate"]

    lam["penalty"] = {"home": p_pen_home * conversion_rate, "away": p_pen_away * conversion_rate}

    # ---- free_kick, own_goal: unchanged, live league lookups ----
    fk_share = league_free_kick_share(conn, league, as_of_date)
    lam["free_kick"] = {"home": lam_home * fk_share, "away": lam_away * fk_share}

    og_share = league_own_goal_share(conn, league, as_of_date)
    lam["own_goal"] = {"home": lam_home * og_share, "away": lam_away * og_share}

    # ---- safety clamp per side, then shot as residual ----
    lam_shot = {}
    for side, lam_total in (("home", lam_home), ("away", lam_away)):
        non_shot_total = sum(lam[m][side] for m in METHODS)
        cap = MAX_NON_SHOT_SHARE * lam_total
        if non_shot_total > cap:
            scale = cap / non_shot_total
            for m in METHODS:
                lam[m][side] *= scale
            non_shot_total = cap
        lam_shot[side] = lam_total - non_shot_total

    p_no_goal = tau(0, 0, lam_home, lam_away, rho) * poisson.pmf(0, lam_home) * poisson.pmf(0, lam_away)
    p_any_goal = 1 - p_no_goal
    total_rate = lam_home + lam_away

    result = {"no_goal": p_no_goal}
    result["shot"] = (lam_shot["home"] + lam_shot["away"]) / total_rate * p_any_goal
    for m in METHODS:
        result[m] = (lam[m]["home"] + lam[m]["away"]) / total_rate * p_any_goal

    return result


if __name__ == "__main__":
    conn = get_connection()
    header_constants, penalty_constants = load_constants()
    draw_rate_cache = DrawRateCache(conn, penalty_constants, xi=0.00443)

    row = conn.execute("SELECT * FROM matches WHERE match_id = 29084").fetchone()
    row = dict(row)

    dist = predict_full_frozen(conn, row, "EPL", header_constants, penalty_constants, draw_rate_cache)
    print(f"{row['home_team']} vs {row['away_team']} ({row['datetime']})")
    for method, p in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {method:10s} {p:.4f}")
    print(f"  sum = {sum(dist.values()):.4f}")

    conn.close()
