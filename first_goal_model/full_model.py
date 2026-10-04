"""
Unified first-goal model - the original "10 competing Poisson processes"
design from PLAN.md, now assembled from the validated sub-models:

  - Stage 1 (Dixon-Coles fit to pre-match odds): dc_lambda_home/away,
    dc_rho - cached per match in `matches`.
  - Header: lambda_header(i) = lambda_total(i) * c * a_i * b_j * calib,
    using the per-match cached header_a/b/scale_c columns. `calib` is a
    per-league correction factor (actual header goals / total headed xG,
    leak-safe as-of that match's date) - headed xG systematically
    overstates actual header conversion (well-documented in football
    analytics: headers underperform their xG relative to footed shots).
    Checked directly in this dataset: headed shots convert at ~0.84-0.91x
    their total xG league-wide, vs ~1.05x for non-headed non-penalty shots
    (i.e. those slightly UNDER-predict). Without this correction, a 300-match
    sample showed mean predicted header probability of 18.2% against an
    actual rate of 14.3% - a real, systematic overestimate, not noise.
  - Penalty: lambda_penalty(i) = lambda_total(i) * penalty_share(team) -
    the decayed, shrunk fraction of a team's own goals that are penalties
    (method_shares.penalty_share). No opponent term here (see conversation
    - this simpler form was compared against a fitted logistic regression
    with lambda_opp; the regression won in 4/5 leagues, roughly tied in
    EPL, but this simpler, more consistent-with-the-rest-of-the-model form
    was chosen for the unified model).
  - Free kick / own goal: lambda(i) = lambda_total(i) * league-wide
    constant share (not team-specific - see method_shares.py docstrings
    for why).
  - no_goal: tau(0,0)-adjusted P(0,0) from dc_lambda_home/away/dc_rho
    (dixon_coles.tau).

Every non-shot lambda is subtracted from lambda_total to get lambda_shot as
the residual (with a safety clamp - see MAX_NON_SHOT_SHARE - so the four
non-shot lambdas can never add up to more than 95% of lambda_total and push
lambda_shot negative). Final probabilities: each of the 10 (5 methods x 2
teams) lambdas divided by the total rate (lambda_home + lambda_away),
times P(at least one goal) - the same "race between competing Poisson
processes" logic used everywhere else in this project, just extended from
2 processes (home/away) to 10 (method x side).
"""

import numpy as np
from scipy.stats import poisson

from db import get_connection
from dixon_coles import tau
from method_shares import penalty_share, league_free_kick_share, league_own_goal_share

MAX_NON_SHOT_SHARE = 0.95
METHODS = ("header", "penalty", "free_kick", "own_goal")  # shot handled as residual


def header_calibration_factor(conn, league: str, as_of_date: str) -> float:
    row = conn.execute("""
        SELECT SUM(s.xg) as total_headed_xg, SUM(CASE WHEN s.result='Goal' THEN 1 ELSE 0 END) as actual_goals
        FROM shots s JOIN matches m ON s.match_id = m.match_id
        WHERE s.shot_type='Head' AND m.league=? AND m.datetime<?
    """, (league, as_of_date)).fetchone()
    if row["total_headed_xg"] is None or row["total_headed_xg"] == 0:
        return 0.87  # dataset-wide average fallback
    return row["actual_goals"] / row["total_headed_xg"]


def predict_full(conn, match_row: dict, league: str, as_of_date: str) -> dict:
    lam_home, lam_away, rho = match_row["dc_lambda_home"], match_row["dc_lambda_away"], match_row["dc_rho"]
    c = match_row["header_scale_c"]
    header_calib = header_calibration_factor(conn, league, as_of_date)

    lam = {
        "header": {
            "home": lam_home * c * match_row["header_a_home"] * match_row["header_b_away"] * header_calib,
            "away": lam_away * c * match_row["header_a_away"] * match_row["header_b_home"] * header_calib,
        },
    }

    pen_share_home = penalty_share(conn, match_row["home_team"], league, as_of_date)["rate"]
    pen_share_away = penalty_share(conn, match_row["away_team"], league, as_of_date)["rate"]
    lam["penalty"] = {"home": lam_home * pen_share_home, "away": lam_away * pen_share_away}

    fk_share = league_free_kick_share(conn, league, as_of_date)
    lam["free_kick"] = {"home": lam_home * fk_share, "away": lam_away * fk_share}

    og_share = league_own_goal_share(conn, league, as_of_date)
    lam["own_goal"] = {"home": lam_home * og_share, "away": lam_away * og_share}

    # safety clamp per side, then shot as residual
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
    row = conn.execute("""
        SELECT * FROM matches WHERE match_id = 29084
    """).fetchone()
    row = dict(row)

    dist = predict_full(conn, row, "EPL", row["datetime"])
    print(f"{row['home_team']} vs {row['away_team']} ({row['datetime']})")
    for method, p in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {method:10s} {p:.4f}")
    print(f"  sum = {sum(dist.values()):.4f}")

    conn.close()
