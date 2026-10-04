"""
Fits the header model and penalty model (B) ONCE on a fixed training
window (seasons 2016-2018, i.e. matches from 2016-08-01 to 2019-07-01),
and freezes the result to JSON. This replaces the live per-match refit
that populate_model_columns.py does for header_a/b/scale_c - here every
team gets ONE fixed (a, b) pair and every league gets ONE fixed c and
calibration factor, fit once and reused unchanged for every match in the
test period (2019-present).

Why this is still leak-safe even though it's not a per-match refit: the
test period (2019+) never appears in the training window used here, and
the one feature that still updates live (penalty model B's draw_rate) is
computed from history strictly before each test match's own kickoff, same
as the original design.

Run once:  python3 train_frozen_constants.py
Produces:  header_constants_2016_2018.json
           penalty_constants_2016_2018.json
"""

import json

import numpy as np

from db import get_connection
from header_strength import fit_header_strength_from_panel, DEFAULT_XI as HEADER_XI
from penalty_model import (
    team_penalty_match_history, league_penalty_rate, compute_decayed_series,
    build_feature_matrix, fit_beta, bernoulli_log_likelihood, DEFAULT_XI as PENALTY_XI,
)

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
TRAIN_AFTER = "2016-08-01 00:00:00"   # start of season-2016 window
TRAIN_BEFORE = "2019-07-01 00:00:00"  # end of season-2018 window (exclusive upper bound)
FALLBACK_CONVERSION_RATE = 0.78  # dataset-wide fallback if a league's training window sample is too thin
MIN_PENALTY_ATTEMPTS = 20


# ---------------------------------------------------------------------------
# Header constants
# ---------------------------------------------------------------------------

def build_header_training_panel(conn, league: str, after_date: str, before_date: str):
    """
    Same shape as header_strength.build_team_match_panel / the cached
    variant in populate_model_columns.py, but restricted to
    [after_date, before_date) and reading lambda_total from the already
    -populated dc_lambda_home/away columns (no need to re-solve Stage 1 -
    that fit only ever depends on the match's own odds).
    """
    matches = conn.execute("""
        SELECT match_id, datetime, home_team, away_team, dc_lambda_home, dc_lambda_away
        FROM matches
        WHERE league = ? AND datetime >= ? AND datetime < ? AND is_result = 1
          AND dc_lambda_home IS NOT NULL
        ORDER BY datetime ASC
    """, (league, after_date, before_date)).fetchall()

    panel = []
    for m in matches:
        headed_xg = conn.execute("""
            SELECT h_a, SUM(CASE WHEN shot_type='Head' THEN xg ELSE 0 END) as hxg
            FROM shots WHERE match_id = ? GROUP BY h_a
        """, (m["match_id"],)).fetchall()
        headed_xg_by_side = {r["h_a"]: r["hxg"] for r in headed_xg}

        panel.append({
            "team": m["home_team"], "opponent": m["away_team"], "datetime": m["datetime"],
            "lambda_total": m["dc_lambda_home"], "headed_xg": headed_xg_by_side.get("h", 0.0),
        })
        panel.append({
            "team": m["away_team"], "opponent": m["home_team"], "datetime": m["datetime"],
            "lambda_total": m["dc_lambda_away"], "headed_xg": headed_xg_by_side.get("a", 0.0),
        })

    return panel


def header_calibration_factor_window(conn, league: str, after_date: str, before_date: str) -> float:
    row = conn.execute("""
        SELECT SUM(s.xg) as total_headed_xg, SUM(CASE WHEN s.result='Goal' THEN 1 ELSE 0 END) as actual_goals
        FROM shots s JOIN matches m ON s.match_id = m.match_id
        WHERE s.shot_type='Head' AND m.league=? AND m.datetime>=? AND m.datetime<?
    """, (league, after_date, before_date)).fetchone()
    if row["total_headed_xg"] is None or row["total_headed_xg"] == 0:
        return 0.87
    return row["actual_goals"] / row["total_headed_xg"]


def fit_header_constants(conn) -> dict:
    out = {}
    for league in LEAGUES:
        panel = build_header_training_panel(conn, league, TRAIN_AFTER, TRAIN_BEFORE)
        fit = fit_header_strength_from_panel(panel, TRAIN_BEFORE, HEADER_XI)
        calib = header_calibration_factor_window(conn, league, TRAIN_AFTER, TRAIN_BEFORE)

        out[league] = {
            "strengths": fit["strengths"], "scale": fit["scale"], "calib": calib,
            "n_observations": fit["n_observations"], "converged": fit["converged"],
        }
        print(f"  header[{league}]: n={fit['n_observations']}, converged={fit['converged']}, "
              f"c={fit['scale']:.4f}, calib={calib:.4f}")
    return out


# ---------------------------------------------------------------------------
# Penalty (model B) constants
# ---------------------------------------------------------------------------

def build_penalty_training_panel(conn, league: str, after_date: str, before_date: str):
    """
    Training rows restricted to [after_date, before_date) (the labels the
    regression is fit on), but each team's decay-weighted history feature
    still draws on their FULL history strictly before `before_date` (can
    reach back past after_date) - same leak-safe convention as
    penalty_model.build_penalty_panel, just with the label set windowed.
    """
    matches = conn.execute("""
        SELECT match_id, datetime, home_team, away_team, dc_lambda_home, dc_lambda_away
        FROM matches
        WHERE league = ? AND datetime >= ? AND datetime < ? AND is_result = 1 AND dc_lambda_home IS NOT NULL
        ORDER BY datetime ASC
    """, (league, after_date, before_date)).fetchall()

    history_cache = {}

    def get_history(team):
        if team not in history_cache:
            history_cache[team] = team_penalty_match_history(conn, team, league, before_date)
        return history_cache[team]

    panel = []
    for m in matches:
        get_history(m["home_team"])
        get_history(m["away_team"])

        pens = conn.execute(
            "SELECT h_a, COUNT(*) as n FROM shots WHERE match_id=? AND situation='Penalty' GROUP BY h_a",
            (m["match_id"],),
        ).fetchall()
        pens_by_side = {r["h_a"]: r["n"] for r in pens}

        panel.append({
            "match_id": m["match_id"], "team": m["home_team"], "opponent": m["away_team"], "datetime": m["datetime"],
            "lambda_own": m["dc_lambda_home"], "lambda_opp": m["dc_lambda_away"],
            "got_penalty": 1 if pens_by_side.get("h", 0) > 0 else 0,
        })
        panel.append({
            "match_id": m["match_id"], "team": m["away_team"], "opponent": m["home_team"], "datetime": m["datetime"],
            "lambda_own": m["dc_lambda_away"], "lambda_opp": m["dc_lambda_home"],
            "got_penalty": 1 if pens_by_side.get("a", 0) > 0 else 0,
        })

    return panel, history_cache


def penalty_conversion_rate_window(conn, league: str, after_date: str, before_date: str) -> float:
    row = conn.execute("""
        SELECT SUM(CASE WHEN result='Goal' THEN 1 ELSE 0 END) as scored, COUNT(*) as attempts
        FROM shots s JOIN matches m ON s.match_id = m.match_id
        WHERE s.situation='Penalty' AND m.league=? AND m.datetime>=? AND m.datetime<?
    """, (league, after_date, before_date)).fetchone()
    if row["attempts"] is None or row["attempts"] < MIN_PENALTY_ATTEMPTS:
        return FALLBACK_CONVERSION_RATE
    return row["scored"] / row["attempts"]


# Leagues where the full 4-feature model (intercept, lambda_own, lambda_opp,
# draw_rate) came back unstable on the 2016-2018 window (La_liga's draw_rate
# coefficient sign-flipped to -8.06, vs strongly positive in EPL/Ligue_1/
# Serie_A - an artifact of fitting on ~2280 rows / ~340 positives, not a
# real effect). These leagues drop draw_rate and fit the simpler 3-feature
# model (intercept, lambda_own, lambda_opp) instead.
SIMPLE_MODEL_LEAGUES = {"La_liga"}


def fit_penalty_constants(conn) -> dict:
    out = {}
    for league in LEAGUES:
        panel, history_cache = build_penalty_training_panel(conn, league, TRAIN_AFTER, TRAIN_BEFORE)
        league_rate = league_penalty_rate(conn, league, TRAIN_BEFORE)
        X, y = build_feature_matrix(panel, history_cache, league_rate, PENALTY_XI)

        simple = league in SIMPLE_MODEL_LEAGUES
        if simple:
            X = X[:, :3]  # drop the draw_rate column, keep [intercept, lambda_own, lambda_opp]

        beta, converged = fit_beta(X, y, reg_lambda=0.0)
        conversion_rate = penalty_conversion_rate_window(conn, league, TRAIN_AFTER, TRAIN_BEFORE)

        out[league] = {
            # 4 entries [intercept, lambda_own, lambda_opp, draw_rate] for the full
            # model, 3 entries [intercept, lambda_own, lambda_opp] for the simple one.
            "coefficients": beta.tolist(),
            "model": "simple" if simple else "full",
            "league_rate": league_rate,
            "conversion_rate": conversion_rate,
            "n_observations": len(y), "n_positive": int(y.sum()), "converged": bool(converged),
        }
        print(f"  penalty[{league}] ({'simple' if simple else 'full'}): n={len(y)}, positive={int(y.sum())}, "
              f"converged={converged}, beta={np.round(beta, 4).tolist()}, "
              f"league_rate={league_rate:.4f}, conv={conversion_rate:.4f}")
    return out


if __name__ == "__main__":
    conn = get_connection()

    print(f"Training window: {TRAIN_AFTER} to {TRAIN_BEFORE} (seasons 2016-2018)\n")

    print("Fitting header constants...")
    header_constants = fit_header_constants(conn)
    with open("header_constants_2016_2018.json", "w") as f:
        json.dump(header_constants, f, indent=2)
    print("  -> header_constants_2016_2018.json\n")

    print("Fitting penalty (model B) constants...")
    penalty_constants = fit_penalty_constants(conn)
    with open("penalty_constants_2016_2018.json", "w") as f:
        json.dump(penalty_constants, f, indent=2)
    print("  -> penalty_constants_2016_2018.json")

    conn.close()
