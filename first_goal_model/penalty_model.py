"""
Penalty expected-goals model.

P(team awarded >=1 penalty in a match) via logistic regression, fit PER
LEAGUE:
    logit(P) = b0 + b1*lambda_own + b2*lambda_opp + b3*draw_rate

where lambda_own/lambda_opp are the Stage 1 odds-implied expected goals
(read from the cached dc_lambda_home/away columns) and draw_rate is the
team's own decayed historical penalty-award rate.

A 4th feature, the opponent's decayed penalty-CONCESSION rate, was tried
and dropped: its raw correlation with the target was tiny (0.025), it was
substantially redundant with lambda_opp (correlation -0.27 between the two
features), and a held-out comparison with vs without it showed dropping it
was equal-or-better in 4 of 5 leagues (the 5th showed a difference within
noise). See conversation for the full comparison.

Unlike the header model, there are no per-team parameters here - team
identity flows in only through the decayed rate feature, so the regression
only needs 4 global (per-league) coefficients. That's a much
better-conditioned problem for an event this rare (~0.15-0.2 penalties per
team per match) than trying to fit a header-style per-team parameter would
be.

Decay applies ONLY to the rate feature, not the regression fit itself - the
coefficients are already well-supported by thousands of pooled observations
per league without needing their own recency weighting. xi is fixed to the
header model's CV-validated value (1.5yr half-life) rather than separately
tuned - a per-league xi cross-val came back essentially flat with picks
landing on grid boundaries for 3/5 leagues, not real signal.

No coefficient regularization: an L2-penalty cross-val was tried, but the
per-league "best" values weren't distinguishable from reg_lambda=0 at the
single-fold level (|score difference| / standard error was 0.02-0.39 across
leagues, nowhere close to significant) - the differences were noise, not
tuned precision, so plain unregularized MLE is used instead.

Multiple penalties for one team in one match (rare) get treated identically
to one, a standard and accepted simplification for a binary outcome model.

lambda_penalty for a fixture = P(penalty) * average conversion rate,
treated as a single global constant (conversion rate is high, ~75-80%, and
far less variable across teams than award rate - not worth its own
parameter). Built in penalty_lambda.py once this model is validated.
"""

from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from db import get_connection

HALF_WEEK_DAYS = 3.5
PSEUDO_COUNT = 100.0  # shrinkage strength for the rate feature, toward league average
DEFAULT_XI = 0.00443  # fixed to the header model's CV-validated value (1.5yr half-life) - see module docstring


def _parse(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


def decay_weight(days_ago: float, xi_per_half_week: float) -> float:
    return np.exp(-xi_per_half_week * (days_ago / HALF_WEEK_DAYS))


def team_penalty_match_history(conn, team: str, league: str, before_date: str):
    """Every match `team` played before before_date, with penalties awarded to them."""
    rows = conn.execute("""
        SELECT m.match_id, m.datetime,
               SUM(CASE WHEN ((m.home_team=? AND s.h_a='h') OR (m.away_team=? AND s.h_a='a'))
                        AND s.situation='Penalty' THEN 1 ELSE 0 END) AS pens_for
        FROM matches m LEFT JOIN shots s ON s.match_id = m.match_id
        WHERE m.league = ? AND m.datetime < ? AND m.is_result = 1
          AND (m.home_team = ? OR m.away_team = ?)
        GROUP BY m.match_id
        ORDER BY m.datetime ASC
    """, (team, team, league, before_date, team, team)).fetchall()
    return rows


def league_penalty_rate(conn, league: str, before_date: str) -> float:
    row = conn.execute("""
        SELECT COUNT(DISTINCT s.match_id || '-' || s.h_a) * 1.0
               / (SELECT COUNT(*) * 2 FROM matches WHERE league=? AND datetime<? AND is_result=1)
        FROM shots s JOIN matches m ON s.match_id = m.match_id
        WHERE m.league = ? AND m.datetime < ? AND s.situation = 'Penalty'
    """, (league, before_date, league, before_date)).fetchone()
    return row[0] if row[0] is not None else 0.16


def decayed_rate(history, as_of_date: str, xi: float, prior_rate: float,
                  key: str, pseudo_count: float = PSEUDO_COUNT) -> float:
    """O(n) one-off version - kept for the simple/ad-hoc case (e.g. a single
    live prediction). For building a full training panel, use
    compute_decayed_series instead: computing this fresh per observation
    means O(n) per row, O(n^2) over a team's whole history - the original
    bottleneck (see conversation - a single league's CV took >180s)."""
    as_of = _parse(as_of_date)
    weighted_sum, weighted_total = 0.0, 0.0
    for row in history:
        days_ago = (as_of - _parse(row["datetime"])).total_seconds() / 86400
        w = decay_weight(days_ago, xi)
        weighted_total += w
        weighted_sum += w * row[key]
    return (weighted_sum + pseudo_count * prior_rate) / (weighted_total + pseudo_count)


def compute_decayed_series(history, xi: float, prior_rate: float, key: str,
                            pseudo_count: float = PSEUDO_COUNT) -> dict:
    """
    {match_id: decayed_rate_as_of_just_before_that_match} for one team's
    full chronological history, computed in O(n) instead of O(n^2).

    Exploits the semigroup property of exponential decay - decay(a)*decay(b)
    == decay(a+b) - to update a running weighted sum incrementally as we
    walk forward through time, rather than re-summing the full history from
    scratch for every match (which is what decayed_rate() does, and what
    made a naive per-observation loop O(n^2) over a team's history).

    Uses fractional (not integer) days for the elapsed-time calculation.
    With integer days, floor(a) + floor(b) <= floor(a+b) always - chaining
    many per-hop truncations together (as this function does, one hop per
    historical match) systematically under-decays compared to computing the
    elapsed time in one shot, and the error compounds over a team's history.
    An earlier version of this function used integer days and produced
    materially different fitted coefficients from decayed_rate() on the
    same data before this was caught and fixed.
    """
    result = {}
    running_sum, running_total = 0.0, 0.0
    last_dt = None

    for row in history:
        this_dt = _parse(row["datetime"])
        if last_dt is not None:
            elapsed_days = (this_dt - last_dt).total_seconds() / 86400
            factor = decay_weight(elapsed_days, xi)
            running_sum *= factor
            running_total *= factor

        # state here is decayed history strictly BEFORE this match - the
        # correct, leak-safe feature value to use for it
        result[row["match_id"]] = (running_sum + pseudo_count * prior_rate) / (running_total + pseudo_count)

        running_sum += row[key]
        running_total += 1.0
        last_dt = this_dt

    return result


def build_penalty_panel(conn, league: str, before_date: str):
    """
    One row per team per historical match: team, opponent, datetime,
    lambda_own, lambda_opp (from cached Stage 1), got_penalty (0/1 target),
    plus each team's own full penalty match history (for computing the two
    decayed rate features at whatever xi is chosen later - kept as raw
    history, not pre-decayed, so the same panel can be reused across
    multiple candidate xi values during cross-validation).
    """
    matches = conn.execute("""
        SELECT match_id, datetime, home_team, away_team, dc_lambda_home, dc_lambda_away
        FROM matches
        WHERE league = ? AND datetime < ? AND is_result = 1 AND dc_lambda_home IS NOT NULL
        ORDER BY datetime ASC
    """, (league, before_date)).fetchall()

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


def build_feature_matrix(panel: list, history_cache: dict, league_rate: float, xi: float):
    """
    X, y for any subset of panel rows - shared by fitting (on a training
    subset) and held-out scoring (on a validation subset, using
    coefficients fit elsewhere). Each row's features only ever use that
    row's own team/opponent history strictly before that row's own
    datetime, so this is leak-safe regardless of which subset of panel is
    passed in.

    Decayed rates are precomputed ONCE per team (via compute_decayed_series,
    O(n) per team) rather than recomputed per panel row - the panel may be
    a small subset (e.g. one validation fold) of a much bigger history, but
    the underlying per-team series only needs computing once regardless of
    how many subsets get scored against it.
    """
    draw_series = {team: compute_decayed_series(hist, xi, league_rate, key="pens_for")
                   for team, hist in history_cache.items()}

    X, y = [], []
    for row in panel:
        draw_rate = draw_series[row["team"]][row["match_id"]]

        X.append([1.0, row["lambda_own"], row["lambda_opp"], draw_rate])
        y.append(row["got_penalty"])

    return np.array(X), np.array(y)


def bernoulli_log_likelihood(X, y, beta) -> float:
    z = X @ beta
    log_p = -np.logaddexp(0, -z)
    log_1mp = -np.logaddexp(0, z)
    return float(np.sum(y * log_p + (1 - y) * log_1mp))


def fit_beta(X, y, reg_lambda: float = 0.0):
    """
    Fit logistic regression coefficients by MLE, with an optional L2 penalty
    on the slope coefficients (NOT the intercept, standard practice for
    ridge-regularized logistic regression - the intercept sets the baseline
    rate and regularizing it would bias predictions toward 50/50 regardless
    of the data).
    """
    def neg_log_likelihood(beta):
        penalty = reg_lambda * np.sum(beta[1:] ** 2)
        return -bernoulli_log_likelihood(X, y, beta) + penalty

    result = minimize(neg_log_likelihood, x0=np.zeros(X.shape[1]), method="L-BFGS-B")
    return result.x, result.success


def fit_penalty_model(conn, league: str, panel: list, history_cache: dict, before_date: str,
                       xi: float, reg_lambda: float = 0.0):
    league_rate = league_penalty_rate(conn, league, before_date)
    X, y = build_feature_matrix(panel, history_cache, league_rate, xi)

    beta, converged = fit_beta(X, y, reg_lambda)

    return {"coefficients": beta, "converged": converged,
            "n_observations": len(y), "n_positive": int(y.sum()),
            "final_nll": -bernoulli_log_likelihood(X, y, beta)}


if __name__ == "__main__":
    conn = get_connection()
    league = "EPL"
    as_of = "2026-05-25 00:00:00"

    panel, history_cache = build_penalty_panel(conn, league, as_of)
    fit = fit_penalty_model(conn, league, panel, history_cache, as_of, DEFAULT_XI)

    print(f"{league}: {fit['n_observations']} observations, {fit['n_positive']} positive "
          f"({100*fit['n_positive']/fit['n_observations']:.1f}%), converged={fit['converged']}")
    names = ["intercept", "lambda_own", "lambda_opp", "draw_rate"]
    for name, coef in zip(names, fit["coefficients"]):
        print(f"  {name:16s} {coef:+.4f}")

    conn.close()
