"""
Header expected-goals model, fit Dixon-Coles style.

For team i playing in a given match:
    lambda_header(i) = lambda_total(i) * c * a_i * b_j

where lambda_total(i) is the Stage 1 odds-implied expected goals for team i
in THAT SPECIFIC MATCH (already computed per-match from real odds, not a
team-level average), a_i is team i's header-scoring strength, b_j is
opponent j's header-concession strength, and c is a single global scale
parameter shared by every team (same role as Dixon-Coles' own intercept).

c matters more than it looks: headers are only ~16% of a team's goals, so
an "average" team's a_i and b_i individually need to sit around sqrt(0.16)
~= 0.4, not 1, for their product (times c=1) to reproduce that. Without a
separate c, the regularization below (which shrinks a_i, b_i toward 1) drags
small-sample teams toward the WRONG center - a first version of this model
without c showed exactly that bug: low-sample teams (promoted/relegated
sides) clustered near a=1 while high-sample teams (Man City, Liverpool)
were dragged down toward the true ~0.4-0.5 level, backwards from what
shrinkage should do. c absorbs the overall scale so a_i, b_i can be genuine,
correctly-centered relative strengths.

Fit via decayed maximum likelihood across a league's match history, using
each team's realized headed xG per match as the likelihood target. Headed
xG is continuous, not a count, but we use the standard pragmatic treatment
(common in public xG models) of plugging it into a Poisson likelihood
kernel anyway: per team-match observation,
    ll = -lambda_header + headed_xg * ln(lambda_header)
(the ln(x!) term is dropped - constant w.r.t. parameters, doesn't affect
the fit).

a_i, b_i are optimized in log-space (a_i = exp(alpha_i)) so they stay
positive automatically, with an L2 penalty pulling alpha_i, beta_i toward 0
(i.e. shrinking toward "average team", a_i = b_i = 1). This penalty does
double duty: it regularizes small-sample teams AND resolves the
identifiability problem (a_i, b_i only ever appear as a product, so
shifting all alpha up and all beta down by the same constant would
otherwise leave every prediction unchanged - the penalty breaks that flat
direction since it increases on both sides of the shift).

Fit separately per league - teams from different leagues never actually
play each other, so there's no need to force cross-league comparability,
and it keeps each optimization small (~20-24 teams instead of ~100).

Time decay follows the original Dixon-Coles convention: weight = exp(-xi*t),
t measured in half-weeks before the as-of date (their published value was
xi=0.0065). xi is a FIXED input here, not fit jointly with a_i/b_i/c - an
earlier attempt at including it as a free MLE parameter found a degenerate
optimum (xi drifting to the point of discarding nearly all data, since an
unweighted-sum likelihood always trivially "improves" by having fewer
nonzero-weight terms). xi is instead selected via cross-validation - see
select_xi.py - which doesn't have that failure mode, since a model that's
discarded its data will generalize badly to a held-out season.
"""

import numpy as np
from scipy.optimize import minimize

from db import get_connection
from dixon_coles import solve_match_params
from odds import match_probabilities

REG_LAMBDA = 5.0  # L2 penalty strength on log-strengths, toward a_i=b_i=1
HALF_WEEK_DAYS = 3.5
DEFAULT_XI = 0.00443  # selected via select_xi.py: walk-forward CV on EPL seasons 2014-2023 (2024/25, 2025/26
                       # held out entirely for full-model evaluation), best of 8 candidates -> 1.5yr half-life


def build_team_match_panel(conn, league: str, before_date: str):
    """
    One row per team per historical match: (team, opponent, datetime,
    lambda_total, headed_xg). lambda_total comes from re-solving Stage 1
    on that match's own real odds - the same de-vig + Dixon-Coles inversion
    used for live predictions, just applied retrospectively.
    """
    matches = conn.execute("""
        SELECT match_id, datetime, home_team, away_team,
               avg_home, avg_draw, avg_away, avg_over_2_5, avg_under_2_5
        FROM matches
        WHERE league = ? AND datetime < ? AND is_result = 1
          AND avg_home IS NOT NULL AND avg_over_2_5 IS NOT NULL
        ORDER BY datetime ASC
    """, (league, before_date)).fetchall()

    panel = []
    for m in matches:
        market = match_probabilities(m)
        if market is None:
            continue
        fit = solve_match_params(market)
        lam_home, lam_away = fit["lambda_home"], fit["lambda_away"]

        headed_xg = conn.execute("""
            SELECT h_a, SUM(CASE WHEN shot_type='Head' THEN xg ELSE 0 END) as hxg
            FROM shots WHERE match_id = ? GROUP BY h_a
        """, (m["match_id"],)).fetchall()
        headed_xg_by_side = {r["h_a"]: r["hxg"] for r in headed_xg}

        panel.append({
            "team": m["home_team"], "opponent": m["away_team"], "datetime": m["datetime"],
            "lambda_total": lam_home, "headed_xg": headed_xg_by_side.get("h", 0.0),
        })
        panel.append({
            "team": m["away_team"], "opponent": m["home_team"], "datetime": m["datetime"],
            "lambda_total": lam_away, "headed_xg": headed_xg_by_side.get("a", 0.0),
        })

    return panel


def fit_header_strength_from_panel(panel: list, as_of_date: str, xi: float,
                                    reg_lambda: float = REG_LAMBDA) -> dict:
    """
    Core fit, taking an already-built panel (so callers doing repeated
    fits over overlapping windows - e.g. cross-validation folds - don't
    redundantly re-query and re-solve Stage 1 for the same historical
    matches every time).
    """
    teams = sorted(set(row["team"] for row in panel) | set(row["opponent"] for row in panel))
    idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    as_of = _parse(as_of_date)
    t_half_weeks = np.array([(as_of - _parse(row["datetime"])).days / HALF_WEEK_DAYS for row in panel])
    weights = np.exp(-xi * t_half_weeks)
    lambda_total = np.array([row["lambda_total"] for row in panel])
    headed_xg = np.array([row["headed_xg"] for row in panel])
    team_i = np.array([idx[row["team"]] for row in panel])
    opp_j = np.array([idx[row["opponent"]] for row in panel])

    def neg_log_likelihood(params):
        gamma = params[0]
        alpha, beta = params[1:1 + n], params[1 + n:]
        c = np.exp(gamma)
        a = np.exp(alpha[team_i])
        b = np.exp(beta[opp_j])
        lam_header = lambda_total * c * a * b

        ll = weights * (-lam_header + headed_xg * np.log(lam_header))
        penalty = reg_lambda * (np.sum(alpha ** 2) + np.sum(beta ** 2))
        return -np.sum(ll) + penalty

    x0 = np.zeros(1 + 2 * n)
    x0[0] = np.log(0.16)
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")

    gamma = result.x[0]
    alpha, beta = result.x[1:1 + n], result.x[1 + n:]
    c = float(np.exp(gamma))
    strengths = {
        team: {"a": float(np.exp(alpha[idx[team]])), "b": float(np.exp(beta[idx[team]]))}
        for team in teams
    }

    return {"strengths": strengths, "scale": c, "xi_per_half_week": xi,
            "n_observations": len(panel), "converged": result.success, "final_nll": result.fun}


def fit_header_strength(conn, league: str, as_of_date: str, xi: float = DEFAULT_XI,
                         reg_lambda: float = REG_LAMBDA) -> dict:
    panel = build_team_match_panel(conn, league, as_of_date)
    return fit_header_strength_from_panel(panel, as_of_date, xi, reg_lambda)


def _parse(dt_str: str):
    from datetime import datetime
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    conn = get_connection()
    fit = fit_header_strength(conn, "EPL", "2025-08-01 00:00:00")

    print(f"Fit on {fit['n_observations']} team-match observations, converged={fit['converged']}")
    print(f"Global scale c = {fit['scale']:.4f}, xi = {fit['xi_per_half_week']} per half-week (default, see select_xi.py)")
    print()
    print("Header strengths (a=scoring, b=conceding), sorted by scoring strength:")
    ranked = sorted(fit["strengths"].items(), key=lambda kv: -kv[1]["a"])
    for team, s in ranked:
        print(f"  {team:22s} a={s['a']:.3f}  b={s['b']:.3f}")

    conn.close()
