"""
Dixon-Coles bivariate Poisson model, fit per-match to de-vigged odds.

Rather than a rolling historical team-strength fit, each match's own
pre-match 1X2 + Over/Under 2.5 odds are inverted directly into that match's
implied (lambda_home, lambda_away, rho) - see PLAN.md for why. We get 3
independent constraints from the odds (P(home), P(draw) - P(away) is
implied by these two summing to 1 with them - and P(over 2.5)), which
exactly identifies the 3 unknowns.

rho is the "draw factor": plain independent Poisson underestimates how
often real matches finish as low-scoring draws, so Dixon-Coles applies a
correction tau to the four low-score cells (0-0, 1-0, 0-1, 1-1).
"""

from math import exp

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

MAX_GOALS = 10  # grid size for summing the score distribution; poisson tail
                # beyond this is negligible for realistic football lambdas


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(home=x, away=y) for x, y in [0, max_goals], tau-adjusted."""
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lam)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), mu)
    matrix = np.outer(home_pmf, away_pmf)

    for x in (0, 1):
        for y in (0, 1):
            matrix[x, y] *= tau(x, y, lam, mu, rho)

    return matrix


def market_probabilities_from_params(lam: float, mu: float, rho: float) -> dict:
    """Model-implied 1X2 + O/U 2.5 probabilities for a candidate (lam, mu, rho)."""
    matrix = score_matrix(lam, mu, rho)

    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()

    goals = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    under_2_5 = matrix[goals <= 2].sum()
    over_2_5 = 1 - under_2_5

    return {
        "p_home": home_win,
        "p_draw": draw,
        "p_away": away_win,
        "p_over_2_5": over_2_5,
        "p_under_2_5": under_2_5,
    }


def _residuals(params: np.ndarray, market: dict) -> np.ndarray:
    lam, mu, rho = params
    model = market_probabilities_from_params(lam, mu, rho)
    return np.array([
        model["p_home"] - market["p_home"],
        model["p_draw"] - market["p_draw"],
        model["p_over_2_5"] - market["p_over_2_5"],
    ])


def solve_match_params(market: dict, initial_guess=(1.4, 1.2, -0.1)) -> dict:
    """
    Invert de-vigged 1X2 + O/U 2.5 probabilities (see odds.py) into
    (lambda_home, lambda_away, rho) for a single match.
    """
    result = least_squares(
        _residuals,
        x0=np.array(initial_guess),
        args=(market,),
        bounds=([0.05, 0.05, -0.9], [8.0, 8.0, 0.9]),
    )

    lam, mu, rho = result.x
    return {
        "lambda_home": lam,
        "lambda_away": mu,
        "rho": rho,
        "fit_residual": float(np.abs(result.fun).max()),
    }


def derive_first_goal_quantities(lam: float, mu: float, rho: float) -> dict:
    """P(no goal), P(home scores first), P(away scores first)."""
    p_no_goal = tau(0, 0, lam, mu, rho) * poisson.pmf(0, lam) * poisson.pmf(0, mu)
    p_any_goal = 1 - p_no_goal

    p_home_first = (lam / (lam + mu)) * p_any_goal
    p_away_first = (mu / (lam + mu)) * p_any_goal

    return {
        "p_no_goal": p_no_goal,
        "p_home_first": p_home_first,
        "p_away_first": p_away_first,
    }


if __name__ == "__main__":
    from db import get_connection
    from odds import match_probabilities

    conn = get_connection()
    row = conn.execute("""
        SELECT * FROM matches
        WHERE league='EPL' AND season=2025 AND avg_home IS NOT NULL
        LIMIT 5
    """).fetchall()

    for r in row:
        market = match_probabilities(r)
        if market is None:
            continue

        fit = solve_match_params(market)
        first_goal = derive_first_goal_quantities(fit["lambda_home"], fit["lambda_away"], fit["rho"])

        print(f"{r['home_team']} vs {r['away_team']} "
              f"(actual: {r['home_goals']}-{r['away_goals']})")
        print(f"  market:  H={market['p_home']:.3f} D={market['p_draw']:.3f} A={market['p_away']:.3f} "
              f"O2.5={market['p_over_2_5']:.3f}")
        print(f"  fitted:  lambda_home={fit['lambda_home']:.3f} lambda_away={fit['lambda_away']:.3f} "
              f"rho={fit['rho']:.3f}  (max residual={fit['fit_residual']:.5f})")
        print(f"  derived: P(no_goal)={first_goal['p_no_goal']:.3f} "
              f"P(home_first)={first_goal['p_home_first']:.3f} "
              f"P(away_first)={first_goal['p_away_first']:.3f}")
        print()

    conn.close()
