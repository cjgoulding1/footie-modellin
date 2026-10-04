"""
Select the penalty-rate decay constant xi via walk-forward cross-validation,
PER LEAGUE, restricted entirely to seasons 2014-2023 - seasons 2024 (2024/25)
and 2025 (2025/26) are reserved untouched for full-model evaluation later,
same discipline as select_xi.py for the header model.

Only the two rate features (draw_rate, concession_rate) are decayed -
see penalty_model.py - so xi only affects how those are computed, not the
regression fit itself.
"""

import numpy as np

from db import get_connection
from penalty_model import (
    build_penalty_panel, build_feature_matrix, bernoulli_log_likelihood, league_penalty_rate,
)
from scipy.optimize import minimize

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
CUTOFF_DATE = "2024-08-01 00:00:00"  # safely before every league's season-2024 start
FOLD_SEASONS = [2019, 2020, 2021, 2022, 2023]

XI_CANDIDATES_HALF_LIFE_YEARS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def half_life_years_to_xi(half_life_years: float) -> float:
    half_life_half_weeks = (half_life_years * 365) / 3.5
    return np.log(2) / half_life_half_weeks


def fit_beta(X, y):
    result = minimize(lambda beta: -bernoulli_log_likelihood(X, y, beta), x0=np.zeros(X.shape[1]), method="L-BFGS-B")
    return result.x


def select_xi_for_league(conn, league: str):
    fold_starts = {}
    fold_ends = {}
    for season in FOLD_SEASONS:
        fold_starts[season] = conn.execute(
            "SELECT MIN(datetime) FROM matches WHERE league=? AND season=?", (league, season)
        ).fetchone()[0]
        next_season_start = conn.execute(
            "SELECT MIN(datetime) FROM matches WHERE league=? AND season=?", (league, season + 1)
        ).fetchone()[0]
        fold_ends[season] = next_season_start

    panel, history_cache = build_penalty_panel(conn, league, CUTOFF_DATE)
    league_rate = league_penalty_rate(conn, league, CUTOFF_DATE)

    print(f"\n{league}: {len(panel)} observations through end of season 2023")

    results = {}
    for hl_years in XI_CANDIDATES_HALF_LIFE_YEARS:
        xi = half_life_years_to_xi(hl_years)
        fold_scores = []

        for season in FOLD_SEASONS:
            start, end = fold_starts[season], fold_ends[season]
            train_panel = [r for r in panel if r["datetime"] < start]
            holdout_panel = [r for r in panel if start <= r["datetime"] < end]
            if not holdout_panel or not train_panel:
                continue

            X_train, y_train = build_feature_matrix(train_panel, history_cache, league_rate, xi)
            beta = fit_beta(X_train, y_train)

            X_hold, y_hold = build_feature_matrix(holdout_panel, history_cache, league_rate, xi)
            ll = bernoulli_log_likelihood(X_hold, y_hold, beta)
            fold_scores.append(ll / len(y_hold))

        avg_score = float(np.mean(fold_scores))
        results[hl_years] = avg_score
        print(f"  half-life={hl_years:>4.2f}yr  (xi={xi:.5f}/half-week)  "
              f"avg per-obs held-out log-lik={avg_score:+.5f}")

    best_hl = max(results, key=results.get)
    best_xi = half_life_years_to_xi(best_hl)
    print(f"  -> best for {league}: half-life={best_hl}yr (xi={best_xi:.5f})")
    return best_hl, best_xi


if __name__ == "__main__":
    conn = get_connection()
    selected = {}
    for league in LEAGUES:
        best_hl, best_xi = select_xi_for_league(conn, league)
        selected[league] = {"half_life_years": best_hl, "xi": best_xi}
    conn.close()

    print("\n" + "=" * 50)
    print("SELECTED PER-LEAGUE PENALTY DECAY:")
    for league, v in selected.items():
        print(f"  {league:12s} half-life={v['half_life_years']}yr  xi={v['xi']:.5f}")
