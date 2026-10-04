"""
Select the penalty model's L2 regularization strength via walk-forward
cross-validation, PER LEAGUE, restricted entirely to seasons 2014-2023 -
2024/25 and 2025/26 stay reserved for full-model evaluation, same
discipline as everywhere else in this project.

Decay is no longer separately tuned here - the per-league xi cross-val
(select_penalty_xi.py) came back essentially flat with picks landing on
grid boundaries for 3/5 leagues, not a real signal (see conversation).
Instead this reuses the header model's validated xi (1.5yr half-life,
DEFAULT_XI) for the two rate features, and cross-validates only the
regression's L2 penalty strength.
"""

import numpy as np

from db import get_connection
from header_strength import DEFAULT_XI as HEADER_XI
from penalty_model import build_penalty_panel, build_feature_matrix, bernoulli_log_likelihood, fit_beta, league_penalty_rate

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
CUTOFF_DATE = "2024-08-01 00:00:00"
FOLD_SEASONS = [2019, 2020, 2021, 2022, 2023]

REG_LAMBDA_CANDIDATES = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]


def select_reg_for_league(conn, league: str):
    fold_starts, fold_ends = {}, {}
    for season in FOLD_SEASONS:
        fold_starts[season] = conn.execute(
            "SELECT MIN(datetime) FROM matches WHERE league=? AND season=?", (league, season)
        ).fetchone()[0]
        fold_ends[season] = conn.execute(
            "SELECT MIN(datetime) FROM matches WHERE league=? AND season=?", (league, season + 1)
        ).fetchone()[0]

    panel, history_cache = build_penalty_panel(conn, league, CUTOFF_DATE)
    league_rate = league_penalty_rate(conn, league, CUTOFF_DATE)

    # xi is fixed now, so the decayed-rate features are the same across all
    # folds/candidates - compute the full feature matrix once up front
    X_full, y_full = build_feature_matrix(panel, history_cache, league_rate, HEADER_XI)
    dt_array = np.array([row["datetime"] for row in panel])

    print(f"\n{league}: {len(panel)} observations through end of season 2023")

    results = {}
    for reg_lambda in REG_LAMBDA_CANDIDATES:
        fold_scores = []

        for season in FOLD_SEASONS:
            start, end = fold_starts[season], fold_ends[season]
            train_mask = dt_array < start
            hold_mask = (dt_array >= start) & (dt_array < end)
            if not hold_mask.any() or not train_mask.any():
                continue

            X_train, y_train = X_full[train_mask], y_full[train_mask]
            X_hold, y_hold = X_full[hold_mask], y_full[hold_mask]

            beta, _ = fit_beta(X_train, y_train, reg_lambda)
            ll = bernoulli_log_likelihood(X_hold, y_hold, beta)
            fold_scores.append(ll / hold_mask.sum())

        avg_score = float(np.mean(fold_scores))
        results[reg_lambda] = avg_score
        print(f"  reg_lambda={reg_lambda:>6.1f}  avg per-obs held-out log-lik={avg_score:+.5f}")

    best_reg = max(results, key=results.get)
    print(f"  -> best for {league}: reg_lambda={best_reg}")
    return best_reg


if __name__ == "__main__":
    conn = get_connection()
    selected = {}
    for league in LEAGUES:
        selected[league] = select_reg_for_league(conn, league)
    conn.close()

    print("\n" + "=" * 50)
    print("SELECTED PER-LEAGUE PENALTY REGULARIZATION:")
    for league, reg in selected.items():
        print(f"  {league:12s} reg_lambda={reg}")
