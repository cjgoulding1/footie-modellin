"""
Select the header-model decay constant xi via walk-forward cross-validation,
restricted entirely to EPL seasons 2014-2023 - seasons 2024 (2024/25) and
2025 (2025/26) are reserved untouched for full-model evaluation later, so
they must never influence this choice either.

For each candidate xi and each validation season S in FOLD_SEASONS:
  - fit (a_i, b_i, c) on all matches strictly before S's start (using that
    candidate xi for the decay weights)
  - score the fit on season S's own matches using an UNWEIGHTED
    log-likelihood (this is held-out evaluation, not fitting - every match
    in the validation season counts equally)
  - a team appearing in the validation season with no training history
    (newly promoted) gets a=b=1, consistent with the "no evidence, assume
    average" approach used everywhere else

The winning xi is the one with the best AVERAGE held-out log-likelihood
across all folds - this doesn't have the degenerate-decay failure mode that
free-fitting xi inside the training MLE did, because a model that discarded
its data to fit training noise will generalize badly to a season it never
saw, not trivially "win" the way it does in-sample.
"""

import numpy as np

from db import get_connection
from header_strength import build_team_match_panel, fit_header_strength_from_panel, _parse

LEAGUE = "EPL"
CUTOFF_DATE = "2024-08-16 00:00:00"  # season 2024 (2024/25) start - everything from here on is held out of this entirely
FOLD_SEASON_STARTS = {
    2019: "2019-08-09 00:00:00",
    2020: "2020-09-12 00:00:00",
    2021: "2021-08-13 00:00:00",
    2022: "2022-08-05 00:00:00",
    2023: "2023-08-11 00:00:00",
}
FOLD_SEASON_ENDS = {
    2019: "2020-09-12 00:00:00",
    2020: "2021-08-13 00:00:00",
    2021: "2022-08-05 00:00:00",
    2022: "2023-08-11 00:00:00",
    2023: "2024-08-16 00:00:00",
}

XI_CANDIDATES_HALF_LIFE_YEARS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def half_life_years_to_xi(half_life_years: float) -> float:
    half_life_half_weeks = (half_life_years * 365) / 3.5
    return np.log(2) / half_life_half_weeks


def score_holdout(panel_holdout: list, strengths: dict, scale: float) -> float:
    """Unweighted sum of the Poisson-kernel log-likelihood over held-out rows."""
    total = 0.0
    for row in panel_holdout:
        s_a = strengths.get(row["team"], {"a": 1.0})["a"]
        s_b = strengths.get(row["opponent"], {"b": 1.0})["b"]
        lam_header = row["lambda_total"] * scale * s_a * s_b
        total += -lam_header + row["headed_xg"] * np.log(lam_header)
    return total


def select_xi():
    conn = get_connection()
    full_panel = build_team_match_panel(conn, LEAGUE, CUTOFF_DATE)
    conn.close()

    print(f"Full panel (2014 through end of season 2023): {len(full_panel)} team-match observations")
    print(f"Folds: validating on seasons {sorted(FOLD_SEASON_STARTS)}\n")

    results = {}
    for hl_years in XI_CANDIDATES_HALF_LIFE_YEARS:
        xi = half_life_years_to_xi(hl_years)
        fold_scores = []

        for season, start in FOLD_SEASON_STARTS.items():
            end = FOLD_SEASON_ENDS[season]
            train_panel = [r for r in full_panel if r["datetime"] < start]
            holdout_panel = [r for r in full_panel if start <= r["datetime"] < end]

            fit = fit_header_strength_from_panel(train_panel, start, xi)
            score = score_holdout(holdout_panel, fit["strengths"], fit["scale"])
            # normalize by number of holdout observations so folds with
            # different match counts (e.g. COVID-shortened seasons) are comparable
            fold_scores.append(score / len(holdout_panel))

        avg_score = float(np.mean(fold_scores))
        results[hl_years] = avg_score
        print(f"  half-life={hl_years:>4.2f}yr  (xi={xi:.5f}/half-week)  "
              f"avg per-obs held-out log-lik={avg_score:+.5f}  "
              f"per-fold={[f'{s:+.4f}' for s in fold_scores]}")

    best_hl = max(results, key=results.get)
    best_xi = half_life_years_to_xi(best_hl)
    print()
    print(f"Best: half-life={best_hl}yr (xi={best_xi:.5f} per half-week)")
    return best_hl, best_xi


if __name__ == "__main__":
    select_xi()
