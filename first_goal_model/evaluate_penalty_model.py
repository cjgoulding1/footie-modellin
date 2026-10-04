"""
Fit the final per-league penalty model on data through end of season 2023,
then evaluate its predictive power on the previously untouched 2024
(2024/25) and 2025 (2025/26) seasons - the first time this holdout gets
used for anything, exactly what it was reserved for back when the header
model's decay was being selected.

Metrics:
  - held-out average log-likelihood, vs an intercept-only baseline fit the
    same way (demonstrates real predictive lift, not just "a model exists")
  - AUC (rank-based, Mann-Whitney U - no sklearn dependency)
  - calibration: predictions bucketed into deciles vs actual frequency
"""

import numpy as np
from scipy.stats import rankdata

from db import get_connection
from penalty_model import (
    build_penalty_panel, build_feature_matrix, bernoulli_log_likelihood, fit_beta,
    league_penalty_rate, DEFAULT_XI,
)

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
TRAIN_CUTOFF = "2024-08-01 00:00:00"  # fit only on data strictly before this (seasons 2014-2023)
EVAL_CUTOFF = "2026-05-25 00:00:00"   # evaluate on everything up to end of season 2025

# single shared value across all leagues, not the earlier per-league CV picks
# (those weren't distinguishable from noise - see conversation). Chosen by
# inspection: reins in the most extreme/wrong-signed draw_rate coefficients
# (La_liga, Serie_A) without wiping the feature out entirely the way
# reg=5/10 does almost everywhere.
REG_LAMBDA = 1.0


def auc(y_true, y_score) -> float:
    ranks = rankdata(y_score)
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def calibration_table(y_true, y_score, n_buckets=10):
    order = np.argsort(y_score)
    y_true_sorted, y_score_sorted = y_true[order], y_score[order]
    bucket_edges = np.array_split(np.arange(len(y_true)), n_buckets)

    rows = []
    for edges in bucket_edges:
        if len(edges) == 0:
            continue
        rows.append({
            "n": len(edges),
            "avg_predicted": y_score_sorted[edges].mean(),
            "actual_rate": y_true_sorted[edges].mean(),
        })
    return rows


def evaluate_league(conn, league: str, reg_lambda: float = 0.0):
    # single panel through end of season 2025 covers both training (<2024)
    # and eval (2024-2025) rows - build once, split by date
    panel, history_cache = build_penalty_panel(conn, league, EVAL_CUTOFF)
    league_rate = league_penalty_rate(conn, league, TRAIN_CUTOFF)  # rate as known at training time
    X, y = build_feature_matrix(panel, history_cache, league_rate, DEFAULT_XI)
    dt = np.array([r["datetime"] for r in panel])

    train_mask = dt < TRAIN_CUTOFF
    eval_mask = dt >= TRAIN_CUTOFF

    X_train, y_train = X[train_mask], y[train_mask]
    X_eval, y_eval = X[eval_mask], y[eval_mask]

    beta, converged = fit_beta(X_train, y_train, reg_lambda=reg_lambda)
    beta_baseline, _ = fit_beta(X_train[:, :1], y_train, reg_lambda=0.0)  # intercept-only, never regularized

    ll_model = bernoulli_log_likelihood(X_eval, y_eval, beta) / len(y_eval)
    ll_baseline = bernoulli_log_likelihood(X_eval[:, :1], y_eval, beta_baseline) / len(y_eval)

    p_model = 1 / (1 + np.exp(-(X_eval @ beta)))
    model_auc = auc(y_eval, p_model)
    calib = calibration_table(y_eval, p_model)

    return {
        "league": league, "converged": converged, "coefficients": beta,
        "n_train": len(y_train), "n_eval": len(y_eval), "n_eval_positive": int(y_eval.sum()),
        "ll_model": ll_model, "ll_baseline": ll_baseline, "auc": model_auc, "calibration": calib,
    }


if __name__ == "__main__":
    conn = get_connection()

    names = ["intercept", "lambda_own", "lambda_opp", "draw_rate"]
    all_results = []

    for league in LEAGUES:
        r = evaluate_league(conn, league, reg_lambda=REG_LAMBDA)
        all_results.append(r)

        print(f"\n{'='*60}\n{league}\n{'='*60}")
        print(f"train n={r['n_train']}, eval n={r['n_eval']} ({r['n_eval_positive']} positive, "
              f"{100*r['n_eval_positive']/r['n_eval']:.1f}%)")
        print("coefficients:", ", ".join(f"{n}={c:+.3f}" for n, c in zip(names, r["coefficients"])))
        print(f"held-out avg log-lik:  model={r['ll_model']:+.5f}   intercept-only baseline={r['ll_baseline']:+.5f}"
              f"   lift={r['ll_model']-r['ll_baseline']:+.5f}")
        print(f"AUC: {r['auc']:.4f}")
        print("calibration (predicted vs actual, by decile):")
        for b in r["calibration"]:
            print(f"  n={b['n']:3d}  predicted={100*b['avg_predicted']:5.2f}%  actual={100*b['actual_rate']:5.2f}%")

    conn.close()
