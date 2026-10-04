"""
Half-Kelly backtest using the unified full_model (see full_model.py),
across all 5 leagues, last 2 seasons (2024/25, 2025/26).

Two markets, both keyed off the same avg_under_2_5 threshold (matches
Bet365's actual pricing pattern discovered earlier):
  - shot:   1.44 when avg_under_2_5 > 2.00, else 1.50
  - header: 6.00 when avg_under_2_5 > 2.00, else 6.50

Kelly staking is inherently sequential (stake = fraction of the CURRENT
bankroll), so matches are processed in one chronological order across all
leagues combined, same discipline as backtest_kelly.py.
"""

import numpy as np

from db import get_connection
from full_model import predict_full

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
SEASONS = [2024, 2025]


def shot_price(row):
    return 1.50 if row["avg_under_2_5"] <= 2.00 else 1.44


def header_price(row):
    return 6.50 if row["avg_under_2_5"] <= 2.00 else 6.00


def get_matches(conn):
    placeholders_l = ",".join("?" * len(LEAGUES))
    placeholders_s = ",".join("?" * len(SEASONS))
    rows = conn.execute(f"""
        SELECT m.*, f.method AS actual_method
        FROM matches m JOIN first_goal_labels f ON f.match_id = m.match_id
        WHERE m.league IN ({placeholders_l}) AND m.season IN ({placeholders_s})
          AND m.dc_lambda_home IS NOT NULL AND m.header_a_home IS NOT NULL
          AND m.avg_under_2_5 IS NOT NULL
        ORDER BY m.datetime ASC
    """, (*LEAGUES, *SEASONS)).fetchall()
    return [dict(r) for r in rows]


def run_kelly_backtests(markets: dict, kelly_multiplier: float = 0.5,
                         starting_bankroll: float = 100.0, progress_every: int = 500):
    """
    markets: {method_name: market_price_fn}. Runs all markets in ONE pass -
    predict_full() is the expensive step (~195ms/match, each call does live
    penalty_share/league-share queries), so computing it once per match and
    reusing the same distribution for every market avoids paying that cost
    once per market.
    """
    conn = get_connection()
    matches = get_matches(conn)
    print(f"{len(matches)} matches across {LEAGUES}, seasons {SEASONS}, markets={list(markets)}\n")

    state = {method: {"bankroll": starting_bankroll, "peak": starting_bankroll, "max_drawdown": 0.0, "bets": []}
              for method in markets}

    for i, m in enumerate(matches):
        dist = predict_full(conn, m, m["league"], m["datetime"])

        for method, market_price_fn in markets.items():
            s = state[method]
            if s["bankroll"] <= 0:
                continue

            model_prob = dist[method]
            market_odds = market_price_fn(m)
            b = market_odds - 1
            f_star = (model_prob * market_odds - 1) / b

            if f_star > 0:
                stake = kelly_multiplier * f_star * s["bankroll"]
                won = m["actual_method"] == method
                profit = stake * b if won else -stake
                s["bankroll"] += profit
                s["peak"] = max(s["peak"], s["bankroll"])
                s["max_drawdown"] = max(s["max_drawdown"], (s["peak"] - s["bankroll"]) / s["peak"] if s["peak"] > 0 else 0.0)

                s["bets"].append({
                    "match_id": m["match_id"], "league": m["league"], "datetime": m["datetime"],
                    "home_team": m["home_team"], "away_team": m["away_team"],
                    "market_odds": market_odds, "model_prob": model_prob, "kelly_fraction": f_star,
                    "stake": stake, "bankroll_after": s["bankroll"], "won": won, "profit": profit,
                })

        if (i + 1) % progress_every == 0:
            status = ", ".join(f"{m}={state[m]['bankroll']:.1f}" for m in markets)
            print(f"  ...{i + 1}/{len(matches)} processed  [{status}]")

    conn.close()

    results = {}
    for method in markets:
        s = state[method]
        results[method] = (matches, s["bets"], {
            "final_bankroll": s["bankroll"], "starting_bankroll": starting_bankroll,
            "max_drawdown": s["max_drawdown"],
        })
    return results


def summarize(method, matches, bets, meta):
    n_matches, n_bets = len(matches), len(bets)
    n_won = sum(1 for b in bets if b["won"])
    total_return = (meta["final_bankroll"] - meta["starting_bankroll"]) / meta["starting_bankroll"]
    avg_model_prob = np.mean([b["model_prob"] for b in bets]) if bets else 0.0
    hit_rate = n_won / n_bets if n_bets else 0.0

    print(f"\n{'='*60}\n{method.upper()} MARKET - HALF-KELLY RESULTS\n{'='*60}")
    print(f"Matches evaluated:  {n_matches}")
    print(f"Bets placed:        {n_bets} ({100*n_bets/n_matches:.1f}%)")
    print(f"Starting bankroll:  {meta['starting_bankroll']:.2f}")
    print(f"Final bankroll:     {meta['final_bankroll']:.2f}")
    print(f"Total return:       {100*total_return:+.2f}%")
    print(f"Max drawdown:       {100*meta['max_drawdown']:.2f}%")
    print(f"Bets won:           {n_won}/{n_bets} ({100*hit_rate:.1f}%)")
    print(f"Avg model prob:     {100*avg_model_prob:.2f}%")


if __name__ == "__main__":
    results = run_kelly_backtests({"shot": shot_price, "header": header_price})
    for method, (matches, bets, meta) in results.items():
        summarize(method, matches, bets, meta)
