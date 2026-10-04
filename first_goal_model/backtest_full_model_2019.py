"""
Half-Kelly backtest of the SHOT market using the full model with FROZEN
header/penalty constants (trained once on seasons 2016-2018 by
train_frozen_constants.py - see header_constants_2016_2018.json /
penalty_constants_2016_2018.json), evaluated out-of-sample on seasons
2019-present, all 5 leagues.

Same Bet365 dynamic pricing rule as backtest_full_model.py (discovered by
inspection, keyed off avg_under_2_5): 1.50 when avg_under_2_5 <= 2.00, else
1.44.

Kelly staking is sequential (stake = fraction of CURRENT bankroll), so all
leagues are processed in one chronological stream, same discipline as
backtest.py / backtest_full_model.py.
"""

import numpy as np

from db import get_connection
from full_model_frozen import load_constants, predict_full_frozen, DrawRateCache

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
MIN_SEASON = 2019  # test window: everything from season 2019 onward (train window was 2016-2018)
PENALTY_XI = 0.00443  # must match the xi train_frozen_constants.py used to fit penalty_constants


def shot_price(row):
    return 1.50 if row["avg_under_2_5"] <= 2.00 else 1.44


def get_matches(conn):
    placeholders = ",".join("?" * len(LEAGUES))
    rows = conn.execute(f"""
        SELECT m.*, f.method AS actual_method
        FROM matches m JOIN first_goal_labels f ON f.match_id = m.match_id
        WHERE m.league IN ({placeholders}) AND m.season >= ?
          AND m.dc_lambda_home IS NOT NULL AND m.avg_under_2_5 IS NOT NULL
        ORDER BY m.datetime ASC
    """, (*LEAGUES, MIN_SEASON)).fetchall()
    return [dict(r) for r in rows]


def run(kelly_multiplier: float = 0.5, starting_bankroll: float = 100.0, progress_every: int = 500):
    conn = get_connection()
    header_constants, penalty_constants = load_constants()
    draw_rate_cache = DrawRateCache(conn, penalty_constants, xi=PENALTY_XI)

    matches = get_matches(conn)
    print(f"{len(matches)} matches across {LEAGUES}, season >= {MIN_SEASON} "
          f"(header/penalty constants frozen on seasons 2016-2018)\n")

    bankroll = starting_bankroll
    peak = bankroll
    max_drawdown = 0.0
    bets = []
    bankrupt = False

    for i, m in enumerate(matches):
        if bankroll <= 0:
            bankrupt = True
            break

        dist = predict_full_frozen(conn, m, m["league"], header_constants, penalty_constants, draw_rate_cache)
        model_prob = dist["shot"]

        market_odds = shot_price(m)
        b = market_odds - 1
        f_star = (model_prob * market_odds - 1) / b

        if f_star > 0:
            stake = kelly_multiplier * f_star * bankroll
            won = m["actual_method"] == "shot"
            profit = stake * b if won else -stake
            bankroll += profit
            peak = max(peak, bankroll)
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak > 0 else 0.0)

            bets.append({
                "match_id": m["match_id"], "league": m["league"], "datetime": m["datetime"],
                "home_team": m["home_team"], "away_team": m["away_team"],
                "market_odds": market_odds, "model_prob": model_prob, "kelly_fraction": f_star,
                "stake": stake, "bankroll_after": bankroll, "won": won, "profit": profit,
            })

        if (i + 1) % progress_every == 0:
            print(f"  ...{i + 1}/{len(matches)} processed, {len(bets)} bets, bankroll={bankroll:.2f}")

    conn.close()
    return matches, bets, {"final_bankroll": bankroll, "starting_bankroll": starting_bankroll,
                            "max_drawdown": max_drawdown, "bankrupt": bankrupt}


def summarize(matches, bets, meta):
    n_matches, n_bets = len(matches), len(bets)
    n_won = sum(1 for b in bets if b["won"])
    total_return = (meta["final_bankroll"] - meta["starting_bankroll"]) / meta["starting_bankroll"]
    avg_model_prob = np.mean([b["model_prob"] for b in bets]) if bets else 0.0
    hit_rate = n_won / n_bets if n_bets else 0.0

    print(f"\n{'='*60}\nSHOT MARKET - FROZEN (2016-2018) HEADER+PENALTY MODEL\nTEST: season >= {MIN_SEASON} - HALF-KELLY RESULTS\n{'='*60}")
    print(f"Matches evaluated:  {n_matches}")
    print(f"Bets placed:        {n_bets} ({100*n_bets/n_matches:.1f}%)")
    print(f"Starting bankroll:  {meta['starting_bankroll']:.2f}")
    print(f"Final bankroll:     {meta['final_bankroll']:.2f}")
    print(f"Total return:       {100*total_return:+.2f}%")
    print(f"Max drawdown:       {100*meta['max_drawdown']:.2f}%")
    if meta["bankrupt"]:
        print("BANKRUPT before processing all matches")
    print(f"Bets won:           {n_won}/{n_bets} ({100*hit_rate:.1f}%)")
    print(f"Avg model prob:     {100*avg_model_prob:.2f}%")
    if avg_model_prob > hit_rate:
        print("  -> model was OVERCONFIDENT on the bets it placed (edge may be partly illusory)")
    else:
        print("  -> model was not overconfident on the bets it placed (edge looks real, on this sample)")


if __name__ == "__main__":
    matches, bets, meta = run()
    summarize(matches, bets, meta)
