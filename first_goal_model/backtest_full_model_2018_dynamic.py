"""
Half-Kelly backtest of ALL THREE priced markets at once - shot, header,
penalty - using the full model with FROZEN header/penalty constants
(trained once on seasons 2016-2018 by train_frozen_constants.py - see
header_constants_2016_2018.json / penalty_constants_2016_2018.json),
evaluated on seasons 2018-present, all 5 leagues.

NOTE on overlap: season 2018 (Aug 2018 - Jul 2019) is PARTIALLY INSIDE the
2016-2018 training window used to freeze the header/penalty constants
(training cutoff is 2019-07-01). So season-2018 results here are not
strictly out-of-sample for those two sub-models the way 2019+ is - flagged
per the user's explicit request to run from 2018, not silently dropped.

Same Bet365-style dynamic pricing convention as backtest_full_model.py /
backtest_full_model_2019.py, all three markets keyed off the same
avg_under_2_5 threshold (avg_under_2_5 <= 2.00 vs > 2.00):
  - shot:    1.50 / 1.44
  - header:  6.50 / 6.00
  - penalty: 7.50 / 7.00   (penalty prices supplied directly by the user)

Each market gets its own independent bankroll (own peak/drawdown tracking)
so one market's staking doesn't distort another's, but predict_full_frozen()
is computed ONCE per match and its distribution reused for all three
markets' staking decisions - same perf discipline as backtest_full_model.py.
Kelly staking is sequential per market, so matches are processed in one
chronological stream across all 5 leagues combined.
"""

import numpy as np

from db import get_connection
from full_model_frozen import load_constants, predict_full_frozen, DrawRateCache

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
MIN_SEASON = 2018  # test window: season 2018 onward (overlaps training window - see module docstring)
PENALTY_XI = 0.00443  # must match the xi train_frozen_constants.py used to fit penalty_constants

MARKET_PRICES = {
    "shot":    {"low_under": 1.50, "high_under": 1.44},
    "header":  {"low_under": 6.50, "high_under": 6.00},
    "penalty": {"low_under": 7.50, "high_under": 7.00},
}


def market_price(method: str, row: dict) -> float:
    tiers = MARKET_PRICES[method]
    return tiers["low_under"] if row["avg_under_2_5"] <= 2.00 else tiers["high_under"]


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


# Per-market Kelly fraction. Header gets quarter-Kelly instead of half -
# its calibration was borderline overconfident (avg model prob 17.19% vs
# 16.7% actual hit rate) and half-Kelly on that edge produced a 72.66%
# max drawdown alongside the outsized return - a classic symptom of
# staking too aggressively relative to how noisy/thin the real edge is.
DEFAULT_KELLY_MULTIPLIER = {"shot": 0.5, "header": 0.25, "penalty": 0.5}

# Minimum required edge (model_prob - market_implied_prob) to place a bet
# at all, on top of the Kelly fraction being positive. Applied mainly to
# header to cut down its 35.8%-of-matches bet frequency - small edges are
# the ones most likely to be noise rather than real signal, and they're
# the ones a thin/overconfident model is most likely to get backwards.
DEFAULT_MIN_EDGE = {"shot": 0.0, "header": 0.03, "penalty": 0.0}


def run(kelly_multiplier: dict = None, min_edge: dict = None,
        starting_bankroll: float = 100.0, progress_every: int = 500):
    kelly_multiplier = kelly_multiplier or DEFAULT_KELLY_MULTIPLIER
    min_edge = min_edge or DEFAULT_MIN_EDGE

    conn = get_connection()
    header_constants, penalty_constants = load_constants()
    draw_rate_cache = DrawRateCache(conn, penalty_constants, xi=PENALTY_XI)

    matches = get_matches(conn)
    print(f"{len(matches)} matches across {LEAGUES}, season >= {MIN_SEASON} "
          f"(header/penalty constants frozen on seasons 2016-2018), markets={list(MARKET_PRICES)}")
    print(f"kelly_multiplier={kelly_multiplier}, min_edge={min_edge}\n")

    state = {method: {"bankroll": starting_bankroll, "peak": starting_bankroll, "max_drawdown": 0.0, "bets": [],
                       "bankrupt": False}
              for method in MARKET_PRICES}

    for i, m in enumerate(matches):
        dist = predict_full_frozen(conn, m, m["league"], header_constants, penalty_constants, draw_rate_cache)

        for method in MARKET_PRICES:
            s = state[method]
            if s["bankroll"] <= 0:
                s["bankrupt"] = True
                continue

            model_prob = dist[method]
            odds = market_price(method, m)
            market_prob = 1 / odds
            edge = model_prob - market_prob
            b = odds - 1
            f_star = (model_prob * odds - 1) / b

            if f_star > 0 and edge >= min_edge[method]:
                stake = kelly_multiplier[method] * f_star * s["bankroll"]
                won = m["actual_method"] == method
                profit = stake * b if won else -stake
                s["bankroll"] += profit
                s["peak"] = max(s["peak"], s["bankroll"])
                s["max_drawdown"] = max(s["max_drawdown"], (s["peak"] - s["bankroll"]) / s["peak"] if s["peak"] > 0 else 0.0)

                s["bets"].append({
                    "match_id": m["match_id"], "league": m["league"], "datetime": m["datetime"],
                    "home_team": m["home_team"], "away_team": m["away_team"],
                    "market_odds": odds, "model_prob": model_prob, "kelly_fraction": f_star,
                    "stake": stake, "bankroll_after": s["bankroll"], "won": won, "profit": profit,
                })

        if (i + 1) % progress_every == 0:
            status = ", ".join(f"{m}={state[m]['bankroll']:.1f}" for m in MARKET_PRICES)
            print(f"  ...{i + 1}/{len(matches)} processed  [{status}]")

    conn.close()

    results = {}
    for method in MARKET_PRICES:
        s = state[method]
        results[method] = (matches, s["bets"], {
            "final_bankroll": s["bankroll"], "starting_bankroll": starting_bankroll,
            "max_drawdown": s["max_drawdown"], "bankrupt": s["bankrupt"],
        })
    return results


def run_combined(markets=("shot", "header"), kelly_multiplier: dict = None, min_edge: dict = None,
                  starting_bankroll: float = 100.0, progress_every: int = 500):
    """
    Same model/pricing, but stakes ALL selected markets out of ONE shared
    bankroll instead of independent per-market bankrolls - this is the
    realistic picture, since a real bankroll only exists once. Both
    markets' stakes for a given match are sized off that match's
    PRE-bet bankroll (staked simultaneously, not sequentially reacting to
    each other within the same match), then both profits are applied
    together before moving to the next match.
    """
    kelly_multiplier = kelly_multiplier or DEFAULT_KELLY_MULTIPLIER
    min_edge = min_edge or DEFAULT_MIN_EDGE

    conn = get_connection()
    header_constants, penalty_constants = load_constants()
    draw_rate_cache = DrawRateCache(conn, penalty_constants, xi=PENALTY_XI)

    matches = get_matches(conn)
    print(f"{len(matches)} matches across {LEAGUES}, season >= {MIN_SEASON}, COMBINED bankroll, markets={markets}")
    print(f"kelly_multiplier={kelly_multiplier}, min_edge={min_edge}\n")

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

        match_bets = []
        for method in markets:
            model_prob = dist[method]
            odds = market_price(method, m)
            market_prob = 1 / odds
            edge = model_prob - market_prob
            b = odds - 1
            f_star = (model_prob * odds - 1) / b

            if f_star > 0 and edge >= min_edge[method]:
                stake = kelly_multiplier[method] * f_star * bankroll  # sized off PRE-bet bankroll
                won = m["actual_method"] == method
                profit = stake * b if won else -stake
                match_bets.append({
                    "match_id": m["match_id"], "league": m["league"], "datetime": m["datetime"],
                    "home_team": m["home_team"], "away_team": m["away_team"], "method": method,
                    "market_odds": odds, "model_prob": model_prob, "kelly_fraction": f_star,
                    "stake": stake, "won": won, "profit": profit,
                })

        if match_bets:
            bankroll += sum(bt["profit"] for bt in match_bets)
            peak = max(peak, bankroll)
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak > 0 else 0.0)
            for bt in match_bets:
                bt["bankroll_after"] = bankroll
            bets.extend(match_bets)

        if (i + 1) % progress_every == 0:
            print(f"  ...{i + 1}/{len(matches)} processed, {len(bets)} bets, bankroll={bankroll:.2f}")

    conn.close()
    return matches, bets, {"final_bankroll": bankroll, "starting_bankroll": starting_bankroll,
                            "max_drawdown": max_drawdown, "bankrupt": bankrupt}


def summarize_combined(matches, bets, meta, markets):
    n_matches, n_bets = len(matches), len(bets)
    n_won = sum(1 for b in bets if b["won"])
    total_return = (meta["final_bankroll"] - meta["starting_bankroll"]) / meta["starting_bankroll"]
    avg_model_prob = np.mean([b["model_prob"] for b in bets]) if bets else 0.0
    hit_rate = n_won / n_bets if n_bets else 0.0

    print(f"\n{'='*60}\nCOMBINED {'+'.join(m.upper() for m in markets)} - season >= {MIN_SEASON} - SHARED BANKROLL\n{'='*60}")
    print(f"Matches evaluated:  {n_matches}")
    print(f"Bets placed:        {n_bets}")
    for method in markets:
        n = sum(1 for bt in bets if bt["method"] == method)
        print(f"  {method:10s} {n} ({100*n/n_matches:.1f}% of matches)")
    print(f"Starting bankroll:  {meta['starting_bankroll']:.2f}")
    print(f"Final bankroll:     {meta['final_bankroll']:.2f}")
    print(f"Total return:       {100*total_return:+.2f}%")
    print(f"Max drawdown:       {100*meta['max_drawdown']:.2f}%")
    if meta["bankrupt"]:
        print("BANKRUPT before processing all matches")
    print(f"Bets won (overall): {n_won}/{n_bets} ({100*hit_rate:.1f}%)")
    print(f"Avg model prob:     {100*avg_model_prob:.2f}%")


def summarize(method, matches, bets, meta):
    n_matches, n_bets = len(matches), len(bets)
    n_won = sum(1 for b in bets if b["won"])
    total_return = (meta["final_bankroll"] - meta["starting_bankroll"]) / meta["starting_bankroll"]
    avg_model_prob = np.mean([b["model_prob"] for b in bets]) if bets else 0.0
    hit_rate = n_won / n_bets if n_bets else 0.0

    print(f"\n{'='*60}\n{method.upper()} MARKET - season >= {MIN_SEASON} - HALF-KELLY RESULTS\n{'='*60}")
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
    results = run()
    for method, (matches, bets, meta) in results.items():
        summarize(method, matches, bets, meta)
