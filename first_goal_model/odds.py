"""
De-vig market odds into fair (overround-free) probabilities.

Bookmaker decimal odds imply probabilities that sum to > 1 - the excess is
the bookmaker's margin. We strip it out via simple proportional
normalization: convert each price to its raw implied probability (1/odds),
then divide by the sum so the market's probabilities sum to exactly 1.

This is the simplest de-vigging method. It doesn't correct for
favorite-longshot bias the way Shin's method does, which would be a
reasonable upgrade if backtesting shows the Dixon-Coles fit is
systematically off for short-priced favorites or big underdogs.
"""


def implied_probabilities(odds: list[float]) -> list[float]:
    """Raw implied probabilities from decimal odds, before de-vigging."""
    return [1 / o for o in odds]


def devig_proportional(odds: list[float]) -> list[float]:
    """Fair probabilities for a set of mutually exclusive outcomes."""
    raw = implied_probabilities(odds)
    overround = sum(raw)
    return [p / overround for p in raw]


def match_probabilities(row: dict) -> dict:
    """
    Given a matches row (or dict) with avg_home/avg_draw/avg_away and
    avg_over_2_5/avg_under_2_5, return de-vigged probabilities for both
    markets. Returns None if either market is missing (older seasons don't
    have the O/U market - see import_e0_csv.py).
    """
    if None in (row["avg_home"], row["avg_draw"], row["avg_away"]):
        return None
    if None in (row["avg_over_2_5"], row["avg_under_2_5"]):
        return None

    p_home, p_draw, p_away = devig_proportional([row["avg_home"], row["avg_draw"], row["avg_away"]])
    p_over, p_under = devig_proportional([row["avg_over_2_5"], row["avg_under_2_5"]])

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over_2_5": p_over,
        "p_under_2_5": p_under,
    }


if __name__ == "__main__":
    from db import get_connection

    conn = get_connection()
    row = conn.execute("""
        SELECT * FROM matches
        WHERE league='EPL' AND season=2025 AND avg_home IS NOT NULL
        LIMIT 1
    """).fetchone()

    print(dict(row)["home_team"], "vs", dict(row)["away_team"])
    print("raw odds:", row["avg_home"], row["avg_draw"], row["avg_away"],
          "| O/U:", row["avg_over_2_5"], row["avg_under_2_5"])

    probs = match_probabilities(row)
    print("de-vigged:", probs)
    print("1X2 sums to:", probs["p_home"] + probs["p_draw"] + probs["p_away"])
    print("O/U sums to:", probs["p_over_2_5"] + probs["p_under_2_5"])

    conn.close()
