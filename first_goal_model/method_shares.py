"""
Stage 2: method shares (header, penalty, free_kick, own_goal, shot).

Generic decay-weighting and shrinkage utilities, plus the header/penalty
share builders built on top of them. free_kick and own_goal are not
team-specific (see PLAN.md findings) - free_kick uses a league-average
constant, own_goal a league-average fraction of goals. shot is the
residual after the other four are computed.

header_share and penalty_share are both defined the same way - fraction of
a team's own goals that fall in that category - so they combine cleanly
with a common formula. (Penalty share is deliberately NOT "penalties
awarded per match": that's an absolute rate, not a fraction of goals, and
mixing units would break the multiplicative attack x defense combination
below. Using the goal-share instead still captures the dominant source of
team variation - being awarded a penalty in the first place - since
conversion rates are high and much less variable across teams than award
rates are.)

Every share builder here takes an `as_of_date` and only looks at matches
strictly before it - this is what makes the eventual backtest valid
(otherwise a team's decay-weighted historical rate could leak future
matches into a prediction for an earlier fixture).
"""

import math
from datetime import datetime

from db import get_connection

DEFAULT_HALF_LIFE_DAYS = 365 * 1.5  # starting point, see PLAN.md - tune via backtest


def decay_weight(days_ago: float, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    if days_ago < 0:
        raise ValueError("days_ago must be non-negative (as_of_date must be after the event)")
    return math.exp(-math.log(2) / half_life_days * days_ago)


def shrink(weighted_sum: float, weighted_total: float, prior_rate: float, pseudo_count: float) -> float:
    """
    Empirical-Bayes shrinkage: blend the weighted empirical rate with a
    prior, strength controlled by pseudo_count (higher = more shrinkage).
    weighted_total is decay-weighted effective sample size, not raw count.
    """
    return (weighted_sum + pseudo_count * prior_rate) / (weighted_total + pseudo_count)


def _parse(dt_str: str) -> datetime:
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


def team_goal_events(conn, team: str, league: str, before_date: str, conceded: bool = False):
    """
    Every goal (not own goal) scored BY `team` (or, if conceded=True,
    AGAINST `team`), with its date, shot_type and situation. Own goals are
    excluded - they aren't a signal about either side's attacking method.
    """
    shooter_side, other_side = ("a", "h") if conceded else ("h", "a")
    rows = conn.execute(f"""
        SELECT s.shot_type, s.situation, m.datetime
        FROM shots s
        JOIN matches m ON s.match_id = m.match_id
        WHERE m.league = ? AND m.datetime < ? AND s.result = 'Goal'
          AND ((s.h_a = '{shooter_side}' AND m.home_team = ?) OR (s.h_a = '{other_side}' AND m.away_team = ?))
    """, (league, before_date, team, team)).fetchall()
    return rows


def _weighted_goal_share(conn, team: str, league: str, as_of_date: str, is_category,
                          conceded: bool, prior_rate: float, half_life_days: float,
                          pseudo_count: float) -> dict:
    events = team_goal_events(conn, team, league, as_of_date, conceded=conceded)
    as_of = _parse(as_of_date)

    weighted_hits = 0.0
    weighted_total = 0.0
    for row in events:
        days_ago = (as_of - _parse(row["datetime"])).days
        w = decay_weight(days_ago, half_life_days)
        weighted_total += w
        if is_category(row):
            weighted_hits += w

    rate = shrink(weighted_hits, weighted_total, prior_rate, pseudo_count)
    return {"rate": rate, "effective_n": weighted_total, "raw_n": len(events)}


def header_share(conn, team: str, league: str, as_of_date: str, conceded: bool = False,
                  half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                  prior_rate: float = None, pseudo_count: float = 30.0) -> dict:
    if prior_rate is None:
        prior_rate = league_header_share(conn, league, as_of_date)
    return _weighted_goal_share(
        conn, team, league, as_of_date,
        is_category=lambda row: row["shot_type"] == "Head",
        conceded=conceded, prior_rate=prior_rate,
        half_life_days=half_life_days, pseudo_count=pseudo_count,
    )


def penalty_share(conn, team: str, league: str, as_of_date: str, conceded: bool = False,
                   half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                   prior_rate: float = None, pseudo_count: float = 60.0) -> dict:
    if prior_rate is None:
        prior_rate = league_penalty_share(conn, league, as_of_date)
    return _weighted_goal_share(
        conn, team, league, as_of_date,
        is_category=lambda row: row["situation"] == "Penalty",
        conceded=conceded, prior_rate=prior_rate,
        half_life_days=half_life_days, pseudo_count=pseudo_count,
    )


def league_header_share(conn, league: str, as_of_date: str) -> float:
    row = conn.execute("""
        SELECT
            SUM(CASE WHEN shot_type = 'Head' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS share
        FROM shots s JOIN matches m ON s.match_id = m.match_id
        WHERE m.league = ? AND m.datetime < ? AND s.result = 'Goal'
    """, (league, as_of_date)).fetchone()
    return row["share"] if row["share"] is not None else 0.16  # overall dataset average fallback


def league_penalty_share(conn, league: str, as_of_date: str) -> float:
    row = conn.execute("""
        SELECT
            SUM(CASE WHEN situation = 'Penalty' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS share
        FROM shots s JOIN matches m ON s.match_id = m.match_id
        WHERE m.league = ? AND m.datetime < ? AND s.result = 'Goal'
    """, (league, as_of_date)).fetchone()
    return row["share"] if row["share"] is not None else 0.09  # overall dataset average fallback


def league_free_kick_share(conn, league: str, as_of_date: str) -> float:
    """
    Free kicks are not modeled per-team (see PLAN.md - team variance here is
    a personnel effect, not a durable team trait). This is a plain
    cumulative league average, no decay: a league-wide rate is already
    smoothed over ~20 teams' worth of personnel, so a single departure
    barely moves it, unlike a single-team rate.
    """
    row = conn.execute("""
        SELECT
            SUM(CASE WHEN situation = 'DirectFreekick' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS share
        FROM shots s JOIN matches m ON s.match_id = m.match_id
        WHERE m.league = ? AND m.datetime < ? AND s.result = 'Goal'
    """, (league, as_of_date)).fetchone()
    return row["share"] if row["share"] is not None else 0.02  # overall dataset average fallback


def league_own_goal_share(conn, league: str, as_of_date: str) -> float:
    """
    Own goals scale with a team's own attacking output (see PLAN.md finding
    - correlates with attack volume, not an independent skill) rather than
    getting their own fitted parameter. In share-space (fraction of a
    team's goals that are own-goals-for) this collapses to a plain
    league-wide constant: if own-goal rate is c*lambda and total goal rate
    is lambda, the share own-goal-rate/total-rate = c regardless of lambda.
    """
    row = conn.execute("""
        SELECT
            SUM(CASE WHEN s.result = 'OwnGoal' THEN 1 ELSE 0 END) * 1.0 /
            SUM(CASE WHEN s.result IN ('Goal', 'OwnGoal') THEN 1 ELSE 0 END) AS frac
        FROM shots s JOIN matches m ON s.match_id = m.match_id
        WHERE m.league = ? AND m.datetime < ?
    """, (league, as_of_date)).fetchone()
    return row["frac"] if row["frac"] is not None else 0.03  # overall dataset average fallback


# shares can't exceed this combined total, leaving at least 5% for `shot` -
# guards against a pathological attack x defense combination (e.g. a very
# aerial-strong team facing a very aerial-weak opponent) pushing the other
# four categories past 1 and making the shot residual negative.
MAX_NON_SHOT_SHARE = 0.95


def combine_method_shares(conn, scoring_team: str, opponent_team: str, league: str, as_of_date: str,
                           half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> dict:
    """
    Full 5-way method breakdown for `scoring_team`, conditional on them
    scoring against `opponent_team`. Combines attack x defense / league_avg
    for header and penalty (same multiplicative logic as Dixon-Coles
    itself), league constants for free_kick and own_goal, and shot as the
    residual. Returns shares summing to 1.
    """
    league_header = league_header_share(conn, league, as_of_date)
    league_penalty = league_penalty_share(conn, league, as_of_date)

    header_attack = header_share(conn, scoring_team, league, as_of_date, half_life_days=half_life_days)["rate"]
    header_defense = header_share(conn, opponent_team, league, as_of_date, conceded=True, half_life_days=half_life_days)["rate"]
    header_component = header_attack * header_defense / league_header

    pen_attack = penalty_share(conn, scoring_team, league, as_of_date, half_life_days=half_life_days)["rate"]
    pen_defense = penalty_share(conn, opponent_team, league, as_of_date, conceded=True, half_life_days=half_life_days)["rate"]
    penalty_component = pen_attack * pen_defense / league_penalty

    free_kick_component = league_free_kick_share(conn, league, as_of_date)
    own_goal_component = league_own_goal_share(conn, league, as_of_date)

    non_shot_total = header_component + penalty_component + free_kick_component + own_goal_component
    if non_shot_total > MAX_NON_SHOT_SHARE:
        scale = MAX_NON_SHOT_SHARE / non_shot_total
        header_component *= scale
        penalty_component *= scale
        free_kick_component *= scale
        own_goal_component *= scale
        non_shot_total = MAX_NON_SHOT_SHARE

    shot_component = 1 - non_shot_total

    return {
        "shot": shot_component,
        "header": header_component,
        "penalty": penalty_component,
        "free_kick": free_kick_component,
        "own_goal": own_goal_component,
    }


if __name__ == "__main__":
    conn = get_connection()
    as_of = "2025-08-01 00:00:00"

    print("Header share (as of 2025-08-01), half-life 1.5yr:")
    for team in ["Manchester City", "Burnley", "Luton"]:
        r = header_share(conn, team, "EPL", as_of)
        print(f"  {team:20s} rate={r['rate']:.3f}  effective_n={r['effective_n']:.1f}  raw_n={r['raw_n']}")

    print()
    print("Penalty share (as of 2025-08-01), half-life 1.5yr:")
    for team in ["Real Madrid", "Burnley", "Luton"]:
        lg = "La_liga" if team == "Real Madrid" else "EPL"
        r = penalty_share(conn, team, lg, as_of)
        print(f"  {team:20s} rate={r['rate']:.3f}  effective_n={r['effective_n']:.1f}  raw_n={r['raw_n']}")

    print()
    print("Full method-share breakdown, Manchester City scoring vs Crystal Palace:")
    shares = combine_method_shares(conn, "Manchester City", "Crystal Palace", "EPL", as_of)
    for method, share in shares.items():
        print(f"  {method:10s} {share:.4f}")
    print(f"  sum = {sum(shares.values()):.4f}")

    conn.close()
