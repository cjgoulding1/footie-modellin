"""
Build the first-goal-of-match ground truth table.

For every completed match, finds the earliest goal-scoring shot (by minute)
and classifies it into one of: shot, header, penalty, free_kick, own_goal.
Matches with no goals at all get 'no_goal'.

Classification rule (see PLAN.md), applied in this order:
    1. result == 'OwnGoal'           -> own_goal
    2. situation == 'Penalty'        -> penalty
    3. situation == 'DirectFreekick' -> free_kick
    4. shot_type == 'Head'           -> header
    5. else                          -> shot

Important quirk: for OwnGoal rows, shots.h_a is the shooter's team, not the
team credited with the goal - the goal always belongs to the OTHER side.
Verified against match 6186 (SC Bastia 3-3 Marseille): Romaric's own goal
has h_a='h' (he's a Bastia/home player) but the goal counts for Marseille
(away). This is flipped explicitly below.
"""

from db import get_connection

METHODS = ("shot", "header", "penalty", "free_kick", "own_goal")


def classify(result: str, situation: str, shot_type: str) -> str:
    if result == "OwnGoal":
        return "own_goal"
    if situation == "Penalty":
        return "penalty"
    if situation == "DirectFreekick":
        return "free_kick"
    if shot_type == "Head":
        return "header"
    return "shot"


def scoring_side(result: str, h_a: str) -> str:
    """Which side (h/a) the goal actually counts for."""
    if result == "OwnGoal":
        return "a" if h_a == "h" else "h"
    return h_a


def create_table(conn) -> None:
    conn.execute("DROP TABLE IF EXISTS first_goal_labels")
    conn.execute("""
        CREATE TABLE first_goal_labels (
            match_id INTEGER PRIMARY KEY REFERENCES matches(match_id),
            league TEXT NOT NULL,
            season INTEGER NOT NULL,
            method TEXT NOT NULL,
            minute INTEGER,
            scoring_side TEXT,
            scoring_team TEXT
        )
    """)
    conn.commit()


def build(conn) -> dict:
    create_table(conn)

    matches = conn.execute("""
        SELECT match_id, league, season, home_team, away_team
        FROM matches WHERE is_result = 1
    """).fetchall()

    rows = []
    counts = {m: 0 for m in METHODS + ("no_goal",)}

    for match in matches:
        first_goal = conn.execute("""
            SELECT minute, result, situation, shot_type, h_a
            FROM shots
            WHERE match_id = ? AND result IN ('Goal', 'OwnGoal')
            ORDER BY minute ASC, shot_id ASC
            LIMIT 1
        """, (match["match_id"],)).fetchone()

        if first_goal is None:
            method = "no_goal"
            minute = None
            side = None
            team = None
        else:
            method = classify(first_goal["result"], first_goal["situation"], first_goal["shot_type"])
            minute = first_goal["minute"]
            side = scoring_side(first_goal["result"], first_goal["h_a"])
            team = match["home_team"] if side == "h" else match["away_team"]

        counts[method] += 1
        rows.append((match["match_id"], match["league"], match["season"], method, minute, side, team))

    conn.executemany("""
        INSERT INTO first_goal_labels (match_id, league, season, method, minute, scoring_side, scoring_team)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()

    return counts


if __name__ == "__main__":
    conn = get_connection()
    counts = build(conn)

    total = sum(counts.values())
    print(f"Labeled {total} matches:\n")
    for method, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {method:12s} {n:6d}  ({100*n/total:.1f}%)")

    conn.close()
