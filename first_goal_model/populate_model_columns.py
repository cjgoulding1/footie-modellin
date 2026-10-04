"""
Persist model outputs onto the matches table, for future backtesting.

New columns:
  - dc_lambda_home, dc_lambda_away, dc_rho: Stage 1 Dixon-Coles fit to that
    match's own pre-match odds. Populated for every match with valid odds,
    all 5 leagues, all seasons (2014-2025) - this has no historical
    dependency at all (each match's own odds are self-contained), so there's
    no reason to skip season 2014 here.
  - header_a_home, header_b_home, header_a_away, header_b_away: each team's
    header scoring (a) / conceding (b) strength AS OF immediately before
    that specific match's own kickoff. Only populated from season 2015
    onward - season 2014 is our data's first season, so there's no prior
    history to fit on.
  - header_scale_c: the global per-league-fit scale parameter for that
    match's fit (varies match to match, since every match gets its own fit).

Refit cadence: ONCE PER MATCH. A fixture in October of a season is fit using
that season's own August-September matches too, not just prior seasons -
more granular and more leak-safe than refitting once per season. This is
only tractable because dc_lambda is cached (see below) - measured directly
before running at scale: with the cache, a single match's panel build + fit
runs in well under a second even at the largest (~9000-observation) history,
versus what would otherwise be ~20000 redundant re-fits each rebuilding its
own panel via fresh Dixon-Coles solves.

Performance note: header fitting needs each historical match's lambda_total
as an input. The naive approach - recompute it via a fresh Dixon-Coles solve
inside every one of the ~20000 per-match fits - would redundantly re-solve
the same early matches over and over as the window grows with each
successive match. Instead we populate dc_lambda_home/away FIRST (one pass,
~1.6ms per match), then every header fit reads those cached values directly
- turning what would be tens of thousands of redundant Dixon-Coles solves
into simple SQL reads.
"""

from datetime import datetime

from db import get_connection
from dixon_coles import solve_match_params
from header_strength import DEFAULT_XI, fit_header_strength_from_panel
from odds import match_probabilities

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
FIRST_HEADER_SEASON = 2015  # season 2014 has no prior history to fit on


def add_columns(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(matches)")}
    new_columns = [
        "dc_lambda_home", "dc_lambda_away", "dc_rho",
        "header_a_home", "header_b_home", "header_a_away", "header_b_away",
        "header_scale_c",
    ]
    for col in new_columns:
        if col not in existing:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {col} REAL")
    conn.commit()


def populate_stage1(conn):
    rows = conn.execute("""
        SELECT match_id, avg_home, avg_draw, avg_away, avg_over_2_5, avg_under_2_5
        FROM matches WHERE avg_home IS NOT NULL AND avg_over_2_5 IS NOT NULL
    """).fetchall()

    print(f"Stage 1: fitting {len(rows)} matches...")
    updates = []
    for r in rows:
        market = match_probabilities(r)
        fit = solve_match_params(market)
        updates.append((fit["lambda_home"], fit["lambda_away"], fit["rho"], r["match_id"]))

    conn.executemany("""
        UPDATE matches SET dc_lambda_home = ?, dc_lambda_away = ?, dc_rho = ?
        WHERE match_id = ?
    """, updates)
    conn.commit()
    print(f"  done: {len(updates)} matches updated")


def build_cached_panel(conn, league: str, before_date: str):
    """
    Same shape as header_strength.build_team_match_panel, but reads
    lambda_total from the now-populated dc_lambda_home/away columns instead
    of re-solving Stage 1 - safe because that fit only ever depended on the
    match's own odds, never on any as_of_date, so the cached value is
    identical to what a fresh solve would produce.
    """
    matches = conn.execute("""
        SELECT match_id, datetime, home_team, away_team, dc_lambda_home, dc_lambda_away
        FROM matches
        WHERE league = ? AND datetime < ? AND is_result = 1
          AND dc_lambda_home IS NOT NULL
        ORDER BY datetime ASC
    """, (league, before_date)).fetchall()

    panel = []
    for m in matches:
        headed_xg = conn.execute("""
            SELECT h_a, SUM(CASE WHEN shot_type='Head' THEN xg ELSE 0 END) as hxg
            FROM shots WHERE match_id = ? GROUP BY h_a
        """, (m["match_id"],)).fetchall()
        headed_xg_by_side = {r["h_a"]: r["hxg"] for r in headed_xg}

        panel.append({
            "team": m["home_team"], "opponent": m["away_team"], "datetime": m["datetime"],
            "lambda_total": m["dc_lambda_home"], "headed_xg": headed_xg_by_side.get("h", 0.0),
        })
        panel.append({
            "team": m["away_team"], "opponent": m["home_team"], "datetime": m["datetime"],
            "lambda_total": m["dc_lambda_away"], "headed_xg": headed_xg_by_side.get("a", 0.0),
        })

    return panel


def populate_header_columns(conn, progress_every: int = 500):
    """
    Refit per MATCH, not per season: every match's header coefficients come
    from a fit using only data strictly before THAT match's own kickoff
    (not just before its season started). This is more granular/leak-safe
    than the per-season version - a fixture in October of a season sees
    that season's own August-September matches too, not just prior
    seasons. Only tractable because dc_lambda is cached (see module
    docstring) - each match's panel build + fit runs in well under a
    second even at full ~9000-observation history, measured directly
    before committing to the full run.
    """
    for league in LEAGUES:
        matches = conn.execute("""
            SELECT match_id, datetime, home_team, away_team FROM matches
            WHERE league=? AND season>=? ORDER BY datetime ASC
        """, (league, FIRST_HEADER_SEASON)).fetchall()

        print(f"\n{league}: refitting per match for {len(matches)} matches (season>={FIRST_HEADER_SEASON})")

        for i, m in enumerate(matches):
            panel = build_cached_panel(conn, league, m["datetime"])
            fit = fit_header_strength_from_panel(panel, m["datetime"], DEFAULT_XI)
            strengths, c = fit["strengths"], fit["scale"]

            home_s = strengths.get(m["home_team"], {"a": 1.0, "b": 1.0})
            away_s = strengths.get(m["away_team"], {"a": 1.0, "b": 1.0})

            conn.execute("""
                UPDATE matches SET header_a_home=?, header_b_home=?, header_a_away=?, header_b_away=?, header_scale_c=?
                WHERE match_id=?
            """, (home_s["a"], home_s["b"], away_s["a"], away_s["b"], c, m["match_id"]))

            if (i + 1) % progress_every == 0:
                conn.commit()
                print(f"  ...{i + 1}/{len(matches)} matches refit")

        conn.commit()
        print(f"  {league} done")


if __name__ == "__main__":
    conn = get_connection()
    add_columns(conn)
    populate_stage1(conn)
    populate_header_columns(conn)
    conn.close()
    print("\nDone.")
