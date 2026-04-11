import requests


def get_match_ids(league: str, season: int) -> list[int]:
    """
    Fetch all match IDs for a given league and season from Understat.
    """

    url = f"https://understat.com/getLeagueData/{league}/{season}"

    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{league}/{season}",
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    data = response.json()

    # ✅ Extract match IDs
    match_ids = [int(match["id"]) for match in data["dates"]]

    return match_ids

print(get_match_ids('EPL', 2025))