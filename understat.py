import pandas as pd
import requests
import json
from bs4 import BeautifulSoup

def get_page(league: str, season: int) -> dict:
    """
    Fetch Understat league data and return teams dictionary.

    Parameters
    ----------
    league : str
        League code, e.g. 'EPL', 'LaLiga', 'Bundesliga'
    season : int
        Season start year, e.g. 2025

    Returns
    -------
    dict
        data['teams']
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
    return data['dates']

print(get_page('EPL', 2025))