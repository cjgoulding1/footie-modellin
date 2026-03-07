import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-GB,en;q=0.9"
}

def scrape_match(match_id):

    url = f"https://www.transfermarkt.co.uk/spielbericht/index/spielbericht/{match_id}"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    # --- match info ---
    teams = soup.select(".sb-team .sb-vereinslink")

    home_team = teams[0].text.strip()
    away_team = teams[1].text.strip()

    raw_date = soup.select_one(".sb-datum").get_text(" ", strip=True)

    # remove weird spacing
    clean = " ".join(raw_date.split())

    # extract date + time
    date_match = re.search(r"\w{3},\s*\d{2}/\d{2}/\d{2}\s*\|\s*\d{1,2}:\d{2}\s*[AP]M", clean)

    date = date_match.group(0) if date_match else clean

    date_string = date.replace("|", "").strip()

    dt = datetime.strptime(date_string, "%a, %d/%m/%y %I:%M %p")

    date = dt.strftime("%Y-%m-%d %H:%M")

    match_data = {
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "events": []
    }

    rows = soup.select("li.sb-aktion-heim, li.sb-aktion-gast")

    for row in rows:

        action = row.select_one(".sb-aktion-aktion")
        if not action:
            continue

        text = action.get_text(" ", strip=True).lower()

        if "goal" not in text:
            continue

        scorer_tag = action.select_one("a.wichtig")
        scorer = scorer_tag.text.strip() if scorer_tag else None

        score_tag = row.select_one(".sb-aktion-spielstand b")
        score = score_tag.text.strip() if score_tag else None

        assist_match = re.search(r"Assist:\s*([A-Za-zÀ-ÿ\s\-]+)", action.text)
        assist = assist_match.group(1).strip() if assist_match else None

        # determine team
        team = home_team if "sb-aktion-heim" in row["class"] else away_team

        # goal method
        method = None
        if "header" in text:
            method = "header"
        elif "penalty" in text:
            method = "penalty"
        elif "free kick" in text:
            method = "free_kick"
        elif "left-footed" in text:
            method = "left_foot"
        elif "right-footed" in text:
            method = "right_foot"

        situation = None
        if "counter attack" in text:
            situation = "counter_attack"

        event = {
            "team": team,
            "scorer": scorer,
            "assist": assist,
            "score": score,
            "goal_method": method,
            "goal_situation": situation
        }

        match_data["events"].append(event)

    return match_data

match_ids = [
    4625941,
    4625942,
    4625943
]

dataset = {}

for match_id in match_ids:

    try:
        dataset[match_id] = scrape_match(match_id)
        print(f"Scraped {match_id}")

    except Exception as e:
        print(f"Failed {match_id}: {e}")

print(dataset)