from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
TEMPLATES.mkdir(exist_ok=True)

matches = pd.DataFrame(
    [
        {
            "league_code": "ENG-PL",
            "season": "2025/26",
            "match_date": "2026-08-15",
            "kickoff_time": "10:15",
            "home_team": "Example Home",
            "away_team": "Example Away",
            "home_goals": 2,
            "away_goals": 1,
        }
    ]
)
fixtures = matches.drop(columns=["home_goals", "away_goals"])
matches.to_csv(TEMPLATES / "match_results_template.csv", index=False)
matches.to_excel(TEMPLATES / "match_results_template.xlsx", index=False)
fixtures.to_csv(TEMPLATES / "upcoming_fixtures_template.csv", index=False)
fixtures.to_excel(TEMPLATES / "upcoming_fixtures_template.xlsx", index=False)
print(f"Created templates in {TEMPLATES}")

