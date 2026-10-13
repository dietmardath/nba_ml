from datetime import date
from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


START_DATE = date(2015, 1, 1)
OUTPUT_PATH = Path("/data/nba_season.csv")
SEASON_TYPES = {
    "Regular Season": "regular",
    "Playoffs": "playoffs",
}


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def first_season_to_fetch(today: date) -> int:
    return today.year if today.month >= 10 else today.year - 1


def fetch_season_games(season: str, season_type: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            return leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable="00",
                timeout=60,
            ).get_data_frames()[0]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError("NBA game data could not be fetched")


def main() -> None:
    today = date.today()
    last_season_start = first_season_to_fetch(today)
    games = []

    for season_start in range(2014, last_season_start + 1):
        season = season_label(season_start)
        for api_season_type, output_season_type in SEASON_TYPES.items():
            print(f"Fetching {season} {api_season_type}...")
            rows = fetch_season_games(season, api_season_type)
            if rows.empty:
                continue

            rows = rows[rows["GAME_DATE"].between(START_DATE.isoformat(), today.isoformat())]
            for game_id, game_rows in rows.groupby("GAME_ID"):
                home_rows = game_rows[game_rows["MATCHUP"].str.contains(" vs. ", regex=False)]
                away_rows = game_rows[game_rows["MATCHUP"].str.contains(" @ ", regex=False)]
                if home_rows.empty or away_rows.empty:
                    continue

                home = home_rows.iloc[0]
                away = away_rows.iloc[0]
                if pd.isna(home["PTS"]) or pd.isna(away["PTS"]):
                    continue

                games.append(
                    {
                        "date": home["GAME_DATE"],
                        "home_team": home["TEAM_NAME"],
                        "away_team": away["TEAM_NAME"],
                        "homescore": int(home["PTS"]),
                        "awayscore": int(away["PTS"]),
                        "type": output_season_type,
                    }
                )
            time.sleep(0.6)

    result = pd.DataFrame(
        games,
        columns=["date", "home_team", "away_team", "homescore", "awayscore", "type"],
    )
    if not result.empty:
        result = result.sort_values(["date", "home_team", "away_team"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} games to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()