from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "data" / "nba_season.csv"
FIRST_SEASON_START = 2014
START_DATE = date(2015, 1, 1)
SEASON_TYPES = {
    "Regular Season": "regular",
    "Playoffs": "playoffs",
}
GAME_COLUMNS = ["date", "home_team", "away_team", "homescore", "awayscore", "type"]


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_start(day: date) -> int:
    return day.year if day.month >= 10 else day.year - 1


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


def extract_games(
    rows: pd.DataFrame, output_season_type: str, start: date, end: date
) -> list[dict[str, object]]:
    games = []
    rows = rows[rows["GAME_DATE"].between(start.isoformat(), end.isoformat())]
    for _, game_rows in rows.groupby("GAME_ID"):
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
    return games


def fetch_games(
    first_season_start: int, last_season_start: int, start: date, end: date
) -> pd.DataFrame:
    games = []
    for season_start_year in range(first_season_start, last_season_start + 1):
        season = season_label(season_start_year)
        for api_season_type, output_season_type in SEASON_TYPES.items():
            print(f"Fetching {season} {api_season_type}...")
            rows = fetch_season_games(season, api_season_type)
            if not rows.empty:
                games.extend(extract_games(rows, output_season_type, start, end))
            time.sleep(0.6)

    return pd.DataFrame(games, columns=GAME_COLUMNS)


def update_dataset(path: Path = OUTPUT_PATH, full: bool = False) -> pd.DataFrame:
    today = date.today()
    existing = None
    if not full and path.exists():
        existing = pd.read_csv(path, dtype=str)
        if not existing.empty:
            existing["homescore"] = existing["homescore"].astype(int)
            existing["awayscore"] = existing["awayscore"].astype(int)

    if existing is None or existing.empty:
        fetch_from_season = FIRST_SEASON_START
    else:
        last_played = pd.to_datetime(existing["date"]).max().date()
        fetch_from_season = season_start(last_played)

    fresh = fetch_games(fetch_from_season, first_season_to_fetch(today), START_DATE, today)
    combined = (
        fresh
        if existing is None or existing.empty
        else pd.concat([existing, fresh], ignore_index=True)
    )
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last"
    ).sort_values(["date", "home_team", "away_team"], kind="stable")

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    print(f"Saved {len(combined)} games to {path} ({len(fresh)} rows fetched)")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NBA game results into a CSV.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Refetch all seasons instead of only updating from the latest games.",
    )
    args = parser.parse_args()
    update_dataset(args.output, full=args.full)


if __name__ == "__main__":
    main()
