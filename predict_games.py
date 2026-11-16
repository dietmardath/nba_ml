from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from nba_api.stats.endpoints import scoreboardv3

from elo_backtest import (
    HomeAdvantageTracker,
    INITIAL_RATING,
    home_win_probability,
    simulate,
)
from fetch_nba_games import OUTPUT_PATH, update_dataset
from train_winner_model import FEATURE_COLUMNS, RECENT_GAMES


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "nba_winner_model.joblib"


def default_game_date() -> date:
    try:
        from zoneinfo import ZoneInfo

        nba_today = datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        nba_today = date.today()
    return nba_today + timedelta(days=1)


def build_team_form(predictions: pd.DataFrame) -> pd.DataFrame:
    home_games = pd.DataFrame(
        {
            "date": predictions["date"],
            "team": predictions["home_team"],
            "win": predictions["actual_home_win"].astype(float),
            "margin": predictions["home_score"] - predictions["away_score"],
        }
    )
    away_games = pd.DataFrame(
        {
            "date": predictions["date"],
            "team": predictions["away_team"],
            "win": 1 - predictions["actual_home_win"].astype(float),
            "margin": predictions["away_score"] - predictions["home_score"],
        }
    )
    long = pd.concat([home_games, away_games]).sort_values("date", kind="stable")
    recent = long.groupby("team").tail(RECENT_GAMES)
    return recent.groupby("team").agg(
        win_rate=("win", "mean"), margin=("margin", "mean"), games=("win", "size")
    )


def fetch_schedule(game_date: date) -> pd.DataFrame:
    for attempt in range(3):
        try:
            board = scoreboardv3.ScoreboardV3(
                game_date=game_date.strftime("%m/%d/%Y"), timeout=60
            )
            frames = board.get_data_frames()
            header, teams = frames[1], frames[2]
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    else:
        raise RuntimeError("NBA schedule could not be fetched")

    games = []
    for game in header.itertuples(index=False):
        try:
            away_tricode, home_tricode = (
                game.gameCode.split("/")[1][:3],
                game.gameCode.split("/")[1][3:],
            )
            game_teams = teams[teams["gameId"] == game.gameId]

            def full_name(tricode: str) -> str:
                row = game_teams[game_teams["teamTricode"] == tricode].iloc[0]
                return f"{row['teamCity']} {row['teamName']}"

            home, away = full_name(home_tricode), full_name(away_tricode)
        except (IndexError, KeyError):
            continue
        games.append(
            {"home_team": home, "away_team": away, "status": game.gameStatusText}
        )
    return pd.DataFrame(games, columns=["home_team", "away_team", "status"])


def build_features(
    schedule: pd.DataFrame,
    ratings: dict[str, float],
    form: pd.DataFrame,
    advantages: HomeAdvantageTracker,
) -> pd.DataFrame:
    rows = []
    for game in schedule.itertuples(index=False):
        home, away = game.home_team, game.away_team
        home_rating = ratings.get(home, INITIAL_RATING)
        away_rating = ratings.get(away, INITIAL_RATING)
        home_form = form.loc[home] if home in form.index else None
        away_form = form.loc[away] if away in form.index else None
        home_advantage = advantages.advantage(home)
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "status": game.status,
                "home_rating_pre": home_rating,
                "away_rating_pre": away_rating,
                "home_home_advantage": home_advantage,
                "away_home_advantage": advantages.advantage(away),
                "home_win_probability": home_win_probability(
                    home_rating, away_rating, home_advantage
                ),
                "home_recent_win_rate": (
                    home_form["win_rate"] if home_form is not None else 0.5
                ),
                "away_recent_win_rate": (
                    away_form["win_rate"] if away_form is not None else 0.5
                ),
                "home_recent_margin": (
                    home_form["margin"] if home_form is not None else 0.0
                ),
                "away_recent_margin": (
                    away_form["margin"] if away_form is not None else 0.0
                ),
                "home_recent_games": (
                    int(home_form["games"]) if home_form is not None else 0
                ),
                "away_recent_games": (
                    int(away_form["games"]) if away_form is not None else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict winners and probabilities for upcoming NBA games."
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="NBA game date, YYYY-MM-DD (default: tomorrow, US Eastern time).",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Skip refreshing the game history from the NBA API.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    game_date = args.date or default_game_date()

    if args.no_update:
        games = pd.read_csv(OUTPUT_PATH)
    else:
        games = update_dataset()

    predictions, ratings, advantages = simulate(games)
    form = build_team_form(predictions)

    schedule = fetch_schedule(game_date)
    if schedule.empty:
        print(f"Keine NBA-Spiele am {game_date:%d.%m.%Y}.")
        return

    matchups = build_features(schedule, ratings, form, advantages)

    model_probabilities = None
    try:
        bundle = joblib.load(args.model)
        model = bundle["model"]
        model_probabilities = model.predict_proba(matchups[FEATURE_COLUMNS])[:, 1]
    except Exception as error:
        print(f"Hinweis: ML-Modell nicht nutzbar ({error}), zeige nur Elo.")

    print(f"\nNBA-Spiele am {game_date:%d.%m.%Y} ({len(matchups)} Spiele):\n")
    matchup_width = max(
        len(f"{row.away_team} @ {row.home_team}")
        for row in matchups.itertuples(index=False)
    )
    for index, game in enumerate(matchups.itertuples(index=False)):
        elo_probability = game.home_win_probability
        if model_probabilities is not None:
            probability = float(model_probabilities[index])
            source = "ML "
        else:
            probability = elo_probability
            source = "Elo"
        favorite = game.home_team if probability >= 0.5 else game.away_team
        matchup = f"{game.away_team} @ {game.home_team}"
        print(
            f"{matchup:<{matchup_width}}  {game.status:<12}"
            f"{source}: {favorite} {max(probability, 1 - probability):6.1%}"
            f"  (Elo: {max(elo_probability, 1 - elo_probability):6.1%})"
        )


if __name__ == "__main__":
    main()
