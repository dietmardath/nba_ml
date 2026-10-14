from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "nba_season.csv"
INITIAL_RATING = 1500.0
DEFAULT_HOME_ADVANTAGE = 80.0
DEFAULT_K_FACTOR = 20.0
SEASON_REGRESSION = 1 / 3
REQUIRED_COLUMNS = {"date", "home_team", "away_team", "homescore", "awayscore"}


def season_start(date: pd.Timestamp) -> int:
    return date.year if date.month >= 10 else date.year - 1


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def home_win_probability(
    home_rating: float, away_rating: float, home_advantage: float
) -> float:
    rating_difference = home_rating + home_advantage - away_rating
    return 1 / (1 + 10 ** (-rating_difference / 400))


def margin_multiplier(margin: int, rating_difference: float) -> float:
    return math.log(abs(margin) + 1) * 2.2 / (2.2 + 0.001 * abs(rating_difference))


def run_backtest(
    games: pd.DataFrame,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    k_factor: float = DEFAULT_K_FACTOR,
    season_regression: float = SEASON_REGRESSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_COLUMNS - set(games.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not 0 <= season_regression <= 1:
        raise ValueError("season_regression must be between 0 and 1")

    games = games.copy()
    games["date"] = pd.to_datetime(games["date"], errors="raise")
    games["homescore"] = pd.to_numeric(games["homescore"], errors="raise")
    games["awayscore"] = pd.to_numeric(games["awayscore"], errors="raise")
    if games[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Game dates, teams, and scores must not be missing")
    games["season"] = games["date"].map(lambda date: season_label(season_start(date)))
    games = games.sort_values(["date", "home_team", "away_team"], kind="stable")

    ratings: dict[str, float] = {}
    previous_season: int | None = None
    predictions: list[dict[str, object]] = []

    for game_date, day_games in games.groupby("date", sort=True):
        current_season = season_start(game_date)
        if previous_season is not None and current_season != previous_season:
            for team in ratings:
                ratings[team] += (INITIAL_RATING - ratings[team]) * season_regression
        previous_season = current_season

        daily_predictions = []
        for game in day_games.itertuples(index=False):
            home_rating = ratings.get(game.home_team, INITIAL_RATING)
            away_rating = ratings.get(game.away_team, INITIAL_RATING)
            adjusted_difference = home_rating + home_advantage - away_rating
            probability = home_win_probability(home_rating, away_rating, home_advantage)
            actual = float(game.homescore > game.awayscore)
            prediction = {
                "date": game_date,
                "season": game.season,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_score": game.homescore,
                "away_score": game.awayscore,
                "home_rating_pre": home_rating,
                "away_rating_pre": away_rating,
                "home_win_probability": probability,
                "actual_home_win": actual,
                "predicted_home_win": probability >= 0.5,
            }
            daily_predictions.append((prediction, adjusted_difference))

        # All games on a date are forecast from ratings available before that date.
        rating_changes: dict[str, float] = {}
        for prediction, adjusted_difference in daily_predictions:
            margin = prediction["home_score"] - prediction["away_score"]
            multiplier = margin_multiplier(margin, adjusted_difference)
            change = k_factor * multiplier * (
                prediction["actual_home_win"] - prediction["home_win_probability"]
            )
            home_team = prediction["home_team"]
            away_team = prediction["away_team"]
            rating_changes[home_team] = rating_changes.get(home_team, 0.0) + change
            rating_changes[away_team] = rating_changes.get(away_team, 0.0) - change
            predictions.append(prediction)

        for team, change in rating_changes.items():
            ratings[team] = ratings.get(team, INITIAL_RATING) + change

    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        return prediction_frame, pd.DataFrame(
            columns=["season", "games", "home_accuracy", "brier_score", "log_loss"]
        )

    prediction_frame["brier_component"] = (
        prediction_frame["home_win_probability"] - prediction_frame["actual_home_win"]
    ) ** 2
    epsilon = 1e-15
    probability = prediction_frame["home_win_probability"].clip(epsilon, 1 - epsilon)
    prediction_frame["log_loss_component"] = -(
        prediction_frame["actual_home_win"] * probability.map(math.log)
        + (1 - prediction_frame["actual_home_win"])
        * (1 - probability).map(math.log)
    )
    prediction_frame["correct"] = (
        prediction_frame["predicted_home_win"]
        == prediction_frame["actual_home_win"].astype(bool)
    )
    summary = (
        prediction_frame.groupby("season", sort=True)
        .agg(
            games=("date", "size"),
            home_accuracy=("correct", "mean"),
            brier_score=("brier_component", "mean"),
            log_loss=("log_loss_component", "mean"),
        )
        .reset_index()
    )
    return prediction_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest pre-game NBA Elo ratings season by season."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--home-advantage", type=float, default=DEFAULT_HOME_ADVANTAGE)
    parser.add_argument("--k-factor", type=float, default=DEFAULT_K_FACTOR)
    args = parser.parse_args()

    games = pd.read_csv(args.input)
    predictions, summary = run_backtest(
        games,
        home_advantage=args.home_advantage,
        k_factor=args.k_factor,
    )
    if summary.empty:
        print("No games to backtest.")
        return

    display_summary = summary.copy()
    for column in ("home_accuracy", "brier_score", "log_loss"):
        display_summary[column] = display_summary[column].map(lambda value: f"{value:.3f}")
    print(display_summary.to_string(index=False))

    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions_output, index=False)
        print(f"\nSaved game predictions to {args.predictions_output}")


if __name__ == "__main__":
    main()
