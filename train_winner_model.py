from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from elo_backtest import run_backtest


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "nba_season.csv"
DEFAULT_MODEL_OUTPUT = ROOT / "models" / "nba_winner_model.joblib"
DEFAULT_PREDICTIONS_OUTPUT = ROOT / "data" / "ml_winner_predictions.csv"
DEFAULT_METRICS_OUTPUT = ROOT / "data" / "ml_winner_validation.csv"
TRAIN_SEASONS = range(2015, 2022)
VALIDATION_SEASONS = (2022, 2023)
RECENT_GAMES = 10
FEATURE_COLUMNS = [
    "home_rating_pre",
    "away_rating_pre",
    "home_win_probability",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_recent_margin",
    "away_recent_margin",
    "home_recent_games",
    "away_recent_games",
]


def add_pregame_form_features(games: pd.DataFrame) -> pd.DataFrame:
    predictions, _ = run_backtest(games)
    recent: dict[str, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=RECENT_GAMES)
    )
    rows = []

    for _, day in predictions.groupby("date", sort=True):
        day_rows = []
        for game in day.itertuples(index=False):
            home_history = list(recent[game.home_team])
            away_history = list(recent[game.away_team])
            row = game._asdict()
            row.update(
                {
                    "home_recent_win_rate": (
                        sum(result for result, _ in home_history) / len(home_history)
                        if home_history
                        else 0.5
                    ),
                    "away_recent_win_rate": (
                        sum(result for result, _ in away_history) / len(away_history)
                        if away_history
                        else 0.5
                    ),
                    "home_recent_margin": (
                        sum(margin for _, margin in home_history) / len(home_history)
                        if home_history
                        else 0.0
                    ),
                    "away_recent_margin": (
                        sum(margin for _, margin in away_history) / len(away_history)
                        if away_history
                        else 0.0
                    ),
                    "home_recent_games": len(home_history),
                    "away_recent_games": len(away_history),
                }
            )
            day_rows.append(row)
            rows.append(row)

        for game in day.itertuples(index=False):
            home_win = int(game.actual_home_win)
            margin = int(game.home_score - game.away_score)
            recent[game.home_team].append((home_win, margin))
            recent[game.away_team].append((1 - home_win, -margin))

    return pd.DataFrame(rows)


def season_start(year_label: str) -> int:
    return int(year_label.split("-")[0])


def evaluate_validation(
    validation: pd.DataFrame, model_probabilities: pd.Series
) -> pd.DataFrame:
    rows = []
    for season, games in validation.groupby("season", sort=True):
        probabilities = model_probabilities.loc[games.index]
        actual = games["actual_home_win"].astype(int)
        elo_probabilities = games["home_win_probability"]
        rows.append(
            {
                "season": season,
                "games": len(games),
                "model_accuracy": accuracy_score(actual, probabilities >= 0.5),
                "model_brier_score": brier_score_loss(actual, probabilities),
                "model_log_loss": log_loss(actual, probabilities, labels=[0, 1]),
                "model_roc_auc": roc_auc_score(actual, probabilities),
                "elo_accuracy": accuracy_score(actual, elo_probabilities >= 0.5),
                "elo_brier_score": brier_score_loss(actual, elo_probabilities),
                "elo_log_loss": log_loss(actual, elo_probabilities, labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an NBA winner classifier on 2015-2021 and validate on 2022-2023."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    args = parser.parse_args()

    games = pd.read_csv(args.input)
    dataset = add_pregame_form_features(games)
    dataset["season_start"] = dataset["season"].map(season_start)

    training = dataset[dataset["season_start"].between(2015, 2021)].copy()
    validation = dataset[dataset["season_start"].isin(VALIDATION_SEASONS)].copy()
    if training.empty or validation.empty:
        raise ValueError("Training or validation games are missing from the input data")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    )
    model.fit(training[FEATURE_COLUMNS], training["actual_home_win"].astype(int))

    validation = validation.copy()
    probabilities = pd.Series(
        model.predict_proba(validation[FEATURE_COLUMNS])[:, 1], index=validation.index
    )
    validation["model_home_win_probability"] = probabilities
    validation["model_predicted_home_win"] = probabilities >= 0.5
    validation["correct"] = (
        validation["model_predicted_home_win"]
        == validation["actual_home_win"].astype(bool)
    )
    metrics = evaluate_validation(validation, probabilities)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS}, args.model_output
    )
    validation.to_csv(args.predictions_output, index=False)
    metrics.to_csv(args.metrics_output, index=False)

    display_metrics = metrics.copy()
    for column in display_metrics.select_dtypes(include="number").columns:
        if column != "games":
            display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.3f}")
    print(f"Training games (2015-16 to 2021-22): {len(training)}")
    print(display_metrics.to_string(index=False))
    print(f"\nModel saved to {args.model_output}")
    print(f"Validation predictions saved to {args.predictions_output}")
    print(f"Validation metrics saved to {args.metrics_output}")


if __name__ == "__main__":
    main()
