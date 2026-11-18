from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from elo_backtest import REQUIRED_COLUMNS, season_label, season_start


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "nba_season.csv"
DEFAULT_FEATURES_OUTPUT = ROOT / "data" / "team_season_features.csv"
DEFAULT_ASSIGNMENTS_OUTPUT = ROOT / "data" / "team_cluster_assignments.csv"
DEFAULT_PAIR_ADVANTAGES_OUTPUT = ROOT / "data" / "cluster_pair_advantages.csv"
DEFAULT_N_CLUSTERS = 5
DEFAULT_BASE_ADVANTAGE = 3.0
DEFAULT_PRIOR_GAMES = 150
UNKNOWN_CLUSTER = -1
FEATURE_COLUMNS = [
    "avg_margin_wins",
    "avg_margin_losses",
    "avg_win_streak_len",
    "avg_loss_streak_len",
]


def team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape games into one row per team per game, in chronological order."""
    missing = REQUIRED_COLUMNS - set(games.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    games = games.copy()
    games["date"] = pd.to_datetime(games["date"], errors="raise")
    games["homescore"] = pd.to_numeric(games["homescore"], errors="raise")
    games["awayscore"] = pd.to_numeric(games["awayscore"], errors="raise")
    if games[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Game dates, teams, and scores must not be missing")

    home = pd.DataFrame(
        {
            "date": games["date"],
            "team": games["home_team"],
            "margin": games["homescore"] - games["awayscore"],
        }
    )
    away = pd.DataFrame(
        {
            "date": games["date"],
            "team": games["away_team"],
            "margin": games["awayscore"] - games["homescore"],
        }
    )
    log = pd.concat([home, away], ignore_index=True)
    log["season"] = log["date"].map(lambda date: season_label(season_start(date)))
    log["win"] = log["margin"] > 0
    return log.sort_values(["team", "date"], kind="stable")


def streak_lengths(results: list[bool], target: bool) -> list[int]:
    """Return the lengths of completed runs of `target` results."""
    runs: list[int] = []
    current = 0
    for result in results:
        if result == target:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def build_team_season_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team per-season margin and streak features."""
    log = team_game_log(games)
    rows = []
    for (team, season), group in log.groupby(["team", "season"], sort=True):
        wins = group.loc[group["win"], "margin"]
        losses = group.loc[~group["win"], "margin"]
        results = group["win"].tolist()
        win_streaks = streak_lengths(results, True)
        loss_streaks = streak_lengths(results, False)
        rows.append(
            {
                "team": team,
                "season": season,
                "games": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "avg_margin_wins": wins.mean(),
                "avg_margin_losses": losses.mean(),
                "avg_win_streak_len": (
                    sum(win_streaks) / len(win_streaks) if win_streaks else 0.0
                ),
                "avg_loss_streak_len": (
                    sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def assign_clusters(
    features: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS
) -> pd.DataFrame:
    """Assign cluster labels leakage-safe: season t uses a model fit on seasons < t."""
    seasons = sorted(features["season"].unique())
    assignments = []
    for index, season in enumerate(seasons):
        current = features[features["season"] == season]
        if index == 0 or n_clusters < 2:
            clusters = pd.Series(UNKNOWN_CLUSTER, index=current.index)
        else:
            history = features[features["season"].isin(seasons[:index])]
            fit_count = min(n_clusters, len(history))
            scaler = StandardScaler()
            history_scaled = scaler.fit_transform(history[FEATURE_COLUMNS])
            kmeans = KMeans(n_clusters=fit_count, n_init=10, random_state=42)
            kmeans.fit(history_scaled)
            clusters = pd.Series(
                kmeans.predict(scaler.transform(current[FEATURE_COLUMNS])),
                index=current.index,
            )
        assignments.append(
            pd.DataFrame(
                {"season": season, "team": current["team"], "cluster": clusters}
            )
        )
    return pd.concat(assignments, ignore_index=True)


class ClusterAdvantageTracker:
    """Per-cluster-pair home advantage estimated from past games only.

    Mirrors HomeAdvantageTracker but keyed by (home_cluster, away_cluster):
    tracks the pair's mean home margin, shrinks its deviation from the
    league mean toward zero, and adds it to the base advantage in points.
    """

    def __init__(
        self,
        base_advantage: float = DEFAULT_BASE_ADVANTAGE,
        prior_games: int = DEFAULT_PRIOR_GAMES,
    ) -> None:
        self.base_advantage = base_advantage
        self.prior_games = prior_games
        self.pair_margin: dict[tuple[int, int], float] = {}
        self.pair_games: dict[tuple[int, int], int] = {}
        self.league_margin = 0.0
        self.league_games = 0

    def advantage(self, home_cluster: int, away_cluster: int) -> float:
        key = (home_cluster, away_cluster)
        games = self.pair_games.get(key, 0)
        if not games or not self.league_games:
            return self.base_advantage
        pair_mean = self.pair_margin[key] / games
        league_mean = self.league_margin / self.league_games
        shrinkage = games / (games + self.prior_games)
        return self.base_advantage + (pair_mean - league_mean) * shrinkage

    def update(self, home_cluster: int, away_cluster: int, margin: float) -> None:
        key = (home_cluster, away_cluster)
        self.pair_margin[key] = self.pair_margin.get(key, 0.0) + margin
        self.pair_games[key] = self.pair_games.get(key, 0) + 1
        self.league_margin += margin
        self.league_games += 1


def replay_cluster_advantages(
    games: pd.DataFrame,
    assignments: pd.DataFrame,
    tracker: ClusterAdvantageTracker | None = None,
) -> tuple[pd.DataFrame, ClusterAdvantageTracker]:
    """Replay games chronologically, recording the pre-game pair advantage."""
    if tracker is None:
        tracker = ClusterAdvantageTracker()

    games = games.copy()
    games["date"] = pd.to_datetime(games["date"], errors="raise")
    games["season"] = games["date"].map(lambda date: season_label(season_start(date)))
    games = games.sort_values(["date", "home_team", "away_team"], kind="stable")

    cluster_lookup = {
        (row.season, row.team): row.cluster
        for row in assignments.itertuples(index=False)
    }

    rows = []
    for game in games.itertuples(index=False):
        home_cluster = cluster_lookup.get((game.season, game.home_team), UNKNOWN_CLUSTER)
        away_cluster = cluster_lookup.get((game.season, game.away_team), UNKNOWN_CLUSTER)
        margin = game.homescore - game.awayscore
        rows.append(
            {
                "date": game.date,
                "season": game.season,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_cluster": home_cluster,
                "away_cluster": away_cluster,
                "cluster_home_advantage": tracker.advantage(home_cluster, away_cluster),
                "actual_margin": margin,
            }
        )
        tracker.update(home_cluster, away_cluster, float(margin))
    return pd.DataFrame(rows), tracker


def pair_advantage_table(tracker: ClusterAdvantageTracker) -> pd.DataFrame:
    """Summarize the tracker's current per-pair advantage estimates."""
    rows = [
        {
            "home_cluster": home_cluster,
            "away_cluster": away_cluster,
            "games": tracker.pair_games[key],
            "mean_home_margin": tracker.pair_margin[key] / tracker.pair_games[key],
            "advantage": tracker.advantage(home_cluster, away_cluster),
        }
        for key in tracker.pair_games
        for home_cluster, away_cluster in [key]
    ]
    return pd.DataFrame(rows).sort_values(
        ["home_cluster", "away_cluster"], ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-team per-season margin and streak features and cluster them."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--features-output", type=Path, default=DEFAULT_FEATURES_OUTPUT)
    parser.add_argument(
        "--assignments-output", type=Path, default=DEFAULT_ASSIGNMENTS_OUTPUT
    )
    parser.add_argument(
        "--pair-advantages-output", type=Path, default=DEFAULT_PAIR_ADVANTAGES_OUTPUT
    )
    parser.add_argument("--clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--prior-games", type=int, default=DEFAULT_PRIOR_GAMES)
    args = parser.parse_args()

    games = pd.read_csv(args.input)
    features = build_team_season_features(games)
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_output, index=False)

    assignments = assign_clusters(features, n_clusters=args.clusters)
    args.assignments_output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.assignments_output, index=False)

    print(f"Saved {len(features)} team-season rows to {args.features_output}")
    print(
        f"Saved cluster assignments (k={args.clusters}) "
        f"to {args.assignments_output}"
    )

    tracker = ClusterAdvantageTracker(prior_games=args.prior_games)
    game_advantages, tracker = replay_cluster_advantages(games, assignments, tracker)
    pair_table = pair_advantage_table(tracker)
    args.pair_advantages_output.parent.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(args.pair_advantages_output, index=False)
    print(
        f"Saved {len(pair_table)} cluster-pair advantages "
        f"to {args.pair_advantages_output}"
    )
    print(pair_table.to_string(index=False))


if __name__ == "__main__":
    main()
