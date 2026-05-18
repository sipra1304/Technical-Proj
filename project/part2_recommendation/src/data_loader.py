import pandas as pd
import os


def load_movielens_1m(data_dir="data/ml-1m"):
    """Load MovieLens 1M dataset from .dat files."""
    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path = os.path.join(data_dir, "movies.dat")
    users_path = os.path.join(data_dir, "users.dat")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    users = pd.read_csv(
        users_path,
        sep="::",
        names=["userId", "gender", "age", "occupation", "zip"],
        engine="python",
        encoding="latin-1",
    )

    # Binary label: rating >= 3 is positive interaction (used for graph edges)
    ratings["label"] = (ratings["rating"] >= 4).astype(int)

    return ratings, movies, users


def train_test_split_by_time(ratings, test_ratio=0.2):
    """Split ratings by timestamp: latest interactions go to test set."""
    ratings_sorted = ratings.sort_values("timestamp")
    split_idx = int(len(ratings_sorted) * (1 - test_ratio))
    train = ratings_sorted.iloc[:split_idx].reset_index(drop=True)
    test = ratings_sorted.iloc[split_idx:].reset_index(drop=True)
    return train, test
