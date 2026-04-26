from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import normalize

from src.movie_data import clean_text


@dataclass
class MovieRecommendation:
    movie_id: str
    title: str
    year: int
    rating: float
    votes: int
    genres: str
    director: str
    cast: str
    overview: str
    similarity: float


class MovieRecommender:
    def __init__(self, artifacts_path: str | Path = "artifacts/movie_recommender.joblib") -> None:
        artifacts_path = Path(artifacts_path)
        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at '{artifacts_path}'. Train the model first."
            )

        artifacts = joblib.load(artifacts_path)
        self.vectorizer = artifacts["vectorizer"]
        self.svd = artifacts["svd"]
        self.nn_model = artifacts["nn_model"]
        self.movie_df = artifacts["movie_df"]
        self.title_to_idx = artifacts["title_to_idx"]

    def _encode(self, text: str) -> np.ndarray:
        x = self.vectorizer.transform([clean_text(text)])
        if self.svd is not None:
            x = self.svd.transform(x)
        else:
            x = x.toarray()
        return normalize(x)

    def titles(self) -> list[str]:
        return self.movie_df["title"].tolist()

    def recommend_by_title(self, movie_title: str, top_k: int = 12) -> list[MovieRecommendation]:
        idx = self.title_to_idx.get(movie_title.lower())
        if idx is None:
            raise ValueError(f"Movie '{movie_title}' was not found in trained metadata")

        row = self.movie_df.iloc[idx]
        seed_text = " ".join(
            [
                str(row["title"]),
                str(row["overview"]),
                str(row["genres"]),
                str(row["keywords"]),
                str(row["cast"]),
                str(row["director"]),
            ]
        )

        seed_vector = self._encode(seed_text)
        distances, indices = self.nn_model.kneighbors(seed_vector, n_neighbors=min(top_k + 1, len(self.movie_df)))

        results: list[MovieRecommendation] = []
        for distance, neighbor_idx in zip(distances[0], indices[0]):
            if int(neighbor_idx) == int(idx):
                continue
            movie = self.movie_df.iloc[int(neighbor_idx)]
            similarity = max(0.0, 1.0 - float(distance))
            results.append(
                MovieRecommendation(
                    movie_id=str(movie["movie_id"]),
                    title=str(movie["title"]),
                    year=int(movie["year"]),
                    rating=float(movie["rating"]),
                    votes=int(movie["votes"]),
                    genres=str(movie["genres"]),
                    director=str(movie["director"]),
                    cast=str(movie["cast"]),
                    overview=str(movie["overview"]),
                    similarity=similarity,
                )
            )

        return results[:top_k]
