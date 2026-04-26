from __future__ import annotations

from pathlib import Path

import numpy as np

from src.movie_recommender import MovieRecommender


def _genre_set(genres: str) -> set[str]:
    return {g.strip().lower() for g in str(genres).split("|") if g.strip()}


def evaluate_genre_consistency(
    artifacts_path: str | Path = "artifacts/movie_recommender.joblib",
    sample_size: int = 300,
    k: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    recommender = MovieRecommender(artifacts_path)
    df = recommender.movie_df

    sample_size = min(sample_size, len(df))
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(df), size=sample_size, replace=False)

    hit_count = 0
    overlaps: list[float] = []

    for idx in indices:
        seed = df.iloc[int(idx)]
        seed_genres = _genre_set(seed["genres"])
        if not seed_genres:
            continue

        recs = recommender.recommend_by_title(str(seed["title"]), top_k=k)
        shared_any = False

        for rec in recs:
            rec_genres = _genre_set(rec.genres)
            inter = len(seed_genres & rec_genres)
            union = max(len(seed_genres | rec_genres), 1)
            jaccard = inter / union
            overlaps.append(jaccard)
            if inter > 0:
                shared_any = True

        if shared_any:
            hit_count += 1

    total = max(sample_size, 1)
    return {
        "sample_size": float(sample_size),
        "k": float(k),
        "hit_rate_at_k": float(hit_count / total),
        "avg_genre_jaccard": float(np.mean(overlaps) if overlaps else 0.0),
    }


if __name__ == "__main__":
    metrics = evaluate_genre_consistency()
    for key, value in metrics.items():
        print(f"{key}: {value}")
