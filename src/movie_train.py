from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.movie_data import build_content_text, load_movies


def _similarity_stats(embeddings: np.ndarray) -> dict[str, float]:
    n = embeddings.shape[0]
    if n < 2:
        return {
            "avg_pair_similarity": 0.0,
            "max_pair_similarity": 0.0,
            "min_pair_similarity": 0.0,
        }

    # For large datasets, estimate similarity stats from random pairs to avoid O(n^2) memory.
    rng = np.random.default_rng(42)
    sample_pairs = min(20000, n * 4)
    i_idx = rng.integers(0, n, size=sample_pairs)
    j_idx = rng.integers(0, n, size=sample_pairs)

    mask = i_idx != j_idx
    if not np.any(mask):
        return {
            "avg_pair_similarity": 0.0,
            "max_pair_similarity": 0.0,
            "min_pair_similarity": 0.0,
        }

    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    pair_sim = np.sum(embeddings[i_idx] * embeddings[j_idx], axis=1)

    return {
        "avg_pair_similarity": float(np.mean(pair_sim)),
        "max_pair_similarity": float(np.max(pair_sim)),
        "min_pair_similarity": float(np.min(pair_sim)),
    }


def train_movie_model(
    dataset_path: str | Path,
    artifacts_path: str | Path,
    max_features: int = 12000,
    n_components: int = 100,
) -> dict[str, Any]:
    df = load_movies(dataset_path)
    texts = build_content_text(df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
    )
    X_tfidf = vectorizer.fit_transform(texts)

    use_svd = X_tfidf.shape[1] > n_components + 1
    if use_svd:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_dense = svd.fit_transform(X_tfidf)
    else:
        svd = None
        X_dense = X_tfidf.toarray()

    X_embed = normalize(X_dense)

    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(X_embed)

    title_to_idx = {title.lower(): i for i, title in enumerate(df["title"].tolist())}

    artifacts = {
        "vectorizer": vectorizer,
        "svd": svd,
        "nn_model": nn_model,
        "movie_df": df,
        "title_to_idx": title_to_idx,
    }

    artifacts_path = Path(artifacts_path)
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, artifacts_path)

    stats = _similarity_stats(X_embed)
    return {
        "num_movies": int(len(df)),
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "embedding_dim": int(X_embed.shape[1]),
        "used_svd": bool(use_svd),
        "svd_components": int(n_components if use_svd else X_embed.shape[1]),
        **stats,
    }
