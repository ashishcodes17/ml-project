from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_MOVIE_COLS = {
    "movie_id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "director",
    "year",
    "rating",
    "votes",
}


def clean_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def validate_movies_frame(df: pd.DataFrame) -> None:
    missing = REQUIRED_MOVIE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"movie dataset is missing required columns: {sorted(missing)}")


def load_movies(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_movies_frame(df)

    for col in ["movie_id", "title", "overview", "genres", "keywords", "cast", "director"]:
        df[col] = df[col].astype(str)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["title", "overview"]).copy()
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    return df


def build_content_text(df: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for _, row in df.iterrows():
        genres = clean_text(row["genres"])
        # Repeat genres to slightly increase their weight in TF-IDF.
        genre_boost = f"{genres} {genres}"
        combined = " ".join(
            [
                clean_text(row["title"]),
                clean_text(row["overview"]),
                genre_boost,
                clean_text(row["keywords"]),
                clean_text(row["cast"]),
                clean_text(row["director"]),
            ]
        )
        texts.append(combined)
    return texts
