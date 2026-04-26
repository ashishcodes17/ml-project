from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from typing import Iterable

import pandas as pd


IMDB_BASE = "https://datasets.imdbws.com"
DATA_FILES = {
    "title.basics.tsv.gz": "title.basics.tsv.gz",
    "title.ratings.tsv.gz": "title.ratings.tsv.gz",
    "title.akas.tsv.gz": "title.akas.tsv.gz",
    "title.crew.tsv.gz": "title.crew.tsv.gz",
    "name.basics.tsv.gz": "name.basics.tsv.gz",
}

INDIAN_LANGS = {"hi", "ta", "te", "ml", "kn", "mr", "bn", "pa", "gu", "or", "as", "ur"}


def _download_if_missing(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name, remote_name in DATA_FILES.items():
        target = data_dir / file_name
        if target.exists():
            continue
        url = f"{IMDB_BASE}/{remote_name}"
        print(f"Downloading {url}")
        urlretrieve(url, target)


def _read_tsv_gz(path: Path, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        na_values="\\N",
        low_memory=False,
        compression="gzip",
    )


def _read_tsv_gz_chunks(path: Path, usecols: list[str], chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        na_values="\\N",
        low_memory=False,
        compression="gzip",
        chunksize=chunksize,
    )


def _collect_indian_title_ids(akas_path: Path, chunksize: int = 1_000_000) -> set[str]:
    indian_ids: set[str] = set()
    for chunk in _read_tsv_gz_chunks(akas_path, ["titleId", "region", "language"], chunksize):
        is_indian = (chunk["region"] == "IN") | (chunk["language"].isin(INDIAN_LANGS))
        ids = chunk.loc[is_indian, "titleId"].dropna().astype(str)
        indian_ids.update(ids.tolist())
    return indian_ids


def _load_basics_for_ids(basics_path: Path, indian_ids: set[str], chunksize: int = 750_000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    cols = ["tconst", "titleType", "primaryTitle", "startYear", "genres", "isAdult"]
    for chunk in _read_tsv_gz_chunks(basics_path, cols, chunksize):
        filtered = chunk[
            (chunk["titleType"] == "movie")
            & (chunk["isAdult"] == 0)
            & (chunk["tconst"].isin(indian_ids))
        ]
        filtered = filtered.dropna(subset=["tconst", "primaryTitle", "startYear"])
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=cols)
    return pd.concat(chunks, ignore_index=True)


def _load_ratings_for_ids(ratings_path: Path, target_ids: set[str], chunksize: int = 1_000_000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in _read_tsv_gz_chunks(ratings_path, ["tconst", "averageRating", "numVotes"], chunksize):
        filtered = chunk[chunk["tconst"].isin(target_ids)]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=["tconst", "averageRating", "numVotes"])
    return pd.concat(chunks, ignore_index=True)


def _load_crew_for_ids(crew_path: Path, target_ids: set[str], chunksize: int = 1_000_000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in _read_tsv_gz_chunks(crew_path, ["tconst", "directors"], chunksize):
        filtered = chunk[chunk["tconst"].isin(target_ids)].copy()
        if not filtered.empty:
            filtered["director_nconst"] = filtered["directors"].astype(str).str.split(",").str[0]
            chunks.append(filtered[["tconst", "director_nconst"]])
    if not chunks:
        return pd.DataFrame(columns=["tconst", "director_nconst"])
    return pd.concat(chunks, ignore_index=True)


def _load_names_for_ids(names_path: Path, director_ids: set[str], chunksize: int = 1_000_000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in _read_tsv_gz_chunks(names_path, ["nconst", "primaryName"], chunksize):
        filtered = chunk[chunk["nconst"].isin(director_ids)]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=["director_nconst", "director_name"])
    names = pd.concat(chunks, ignore_index=True)
    names = names.rename(columns={"nconst": "director_nconst", "primaryName": "director_name"})
    return names


def prepare_indian_movies_dataset(
    raw_dir: str | Path = "data/raw/imdb",
    output_csv: str | Path = "data/movies_india_100k.csv",
    min_votes: int = 50,
    max_movies: int = 120000,
) -> Path:
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)

    _download_if_missing(raw_dir)

    print("Reading title.akas in chunks")
    indian_ids = _collect_indian_title_ids(raw_dir / "title.akas.tsv.gz")

    print("Reading title.basics in chunks")
    basics = _load_basics_for_ids(raw_dir / "title.basics.tsv.gz", indian_ids)
    target_ids = set(basics["tconst"].astype(str).tolist())

    print("Reading title.ratings in chunks")
    ratings = _load_ratings_for_ids(raw_dir / "title.ratings.tsv.gz", target_ids)
    ratings = ratings.dropna(subset=["tconst"])

    merged = basics.merge(ratings, on="tconst", how="left")
    merged["numVotes"] = pd.to_numeric(merged["numVotes"], errors="coerce").fillna(0).astype(int)
    merged["averageRating"] = pd.to_numeric(merged["averageRating"], errors="coerce").fillna(0.0)
    merged = merged[merged["numVotes"] >= min_votes].copy()

    print("Reading title.crew in chunks")
    crew = _load_crew_for_ids(raw_dir / "title.crew.tsv.gz", target_ids)

    director_ids = set(crew["director_nconst"].dropna().astype(str).tolist())
    print("Reading name.basics in chunks")
    names = _load_names_for_ids(raw_dir / "name.basics.tsv.gz", director_ids)

    merged = merged.merge(crew, on="tconst", how="left")
    merged = merged.merge(names, on="director_nconst", how="left")

    merged["startYear"] = pd.to_numeric(merged["startYear"], errors="coerce").fillna(0).astype(int)
    merged = merged.sort_values(["numVotes", "averageRating"], ascending=[False, False])

    if max_movies > 0:
        merged = merged.head(max_movies).copy()

    merged["genres"] = merged["genres"].fillna("").astype(str).str.replace(",", "|", regex=False)
    merged["keywords"] = merged["genres"].str.replace("|", " ", regex=False)
    merged["overview"] = (
        "Indian movie. Title: "
        + merged["primaryTitle"].astype(str)
        + ". Genres: "
        + merged["genres"].astype(str)
        + "."
    )

    final_df = pd.DataFrame(
        {
            "movie_id": merged["tconst"].astype(str),
            "title": merged["primaryTitle"].astype(str),
            "overview": merged["overview"].astype(str),
            "genres": merged["genres"].astype(str),
            "keywords": merged["keywords"].astype(str),
            "cast": "",
            "director": merged["director_name"].fillna("Unknown").astype(str),
            "year": merged["startYear"].astype(int),
            "rating": merged["averageRating"].astype(float),
            "votes": merged["numVotes"].astype(int),
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print(f"Saved {len(final_df)} movies to {output_csv}")
    return output_csv


if __name__ == "__main__":
    prepare_indian_movies_dataset()
