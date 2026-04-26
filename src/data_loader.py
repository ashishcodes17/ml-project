from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


REQUIRED_JOB_COLS = {"job_id", "title", "description"}
REQUIRED_RESUME_COLS = {"resume_id", "candidate_name", "resume_text"}


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def load_data(jobs_path: str | Path, resumes_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    jobs_df = pd.read_csv(jobs_path)
    resumes_df = pd.read_csv(resumes_path)

    _validate_columns(jobs_df, REQUIRED_JOB_COLS, "jobs file")
    _validate_columns(resumes_df, REQUIRED_RESUME_COLS, "resumes file")

    jobs_df = jobs_df.dropna(subset=["job_id", "title", "description"]).copy()
    resumes_df = resumes_df.dropna(subset=["resume_id", "candidate_name", "resume_text"]).copy()

    jobs_df["job_id"] = jobs_df["job_id"].astype(str)
    resumes_df["resume_id"] = resumes_df["resume_id"].astype(str)

    return jobs_df, resumes_df
