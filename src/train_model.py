from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.preprocess import clean_text, extract_skills


REQUIRED_TRAIN_COLS = {"job_description", "resume_text", "label"}


def pair_to_text(job_text: str, resume_text: str) -> str:
    clean_job = clean_text(job_text)
    clean_resume = clean_text(resume_text)

    job_skills = extract_skills(clean_job)
    resume_skills = extract_skills(clean_resume)
    overlap = sorted(job_skills & resume_skills)

    overlap_text = " ".join(overlap) if overlap else "no_shared_skill"
    return f"job {clean_job} [SEP] resume {clean_resume} [SEP] shared {overlap_text}"


def _validate_training_frame(df: pd.DataFrame) -> None:
    missing = REQUIRED_TRAIN_COLS - set(df.columns)
    if missing:
        raise ValueError(f"training file is missing required columns: {sorted(missing)}")


def train_and_save_model(
    train_csv_path: str | Path,
    artifacts_path: str | Path,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    df = pd.read_csv(train_csv_path)
    _validate_training_frame(df)

    df = df.dropna(subset=["job_description", "resume_text", "label"]).copy()
    df["label"] = df["label"].astype(int)

    X_text = [pair_to_text(j, r) for j, r in zip(df["job_description"].astype(str), df["resume_text"].astype(str))]
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    model.fit(X_train_vec, y_train)

    prob = model.predict_proba(X_test_vec)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    artifacts = {
        "vectorizer": vectorizer,
        "model": model,
        "metrics": metrics,
    }

    artifacts_path = Path(artifacts_path)
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, artifacts_path)

    return metrics
