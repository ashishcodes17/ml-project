from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import pandas as pd

from src.preprocess import extract_skills
from src.train_model import pair_to_text


@dataclass
class MatchResult:
    resume_id: str
    candidate_name: str
    model_score: float
    skill_overlap: float
    final_score: float
    matched_skills: list[str]
    missing_skills: list[str]


class ResumeJobMatcher:
    def __init__(self, artifacts_path: str | Path = "artifacts/matcher.joblib", w_model: float = 0.85, w_skill: float = 0.15) -> None:
        if abs((w_model + w_skill) - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.0")

        artifacts_path = Path(artifacts_path)
        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"Trained model artifacts not found at '{artifacts_path}'. Train the model first."
            )

        artifacts = joblib.load(artifacts_path)
        self.vectorizer = artifacts["vectorizer"]
        self.model = artifacts["model"]
        self.w_model = w_model
        self.w_skill = w_skill

    def rank_resumes(
        self,
        job_row: pd.Series,
        resumes_df: pd.DataFrame,
        top_k: int = 5,
    ) -> List[MatchResult]:
        job_text = str(job_row["description"])
        resume_texts = resumes_df["resume_text"].astype(str).tolist()

        pair_texts = [pair_to_text(job_text, resume_text) for resume_text in resume_texts]
        X_pairs = self.vectorizer.transform(pair_texts)
        model_scores = self.model.predict_proba(X_pairs)[:, 1]

        job_skills = extract_skills(job_text)
        results: list[MatchResult] = []

        for i, (_, row) in enumerate(resumes_df.iterrows()):
            resume_skills = extract_skills(str(row["resume_text"]))
            overlap = len(job_skills & resume_skills) / max(len(job_skills), 1)
            final = self.w_model * float(model_scores[i]) + self.w_skill * float(overlap)

            matched = sorted(job_skills & resume_skills)
            missing = sorted(job_skills - resume_skills)

            results.append(
                MatchResult(
                    resume_id=str(row["resume_id"]),
                    candidate_name=str(row["candidate_name"]),
                    model_score=float(model_scores[i]),
                    skill_overlap=float(overlap),
                    final_score=float(final),
                    matched_skills=matched,
                    missing_skills=missing,
                )
            )

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]
