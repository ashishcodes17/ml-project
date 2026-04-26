from __future__ import annotations

import re
from typing import Set


SKILL_KEYWORDS = {
    "python",
    "sql",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "machine learning",
    "deep learning",
    "nlp",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "git",
    "power bi",
    "tableau",
    "excel",
    "data analysis",
}


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_skills(text: str) -> Set[str]:
    cleaned = clean_text(text)
    found = set()
    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned):
            found.add(skill)
    return found
