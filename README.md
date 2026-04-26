# CineMatch AI

<p align="center">
  <b>Production-style Movie Recommendation System</b><br/>
  <i>Trainable ML pipeline + Netflix-inspired UI + 100k+ India-focused movie data</i>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" />
  <img alt="Status" src="https://img.shields.io/badge/Status-Working-success" />
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-100k%2B%20Indian%20Movies-informational" />
</p>

## Table of Contents

1. Overview
2. Why this project is strong for ML coursework
3. Live project highlights
4. Architecture
5. Repository structure
6. Quick start (Windows)
7. Dataset collection (100k+ Indian movies)
8. Model training
9. Model testing and evaluation
10. Running the app
11. Key results from your run
12. Technical deep dive
13. Limitations and risk notes
14. Viva-ready explanation points
15. Future improvements

## Overview

CineMatch AI is a content-based movie recommendation system designed for an ML-only academic project.
It trains a recommendation model from scratch using classical machine learning and NLP techniques,
then serves recommendations through a polished Netflix-style interface.

This project demonstrates the complete ML lifecycle:

- Data acquisition from online public datasets
- Data preprocessing and feature engineering
- Model training and persistence
- Offline evaluation
- Inference in a user-facing application

## Why this project is strong for ML coursework

- End-to-end ML pipeline, not a toy notebook
- No external recommendation API dependency
- Real large-scale dataset handling (100k+ movies)
- Practical engineering constraints solved (memory-safe preprocessing/training)
- Explainable design choices (TF-IDF + SVD + cosine KNN)

## Live project highlights

- Netflix-inspired UI experience in Streamlit
- Fast recommendation retrieval via nearest-neighbor search
- Configurable training knobs in app sidebar
- Automated evaluation script for objective quality checks

## Architecture

```text
IMDb Public Data (online)
	|
	v
prepare_indian_movies_imdb.py
	|
	v
movies_india_100k.csv
	|
	v
movie_data.py -> feature text building
	|
	v
movie_train.py
  - TF-IDF
  - TruncatedSVD
  - NearestNeighbors (cosine)
	|
	v
artifacts/movie_recommender.joblib
	|
	v
movie_recommender.py -> app.py (Streamlit UI)
```

## Repository structure

```text
ml-proj/
|- app.py
|- requirements.txt
|- README.md
|- data/
|  |- movies_sample.csv
|  |- movies_india_100k.csv              # generated
|  |- raw/imdb/                           # downloaded IMDb dumps
|- artifacts/
|  |- movie_recommender.joblib            # generated
|- src/
   |- movie_data.py
   |- movie_train.py
   |- movie_recommender.py
   |- movie_eval.py
   |- prepare_indian_movies_imdb.py
```

## Quick start (Windows)

```powershell
cd /d F:PROJPATH\ml-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset collection (100k+ Indian movies)

This step downloads IMDb public files and creates an India-focused movie dataset.

```powershell
python -c "from src.prepare_indian_movies_imdb import prepare_indian_movies_dataset; prepare_indian_movies_dataset(raw_dir='data/raw/imdb', output_csv='data/movies_india_100k.csv', min_votes=1, max_movies=200000)"
```

Verify row count:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/movies_india_100k.csv'); print('rows=',len(df))"
```

Note:

- The collector is implemented with chunk-based reading for memory safety.
- Filtering logic uses region/language signals to focus on Indian titles.

## Model training

Train on the generated dataset:

```powershell
python -c "from src.movie_train import train_movie_model; print(train_movie_model(dataset_path='data/movies_india_100k.csv', artifacts_path='artifacts/movie_recommender.joblib', max_features=30000, n_components=256))"
```

Recommended training parameters for 100k-scale data:

| Parameter | Value | Reason |
|---|---:|---|
| max_features | 30000 | Rich enough text vocabulary without excessive memory |
| n_components | 256 | Good compression/performance tradeoff |
| metric | cosine | Suitable for text embedding similarity |

Output artifact:

- `artifacts/movie_recommender.joblib`

## Model testing and evaluation

Run offline evaluation:

```powershell
python -m src.movie_eval
```

Current metrics produced:

- `hit_rate_at_k`: fraction of sampled seed movies where at least one recommendation shares genre
- `avg_genre_jaccard`: average genre overlap between seed and recommended movies

## Running the app

```powershell
streamlit run app.py
```

App flow:

1. Train/retrain model from sidebar (or train via CLI first)
2. Select a seed movie
3. Click "Get Recommendations"
4. Review ranked movie cards with similarity score and metadata

## Key results from your run

From your actual execution:

- Collected rows: `100255`
- Usable rows after cleaning/deduplication for training: `89560`
- Training summary:
  - `vocab_size`: `30000`
  - `embedding_dim`: `256`
  - `used_svd`: `True`
- Evaluation summary:
  - `hit_rate_at_k`: `0.9967`
  - `avg_genre_jaccard`: `0.9891`

Interpretation:

- The system strongly preserves genre consistency.
- Report these as proxy quality metrics for content-based recommendation.

## Technical deep dive

### Feature engineering

For each movie, the model combines:

- title
- overview
- genres
- keywords
- cast
- director

This creates a single content representation per movie.

### Modeling strategy

1. TF-IDF converts content text to sparse vectors.
2. TruncatedSVD compresses sparse vectors into dense latent embeddings.
3. L2 normalization standardizes vector magnitude.
4. NearestNeighbors with cosine distance retrieves the most similar movies.

### Why this choice

- Works well without user interaction logs
- Interpretable and easy to explain in viva
- Scales to large text corpora with controlled memory

## Limitations and risk notes

- Current evaluation is genre-driven and optimistic for semantic quality.
- Content-based systems can over-specialize and reduce novelty.
- No collaborative filtering yet (no user behavior modeling).
- Metadata quality from source datasets can affect recommendation quality.





1. Problem: recommend similar Indian movies at scale.
2. Data: built from IMDb online dumps; filtered and cleaned.
3. Model: TF-IDF + SVD + cosine KNN chosen for ML-only syllabus alignment.
4. Engineering: chunked ingestion and memory-safe stats for 100k+ records.
5. Outcome: successful training on ~90k usable movies with strong offline consistency.
6. Product: deployed as an interactive recommendation app with polished UI.

## Future improvements

1. Add stricter evaluation metrics beyond genre overlap (human-labeled relevance set).
2. Integrate plot summaries from richer sources for better semantic depth.
3. Add hybrid recommender (content + collaborative filtering).
4. Add multilingual query support for Hindi, Tamil, Telugu titles.
5. Add caching/indexing for lower-latency production inference.

---
 