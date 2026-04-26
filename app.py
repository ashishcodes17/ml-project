from __future__ import annotations

from pathlib import Path
from typing import Iterable

import streamlit as st

from src.movie_recommender import MovieRecommender
from src.movie_train import train_movie_model


st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")

ARTIFACTS_PATH = Path("artifacts/movie_recommender.joblib")
DATASET_PATH = Path("data/movies_sample.csv")


if "sidebar_visible" not in st.session_state:
    st.session_state["sidebar_visible"] = True


def render_styles(sidebar_visible: bool) -> None:
    sidebar_css = ""
    if not sidebar_visible:
        sidebar_css = """
            [data-testid="stSidebar"] {
                display: none;
            }
            [data-testid="collapsedControl"] {
                display: none;
            }
        """

    base_css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Manrope:wght@400;600;700;800&display=swap');

            .stApp {
                background: radial-gradient(circle at 20% 10%, #3f0d12 0%, #1a0a0d 40%, #0b0b0f 100%);
                color: #f8f5f2;
                font-family: 'Manrope', sans-serif;
            }

            .hero {
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 22px;
                padding: 2.2rem;
                background: linear-gradient(135deg, rgba(214, 18, 18, 0.25), rgba(15, 15, 20, 0.75));
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
                margin-bottom: 1.2rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 3.2rem;
                font-family: 'Bebas Neue', sans-serif;
                letter-spacing: 2px;
                line-height: 1.0;
            }

            .hero p {
                margin-top: 0.6rem;
                max-width: 860px;
                opacity: 0.92;
            }

            .metric-chip {
                display: inline-block;
                padding: 0.4rem 0.7rem;
                margin-right: 0.4rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.18);
                font-size: 0.84rem;
            }

            .movie-card {
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 16px;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
                padding: 1rem;
                min-height: 260px;
                margin-bottom: 1rem;
            }

            .movie-title {
                font-weight: 800;
                font-size: 1.05rem;
                margin-bottom: 0.25rem;
            }

            .movie-meta {
                font-size: 0.8rem;
                opacity: 0.85;
                margin-bottom: 0.6rem;
            }

            .movie-overview {
                font-size: 0.88rem;
                opacity: 0.92;
                line-height: 1.4;
            }
        </style>
    """

    st.markdown(
        base_css + (f"<style>{sidebar_css}</style>" if sidebar_css else ""),
        unsafe_allow_html=True,
    )


def safe_text(text: str, max_len: int = 160) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= max_len else f"{clean[:max_len - 3]}..."


def format_genres(genres: str) -> str:
    return ", ".join(part.strip() for part in str(genres).split("|") if part.strip())


def render_card(title: str, year: int, rating: float, score: float, genres: str, overview: str) -> None:
    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-title">{title}</div>
            <div class="movie-meta">{year} • ⭐ {rating:.1f} • 🎯 {(score * 100):.1f}% match</div>
            <div class="metric-chip">{genres}</div>
            <p class="movie-overview">{safe_text(overview)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def split_chunks(items: Iterable, size: int) -> list[list]:
    items = list(items)
    return [items[i : i + size] for i in range(0, len(items), size)]


render_styles(st.session_state["sidebar_visible"])

if not st.session_state["sidebar_visible"]:
    if st.button("Open Sidebar", use_container_width=False):
        st.session_state["sidebar_visible"] = True
        st.rerun()

st.markdown(
    """
    <section class="hero">
        <h1>CINEMATCH AI</h1>
        <p>
            Build and train your own large-scale movie recommender on Indian cinema metadata,
            then explore high-quality suggestions in a cinematic Netflix-style experience.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

top_k = 12
if st.session_state["sidebar_visible"]:
    with st.sidebar:
        st.header("Model Control")
        st.caption("Train and run your own recommender model")

        if st.button("Close Sidebar", use_container_width=True):
            st.session_state["sidebar_visible"] = False
            st.rerun()

        max_features = st.slider("TF-IDF max features", 2000, 25000, 12000, 500)
        svd_components = st.slider("SVD components", 20, 250, 100, 5)

        if st.button("Train / Retrain Model"):
            with st.spinner("Training model..."):
                try:
                    metrics = train_movie_model(
                        dataset_path=DATASET_PATH,
                        artifacts_path=ARTIFACTS_PATH,
                        max_features=max_features,
                        n_components=svd_components,
                    )
                    st.success(
                        f"Model trained on {metrics['num_movies']} movies. "
                        f"Embedding dim: {metrics['embedding_dim']}"
                    )
                    st.caption(
                        f"avg similarity: {metrics['avg_pair_similarity']:.3f}, "
                        f"vocab: {metrics['vocab_size']}"
                    )
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

        top_k = st.slider("Recommendations", min_value=4, max_value=20, value=12)

if not ARTIFACTS_PATH.exists():
    st.warning("No trained model found. Train the model first from the sidebar.")
    st.stop()

try:
    recommender = MovieRecommender(ARTIFACTS_PATH)
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

titles = recommender.titles()
selected_title = st.selectbox("Pick a movie you like", options=titles, index=0)

if st.button("Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("Generating recommendations..."):
        recommendations = recommender.recommend_by_title(selected_title, top_k=top_k)

    st.session_state["last_recommendations"] = recommendations
    st.session_state["last_title"] = selected_title

if "last_recommendations" in st.session_state:
    recommendations = st.session_state["last_recommendations"]
    title_for_results = st.session_state.get("last_title", selected_title)
    st.markdown(f"### Because you watched: {title_for_results}")
    for chunk in split_chunks(recommendations, 3):
        cols = st.columns(3)
        for col, rec in zip(cols, chunk):
            with col:
                render_card(
                    title=rec.title,
                    year=rec.year,
                    rating=rec.rating,
                    score=rec.similarity,
                    genres=format_genres(rec.genres),
                    overview=rec.overview,
                )
else:
    st.info("Select a movie and click Get Recommendations.")
