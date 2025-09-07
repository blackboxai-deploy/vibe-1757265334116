"""
Configuration file for Movie Recommendation System
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
LINKS_FILE = DATA_DIR / "links.csv"

# Model files
CONTENT_MODEL_FILE = MODELS_DIR / "content_model.pkl"
COLLABORATIVE_MODEL_FILE = MODELS_DIR / "collaborative_model.pkl"
TFIDF_VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"
SIMILARITY_MATRIX_FILE = MODELS_DIR / "similarity_matrix.pkl"

# API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "your_api_key_here")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Model parameters
CONTENT_BASED_PARAMS = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.8,
    'ngram_range': (1, 2)
}

COLLABORATIVE_PARAMS = {
    'n_factors': 50,
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02
}

# Recommendation settings
DEFAULT_N_RECOMMENDATIONS = 10
MIN_RATINGS_FOR_USER = 5
MIN_RATINGS_FOR_MOVIE = 10

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': '🎬 Movie Recommender',
    'page_icon': '🎬',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}