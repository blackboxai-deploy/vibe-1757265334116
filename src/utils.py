"""
Utility functions for Movie Recommendation System
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import re
from datetime import datetime

def extract_year_from_title(title: str) -> Tuple[str, int]:
    """Extract year from movie title"""
    # Pattern to match year in parentheses at the end
    pattern = r'^(.*?)\s*\((\d{4})\)$'
    match = re.match(pattern, title)
    
    if match:
        clean_title = match.group(1).strip()
        year = int(match.group(2))
        return clean_title, year
    else:
        return title, None

def parse_genres(genres_str: str) -> List[str]:
    """Parse genres string into list"""
    if pd.isna(genres_str) or genres_str == '':
        return []
    return [genre.strip() for genre in genres_str.split('|')]

def get_unique_genres(movies_df: pd.DataFrame) -> List[str]:
    """Get all unique genres from movies dataset"""
    all_genres = set()
    for genres_str in movies_df['genres']:
        genres = parse_genres(genres_str)
        all_genres.update(genres)
    
    return sorted(list(all_genres))

def calculate_movie_popularity(ratings_df: pd.DataFrame, 
                             min_ratings: int = 10) -> pd.Series:
    """Calculate movie popularity based on rating count and average"""
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    })
    
    movie_stats.columns = ['count', 'mean']
    
    # Filter movies with minimum number of ratings
    popular_movies = movie_stats[movie_stats['count'] >= min_ratings]
    
    # Calculate popularity score (weighted rating)
    C = popular_movies['mean'].mean()  # Average rating across all movies
    m = min_ratings  # Minimum ratings required
    
    def weighted_rating(row):
        v = row['count']
        R = row['mean']
        return (v / (v + m) * R) + (m / (v + m) * C)
    
    popular_movies['popularity_score'] = popular_movies.apply(
        weighted_rating, axis=1
    )
    
    return popular_movies['popularity_score'].sort_values(ascending=False)

def create_user_profile(user_ratings: pd.DataFrame, 
                       movies_df: pd.DataFrame) -> Dict[str, Any]:
    """Create user profile based on rating history"""
    if user_ratings.empty:
        return {
            'favorite_genres': [],
            'avg_rating': 0,
            'total_ratings': 0,
            'rating_distribution': {},
            'preferred_years': []
        }
    
    # Merge with movie data to get genres
    user_movies = user_ratings.merge(movies_df, on='movieId')
    
    # Calculate favorite genres
    genre_ratings = {}
    for _, row in user_movies.iterrows():
        genres = parse_genres(row['genres'])
        for genre in genres:
            if genre not in genre_ratings:
                genre_ratings[genre] = []
            genre_ratings[genre].append(row['rating'])
    
    # Average rating per genre
    genre_avg_ratings = {
        genre: np.mean(ratings) 
        for genre, ratings in genre_ratings.items()
    }
    
    favorite_genres = sorted(
        genre_avg_ratings.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Extract years and find preferred decades
    years = []
    for title in user_movies['title']:
        _, year = extract_year_from_title(title)
        if year:
            years.append(year)
    
    # Rating distribution
    rating_dist = user_ratings['rating'].value_counts().to_dict()
    
    return {
        'favorite_genres': [genre for genre, _ in favorite_genres[:5]],
        'avg_rating': user_ratings['rating'].mean(),
        'total_ratings': len(user_ratings),
        'rating_distribution': rating_dist,
        'preferred_years': sorted(years) if years else [],
        'genre_preferences': dict(favorite_genres)
    }

def calculate_movie_similarity(movie1_genres: List[str], 
                             movie2_genres: List[str]) -> float:
    """Calculate Jaccard similarity between two movies based on genres"""
    set1 = set(movie1_genres)
    set2 = set(movie2_genres)
    
    if not set1 and not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def format_runtime(minutes: int) -> str:
    """Format runtime in minutes to hours and minutes"""
    if not minutes or pd.isna(minutes):
        return "Runtime not available"
    
    hours = minutes // 60
    mins = minutes % 60
    
    if hours > 0:
        return f"{hours}h {mins}m"
    else:
        return f"{mins}m"

def get_recommendation_explanation(recommended_movie: Dict, 
                                user_profile: Dict, 
                                method: str = "content") -> str:
    """Generate explanation for why a movie was recommended"""
    explanations = []
    
    if method == "content":
        # Content-based explanations
        movie_genres = parse_genres(recommended_movie.get('genres', ''))
        user_fav_genres = user_profile.get('favorite_genres', [])
        
        common_genres = set(movie_genres).intersection(set(user_fav_genres))
        if common_genres:
            explanations.append(
                f"You like {', '.join(common_genres)} movies"
            )
    
    elif method == "collaborative":
        # Collaborative filtering explanations
        explanations.append(
            "Users with similar taste also enjoyed this movie"
        )
    
    elif method == "popularity":
        # Popularity-based explanations
        explanations.append(
            "This is a highly rated and popular movie"
        )
    
    else:
        # Hybrid explanations
        explanations.append(
            "Based on your preferences and similar users' choices"
        )
    
    return "; ".join(explanations) if explanations else "Recommended for you"

def filter_movies_by_criteria(movies_df: pd.DataFrame, 
                            ratings_df: pd.DataFrame,
                            genre: str = None,
                            min_year: int = None,
                            max_year: int = None,
                            min_rating: float = None,
                            min_rating_count: int = None) -> pd.DataFrame:
    """Filter movies based on various criteria"""
    filtered_movies = movies_df.copy()
    
    # Filter by genre
    if genre and genre != "All":
        filtered_movies = filtered_movies[
            filtered_movies['genres'].str.contains(genre, na=False)
        ]
    
    # Filter by year
    if min_year or max_year:
        years = filtered_movies['title'].apply(
            lambda x: extract_year_from_title(x)[1]
        )
        
        if min_year:
            filtered_movies = filtered_movies[years >= min_year]
        if max_year:
            filtered_movies = filtered_movies[years <= max_year]
    
    # Filter by rating criteria
    if min_rating or min_rating_count:
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        })
        movie_stats.columns = ['avg_rating', 'rating_count']
        
        if min_rating:
            valid_movies = movie_stats[
                movie_stats['avg_rating'] >= min_rating
            ].index
            filtered_movies = filtered_movies[
                filtered_movies['movieId'].isin(valid_movies)
            ]
        
        if min_rating_count:
            valid_movies = movie_stats[
                movie_stats['rating_count'] >= min_rating_count
            ].index
            filtered_movies = filtered_movies[
                filtered_movies['movieId'].isin(valid_movies)
            ]
    
    return filtered_movies

def timestamp_to_datetime(timestamp: int) -> datetime:
    """Convert Unix timestamp to datetime"""
    try:
        return datetime.fromtimestamp(timestamp)
    except (ValueError, TypeError):
        return datetime.now()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def normalize_ratings(ratings: np.ndarray, method: str = "min_max") -> np.ndarray:
    """Normalize ratings array"""
    if method == "min_max":
        min_val = np.min(ratings)
        max_val = np.max(ratings)
        if max_val == min_val:
            return np.zeros_like(ratings)
        return (ratings - min_val) / (max_val - min_val)
    
    elif method == "z_score":
        mean_val = np.mean(ratings)
        std_val = np.std(ratings)
        if std_val == 0:
            return np.zeros_like(ratings)
        return (ratings - mean_val) / std_val
    
    else:
        return ratings

def get_top_n_items(scores: Dict[Any, float], n: int = 10) -> List[Tuple[Any, float]]:
    """Get top N items from a dictionary of scores"""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]