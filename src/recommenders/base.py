"""
Base recommender class for Movie Recommendation System
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

class BaseRecommender(ABC):
    """Abstract base class for all recommendation algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.movies_df = None
        self.ratings_df = None
    
    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, **kwargs) -> 'BaseRecommender':
        """Fit the recommender model to the data"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int = None, movie_id: int = None, 
                 n_recommendations: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Generate recommendations"""
        pass
    
    def get_movie_info(self, movie_id: int) -> Dict[str, Any]:
        """Get movie information by ID"""
        if self.movies_df is None:
            return {}
        
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_row.empty:
            return {}
        
        movie = movie_row.iloc[0]
        return {
            'movieId': movie['movieId'],
            'title': movie['title'],
            'genres': movie['genres']
        }
    
    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """Get all ratings for a specific user"""
        if self.ratings_df is None:
            return pd.DataFrame()
        
        return self.ratings_df[self.ratings_df['userId'] == user_id]
    
    def calculate_movie_stats(self, movie_id: int) -> Dict[str, float]:
        """Calculate statistics for a movie"""
        if self.ratings_df is None:
            return {}
        
        movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating']
        
        if movie_ratings.empty:
            return {
                'avg_rating': 0.0,
                'rating_count': 0,
                'rating_std': 0.0
            }
        
        return {
            'avg_rating': movie_ratings.mean(),
            'rating_count': len(movie_ratings),
            'rating_std': movie_ratings.std()
        }
    
    def filter_recommendations(self, recommendations: List[Dict[str, Any]], 
                             user_id: int = None,
                             exclude_rated: bool = True) -> List[Dict[str, Any]]:
        """Filter recommendations based on criteria"""
        if not exclude_rated or user_id is None or self.ratings_df is None:
            return recommendations
        
        # Get movies already rated by user
        user_ratings = self.get_user_ratings(user_id)
        rated_movie_ids = set(user_ratings['movieId'].values)
        
        # Filter out already rated movies
        filtered_recs = [
            rec for rec in recommendations 
            if rec['movieId'] not in rated_movie_ids
        ]
        
        return filtered_recs
    
    def add_movie_details(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add detailed movie information to recommendations"""
        detailed_recs = []
        
        for rec in recommendations:
            movie_info = self.get_movie_info(rec['movieId'])
            movie_stats = self.calculate_movie_stats(rec['movieId'])
            
            detailed_rec = {**rec, **movie_info, **movie_stats}
            detailed_recs.append(detailed_rec)
        
        return detailed_recs
    
    def validate_input(self, user_id: int = None, movie_id: int = None) -> bool:
        """Validate input parameters"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} model is not fitted yet")
        
        if user_id is not None:
            if self.ratings_df is None or user_id not in self.ratings_df['userId'].values:
                return False
        
        if movie_id is not None:
            if self.movies_df is None or movie_id not in self.movies_df['movieId'].values:
                return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'n_movies': len(self.movies_df) if self.movies_df is not None else 0,
            'n_ratings': len(self.ratings_df) if self.ratings_df is not None else 0,
            'n_users': len(self.ratings_df['userId'].unique()) if self.ratings_df is not None else 0
        }

class PopularityRecommender(BaseRecommender):
    """Simple popularity-based recommender"""
    
    def __init__(self):
        super().__init__("Popularity-Based")
        self.popularity_scores = None
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
           min_ratings: int = 10) -> 'PopularityRecommender':
        """Fit popularity model"""
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Calculate popularity scores
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        movie_stats.columns = ['count', 'mean']
        
        # Filter movies with minimum ratings
        popular_movies = movie_stats[movie_stats['count'] >= min_ratings]
        
        if popular_movies.empty:
            # Fallback: use all movies if none meet minimum criteria
            popular_movies = movie_stats
        
        # Calculate weighted rating (IMDB formula)
        C = popular_movies['mean'].mean()  # Average rating
        m = min_ratings  # Minimum ratings required
        
        def weighted_rating(row):
            v = row['count']
            R = row['mean']
            return (v / (v + m) * R) + (m / (v + m) * C)
        
        popular_movies['popularity_score'] = popular_movies.apply(
            weighted_rating, axis=1
        )
        
        self.popularity_scores = popular_movies['popularity_score'].sort_values(
            ascending=False
        )
        
        self.is_fitted = True
        return self
    
    def recommend(self, user_id: int = None, movie_id: int = None, 
                 n_recommendations: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Generate popularity-based recommendations"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Get top popular movies
        top_movies = self.popularity_scores.head(n_recommendations * 2)  # Get extra for filtering
        
        recommendations = []
        for movie_id_pop, score in top_movies.items():
            recommendations.append({
                'movieId': movie_id_pop,
                'score': score,
                'method': 'popularity',
                'explanation': 'Highly rated and popular movie'
            })
        
        # Filter out movies already rated by user (if user_id provided)
        recommendations = self.filter_recommendations(recommendations, user_id)
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations[:n_recommendations]

class RandomRecommender(BaseRecommender):
    """Random recommender for baseline comparison"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Random")
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
           **kwargs) -> 'RandomRecommender':
        """Fit random model (no actual fitting needed)"""
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.is_fitted = True
        return self
    
    def recommend(self, user_id: int = None, movie_id: int = None, 
                 n_recommendations: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Generate random recommendations"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Get random movies
        available_movies = self.movies_df['movieId'].values
        random_movies = np.random.choice(
            available_movies, 
            size=min(n_recommendations * 2, len(available_movies)), 
            replace=False
        )
        
        recommendations = []
        for movie_id_rand in random_movies:
            recommendations.append({
                'movieId': int(movie_id_rand),
                'score': np.random.random(),
                'method': 'random',
                'explanation': 'Randomly selected movie'
            })
        
        # Filter out movies already rated by user
        recommendations = self.filter_recommendations(recommendations, user_id)
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations[:n_recommendations]