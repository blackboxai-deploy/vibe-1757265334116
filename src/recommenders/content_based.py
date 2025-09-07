"""
Content-based recommender for Movie Recommendation System
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from .base import BaseRecommender
import config
from src.utils import parse_genres

class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using movie features"""
    
    def __init__(self):
        super().__init__("Content-Based")
        self.tfidf_vectorizer = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.genre_similarity_matrix = None
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
           use_genres: bool = True, use_titles: bool = True, 
           **kwargs) -> 'ContentBasedRecommender':
        """Fit content-based model using movie features"""
        self.ratings_df = ratings_df
        self.movies_df = movies_df.copy()
        
        # Prepare content features
        content_features = self._prepare_content_features(
            movies_df, use_genres, use_titles
        )
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.CONTENT_BASED_PARAMS.get('max_features', 5000),
            min_df=config.CONTENT_BASED_PARAMS.get('min_df', 2),
            max_df=config.CONTENT_BASED_PARAMS.get('max_df', 0.8),
            ngram_range=config.CONTENT_BASED_PARAMS.get('ngram_range', (1, 2)),
            stop_words='english'
        )
        
        self.feature_matrix = self.tfidf_vectorizer.fit_transform(content_features)
        
        # Calculate similarity matrices
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        if use_genres:
            self.genre_similarity_matrix = self._calculate_genre_similarity(movies_df)
        
        self.is_fitted = True
        return self
    
    def _prepare_content_features(self, movies_df: pd.DataFrame, 
                                use_genres: bool, use_titles: bool) -> List[str]:
        """Prepare content features for TF-IDF"""
        content_features = []
        
        for _, movie in movies_df.iterrows():
            features = []
            
            # Add genres
            if use_genres and pd.notna(movie['genres']):
                genres = parse_genres(movie['genres'])
                # Repeat genres to give them more weight
                features.extend(genres * 3)
            
            # Add title (cleaned)
            if use_titles and pd.notna(movie['title']):
                # Extract title without year
                title = movie['title']
                if '(' in title:
                    title = title[:title.rfind('(')].strip()
                
                # Split title into words and add them
                title_words = title.lower().split()
                features.extend(title_words)
            
            content_features.append(' '.join(features))
        
        return content_features
    
    def _calculate_genre_similarity(self, movies_df: pd.DataFrame) -> np.ndarray:
        """Calculate genre-based similarity matrix using Jaccard similarity"""
        n_movies = len(movies_df)
        genre_sim_matrix = np.zeros((n_movies, n_movies))
        
        movie_genres = []
        for genres_str in movies_df['genres']:
            movie_genres.append(set(parse_genres(genres_str)))
        
        for i in range(n_movies):
            for j in range(n_movies):
                if i == j:
                    genre_sim_matrix[i][j] = 1.0
                else:
                    # Jaccard similarity
                    intersection = len(movie_genres[i].intersection(movie_genres[j]))
                    union = len(movie_genres[i].union(movie_genres[j]))
                    
                    if union > 0:
                        genre_sim_matrix[i][j] = intersection / union
                    else:
                        genre_sim_matrix[i][j] = 0.0
        
        return genre_sim_matrix
    
    def recommend(self, user_id: int = None, movie_id: int = None, 
                 n_recommendations: int = 10, 
                 combine_methods: bool = True,
                 **kwargs) -> List[Dict[str, Any]]:
        """Generate content-based recommendations"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if movie_id is not None:
            return self._recommend_similar_movies(movie_id, n_recommendations)
        elif user_id is not None:
            return self._recommend_for_user(user_id, n_recommendations, combine_methods)
        else:
            # Return popular movies as fallback
            return self._recommend_popular_content(n_recommendations)
    
    def _recommend_similar_movies(self, movie_id: int, 
                                n_recommendations: int) -> List[Dict[str, Any]]:
        """Recommend movies similar to a given movie"""
        # Find movie index
        movie_indices = self.movies_df[self.movies_df['movieId'] == movie_id].index
        
        if movie_indices.empty:
            return []
        
        movie_idx = movie_indices[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Combine with genre similarity if available
        if self.genre_similarity_matrix is not None:
            genre_scores = list(enumerate(self.genre_similarity_matrix[movie_idx]))
            
            # Weighted combination (70% content, 30% genre)
            combined_scores = []
            for i, (content_score, genre_score) in enumerate(zip(sim_scores, genre_scores)):
                combined_score = 0.7 * content_score[1] + 0.3 * genre_score[1]
                combined_scores.append((i, combined_score))
            
            sim_scores = combined_scores
        
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar movies (excluding the input movie itself)
        recommendations = []
        for i, score in sim_scores[1:n_recommendations+1]:
            movie_row = self.movies_df.iloc[i]
            recommendations.append({
                'movieId': movie_row['movieId'],
                'score': score,
                'method': 'content_similarity',
                'explanation': f'Similar content to the selected movie'
            })
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations
    
    def _recommend_for_user(self, user_id: int, n_recommendations: int,
                          combine_methods: bool = True) -> List[Dict[str, Any]]:
        """Recommend movies for a user based on their rating history"""
        user_ratings = self.get_user_ratings(user_id)
        
        if user_ratings.empty:
            return self._recommend_popular_content(n_recommendations)
        
        # Get user's highly rated movies (rating >= 4)
        liked_movies = user_ratings[user_ratings['rating'] >= 4]
        
        if liked_movies.empty:
            # Use all rated movies if no high ratings
            liked_movies = user_ratings
        
        # Calculate user profile based on liked movies
        user_profile_scores = self._calculate_user_profile(liked_movies)
        
        # Combine with content similarity
        if combine_methods:
            content_scores = self._calculate_content_based_scores(liked_movies)
            
            # Weighted combination
            final_scores = {}
            all_movies = set(list(user_profile_scores.keys()) + list(content_scores.keys()))
            
            for movie_id in all_movies:
                profile_score = user_profile_scores.get(movie_id, 0)
                content_score = content_scores.get(movie_id, 0)
                final_scores[movie_id] = 0.6 * profile_score + 0.4 * content_score
        else:
            final_scores = user_profile_scores
        
        # Sort and get top recommendations
        sorted_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_id, score in sorted_movies[:n_recommendations*2]:  # Get extra for filtering
            if movie_id not in user_ratings['movieId'].values:  # Exclude already rated
                recommendations.append({
                    'movieId': movie_id,
                    'score': score,
                    'method': 'user_profile',
                    'explanation': 'Based on your movie preferences'
                })
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations[:n_recommendations]
    
    def _calculate_user_profile(self, user_ratings: pd.DataFrame) -> Dict[int, float]:
        """Calculate user profile based on genre preferences"""
        user_genre_scores = {}
        
        # Calculate weighted genre preferences
        for _, rating in user_ratings.iterrows():
            movie_info = self.get_movie_info(rating['movieId'])
            genres = parse_genres(movie_info.get('genres', ''))
            
            rating_weight = (rating['rating'] - 2.5) / 2.5  # Normalize to [-1, 1]
            
            for genre in genres:
                if genre not in user_genre_scores:
                    user_genre_scores[genre] = 0
                user_genre_scores[genre] += rating_weight
        
        # Calculate scores for all movies based on genre preferences
        movie_scores = {}
        
        for _, movie in self.movies_df.iterrows():
            movie_genres = parse_genres(movie['genres'])
            score = 0
            
            for genre in movie_genres:
                if genre in user_genre_scores:
                    score += user_genre_scores[genre]
            
            if movie_genres:  # Normalize by number of genres
                score /= len(movie_genres)
            
            movie_scores[movie['movieId']] = max(0, score)  # Ensure non-negative
        
        return movie_scores
    
    def _calculate_content_based_scores(self, user_ratings: pd.DataFrame) -> Dict[int, float]:
        """Calculate content-based scores using similarity matrix"""
        movie_scores = {}
        
        for _, rating in user_ratings.iterrows():
            # Find movie index
            movie_indices = self.movies_df[self.movies_df['movieId'] == rating['movieId']].index
            
            if movie_indices.empty:
                continue
            
            movie_idx = movie_indices[0]
            rating_weight = (rating['rating'] - 2.5) / 2.5  # Normalize rating
            
            # Add similarity scores weighted by rating
            sim_scores = self.similarity_matrix[movie_idx]
            
            for i, sim_score in enumerate(sim_scores):
                other_movie_id = self.movies_df.iloc[i]['movieId']
                
                if other_movie_id not in movie_scores:
                    movie_scores[other_movie_id] = 0
                
                movie_scores[other_movie_id] += rating_weight * sim_score
        
        # Normalize scores
        if movie_scores:
            max_score = max(movie_scores.values())
            if max_score > 0:
                movie_scores = {k: v/max_score for k, v in movie_scores.items()}
        
        return movie_scores
    
    def _recommend_popular_content(self, n_recommendations: int) -> List[Dict[str, Any]]:
        """Fallback: recommend popular movies"""
        if self.ratings_df is None:
            return []
        
        # Calculate movie popularity
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        movie_stats.columns = ['count', 'mean']
        
        # Filter movies with at least 5 ratings
        popular_movies = movie_stats[movie_stats['count'] >= 5]
        
        if popular_movies.empty:
            popular_movies = movie_stats
        
        # Sort by average rating
        top_movies = popular_movies.sort_values('mean', ascending=False)
        
        recommendations = []
        for movie_id, stats in top_movies.head(n_recommendations).iterrows():
            recommendations.append({
                'movieId': movie_id,
                'score': stats['mean'],
                'method': 'popular_content',
                'explanation': 'Popular highly-rated movie'
            })
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations
    
    def get_feature_importance(self, movie_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most important features for a movie"""
        if not self.is_fitted or self.tfidf_vectorizer is None:
            return []
        
        # Find movie index
        movie_indices = self.movies_df[self.movies_df['movieId'] == movie_id].index
        
        if movie_indices.empty:
            return []
        
        movie_idx = movie_indices[0]
        
        # Get feature vector for the movie
        feature_vector = self.feature_matrix[movie_idx].toarray()[0]
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get top features
        feature_scores = [(feature_names[i], score) for i, score in enumerate(feature_vector)]
        feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        
        return feature_scores[:top_n]