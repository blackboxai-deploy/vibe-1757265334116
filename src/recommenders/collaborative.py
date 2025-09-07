"""
Collaborative filtering recommender for Movie Recommendation System
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from typing import List, Dict, Any, Tuple
from .base import BaseRecommender
import config

try:
    from surprise import Dataset, Reader, SVD, KNNBasic, NMF as SurpriseNMF
    from surprise.model_selection import train_test_split, cross_validate
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("Surprise library not available. Using basic collaborative filtering implementation.")

class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative filtering recommender using user-item interactions"""
    
    def __init__(self, method='memory_based', similarity='cosine', k=20):
        super().__init__("Collaborative Filtering")
        self.method = method  # 'memory_based', 'matrix_factorization', 'surprise_svd'
        self.similarity = similarity  # 'cosine', 'pearson'
        self.k = k  # Number of neighbors for memory-based CF
        
        self.user_item_matrix = None
        self.item_item_similarity = None
        self.user_user_similarity = None
        self.model = None
        self.user_means = None
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
           **kwargs) -> 'CollaborativeFilteringRecommender':
        """Fit collaborative filtering model"""
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Create user-item matrix
        self.user_item_matrix = self._create_user_item_matrix(ratings_df)
        
        # Calculate user means for mean-centered predictions
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        if self.method == 'memory_based':
            self._fit_memory_based()
        elif self.method == 'matrix_factorization':
            self._fit_matrix_factorization()
        elif self.method == 'surprise_svd' and SURPRISE_AVAILABLE:
            self._fit_surprise_svd(ratings_df)
        else:
            # Fallback to memory-based if method not available
            self.method = 'memory_based'
            self._fit_memory_based()
        
        self.is_fitted = True
        return self
    
    def _create_user_item_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item rating matrix"""
        user_item_matrix = ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        return user_item_matrix
    
    def _fit_memory_based(self):
        """Fit memory-based collaborative filtering"""
        # Calculate item-item similarity
        # Use only movies that have been rated
        non_zero_cols = (self.user_item_matrix != 0).any(axis=0)
        active_matrix = self.user_item_matrix.loc[:, non_zero_cols]
        
        if self.similarity == 'cosine':
            # Item-item similarity
            item_similarity = cosine_similarity(active_matrix.T)
            self.item_item_similarity = pd.DataFrame(
                item_similarity,
                index=active_matrix.columns,
                columns=active_matrix.columns
            )
            
            # User-user similarity
            user_similarity = cosine_similarity(active_matrix)
            self.user_user_similarity = pd.DataFrame(
                user_similarity,
                index=active_matrix.index,
                columns=active_matrix.index
            )
        
        # Remove self-similarities
        np.fill_diagonal(self.item_item_similarity.values, 0)
        np.fill_diagonal(self.user_user_similarity.values, 0)
    
    def _fit_matrix_factorization(self):
        """Fit matrix factorization model using NMF"""
        try:
            # Use Non-negative Matrix Factorization
            n_components = min(50, min(self.user_item_matrix.shape) - 1)
            
            self.model = NMF(
                n_components=n_components,
                init='random',
                random_state=42,
                max_iter=200,
                alpha_W=0.01,
                alpha_H=0.01
            )
            
            # Fit the model
            W = self.model.fit_transform(self.user_item_matrix)
            H = self.model.components_
            
            # Store factorized matrices
            self.user_factors = pd.DataFrame(
                W, 
                index=self.user_item_matrix.index
            )
            self.item_factors = pd.DataFrame(
                H.T, 
                index=self.user_item_matrix.columns
            )
            
        except Exception as e:
            print(f"Matrix factorization failed: {e}")
            print("Falling back to memory-based approach")
            self.method = 'memory_based'
            self._fit_memory_based()
    
    def _fit_surprise_svd(self, ratings_df: pd.DataFrame):
        """Fit SVD model using Surprise library"""
        try:
            # Prepare data for Surprise
            reader = Reader(rating_scale=(0.5, 5.0))
            data = Dataset.load_from_df(
                ratings_df[['userId', 'movieId', 'rating']], 
                reader
            )
            
            # Create and train SVD model
            self.model = SVD(
                n_factors=config.COLLABORATIVE_PARAMS.get('n_factors', 50),
                n_epochs=config.COLLABORATIVE_PARAMS.get('n_epochs', 20),
                lr_all=config.COLLABORATIVE_PARAMS.get('lr_all', 0.005),
                reg_all=config.COLLABORATIVE_PARAMS.get('reg_all', 0.02)
            )
            
            # Train on full dataset
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            
        except Exception as e:
            print(f"Surprise SVD fitting failed: {e}")
            print("Falling back to memory-based approach")
            self.method = 'memory_based'
            self._fit_memory_based()
    
    def recommend(self, user_id: int = None, movie_id: int = None, 
                 n_recommendations: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Generate collaborative filtering recommendations"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if user_id is not None:
            return self._recommend_for_user(user_id, n_recommendations)
        elif movie_id is not None:
            return self._recommend_similar_items(movie_id, n_recommendations)
        else:
            # Return popular movies as fallback
            return self._recommend_popular_items(n_recommendations)
    
    def _recommend_for_user(self, user_id: int, 
                          n_recommendations: int) -> List[Dict[str, Any]]:
        """Generate user-based recommendations"""
        if user_id not in self.user_item_matrix.index:
            return self._recommend_popular_items(n_recommendations)
        
        if self.method == 'surprise_svd' and self.model is not None:
            return self._recommend_user_svd(user_id, n_recommendations)
        elif self.method == 'matrix_factorization' and hasattr(self, 'user_factors'):
            return self._recommend_user_mf(user_id, n_recommendations)
        else:
            return self._recommend_user_memory_based(user_id, n_recommendations)
    
    def _recommend_user_memory_based(self, user_id: int, 
                                   n_recommendations: int) -> List[Dict[str, Any]]:
        """User-based memory-based collaborative filtering"""
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        if len(unrated_movies) == 0:
            return []
        
        # Find similar users
        user_similarities = self.user_user_similarity.loc[user_id]
        similar_users = user_similarities.nlargest(self.k + 1)[1:]  # Exclude self
        
        # Predict ratings for unrated movies
        predictions = {}
        
        for movie_id in unrated_movies:
            if movie_id not in self.user_item_matrix.columns:
                continue
            
            # Get ratings from similar users
            similar_user_ratings = []
            similar_user_sims = []
            
            for sim_user_id, similarity in similar_users.items():
                if similarity > 0:  # Only positive similarities
                    sim_user_rating = self.user_item_matrix.loc[sim_user_id, movie_id]
                    if sim_user_rating > 0:  # User has rated this movie
                        similar_user_ratings.append(sim_user_rating)
                        similar_user_sims.append(similarity)
            
            # Calculate weighted average prediction
            if similar_user_ratings:
                weighted_sum = sum(
                    rating * sim 
                    for rating, sim in zip(similar_user_ratings, similar_user_sims)
                )
                similarity_sum = sum(similar_user_sims)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    predictions[movie_id] = predicted_rating
        
        # Sort predictions and get top N
        sorted_predictions = sorted(
            predictions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        recommendations = []
        for movie_id, score in sorted_predictions[:n_recommendations]:
            recommendations.append({
                'movieId': movie_id,
                'score': score,
                'method': 'user_based_cf',
                'explanation': 'Users with similar taste also liked this movie'
            })
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations
    
    def _recommend_user_mf(self, user_id: int, 
                         n_recommendations: int) -> List[Dict[str, Any]]:
        """Matrix factorization based recommendations"""
        if user_id not in self.user_factors.index:
            return self._recommend_popular_items(n_recommendations)
        
        user_vector = self.user_factors.loc[user_id].values
        
        # Calculate predicted ratings for all items
        predicted_ratings = self.item_factors.dot(user_vector)
        
        # Get user's existing ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Filter to unrated movies only
        unrated_predictions = predicted_ratings[
            predicted_ratings.index.isin(unrated_movies)
        ]
        
        # Sort and get top N
        top_predictions = unrated_predictions.nlargest(n_recommendations)
        
        recommendations = []
        for movie_id, score in top_predictions.items():
            recommendations.append({
                'movieId': movie_id,
                'score': score,
                'method': 'matrix_factorization',
                'explanation': 'Based on latent factors learned from user behavior'
            })
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations
    
    def _recommend_user_svd(self, user_id: int, 
                          n_recommendations: int) -> List[Dict[str, Any]]:
        """SVD-based recommendations using Surprise"""
        # Get user's existing ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Generate predictions for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            pred = self.model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_id, score in predictions[:n_recommendations]:
            recommendations.append({
                'movieId': movie_id,
                'score': score,
                'method': 'svd_cf',
                'explanation': 'Predicted using SVD matrix factorization'
            })
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations
    
    def _recommend_similar_items(self, movie_id: int, 
                               n_recommendations: int) -> List[Dict[str, Any]]:
        """Item-based collaborative filtering"""
        if (self.item_item_similarity is None or 
            movie_id not in self.item_item_similarity.index):
            return []
        
        # Get similar movies
        similar_movies = self.item_item_similarity.loc[movie_id]
        top_similar = similar_movies.nlargest(n_recommendations)
        
        recommendations = []
        for similar_movie_id, similarity in top_similar.items():
            if similarity > 0:  # Only positive similarities
                recommendations.append({
                    'movieId': similar_movie_id,
                    'score': similarity,
                    'method': 'item_based_cf',
                    'explanation': 'Users who liked the selected movie also liked this'
                })
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations
    
    def _recommend_popular_items(self, n_recommendations: int) -> List[Dict[str, Any]]:
        """Fallback: recommend popular items"""
        if self.ratings_df is None:
            return []
        
        # Calculate item popularity (rating count and average)
        item_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        item_stats.columns = ['count', 'mean']
        
        # Filter items with minimum ratings
        min_ratings = 5
        popular_items = item_stats[item_stats['count'] >= min_ratings]
        
        if popular_items.empty:
            popular_items = item_stats
        
        # Sort by average rating
        top_items = popular_items.sort_values('mean', ascending=False)
        
        recommendations = []
        for movie_id, stats in top_items.head(n_recommendations).iterrows():
            recommendations.append({
                'movieId': movie_id,
                'score': stats['mean'],
                'method': 'popular_fallback',
                'explanation': 'Popular highly-rated movie'
            })
        
        # Add movie details
        recommendations = self.add_movie_details(recommendations)
        
        return recommendations
    
    def evaluate_model(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """Evaluate collaborative filtering model"""
        if not self.is_fitted:
            return {}
        
        predictions = []
        actuals = []
        
        for _, rating in test_ratings.iterrows():
            user_id = rating['userId']
            movie_id = rating['movieId']
            actual_rating = rating['rating']
            
            # Get prediction
            if self.method == 'surprise_svd' and self.model is not None:
                try:
                    pred = self.model.predict(user_id, movie_id)
                    predicted_rating = pred.est
                except:
                    continue
            else:
                # Use memory-based prediction
                predicted_rating = self._predict_rating(user_id, movie_id)
                if predicted_rating is None:
                    continue
            
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        if not predictions:
            return {'rmse': float('inf'), 'mae': float('inf')}
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions)
        }
    
    def _predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a user-movie pair"""
        if (user_id not in self.user_item_matrix.index or 
            movie_id not in self.user_item_matrix.columns):
            return None
        
        if self.method == 'matrix_factorization' and hasattr(self, 'user_factors'):
            user_vector = self.user_factors.loc[user_id].values
            item_vector = self.item_factors.loc[movie_id].values
            return np.dot(user_vector, item_vector)
        
        elif self.user_user_similarity is not None:
            # User-based prediction
            user_similarities = self.user_user_similarity.loc[user_id]
            similar_users = user_similarities.nlargest(self.k + 1)[1:]
            
            weighted_sum = 0
            similarity_sum = 0
            
            for sim_user_id, similarity in similar_users.items():
                if similarity > 0:
                    sim_user_rating = self.user_item_matrix.loc[sim_user_id, movie_id]
                    if sim_user_rating > 0:
                        weighted_sum += similarity * sim_user_rating
                        similarity_sum += similarity
            
            if similarity_sum > 0:
                return weighted_sum / similarity_sum
        
        # Fallback to user mean
        user_mean = self.user_means.get(user_id, 3.0)
        return user_mean
    
    def get_user_similarity(self, user1_id: int, user2_id: int) -> float:
        """Get similarity between two users"""
        if (self.user_user_similarity is not None and 
            user1_id in self.user_user_similarity.index and 
            user2_id in self.user_user_similarity.columns):
            return self.user_user_similarity.loc[user1_id, user2_id]
        return 0.0
    
    def get_item_similarity(self, movie1_id: int, movie2_id: int) -> float:
        """Get similarity between two movies"""
        if (self.item_item_similarity is not None and 
            movie1_id in self.item_item_similarity.index and 
            movie2_id in self.item_item_similarity.columns):
            return self.item_item_similarity.loc[movie1_id, movie2_id]
        return 0.0