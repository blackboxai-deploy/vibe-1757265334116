"""
Data loading and preprocessing utilities for Movie Recommendation System
"""
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from typing import Dict, List, Tuple, Optional
import config

class MovieDataLoader:
    """Load and preprocess movie dataset"""
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.links_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all dataset files"""
        try:
            # Load movies data
            if config.MOVIES_FILE.exists():
                self.movies_df = pd.read_csv(config.MOVIES_FILE)
            else:
                self.movies_df = self._create_sample_movies_data()
                
            # Load ratings data
            if config.RATINGS_FILE.exists():
                self.ratings_df = pd.read_csv(config.RATINGS_FILE)
            else:
                self.ratings_df = self._create_sample_ratings_data()
                
            # Load links data
            if config.LINKS_FILE.exists():
                self.links_df = pd.read_csv(config.LINKS_FILE)
            else:
                self.links_df = self._create_sample_links_data()
                
            return self.movies_df, self.ratings_df, self.links_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_sample_data()
    
    def _create_sample_movies_data(self) -> pd.DataFrame:
        """Create sample movies dataset"""
        sample_movies = [
            {"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure|Animation|Children|Comedy|Fantasy"},
            {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy"},
            {"movieId": 3, "title": "Grumpier Old Men (1995)", "genres": "Comedy|Romance"},
            {"movieId": 4, "title": "Waiting to Exhale (1995)", "genres": "Comedy|Drama|Romance"},
            {"movieId": 5, "title": "Father of the Bride Part II (1995)", "genres": "Comedy"},
            {"movieId": 6, "title": "Heat (1995)", "genres": "Action|Crime|Thriller"},
            {"movieId": 7, "title": "Sabrina (1995)", "genres": "Comedy|Romance"},
            {"movieId": 8, "title": "Tom and Huck (1995)", "genres": "Adventure|Children"},
            {"movieId": 9, "title": "Sudden Death (1995)", "genres": "Action"},
            {"movieId": 10, "title": "GoldenEye (1995)", "genres": "Action|Adventure|Thriller"},
            {"movieId": 11, "title": "American Beauty (1999)", "genres": "Drama|Romance"},
            {"movieId": 12, "title": "Star Wars: Episode IV (1977)", "genres": "Action|Adventure|Sci-Fi"},
            {"movieId": 13, "title": "The Lion King (1994)", "genres": "Animation|Children|Drama|Musical"},
            {"movieId": 14, "title": "Pulp Fiction (1994)", "genres": "Comedy|Crime|Drama|Thriller"},
            {"movieId": 15, "title": "Forrest Gump (1994)", "genres": "Comedy|Drama|Romance|War"},
            {"movieId": 16, "title": "Silence of the Lambs (1991)", "genres": "Crime|Horror|Thriller"},
            {"movieId": 17, "title": "Matrix, The (1999)", "genres": "Action|Sci-Fi|Thriller"},
            {"movieId": 18, "title": "Goodfellas (1990)", "genres": "Crime|Drama"},
            {"movieId": 19, "title": "Seven (1995)", "genres": "Mystery|Thriller"},
            {"movieId": 20, "title": "Usual Suspects, The (1995)", "genres": "Crime|Mystery|Thriller"},
            {"movieId": 21, "title": "Schindler's List (1993)", "genres": "Drama|War"},
            {"movieId": 22, "title": "Shawshank Redemption, The (1994)", "genres": "Crime|Drama"},
            {"movieId": 23, "title": "Casablanca (1942)", "genres": "Drama|Romance"},
            {"movieId": 24, "title": "Citizen Kane (1941)", "genres": "Drama|Mystery"},
            {"movieId": 25, "title": "Wizard of Oz, The (1939)", "genres": "Adventure|Children|Fantasy|Musical"}
        ]
        
        movies_df = pd.DataFrame(sample_movies)
        
        # Save to file
        config.DATA_DIR.mkdir(exist_ok=True)
        movies_df.to_csv(config.MOVIES_FILE, index=False)
        
        return movies_df
    
    def _create_sample_ratings_data(self) -> pd.DataFrame:
        """Create sample ratings dataset"""
        np.random.seed(42)
        
        # Generate ratings for 50 users and 25 movies
        ratings_data = []
        for user_id in range(1, 51):
            # Each user rates 5-15 movies randomly
            n_ratings = np.random.randint(5, 16)
            movie_ids = np.random.choice(range(1, 26), size=n_ratings, replace=False)
            
            for movie_id in movie_ids:
                # Generate realistic ratings (more 3-5 stars)
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.35, 0.25])
                timestamp = np.random.randint(978300760, 1672531200)  # Random timestamp
                
                ratings_data.append({
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": rating,
                    "timestamp": timestamp
                })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        # Save to file
        ratings_df.to_csv(config.RATINGS_FILE, index=False)
        
        return ratings_df
    
    def _create_sample_links_data(self) -> pd.DataFrame:
        """Create sample links dataset"""
        # Sample IMDB and TMDB IDs for movies
        links_data = [
            {"movieId": 1, "imdbId": "0114709", "tmdbId": "862"},
            {"movieId": 2, "imdbId": "0113497", "tmdbId": "8844"},
            {"movieId": 3, "imdbId": "0113228", "tmdbId": "15602"},
            {"movieId": 4, "imdbId": "0114885", "tmdbId": "31357"},
            {"movieId": 5, "imdbId": "0113041", "tmdbId": "11862"},
            {"movieId": 6, "imdbId": "0113277", "tmdbId": "949"},
            {"movieId": 7, "imdbId": "0114319", "tmdbId": "11860"},
            {"movieId": 8, "imdbId": "0112302", "tmdbId": "45325"},
            {"movieId": 9, "imdbId": "0114576", "tmdbId": "9091"},
            {"movieId": 10, "imdbId": "0113189", "tmdbId": "710"},
            {"movieId": 11, "imdbId": "0169547", "tmdbId": "14"},
            {"movieId": 12, "imdbId": "0076759", "tmdbId": "11"},
            {"movieId": 13, "imdbId": "0110357", "tmdbId": "8587"},
            {"movieId": 14, "imdbId": "0110912", "tmdbId": "680"},
            {"movieId": 15, "imdbId": "0109830", "tmdbId": "13"},
            {"movieId": 16, "imdbId": "0102926", "tmdbId": "274"},
            {"movieId": 17, "imdbId": "0133093", "tmdbId": "603"},
            {"movieId": 18, "imdbId": "0099685", "tmdbId": "769"},
            {"movieId": 19, "imdbId": "0114369", "tmdbId": "807"},
            {"movieId": 20, "imdbId": "0114814", "tmdbId": "629"},
            {"movieId": 21, "imdbId": "0108052", "tmdbId": "424"},
            {"movieId": 22, "imdbId": "0111161", "tmdbId": "278"},
            {"movieId": 23, "imdbId": "0034583", "tmdbId": "289"},
            {"movieId": 24, "imdbId": "0033467", "tmdbId": "15"},
            {"movieId": 25, "imdbId": "0032138", "tmdbId": "630"}
        ]
        
        links_df = pd.DataFrame(links_data)
        
        # Save to file
        links_df.to_csv(config.LINKS_FILE, index=False)
        
        return links_df
    
    def _create_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create all sample datasets"""
        movies_df = self._create_sample_movies_data()
        ratings_df = self._create_sample_ratings_data()
        links_df = self._create_sample_links_data()
        
        return movies_df, ratings_df, links_df
    
    def preprocess_data(self) -> Dict:
        """Preprocess data for recommendation systems"""
        if self.movies_df is None or self.ratings_df is None:
            self.load_data()
        
        # Create user-item matrix
        user_item_matrix = self.ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Extract genres for content-based filtering
        genres_df = self.movies_df['genres'].str.get_dummies('|')
        
        # Calculate movie statistics
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        
        movie_stats.columns = ['rating_count', 'rating_mean', 'rating_std']
        movie_stats = movie_stats.fillna(0)
        
        return {
            'user_item_matrix': user_item_matrix,
            'genres_df': genres_df,
            'movie_stats': movie_stats,
            'movies_df': self.movies_df,
            'ratings_df': self.ratings_df
        }

class TMDBAPIClient:
    """TMDB API client for fetching movie metadata"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.TMDB_API_KEY
        self.base_url = config.TMDB_BASE_URL
        self.image_base_url = config.TMDB_IMAGE_BASE_URL
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch movie details from TMDB"""
        if not self.api_key or self.api_key == "your_api_key_here":
            return self._get_placeholder_movie_details(tmdb_id)
        
        try:
            url = f"{self.base_url}/movie/{tmdb_id}"
            params = {'api_key': self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error fetching movie details for TMDB ID {tmdb_id}: {e}")
            return self._get_placeholder_movie_details(tmdb_id)
    
    def _get_placeholder_movie_details(self, tmdb_id: int) -> Dict:
        """Return placeholder movie details when API is not available"""
        return {
            'id': tmdb_id,
            'title': f'Movie {tmdb_id}',
            'overview': 'Movie overview not available',
            'poster_path': f'https://placehold.co/300x450?text=Movie+Poster+{tmdb_id}',
            'release_date': '2020-01-01',
            'vote_average': 7.0,
            'vote_count': 100,
            'runtime': 120
        }
    
    def get_poster_url(self, poster_path: str) -> str:
        """Get full URL for movie poster"""
        if poster_path and not poster_path.startswith('http'):
            return f"{self.image_base_url}{poster_path}"
        return poster_path or f'https://placehold.co/300x450?text=No+Poster+Available'