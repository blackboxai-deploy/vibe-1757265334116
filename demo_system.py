#!/usr/bin/env python3
"""
Movie Recommendation System - Basic Demo
Demonstrates the system architecture without external dependencies
"""
import json
import random
from typing import List, Dict, Any

class SimpleMovieData:
    """Simple movie data generator for demo purposes"""
    
    def __init__(self):
        self.movies = [
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
        ]
        
        # Generate sample ratings
        random.seed(42)
        self.ratings = []
        for user_id in range(1, 21):  # 20 users
            for _ in range(random.randint(5, 15)):  # Each user rates 5-15 movies
                movie_id = random.choice([m['movieId'] for m in self.movies])
                rating = random.choice([1, 2, 3, 4, 5])
                self.ratings.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': rating
                })
    
    def get_movie_by_id(self, movie_id: int) -> Dict[str, Any]:
        """Get movie by ID"""
        for movie in self.movies:
            if movie['movieId'] == movie_id:
                return movie
        return {}
    
    def get_user_ratings(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all ratings for a user"""
        return [r for r in self.ratings if r['userId'] == user_id]

class SimpleContentRecommender:
    """Simple content-based recommender using genre similarity"""
    
    def __init__(self, data: SimpleMovieData):
        self.data = data
        self.genre_similarity = {}
        self._calculate_genre_similarity()
    
    def _parse_genres(self, genres_str: str) -> List[str]:
        """Parse genres string into list"""
        return genres_str.split('|') if genres_str else []
    
    def _calculate_genre_similarity(self):
        """Calculate genre-based similarity between movies"""
        movies = self.data.movies
        
        for i, movie1 in enumerate(movies):
            self.genre_similarity[movie1['movieId']] = {}
            genres1 = set(self._parse_genres(movie1['genres']))
            
            for j, movie2 in enumerate(movies):
                if i != j:
                    genres2 = set(self._parse_genres(movie2['genres']))
                    
                    # Jaccard similarity
                    intersection = len(genres1.intersection(genres2))
                    union = len(genres1.union(genres2))
                    similarity = intersection / union if union > 0 else 0
                    
                    self.genre_similarity[movie1['movieId']][movie2['movieId']] = similarity
    
    def recommend_similar_movies(self, movie_id: int, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Recommend movies similar to a given movie"""
        if movie_id not in self.genre_similarity:
            return []
        
        similar_movies = self.genre_similarity[movie_id]
        sorted_similar = sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for sim_movie_id, similarity in sorted_similar[:n_recommendations]:
            movie_info = self.data.get_movie_by_id(sim_movie_id)
            if movie_info:
                recommendations.append({
                    'movieId': sim_movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'similarity_score': similarity,
                    'method': 'content_based'
                })
        
        return recommendations
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Recommend movies for a user based on their preferences"""
        user_ratings = self.data.get_user_ratings(user_id)
        
        if not user_ratings:
            return self._get_popular_movies(n_recommendations)
        
        # Find highly rated movies (4+ stars)
        liked_movies = [r for r in user_ratings if r['rating'] >= 4]
        
        if not liked_movies:
            liked_movies = user_ratings  # Use all ratings if no high ratings
        
        # Get genre preferences
        genre_scores = {}
        for rating in liked_movies:
            movie_info = self.data.get_movie_by_id(rating['movieId'])
            if movie_info:
                genres = self._parse_genres(movie_info['genres'])
                for genre in genres:
                    genre_scores[genre] = genre_scores.get(genre, 0) + rating['rating']
        
        # Score all movies based on genre preferences
        movie_scores = {}
        rated_movie_ids = {r['movieId'] for r in user_ratings}
        
        for movie in self.data.movies:
            if movie['movieId'] not in rated_movie_ids:  # Don't recommend already rated movies
                genres = self._parse_genres(movie['genres'])
                score = sum(genre_scores.get(genre, 0) for genre in genres)
                if genres:
                    score = score / len(genres)  # Normalize by number of genres
                movie_scores[movie['movieId']] = score
        
        # Sort and get top recommendations
        sorted_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_id, score in sorted_scores[:n_recommendations]:
            movie_info = self.data.get_movie_by_id(movie_id)
            if movie_info:
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'preference_score': score,
                    'method': 'user_profile'
                })
        
        return recommendations
    
    def _get_popular_movies(self, n_recommendations: int) -> List[Dict[str, Any]]:
        """Get popular movies as fallback"""
        # Calculate movie popularity (rating count)
        movie_rating_counts = {}
        for rating in self.data.ratings:
            movie_id = rating['movieId']
            movie_rating_counts[movie_id] = movie_rating_counts.get(movie_id, 0) + 1
        
        # Sort by popularity
        popular_movies = sorted(movie_rating_counts.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_id, count in popular_movies[:n_recommendations]:
            movie_info = self.data.get_movie_by_id(movie_id)
            if movie_info:
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'rating_count': count,
                    'method': 'popularity'
                })
        
        return recommendations

class SimpleCollaborativeRecommender:
    """Simple collaborative filtering using user similarity"""
    
    def __init__(self, data: SimpleMovieData):
        self.data = data
        self.user_similarity = {}
        self._calculate_user_similarity()
    
    def _calculate_user_similarity(self):
        """Calculate user-user similarity based on rating patterns"""
        users = list(set(r['userId'] for r in self.data.ratings))
        
        # Create user rating profiles
        user_profiles = {}
        for user_id in users:
            user_ratings = self.data.get_user_ratings(user_id)
            user_profiles[user_id] = {r['movieId']: r['rating'] for r in user_ratings}
        
        # Calculate similarity between users
        for user1 in users:
            self.user_similarity[user1] = {}
            
            for user2 in users:
                if user1 != user2:
                    # Find commonly rated movies
                    common_movies = set(user_profiles[user1].keys()).intersection(
                        set(user_profiles[user2].keys())
                    )
                    
                    if len(common_movies) >= 2:  # Need at least 2 common movies
                        # Calculate Pearson correlation (simplified)
                        ratings1 = [user_profiles[user1][m] for m in common_movies]
                        ratings2 = [user_profiles[user2][m] for m in common_movies]
                        
                        mean1 = sum(ratings1) / len(ratings1)
                        mean2 = sum(ratings2) / len(ratings2)
                        
                        num = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(ratings1, ratings2))
                        den1 = sum((r1 - mean1) ** 2 for r1 in ratings1) ** 0.5
                        den2 = sum((r2 - mean2) ** 2 for r2 in ratings2) ** 0.5
                        
                        if den1 > 0 and den2 > 0:
                            correlation = num / (den1 * den2)
                            self.user_similarity[user1][user2] = max(0, correlation)  # Only positive correlations
                        else:
                            self.user_similarity[user1][user2] = 0
                    else:
                        self.user_similarity[user1][user2] = 0
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Recommend movies using collaborative filtering"""
        if user_id not in self.user_similarity:
            return []
        
        # Find similar users
        similar_users = self.user_similarity[user_id]
        if not similar_users:
            return []
        
        # Sort users by similarity
        sorted_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)
        top_similar_users = [user for user, sim in sorted_users[:5] if sim > 0]
        
        if not top_similar_users:
            return []
        
        # Get movies rated highly by similar users
        user_rated_movies = {r['movieId'] for r in self.data.get_user_ratings(user_id)}
        candidate_movies = {}
        
        for similar_user in top_similar_users:
            similar_user_ratings = self.data.get_user_ratings(similar_user)
            similarity_score = similar_users[similar_user]
            
            for rating in similar_user_ratings:
                movie_id = rating['movieId']
                if movie_id not in user_rated_movies and rating['rating'] >= 4:
                    # Weight rating by user similarity
                    weighted_rating = rating['rating'] * similarity_score
                    candidate_movies[movie_id] = candidate_movies.get(movie_id, 0) + weighted_rating
        
        # Sort and get top recommendations
        sorted_candidates = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_id, score in sorted_candidates[:n_recommendations]:
            movie_info = self.data.get_movie_by_id(movie_id)
            if movie_info:
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'collaborative_score': score,
                    'method': 'collaborative_filtering'
                })
        
        return recommendations

def demo_system():
    """Demonstrate the movie recommendation system"""
    print("🎬 Movie Recommendation System - Live Demo")
    print("=" * 60)
    
    # Initialize data and recommenders
    print("🔄 Initializing system...")
    data = SimpleMovieData()
    content_rec = SimpleContentRecommender(data)
    collab_rec = SimpleCollaborativeRecommender(data)
    
    print(f"✅ System initialized!")
    print(f"   📽️  Movies: {len(data.movies)}")
    print(f"   ⭐ Ratings: {len(data.ratings)}")
    print(f"   👤 Users: {len(set(r['userId'] for r in data.ratings))}")
    
    # Demo 1: Movie-to-Movie Similarity
    print(f"\n" + "="*60)
    print(f"🎯 DEMO 1: Movie-to-Movie Similarity")
    print(f"="*60)
    
    toy_story_id = 1  # Toy Story
    print(f"🎬 Movies similar to 'Toy Story':")
    similar_movies = content_rec.recommend_similar_movies(toy_story_id, 5)
    
    for i, movie in enumerate(similar_movies, 1):
        print(f"   {i}. {movie['title']}")
        print(f"      Similarity: {movie['similarity_score']:.3f} | Genres: {movie['genres']}")
    
    # Demo 2: User-Based Content Recommendations
    print(f"\n" + "="*60)
    print(f"🎯 DEMO 2: Content-Based User Recommendations")
    print(f"="*60)
    
    test_user = 1
    user_ratings = data.get_user_ratings(test_user)
    print(f"👤 User {test_user} Profile:")
    print(f"   Total ratings: {len(user_ratings)}")
    print(f"   Favorite movies:")
    
    # Show user's highly rated movies
    high_rated = [r for r in user_ratings if r['rating'] >= 4]
    for rating in high_rated[:3]:
        movie = data.get_movie_by_id(rating['movieId'])
        print(f"      ⭐ {rating['rating']}/5 - {movie['title']}")
    
    print(f"\n📋 Content-based recommendations for User {test_user}:")
    content_recs = content_rec.recommend_for_user(test_user, 5)
    for i, rec in enumerate(content_recs, 1):
        print(f"   {i}. {rec['title']}")
        print(f"      Score: {rec['preference_score']:.2f} | Genres: {rec['genres']}")
    
    # Demo 3: Collaborative Filtering
    print(f"\n" + "="*60)
    print(f"🎯 DEMO 3: Collaborative Filtering Recommendations")
    print(f"="*60)
    
    print(f"👥 Collaborative filtering recommendations for User {test_user}:")
    collab_recs = collab_rec.recommend_for_user(test_user, 5)
    
    if collab_recs:
        for i, rec in enumerate(collab_recs, 1):
            print(f"   {i}. {rec['title']}")
            print(f"      Score: {rec['collaborative_score']:.2f} | Genres: {rec['genres']}")
    else:
        print("   No collaborative recommendations available (insufficient user overlap)")
    
    # Demo 4: System Comparison
    print(f"\n" + "="*60)
    print(f"🎯 DEMO 4: Algorithm Comparison")
    print(f"="*60)
    
    print("🔄 Comparing different recommendation approaches...")
    
    methods = {
        "Content-Based": content_recs[:3],
        "Collaborative": collab_recs[:3] if collab_recs else []
    }
    
    for method_name, recs in methods.items():
        print(f"\n📊 {method_name} (Top 3):")
        if recs:
            for i, rec in enumerate(recs, 1):
                print(f"   {i}. {rec['title']}")
        else:
            print("   No recommendations available")
    
    # System Statistics
    print(f"\n" + "="*60)
    print(f"📊 SYSTEM STATISTICS")
    print(f"="*60)
    
    # Genre analysis
    all_genres = set()
    for movie in data.movies:
        all_genres.update(movie['genres'].split('|'))
    
    print(f"🏷️  Unique genres: {len(all_genres)}")
    print(f"🎬 Average genres per movie: {sum(len(m['genres'].split('|')) for m in data.movies) / len(data.movies):.1f}")
    
    # Rating analysis
    all_ratings = [r['rating'] for r in data.ratings]
    avg_rating = sum(all_ratings) / len(all_ratings)
    
    print(f"⭐ Average rating: {avg_rating:.2f}")
    print(f"📈 Rating distribution: {dict(sorted({r: all_ratings.count(r) for r in set(all_ratings)}.items()))}")
    
    # User activity
    user_activity = {}
    for rating in data.ratings:
        user_id = rating['userId']
        user_activity[user_id] = user_activity.get(user_id, 0) + 1
    
    avg_ratings_per_user = sum(user_activity.values()) / len(user_activity)
    print(f"👤 Average ratings per user: {avg_ratings_per_user:.1f}")
    print(f"🏆 Most active user: {max(user_activity.values())} ratings")
    
    print(f"\n" + "="*60)
    print(f"✅ DEMO COMPLETE - System Working Successfully!")
    print(f"="*60)
    
    print(f"\n🚀 Key Features Demonstrated:")
    print(f"   ✅ Content-based filtering using genre similarity")
    print(f"   ✅ User preference learning from rating history")
    print(f"   ✅ Collaborative filtering using user similarity")
    print(f"   ✅ Movie-to-movie similarity recommendations")
    print(f"   ✅ Multi-algorithm comparison and evaluation")
    
    print(f"\n📝 Next Steps:")
    print(f"   📊 Advanced ML models (Matrix Factorization, Deep Learning)")
    print(f"   🌐 Full Streamlit web interface deployment")
    print(f"   📈 Performance evaluation and A/B testing")
    print(f"   🔄 Hybrid recommendation algorithms")
    
    return True

if __name__ == "__main__":
    demo_system()