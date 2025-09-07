"""
Movie Recommendation System - Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import config
from src.data_loader import MovieDataLoader, TMDBAPIClient
from src.recommenders.base import PopularityRecommender, RandomRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.utils import get_unique_genres, parse_genres, create_user_profile

# Set page configuration
st.set_page_config(
    page_title=config.STREAMLIT_CONFIG['page_title'],
    page_icon=config.STREAMLIT_CONFIG['page_icon'],
    layout=config.STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=config.STREAMLIT_CONFIG['initial_sidebar_state']
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .recommendation-score {
        color: #FF6B6B;
        font-weight: bold;
    }
    
    .genre-tag {
        display: inline-block;
        background-color: #4ECDC4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the movie data"""
    data_loader = MovieDataLoader()
    movies_df, ratings_df, links_df = data_loader.load_data()
    return movies_df, ratings_df, links_df

@st.cache_resource
def initialize_recommenders(movies_df, ratings_df):
    """Initialize and cache the recommendation models"""
    recommenders = {}
    
    # Popularity-based recommender
    popularity_rec = PopularityRecommender()
    popularity_rec.fit(ratings_df, movies_df)
    recommenders['popularity'] = popularity_rec
    
    # Content-based recommender
    content_rec = ContentBasedRecommender()
    content_rec.fit(ratings_df, movies_df)
    recommenders['content'] = content_rec
    
    # Random recommender (baseline)
    random_rec = RandomRecommender()
    random_rec.fit(ratings_df, movies_df)
    recommenders['random'] = random_rec
    
    return recommenders

def display_movie_card(movie_info, show_score=False, tmdb_client=None):
    """Display a movie card with information"""
    title = movie_info.get('title', 'Unknown Title')
    genres = movie_info.get('genres', '')
    score = movie_info.get('score', 0)
    explanation = movie_info.get('explanation', '')
    avg_rating = movie_info.get('avg_rating', 0)
    rating_count = movie_info.get('rating_count', 0)
    
    # Get movie poster if TMDB client is available
    poster_url = "https://placehold.co/200x300?text=Movie+Poster"
    if tmdb_client and 'tmdbId' in movie_info:
        try:
            movie_details = tmdb_client.get_movie_details(movie_info['tmdbId'])
            if movie_details and 'poster_path' in movie_details:
                poster_url = tmdb_client.get_poster_url(movie_details['poster_path'])
        except:
            pass
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image(poster_url, width=150)
    
    with col2:
        st.markdown(f"**{title}**")
        
        if genres:
            genre_list = parse_genres(genres)
            genre_html = ""
            for genre in genre_list[:5]:  # Limit to first 5 genres
                genre_html += f'<span class="genre-tag">{genre}</span> '
            st.markdown(genre_html, unsafe_allow_html=True)
        
        if avg_rating > 0:
            st.write(f"⭐ Average Rating: {avg_rating:.1f}/5.0 ({rating_count} ratings)")
        
        if show_score and score > 0:
            st.markdown(f'<p class="recommendation-score">Recommendation Score: {score:.2f}</p>', 
                       unsafe_allow_html=True)
        
        if explanation:
            st.write(f"💡 {explanation}")
    
    st.markdown("---")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI-Powered Movie Recommendation System! This application demonstrates 
    multiple machine learning approaches for movie recommendations including content-based 
    filtering, collaborative filtering, and hybrid methods.
    """)
    
    # Load data
    with st.spinner("Loading movie data..."):
        movies_df, ratings_df, links_df = load_data()
    
    # Initialize TMDB client
    tmdb_client = TMDBAPIClient()
    
    # Initialize recommenders
    with st.spinner("Initializing recommendation models..."):
        recommenders = initialize_recommenders(movies_df, ratings_df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Discover Movies", "🎯 Get Recommendations", "⭐ Rate Movies", "📊 Analytics"]
    )
    
    # Dataset overview in sidebar
    st.sidebar.markdown("## Dataset Overview")
    st.sidebar.metric("Movies", f"{len(movies_df):,}")
    st.sidebar.metric("Ratings", f"{len(ratings_df):,}")
    st.sidebar.metric("Users", f"{ratings_df['userId'].nunique():,}")
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page(movies_df, ratings_df, recommenders)
    
    elif page == "🔍 Discover Movies":
        show_discovery_page(movies_df, ratings_df, tmdb_client)
    
    elif page == "🎯 Get Recommendations":
        show_recommendations_page(movies_df, ratings_df, recommenders, tmdb_client)
    
    elif page == "⭐ Rate Movies":
        show_rating_page(movies_df, ratings_df)
    
    elif page == "📊 Analytics":
        show_analytics_page(movies_df, ratings_df)

def show_home_page(movies_df, ratings_df, recommenders):
    """Show the home page with system overview"""
    
    st.markdown("## 🚀 System Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Content-Based Filtering
        - Analyzes movie features (genres, titles)
        - Recommends similar movies
        - Great for discovering new content
        """)
    
    with col2:
        st.markdown("""
        ### 👥 Collaborative Filtering  
        - Uses user behavior patterns
        - Finds users with similar tastes
        - Leverages community wisdom
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 Hybrid Approach
        - Combines multiple methods
        - Handles cold start problems
        - Provides robust recommendations
        """)
    
    st.markdown("## 📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate some interesting stats
    avg_rating = ratings_df['rating'].mean()
    most_rated_movie = ratings_df.groupby('movieId').size().idxmax()
    most_rated_title = movies_df[movies_df['movieId'] == most_rated_movie]['title'].iloc[0]
    unique_genres = len(get_unique_genres(movies_df))
    
    with col1:
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
    
    with col2:
        st.metric("Unique Genres", f"{unique_genres}")
    
    with col3:
        movie_stats = ratings_df.groupby('movieId').size()
        st.metric("Most Ratings", f"{movie_stats.max()}")
    
    with col4:
        user_stats = ratings_df.groupby('userId').size()
        st.metric("Most Active User", f"{user_stats.max()} ratings")
    
    st.markdown("## 🎬 Featured Movies")
    
    # Show some popular movies
    popular_recommendations = recommenders['popularity'].recommend(n_recommendations=6)
    
    for i in range(0, len(popular_recommendations), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(popular_recommendations):
                movie = popular_recommendations[i]
                with st.container():
                    st.markdown(f"**{movie['title']}**")
                    genres = parse_genres(movie.get('genres', ''))
                    if genres:
                        st.write(f"🏷️ {', '.join(genres[:3])}")
                    st.write(f"⭐ {movie.get('avg_rating', 0):.1f}/5.0 ({movie.get('rating_count', 0)} ratings)")
        
        with col2:
            if i + 1 < len(popular_recommendations):
                movie = popular_recommendations[i + 1]
                with st.container():
                    st.markdown(f"**{movie['title']}**")
                    genres = parse_genres(movie.get('genres', ''))
                    if genres:
                        st.write(f"🏷️ {', '.join(genres[:3])}")
                    st.write(f"⭐ {movie.get('avg_rating', 0):.1f}/5.0 ({movie.get('rating_count', 0)} ratings)")
    
    st.markdown("## 🔧 How to Use")
    st.markdown("""
    1. **Discover Movies**: Browse and search through our movie catalog
    2. **Get Recommendations**: Select a user ID or movie to get personalized suggestions
    3. **Rate Movies**: Simulate user ratings to see how recommendations change
    4. **Analytics**: Explore dataset insights and model performance
    """)

def show_discovery_page(movies_df, ratings_df, tmdb_client):
    """Show movie discovery page"""
    
    st.markdown("## 🔍 Discover Movies")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Genre filter
        all_genres = ["All"] + get_unique_genres(movies_df)
        selected_genre = st.selectbox("Filter by Genre:", all_genres)
    
    with col2:
        # Rating filter
        min_rating = st.slider("Minimum Average Rating:", 1.0, 5.0, 3.0, 0.1)
    
    with col3:
        # Minimum ratings filter
        min_ratings_count = st.slider("Minimum Number of Ratings:", 1, 50, 10)
    
    # Search
    search_query = st.text_input("Search movies by title:", "")
    
    # Filter movies
    filtered_movies = movies_df.copy()
    
    if selected_genre != "All":
        filtered_movies = filtered_movies[
            filtered_movies['genres'].str.contains(selected_genre, na=False)
        ]
    
    if search_query:
        filtered_movies = filtered_movies[
            filtered_movies['title'].str.contains(search_query, case=False, na=False)
        ]
    
    # Add rating statistics
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    })
    movie_stats.columns = ['avg_rating', 'rating_count']
    
    # Merge with filtered movies
    filtered_movies_with_stats = filtered_movies.merge(
        movie_stats, 
        left_on='movieId', 
        right_index=True, 
        how='left'
    ).fillna({'avg_rating': 0, 'rating_count': 0})
    
    # Apply rating filters
    filtered_movies_with_stats = filtered_movies_with_stats[
        (filtered_movies_with_stats['avg_rating'] >= min_rating) &
        (filtered_movies_with_stats['rating_count'] >= min_ratings_count)
    ]
    
    # Sort by average rating
    filtered_movies_with_stats = filtered_movies_with_stats.sort_values(
        'avg_rating', ascending=False
    )
    
    st.markdown(f"**Found {len(filtered_movies_with_stats)} movies**")
    
    # Pagination
    page_size = 10
    total_pages = len(filtered_movies_with_stats) // page_size + 1
    
    if total_pages > 1:
        page_num = st.selectbox(
            f"Page (showing {page_size} movies per page):", 
            range(1, total_pages + 1)
        )
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        movies_to_show = filtered_movies_with_stats.iloc[start_idx:end_idx]
    else:
        movies_to_show = filtered_movies_with_stats.head(page_size)
    
    # Display movies
    if not movies_to_show.empty:
        for _, movie in movies_to_show.iterrows():
            movie_info = {
                'title': movie['title'],
                'genres': movie['genres'],
                'avg_rating': movie['avg_rating'],
                'rating_count': movie['rating_count'],
                'movieId': movie['movieId']
            }
            display_movie_card(movie_info, tmdb_client=tmdb_client)
    else:
        st.write("No movies found matching the criteria.")

def show_recommendations_page(movies_df, ratings_df, recommenders, tmdb_client):
    """Show recommendations page"""
    
    st.markdown("## 🎯 Get Personalized Recommendations")
    
    # Recommendation type selection
    rec_type = st.radio(
        "Choose recommendation type:",
        ["👤 User-based Recommendations", "🎬 Movie-based Recommendations", "📈 Popular Movies"]
    )
    
    if rec_type == "👤 User-based Recommendations":
        st.markdown("### Recommendations based on User Preferences")
        
        # User selection
        available_users = sorted(ratings_df['userId'].unique())
        selected_user = st.selectbox(
            "Select a User ID:", 
            available_users,
            help="Users with more ratings will get better recommendations"
        )
        
        # Show user profile
        if selected_user:
            user_ratings = ratings_df[ratings_df['userId'] == selected_user]
            user_profile = create_user_profile(user_ratings, movies_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**User Profile:**")
                st.write(f"Total ratings: {user_profile['total_ratings']}")
                st.write(f"Average rating: {user_profile['avg_rating']:.2f}")
                
                if user_profile['favorite_genres']:
                    st.write("Favorite genres:")
                    for genre in user_profile['favorite_genres'][:5]:
                        st.write(f"  • {genre}")
            
            with col2:
                st.markdown("**Recent Ratings:**")
                recent_ratings = user_ratings.merge(
                    movies_df[['movieId', 'title']], on='movieId'
                ).sort_values('timestamp', ascending=False).head(5)
                
                for _, rating in recent_ratings.iterrows():
                    st.write(f"⭐ {rating['rating']:.1f} - {rating['title'][:30]}...")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Choose recommendation algorithm:",
            ["Content-Based", "Popularity-Based", "Random (Baseline)"]
        )
        
        # Get recommendations
        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                if algorithm == "Content-Based":
                    recommendations = recommenders['content'].recommend(
                        user_id=selected_user, n_recommendations=10
                    )
                elif algorithm == "Popularity-Based":
                    recommendations = recommenders['popularity'].recommend(
                        user_id=selected_user, n_recommendations=10
                    )
                else:  # Random
                    recommendations = recommenders['random'].recommend(
                        user_id=selected_user, n_recommendations=10
                    )
                
                st.markdown("### 🎬 Your Recommendations")
                
                if recommendations:
                    for rec in recommendations:
                        display_movie_card(rec, show_score=True, tmdb_client=tmdb_client)
                else:
                    st.write("No recommendations available for this user.")
    
    elif rec_type == "🎬 Movie-based Recommendations":
        st.markdown("### Find Movies Similar to One You Like")
        
        # Movie selection
        movie_titles = dict(zip(movies_df['title'], movies_df['movieId']))
        selected_title = st.selectbox(
            "Select a movie:", 
            sorted(movie_titles.keys()),
            help="Get recommendations based on movies similar to this one"
        )
        
        selected_movie_id = movie_titles[selected_title]
        
        # Show selected movie info
        selected_movie = movies_df[movies_df['movieId'] == selected_movie_id].iloc[0]
        st.markdown("**Selected Movie:**")
        movie_info = {
            'title': selected_movie['title'],
            'genres': selected_movie['genres'],
            'movieId': selected_movie['movieId']
        }
        
        # Add rating info
        movie_ratings = ratings_df[ratings_df['movieId'] == selected_movie_id]
        if not movie_ratings.empty:
            movie_info['avg_rating'] = movie_ratings['rating'].mean()
            movie_info['rating_count'] = len(movie_ratings)
        
        display_movie_card(movie_info, tmdb_client=tmdb_client)
        
        # Get similar movies
        if st.button("Find Similar Movies"):
            with st.spinner("Finding similar movies..."):
                recommendations = recommenders['content'].recommend(
                    movie_id=selected_movie_id, n_recommendations=10
                )
                
                st.markdown("### 🎯 Movies Similar to Your Selection")
                
                if recommendations:
                    for rec in recommendations:
                        display_movie_card(rec, show_score=True, tmdb_client=tmdb_client)
                else:
                    st.write("No similar movies found.")
    
    else:  # Popular Movies
        st.markdown("### 📈 Currently Popular Movies")
        
        n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        recommendations = recommenders['popularity'].recommend(
            n_recommendations=n_recommendations
        )
        
        st.markdown("### 🏆 Most Popular Movies")
        
        for rec in recommendations:
            display_movie_card(rec, show_score=True, tmdb_client=tmdb_client)

def show_rating_page(movies_df, ratings_df):
    """Show movie rating simulation page"""
    
    st.markdown("## ⭐ Rate Movies")
    st.markdown("Simulate user ratings to see how they affect recommendations.")
    
    # User selection or creation
    existing_users = sorted(ratings_df['userId'].unique())
    
    user_choice = st.radio(
        "Choose user option:",
        ["Select existing user", "Create new user simulation"]
    )
    
    if user_choice == "Select existing user":
        user_id = st.selectbox("Select User ID:", existing_users)
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        st.markdown(f"### Current Ratings for User {user_id}")
        
        # Show existing ratings
        user_movies = user_ratings.merge(
            movies_df[['movieId', 'title', 'genres']], on='movieId'
        ).sort_values('rating', ascending=False)
        
        for _, rating in user_movies.head(10).iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(rating['title'])
            with col2:
                st.write(f"⭐ {rating['rating']}")
            with col3:
                genres = parse_genres(rating['genres'])
                if genres:
                    st.write(genres[0])
    
    else:
        st.markdown("### Simulate New User Ratings")
        st.write("Rate some movies to see what recommendations you would get!")
        
        # Initialize session state for ratings
        if 'user_ratings' not in st.session_state:
            st.session_state.user_ratings = {}
        
        # Movie selection for rating
        popular_movies = ratings_df.groupby('movieId').size().nlargest(50)
        popular_movie_ids = popular_movies.index.tolist()
        popular_movie_titles = movies_df[
            movies_df['movieId'].isin(popular_movie_ids)
        ].set_index('movieId')['title'].to_dict()
        
        st.markdown("**Rate some popular movies:**")
        
        # Create rating interface
        for movie_id in popular_movie_ids[:10]:
            if movie_id in popular_movie_titles:
                title = popular_movie_titles[movie_id]
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"{title}")
                
                with col2:
                    rating = st.selectbox(
                        f"Rating for {movie_id}:",
                        [0, 1, 2, 3, 4, 5],
                        key=f"rating_{movie_id}",
                        format_func=lambda x: "Not rated" if x == 0 else f"{x} ⭐"
                    )
                    
                    if rating > 0:
                        st.session_state.user_ratings[movie_id] = rating
        
        # Show current ratings
        if st.session_state.user_ratings:
            st.markdown("### Your Current Ratings")
            for movie_id, rating in st.session_state.user_ratings.items():
                title = popular_movie_titles.get(movie_id, f"Movie {movie_id}")
                st.write(f"⭐ {rating} - {title}")
            
            # Clear ratings button
            if st.button("Clear All Ratings"):
                st.session_state.user_ratings = {}
                st.rerun()

def show_analytics_page(movies_df, ratings_df):
    """Show analytics and insights page"""
    
    st.markdown("## 📊 Dataset Analytics & Insights")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", f"{len(movies_df):,}")
    
    with col2:
        st.metric("Total Ratings", f"{len(ratings_df):,}")
    
    with col3:
        st.metric("Unique Users", f"{ratings_df['userId'].nunique():,}")
    
    with col4:
        avg_rating = ratings_df['rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f} ⭐")
    
    # Rating distribution
    st.markdown("### Rating Distribution")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.bar_chart(rating_counts)
    
    with col2:
        st.write("**Rating Statistics:**")
        for rating, count in rating_counts.items():
            percentage = (count / len(ratings_df)) * 100
            st.write(f"{rating} ⭐: {percentage:.1f}%")
    
    # Genre analysis
    st.markdown("### Genre Analysis")
    
    # Count movies per genre
    genre_counts = {}
    for _, movie in movies_df.iterrows():
        genres = parse_genres(movie['genres'])
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    genre_df = pd.DataFrame(
        list(genre_counts.items()), 
        columns=['Genre', 'Count']
    ).sort_values('Count', ascending=True)
    
    st.bar_chart(genre_df.set_index('Genre'))
    
    # User activity
    st.markdown("### User Activity Analysis")
    
    user_stats = ratings_df.groupby('userId').size()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**User Activity Distribution:**")
        st.write(f"Most active user: {user_stats.max()} ratings")
        st.write(f"Average ratings per user: {user_stats.mean():.1f}")
        st.write(f"Users with 10+ ratings: {(user_stats >= 10).sum()}")
        st.write(f"Users with 20+ ratings: {(user_stats >= 20).sum()}")
    
    with col2:
        # User activity histogram
        st.write("**Ratings per User Distribution:**")
        activity_bins = [1, 5, 10, 20, 50, 100, user_stats.max()]
        activity_labels = ['1-4', '5-9', '10-19', '20-49', '50-99', '100+']
        
        activity_dist = pd.cut(user_stats, bins=activity_bins, labels=activity_labels, include_lowest=True).value_counts()
        st.bar_chart(activity_dist)
    
    # Movie popularity
    st.markdown("### Movie Popularity Analysis")
    
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    })
    movie_stats.columns = ['rating_count', 'avg_rating']
    
    # Merge with movie titles
    popular_movies = movie_stats.merge(
        movies_df[['movieId', 'title']], 
        left_index=True, 
        right_on='movieId'
    ).sort_values('rating_count', ascending=False)
    
    st.write("**Top 10 Most Rated Movies:**")
    for i, (_, movie) in enumerate(popular_movies.head(10).iterrows(), 1):
        st.write(f"{i}. {movie['title']} - {movie['rating_count']} ratings (avg: {movie['avg_rating']:.1f})")
    
    # Data sparsity
    st.markdown("### Data Sparsity Analysis")
    
    n_users = ratings_df['userId'].nunique()
    n_movies = ratings_df['movieId'].nunique()
    n_ratings = len(ratings_df)
    total_possible = n_users * n_movies
    sparsity = 1 - (n_ratings / total_possible)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Matrix Sparsity", f"{sparsity:.1%}")
        st.metric("Matrix Density", f"{1-sparsity:.1%}")
    
    with col2:
        st.write(f"**Sparsity Details:**")
        st.write(f"Possible user-movie pairs: {total_possible:,}")
        st.write(f"Actual ratings: {n_ratings:,}")
        st.write(f"Missing ratings: {total_possible - n_ratings:,}")

if __name__ == "__main__":
    main()