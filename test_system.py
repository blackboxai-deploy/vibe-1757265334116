#!/usr/bin/env python3
"""
Test script to verify the movie recommendation system works
"""
import sys
import os
import traceback

# Add current directory to path
sys.path.append('.')

def test_data_loading():
    """Test data loading functionality"""
    print("🔄 Testing data loading...")
    try:
        from src.data_loader import MovieDataLoader
        
        # Load data
        loader = MovieDataLoader()
        movies_df, ratings_df, links_df = loader.load_data()
        
        print(f"✅ Data loaded successfully!")
        print(f"   📽️  Movies: {len(movies_df)}")
        print(f"   ⭐ Ratings: {len(ratings_df)}")
        print(f"   👤 Users: {ratings_df['userId'].nunique()}")
        print(f"   🔗 Links: {len(links_df)}")
        
        return True, (movies_df, ratings_df, links_df)
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False, None

def test_content_recommender(data):
    """Test content-based recommender"""
    print("\n🔄 Testing content-based recommender...")
    try:
        from src.recommenders.content_based import ContentBasedRecommender
        
        movies_df, ratings_df, links_df = data
        
        # Create and fit recommender
        recommender = ContentBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        print("✅ Content-based recommender trained successfully!")
        
        # Test movie-to-movie recommendations
        toy_story = movies_df[movies_df['title'].str.contains('Toy Story', case=False, na=False)]
        if not toy_story.empty:
            toy_story_id = toy_story.iloc[0]['movieId']
            recommendations = recommender.recommend(movie_id=toy_story_id, n_recommendations=5)
            
            print(f"🎬 Movies similar to 'Toy Story':")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec['title']} (Score: {rec['score']:.3f})")
        
        # Test user-based recommendations
        test_user_id = 1
        user_recommendations = recommender.recommend(user_id=test_user_id, n_recommendations=5)
        print(f"\n🎯 Recommendations for User {test_user_id}:")
        for i, rec in enumerate(user_recommendations, 1):
            print(f"   {i}. {rec['title']} (Score: {rec['score']:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Content recommender test failed: {e}")
        traceback.print_exc()
        return False

def test_popularity_recommender(data):
    """Test popularity-based recommender"""
    print("\n🔄 Testing popularity-based recommender...")
    try:
        from src.recommenders.base import PopularityRecommender
        
        movies_df, ratings_df, links_df = data
        
        # Create and fit recommender
        recommender = PopularityRecommender()
        recommender.fit(ratings_df, movies_df)
        
        print("✅ Popularity-based recommender trained successfully!")
        
        # Get popular recommendations
        recommendations = recommender.recommend(n_recommendations=5)
        print(f"🏆 Most popular movies:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['title']} (Score: {rec['score']:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Popularity recommender test failed: {e}")
        traceback.print_exc()
        return False

def test_collaborative_recommender(data):
    """Test collaborative filtering recommender"""
    print("\n🔄 Testing collaborative filtering recommender...")
    try:
        from src.recommenders.collaborative import CollaborativeFilteringRecommender
        
        movies_df, ratings_df, links_df = data
        
        # Create and fit recommender
        recommender = CollaborativeFilteringRecommender(method='memory_based')
        recommender.fit(ratings_df, movies_df)
        
        print("✅ Collaborative filtering recommender trained successfully!")
        
        # Test user-based recommendations
        test_user_id = 1
        recommendations = recommender.recommend(user_id=test_user_id, n_recommendations=5)
        print(f"👥 Collaborative recommendations for User {test_user_id}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['title']} (Score: {rec['score']:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Collaborative recommender test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎬 Movie Recommendation System - Test Suite")
    print("=" * 50)
    
    # Test data loading
    data_success, data = test_data_loading()
    if not data_success:
        print("❌ Cannot proceed without data. Exiting.")
        return False
    
    # Test all recommenders
    content_success = test_content_recommender(data)
    popularity_success = test_popularity_recommender(data)
    collaborative_success = test_collaborative_recommender(data)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Data Loading: {'✅' if data_success else '❌'}")
    print(f"   Content-Based: {'✅' if content_success else '❌'}")
    print(f"   Popularity-Based: {'✅' if popularity_success else '❌'}")
    print(f"   Collaborative: {'✅' if collaborative_success else '❌'}")
    
    all_passed = all([data_success, content_success, popularity_success, collaborative_success])
    
    if all_passed:
        print("\n🎉 All tests passed! Movie recommendation system is working correctly!")
        print("\n🚀 Ready to run Streamlit app:")
        print("   Run: streamlit run streamlit_app/app.py")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)