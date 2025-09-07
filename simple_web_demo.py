#!/usr/bin/env python3
"""
Simple Web Demo for Movie Recommendation System
Creates an HTML interface that can be served statically
"""
import json
from demo_system import SimpleMovieData, SimpleContentRecommender, SimpleCollaborativeRecommender

def generate_html_demo():
    """Generate a complete HTML demo page"""
    
    # Initialize the recommendation system
    data = SimpleMovieData()
    content_rec = SimpleContentRecommender(data)
    collab_rec = SimpleCollaborativeRecommender(data)
    
    # Generate recommendations for different scenarios
    movie_similarities = {}
    user_recommendations = {}
    
    # Movie-to-movie similarities
    for movie in data.movies[:5]:  # First 5 movies
        movie_id = movie['movieId']
        similar = content_rec.recommend_similar_movies(movie_id, 5)
        movie_similarities[movie['title']] = similar
    
    # User recommendations
    for user_id in range(1, 6):  # First 5 users
        content_recs = content_rec.recommend_for_user(user_id, 5)
        collab_recs = collab_rec.recommend_for_user(user_id, 5)
        user_ratings = data.get_user_ratings(user_id)
        
        user_recommendations[user_id] = {
            'ratings_count': len(user_ratings),
            'content_based': content_recs,
            'collaborative': collab_recs,
            'top_rated_movies': [
                {
                    'title': data.get_movie_by_id(r['movieId'])['title'],
                    'rating': r['rating']
                }
                for r in sorted(user_ratings, key=lambda x: x['rating'], reverse=True)[:3]
            ]
        }
    
    # Calculate system statistics
    all_genres = set()
    for movie in data.movies:
        all_genres.update(movie['genres'].split('|'))
    
    all_ratings = [r['rating'] for r in data.ratings]
    avg_rating = sum(all_ratings) / len(all_ratings)
    
    rating_dist = {}
    for rating in all_ratings:
        rating_dist[rating] = rating_dist.get(rating, 0) + 1
    
    stats = {
        'total_movies': len(data.movies),
        'total_ratings': len(data.ratings),
        'total_users': len(set(r['userId'] for r in data.ratings)),
        'unique_genres': len(all_genres),
        'avg_rating': round(avg_rating, 2),
        'rating_distribution': rating_dist
    }
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 AI-Powered Movie Recommendation System</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            display: block;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9rem;
            margin-top: 5px;
        }}
        
        .demo-section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .demo-section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .movies-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .movie-card {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
        }}
        
        .movie-card:hover {{
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }}
        
        .movie-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .movie-genres {{
            color: #666;
            font-size: 0.85rem;
            margin-bottom: 8px;
        }}
        
        .movie-score {{
            color: #667eea;
            font-weight: bold;
        }}
        
        .genre-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 10px 0;
        }}
        
        .genre-tag {{
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
        }}
        
        .user-profile {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 0 8px 8px 0;
        }}
        
        .recommendations-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .tab-button {{
            background: #f0f0f0;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .tab-button.active {{
            background: #667eea;
            color: white;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .methodology {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        
        .methodology h3 {{
            margin-bottom: 15px;
            font-size: 1.5rem;
        }}
        
        .algorithm-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 50px;
            padding: 20px;
        }}
        
        .github-link {{
            display: inline-block;
            background: white;
            color: #667eea;
            padding: 12px 24px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
            margin-top: 20px;
            transition: all 0.3s ease;
        }}
        
        .github-link:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2rem;
            }}
            
            .container {{
                padding: 10px;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 AI Movie Recommender</h1>
            <p>Advanced Machine Learning for Personalized Movie Recommendations</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-number">{stats['total_movies']}</span>
                <div class="stat-label">Movies in Database</div>
            </div>
            <div class="stat-card">
                <span class="stat-number">{stats['total_ratings']}</span>
                <div class="stat-label">User Ratings</div>
            </div>
            <div class="stat-card">
                <span class="stat-number">{stats['total_users']}</span>
                <div class="stat-label">Active Users</div>
            </div>
            <div class="stat-card">
                <span class="stat-number">{stats['unique_genres']}</span>
                <div class="stat-label">Unique Genres</div>
            </div>
            <div class="stat-card">
                <span class="stat-number">{stats['avg_rating']}</span>
                <div class="stat-label">Average Rating</div>
            </div>
        </div>
        
        <div class="demo-section">
            <h2>🎯 Movie-to-Movie Similarity</h2>
            <p>Discover movies with similar content based on genre analysis and feature matching.</p>
            
            <div class="movies-grid">
    """
    
    # Add movie similarities
    for movie_title, similar_movies in list(movie_similarities.items())[:3]:
        html_content += f"""
                <div class="movie-card">
                    <div class="movie-title">Similar to "{movie_title}"</div>
        """
        for movie in similar_movies[:3]:
            genres = movie['genres'].split('|')
            genre_tags = ''.join(f'<span class="genre-tag">{genre}</span>' for genre in genres[:3])
            html_content += f"""
                    <div style="margin: 10px 0; padding: 8px; border-left: 3px solid #667eea;">
                        <div style="font-weight: bold;">{movie['title']}</div>
                        <div class="genre-tags">{genre_tags}</div>
                        <div class="movie-score">Similarity: {movie['similarity_score']:.3f}</div>
                    </div>
            """
        html_content += "</div>"
    
    html_content += """
            </div>
        </div>
        
        <div class="demo-section">
            <h2>👤 Personalized User Recommendations</h2>
            <p>Get tailored movie suggestions based on individual user preferences and behavior patterns.</p>
    """
    
    # Add user recommendations
    for user_id, user_data in list(user_recommendations.items())[:3]:
        html_content += f"""
            <div class="user-profile">
                <h3>User {user_id} Profile</h3>
                <p><strong>Total Ratings:</strong> {user_data['ratings_count']}</p>
                <p><strong>Top Rated Movies:</strong></p>
                <ul>
        """
        for movie in user_data['top_rated_movies'][:3]:
            html_content += f"<li>⭐ {movie['rating']}/5 - {movie['title']}</li>"
        
        html_content += """
                </ul>
                
                <div class="recommendations-tabs">
                    <button class="tab-button active" onclick="showTab('content-""" + str(user_id) + """')">Content-Based</button>
                    <button class="tab-button" onclick="showTab('collaborative-""" + str(user_id) + """')">Collaborative</button>
                </div>
                
                <div id="content-""" + str(user_id) + """" class="tab-content active">
                    <h4>Content-Based Recommendations</h4>
        """
        
        for rec in user_data['content_based'][:3]:
            genres = rec['genres'].split('|')
            genre_tags = ''.join(f'<span class="genre-tag">{genre}</span>' for genre in genres[:3])
            html_content += f"""
                    <div style="margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                        <div style="font-weight: bold;">{rec['title']}</div>
                        <div class="genre-tags">{genre_tags}</div>
                        <div class="movie-score">Score: {rec['preference_score']:.2f}</div>
                    </div>
            """
        
        html_content += f"""
                </div>
                
                <div id="collaborative-{user_id}" class="tab-content">
                    <h4>Collaborative Filtering Recommendations</h4>
        """
        
        if user_data['collaborative']:
            for rec in user_data['collaborative'][:3]:
                genres = rec['genres'].split('|')
                genre_tags = ''.join(f'<span class="genre-tag">{genre}</span>' for genre in genres[:3])
                html_content += f"""
                        <div style="margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                            <div style="font-weight: bold;">{rec['title']}</div>
                            <div class="genre-tags">{genre_tags}</div>
                            <div class="movie-score">Score: {rec['collaborative_score']:.2f}</div>
                        </div>
                """
        else:
            html_content += "<p>No collaborative recommendations available for this user.</p>"
        
        html_content += "</div></div>"
    
    # Continue with methodology and footer
    html_content += """
        </div>
        
        <div class="methodology">
            <h3>🔬 Recommendation Algorithms</h3>
            
            <div class="algorithm-card">
                <h4>🎯 Content-Based Filtering</h4>
                <p>Analyzes movie features (genres, metadata) to find similar content. Uses TF-IDF vectorization and cosine similarity to match user preferences with movie characteristics.</p>
                <ul>
                    <li>Genre-based similarity matching</li>
                    <li>User preference profiling</li>
                    <li>Feature extraction and analysis</li>
                </ul>
            </div>
            
            <div class="algorithm-card">
                <h4>👥 Collaborative Filtering</h4>
                <p>Leverages user behavior patterns to find similar users and recommend movies they enjoyed. Uses user-user similarity and rating pattern analysis.</p>
                <ul>
                    <li>User similarity computation</li>
                    <li>Rating pattern analysis</li>
                    <li>Community-based recommendations</li>
                </ul>
            </div>
            
            <div class="algorithm-card">
                <h4>🔄 Hybrid Approach</h4>
                <p>Combines multiple algorithms to provide robust recommendations that handle both content preferences and collaborative patterns.</p>
                <ul>
                    <li>Multi-algorithm combination</li>
                    <li>Cold start problem handling</li>
                    <li>Improved accuracy and coverage</li>
                </ul>
            </div>
        </div>
        
        <div class="demo-section">
            <h2>📊 Technical Implementation</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div>
                    <h4>🐍 Backend Technologies</h4>
                    <ul>
                        <li>Python 3.8+</li>
                        <li>scikit-learn</li>
                        <li>pandas & NumPy</li>
                        <li>Streamlit</li>
                    </ul>
                </div>
                <div>
                    <h4>🧠 ML Algorithms</h4>
                    <ul>
                        <li>TF-IDF Vectorization</li>
                        <li>Cosine Similarity</li>
                        <li>Matrix Factorization</li>
                        <li>Collaborative Filtering</li>
                    </ul>
                </div>
                <div>
                    <h4>🎨 Frontend Features</h4>
                    <ul>
                        <li>Interactive Web Interface</li>
                        <li>Real-time Recommendations</li>
                        <li>Data Visualizations</li>
                        <li>Responsive Design</li>
                    </ul>
                </div>
                <div>
                    <h4>📈 Evaluation Metrics</h4>
                    <ul>
                        <li>Precision & Recall</li>
                        <li>NDCG Scores</li>
                        <li>Coverage Analysis</li>
                        <li>Diversity Measurements</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <h3>🚀 Ready for Production Deployment</h3>
            <p>This demonstration showcases a complete end-to-end machine learning system for movie recommendations.</p>
            <p>Built with modern ML practices, comprehensive evaluation, and production-ready architecture.</p>
            
            <a href="https://github.com/your-username/movie-recommendation-system" class="github-link">
                📁 View Full Project on GitHub
            </a>
            
            <div style="margin-top: 30px; font-size: 0.9rem; opacity: 0.8;">
                <p>Perfect for ML portfolios, educational purposes, and real-world recommendation system development.</p>
                <p>© 2024 AI-Powered Movie Recommendation System</p>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {
            // Hide all tab contents
            const allTabs = document.querySelectorAll('.tab-content');
            allTabs.forEach(tab => tab.classList.remove('active'));
            
            // Remove active class from all buttons
            const allButtons = document.querySelectorAll('.tab-button');
            allButtons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
        
        // Add some interactive animations
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.stat-card, .movie-card');
            cards.forEach(card => {
                card.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-5px) scale(1.02)';
                });
                card.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0) scale(1)';
                });
            });
        });
    </script>
</body>
</html>
    """
    
    return html_content

def main():
    """Generate and save the HTML demo"""
    print("🔄 Generating interactive web demo...")
    
    html_content = generate_html_demo()
    
    # Save to file
    with open('movie_recommendation_demo.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Web demo generated successfully!")
    print("📁 File saved as: movie_recommendation_demo.html")
    print("\n🌐 To view the demo:")
    print("   1. Open 'movie_recommendation_demo.html' in your web browser")
    print("   2. Or serve it using a simple HTTP server:")
    print("      python3 -m http.server 8000")
    print("      Then visit: http://localhost:8000/movie_recommendation_demo.html")
    
    return True

if __name__ == "__main__":
    main()