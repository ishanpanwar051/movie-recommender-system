import streamlit as st
import pickle
import pandas as pd
import requests
import os
import ast
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")
BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500/"

# Page Config
st.set_page_config(
    page_title="CineMatch | AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main background and font */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460);
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin-top: -50px;
        margin-bottom: 0rem;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
    }
    
    /* Glassmorphism card styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-align: center;
        overflow: hidden;
        margin-bottom: 20px;
    }
    
    .movie-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(0, 210, 255, 0.6);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .movie-poster {
        width: 100%;
        height: auto;
        border-radius: 20px 20px 0 0;
        transition: filter 0.3s ease;
    }
    
    .movie-card:hover .movie-poster {
        filter: brightness(1.1);
    }
    
    .movie-info {
        padding: 15px;
    }
    
    .movie-title {
        font-size: 1rem;
        font-weight: 700;
        color: #fff;
        height: 2.4rem;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-bottom: 5px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border-radius: 50px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.5);
        color: white;
    }
    
    /* Input styling */
    .stSelectbox [data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 25, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def fetch_movie_details(movie_id):
    try:
        url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            full_path = POSTER_BASE_URL + poster_path if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
            return {
                "poster": full_path,
                "rating": data.get('vote_average', 'N/A'),
                "release_date": data.get('release_date', 'N/A')[:4],
                "overview": data.get('overview', '')
            }
    except Exception as e:
        st.error(f"Error fetching details: {e}")
    return {
        "poster": "https://via.placeholder.com/500x750?text=Error",
        "rating": "N/A",
        "release_date": "N/A",
        "overview": ""
    }

def recommend(movie, movies_df, similarity_matrix):
    try:
        index = movies_df[movies_df['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        
        movie_data = []
        for i in distances[1:7]:
            movie_data.append({
                "id": movies_df.iloc[i[0]].movie_id,
                "title": movies_df.iloc[i[0]].title
            })
            
        def get_details(m):
            details = fetch_movie_details(m['id'])
            return {
                "title": m['title'],
                "poster": details['poster'],
                "rating": details['rating'],
                "year": details['release_date']
            }
            
        with ThreadPoolExecutor(max_workers=6) as executor:
            recommendations = list(executor.map(get_details, movie_data))
            
        return recommendations
    except Exception as e:
        st.error(f"Error in recommendation logic: {e}")
        return []

# App Header
st.markdown("<h1 class='main-header'>🎬 CineMatch</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #ccc;'>Discover your next favorite movie using AI-powered recommendations</p>", unsafe_allow_html=True)

# Data Loading with Error Handling
@st.cache_resource
def build_model():
    print("Loading datasets...")
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'popularity', 'vote_average']]
    movies.dropna(inplace=True)

    def convert(text):
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L

    def convert3(text):
        L = []
        counter = 0
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
            counter += 1
        return L

    def fetch_director(text):
        L = []
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
        return L

    def collapse(L):
        L1 = []
        for i in L:
            L1.append(i.replace(" ", ""))
        return L1

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
    new['tags'] = new['tags'].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new['tags']).toarray()
    similarity = cosine_similarity(vector)
    return new, similarity

@st.cache_resource
def load_data():
    try:
        # Check different possible paths
        paths = ['model/movie_list.pkl', 'movie_list.pkl']
        movies = None
        for path in paths:
            if os.path.exists(path):
                movies = pickle.load(open(path, 'rb'))
                break
        
        paths = ['model/similarity.pkl', 'similarity.pkl']
        similarity = None
        for path in paths:
            if os.path.exists(path):
                similarity = pickle.load(open(path, 'rb'))
                break
                
        if movies is not None and similarity is not None:
            return movies, similarity
        else:
            # Auto-generate model from CSV datasets
            print("Model files not found. Building from CSV datasets...")
            return build_model()
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        try:
            print("Attempting to build model from CSV datasets...")
            return build_model()
        except Exception as e2:
            st.error(f"Error building model: {e2}")
            return None, None

movies, similarity = load_data()

if movies is not None and similarity is not None:
    movie_list = movies['title'].values
    
    # Sidebar for filters or settings
    with st.sidebar:
        st.title("Top Rated 🏆")
        if 'vote_average' in movies.columns:
            top_rated = movies.sort_values(by='vote_average', ascending=False).head(5)
            for _, movie in top_rated.iterrows():
                st.write(f"⭐ {movie['vote_average']} | {movie['title']}")
        else:
            st.info("AI-powered filtering coming soon!")
        st.markdown("---")
        st.write("Developed with ❤️ for Movie Buffs")

    # Main search area
    col_search_left, col_search_mid, col_search_right = st.columns([1, 2, 1])
    with col_search_mid:
        selected_movie = st.selectbox(
            "Select a movie you liked:",
            movie_list,
            index=None,
            placeholder="Search for a movie..."
        )
        
        show_rec = st.button('Get Recommendations')

    if show_rec and selected_movie:
        with st.spinner('Generating recommendations...'):
            recommendations = recommend(selected_movie, movies, similarity)
            
            if recommendations:
                st.markdown("### Because you liked **{}**:".format(selected_movie))
                
                # Create a grid of 3x2 for 6 recommendations
                row1 = st.columns(3)
                row2 = st.columns(3)
                
                for idx, movie in enumerate(recommendations):
                    target_col = row1[idx] if idx < 3 else row2[idx-3]
                    with target_col:
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{movie['poster']}" class="movie-poster">
                            <div class="movie-info">
                                <div class="movie-title">{movie['title']}</div>
                                <div style="color: #00d2ff; font-weight: 800; font-size: 0.9rem;">
                                    ⭐ {movie['rating']} <span style="color: rgba(255,255,255,0.4); margin: 0 5px;">|</span> 📅 {movie['year']}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try another movie!")
    elif show_rec:
        st.warning("Please select a movie first!")

else:
    st.error("Model files not found! Please run `generate_model.py` first to generate the recommendation engine.")
    st.info("Make sure `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are in the project root.")
    
    if st.button("Download Data & Generate Model"):
        with st.spinner("Processing... this may take a few minutes."):
            # In a real production app, we would run the scripts here
            st.info("Please run `python download_data.py` and `python generate_model.py` in your terminal.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>© 2024 CineMatch AI. All rights reserved.</p>", unsafe_allow_html=True)
