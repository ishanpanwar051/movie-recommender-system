import pickle
import streamlit as st

st.header('Movie Recommender System')

try:
    movies = pickle.load(open('model/movie_list.pkl','rb'))
    similarity = pickle.load(open('model/similarity.pkl','rb'))
    st.write("Pickle files loaded successfully!")
    st.write(f"Movies shape: {movies.shape}")
    st.write(f"Similarity shape: {similarity.shape}")
    
    movie_list = movies['title'].values
    st.write(f"Number of movies: {len(movie_list)}")
    
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )
    
    st.write(f"Selected movie: {selected_movie}")
    
except Exception as e:
    st.error(f"Error loading pickle files: {e}")
    import traceback
    st.error(traceback.format_exc())
