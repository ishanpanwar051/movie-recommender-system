import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Loading datasets...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print(f"Movies shape: {movies.shape}")
print(f"Credits shape: {credits.shape}")

print("Merging datasets...")
movies = movies.merge(credits, on='title')

print("Selecting columns...")
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'popularity', 'vote_average']]

print("Dropping missing values...")
movies.dropna(inplace=True)

print("Converting genres...")
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

print("Converting keywords...")
movies['keywords'] = movies['keywords'].apply(convert)

print("Converting cast (top 3)...")
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert3)

print("Fetching director...")
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

print("Removing spaces from names...")
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

print("Processing overview...")
movies['overview'] = movies['overview'].apply(lambda x: x.split())

print("Creating tags...")
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

print("Creating new dataframe...")
new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

print(f"Final dataframe shape: {new.shape}")

print("Vectorizing tags...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

print(f"Vector shape: {vector.shape}")

print("Calculating cosine similarity...")
similarity = cosine_similarity(vector)

print(f"Similarity matrix shape: {similarity.shape}")

print("Saving pickle files...")
if not os.path.exists('model'):
    os.makedirs('model')
pickle.dump(new, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

print("Model files generated successfully!")
print("Files created in model/ directory: movie_list.pkl, similarity.pkl")
