import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

top_movie_ids = ratings['movieId'].value_counts().head(1000).index
top_user_ids = ratings['userId'].value_counts().head(1000).index

filtered_movies = movies[movies['movieId'].isin(top_movie_ids)].reset_index(drop=True)
filtered_ratings = ratings[
    ratings['movieId'].isin(top_movie_ids) & ratings['userId'].isin(top_user_ids)
]

filtered_movies['content_soup'] = filtered_movies['genres'].str.replace('|', ' ', regex=False)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filtered_movies['content_soup'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

os.makedirs("models", exist_ok=True)
with open("models/content_model.pkl", "wb") as f:
    pickle.dump((filtered_movies, cosine_sim), f)

user_item_matrix = filtered_ratings.pivot_table(
    index='userId', columns='movieId', values='rating'
).fillna(0)

svd = TruncatedSVD(n_components=20, random_state=42)
user_features = svd.fit_transform(user_item_matrix)
approx_ratings = np.dot(user_features, svd.components_)
predicted_df = pd.DataFrame(approx_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

with open("models/svd_model.pkl", "wb") as f:
    pickle.dump((predicted_df, filtered_movies), f)

print("Models saved successfully!")

