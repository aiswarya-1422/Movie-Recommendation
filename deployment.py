
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

top_movies = ratings['movieId'].value_counts().head(1000).index
filtered_movies = movies[movies['movieId'].isin(top_movies)].reset_index(drop=True)

filtered_movies['content_soup'] = filtered_movies['genres'].str.replace('|', ' ', regex=False)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filtered_movies['content_soup'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
