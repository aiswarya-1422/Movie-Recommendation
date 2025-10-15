import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommender System")

with open("models/content_model.pkl", "rb") as f:
    movies_cb, cosine_sim = pickle.load(f)

with open("models/svd_model.pkl", "rb") as f:
    predicted_cf, movies_cf = pickle.load(f)

st.header("📌 Content-Based Recommendations")
selected_movie = st.selectbox("Select a movie:", movies_cb['title'].tolist())

if selected_movie:
    idx = movies_cb[movies_cb['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:6]]
    recommendations = movies_cb['title'].iloc[top_indices].tolist()
    st.subheader("Top 5 similar movies:")
    for i, title in enumerate(recommendations, 1):
        st.write(f"{i}. {title}")

st.header("👤 Collaborative Filtering Recommendations")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if user_id in predicted_cf.index:
    user_ratings = predicted_cf.loc[user_id]
    top_movies = user_ratings.sort_values(ascending=False).head(5).index
    titles = movies_cf[movies_cf['movieId'].isin(top_movies)]['title'].tolist()
    st.subheader("Top 5 recommended movies:")
    for i, title in enumerate(titles, 1):
        st.write(f"{i}. {title}")
elif user_id:
    st.warning("User ID not found in dataset.")

st.header("🔗 Hybrid Recommendation System")
col1, col2 = st.columns(2)

with col1:
    hybrid_user_id = st.number_input("Hybrid User ID:", min_value=1, step=1, key="hybrid_user")

with col2:
    hybrid_movie = st.selectbox("Favorite Movie:", movies_cb['title'].tolist(), key="hybrid_movie")

def hybrid_recommend(user_id, favorite_movie):

    if favorite_movie not in movies_cb['title'].values:
        return ["Favorite movie not found."]
    idx = movies_cb[movies_cb['title'] == favorite_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    cb_indices = [i[0] for i in sim_scores[1:11]]
    cb_titles = movies_cb['title'].iloc[cb_indices].tolist()

    if user_id not in predicted_cf.index:
        return ["User ID not found."]
    user_ratings = predicted_cf.loc[user_id]
    cf_movie_ids = user_ratings.sort_values(ascending=False).head(10).index
    cf_titles = movies_cb[movies_cb['movieId'].isin(cf_movie_ids)]['title'].tolist()

    combined = cb_titles + cf_titles
    final_list = []
    seen = set()
    for title in combined:
        if title not in seen:
            final_list.append(title)
            seen.add(title)
        if len(final_list) == 10:
            break
    return final_list

if st.button("Get Hybrid Recommendations"):
    hybrid_results = hybrid_recommend(hybrid_user_id, hybrid_movie)
    st.subheader("🎯 Final Hybrid Recommendations:")
    for i, title in enumerate(hybrid_results, 1):
        st.write(f"{i}. {title}")
