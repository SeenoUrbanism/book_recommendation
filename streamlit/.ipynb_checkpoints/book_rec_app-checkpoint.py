# ==============================================
# Hybrid Book Recommender (Enhanced Streamlit App)
# ==============================================
# This app uses your final dataset (books_final.csv)
# and your hybrid recommendation model:
#
#  - Title similarity (TF-IDF)
#  - Genre similarity (one-hot)
#  - Rating similarity (scaled)
#  - Year similarity (scaled)
#  - Cluster boost
#
# UI modeled after the example you provided:
#  - Sidebar interactions
#  - Search by title / author / genre / keyword
#  - Select matching book from a dropdown
#  - Show cluster & details
#  - Apply filters (genre/min rating)
#  - Display recommended books with scores
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------
st.set_page_config(page_title="Hybrid Book Recommender", page_icon="📘", layout="wide")

st.title("Hybrid Book Recommendation System")
st.write("This app uses a hybrid model combining **title, genres, rating, year, and clusters**.")

# -----------------------------------------------
# 2. LOAD FINAL DATASET
# -----------------------------------------------
DATA_PATH = Path(__file__).parent / "../data/clean/books_cluster.csv"

if not DATA_PATH.exists():
    st.error(f"Missing dataset: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.write("### Dataset loaded")
st.write("Shape:", df.shape)

# -----------------------------------------------
# 3. BASIC CLEANING
# -----------------------------------------------
df["title"] = df["title"].astype(str).str.strip()
df["author"] = df["author"].astype(str).str.strip()

df["title_lower"] = df["title"].str.lower()
df["author_lower"] = df["author"].str.lower()

# genres are already one-hot encoded + all_genres exists
genre_cols = [
    "activism","adult","bestseller","children/young adult","fantasy","fiction",
    "guide","history","inspirational","literature","non-fiction","other",
    "poetry","romance","science","spiritual/religious","sport"
]

# -----------------------------------------------
# 4. TEXT FEATURES (TF-IDF)
# -----------------------------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["features_text"])

cos_sim_title = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------------------------
# 5. GENRE SIMILARITY
# -----------------------------------------------
genre_matrix = df[genre_cols].values
cos_sim_genre = cosine_similarity(genre_matrix)

# -----------------------------------------------
# 6. RATING SIMILARITY
# -----------------------------------------------
scaler = MinMaxScaler()
df["rating_scaled"] = scaler.fit_transform(df[["avg_rating"]])
cos_sim_rating = cosine_similarity(df[["rating_scaled"]])

# -----------------------------------------------
# 7. YEAR SIMILARITY
# -----------------------------------------------
df["year_published"] = pd.to_numeric(df["year_published"], errors="coerce").fillna(2000).astype(int)
df["year_scaled"] = scaler.fit_transform(df[["year_published"]])
cos_sim_year = cosine_similarity(df[["year_scaled"]])

# -----------------------------------------------
# 8. SIDEBAR CONTROLS
# -----------------------------------------------
st.sidebar.header("🔍 Search & Controls")

search_mode = st.sidebar.selectbox(
    "Search by:",
    ["Title", "Author", "Genre", "Keyword"]
)

query = st.sidebar.text_input("Type your search:")

N = st.sidebar.slider("Number of recommendations", 3, 20, 10)

min_rating = st.sidebar.slider("Minimum rating", 0.0, 5.0, 0.0, step=0.1)

genre_filter = st.sidebar.multiselect("Filter by genre (optional):", sorted(genre_cols))

st.sidebar.write("---")
st.sidebar.write("Weights for hybrid scoring:")

w_title  = st.sidebar.slider("Title similarity",   0.0, 1.0, 0.45)
w_genre  = st.sidebar.slider("Genre similarity",   0.0, 1.0, 0.25)
w_rating = st.sidebar.slider("Rating influence",   0.0, 1.0, 0.15)
w_year   = st.sidebar.slider("Year influence",     0.0, 1.0, 0.10)
w_cluster = st.sidebar.slider("Cluster boost",     0.0, 1.0, 0.05)

# normalize weights
w_total = w_title + w_genre + w_rating + w_year + w_cluster
w_title, w_genre, w_rating, w_year, w_cluster = [w / w_total for w in [w_title, w_genre, w_rating, w_year, w_cluster]]

# -----------------------------------------------
# 9. FILTER BOOKS BY SEARCH MODE
# -----------------------------------------------
filtered_df = df.copy()

if query:
    q = query.lower().strip()

    if search_mode == "Title":
        mask = filtered_df["title_lower"].str.contains(q, na=False)
    elif search_mode == "Author":
        mask = filtered_df["author_lower"].str.contains(q, na=False)
    elif search_mode == "Genre":
        mask = filtered_df["all_genres"].str.lower().str.contains(q, na=False)
    else:  # Keyword
        mask = filtered_df["features_text"].str.lower().str.contains(q, na=False)

    filtered_df = filtered_df[mask]

# -----------------------------------------------
# 10. SELECT SPECIFIC BOOK FROM MATCHES
# -----------------------------------------------
st.write("### 1️⃣ Choose a book as the starting point")

if query and len(filtered_df) > 0:
    options = [
        (f"{row['title']} — {row['author']}", i)
        for i, row in filtered_df.iterrows()
    ]
    labels = [label for label, _ in options]

    selected_label = st.selectbox("Select a book:", options=labels)

    selected_index = next(idx for label, idx in options if label == selected_label)
else:
    st.write("Search for a book first using the sidebar.")
    selected_index = None

# -----------------------------------------------
# 11. SHOW BOOK DETAILS
# -----------------------------------------------
if selected_index is not None:
    book = df.loc[selected_index]

    st.write("### 2️⃣ Selected Book")
    colA, colB = st.columns([2, 3])

    with colA:
        st.markdown(f"**Title:** {book['title']}")
        st.markdown(f"**Author:** {book['author']}")
        st.markdown(f"**Rating:** {book['avg_rating']:.2f}")
        st.markdown(f"**Year:** {book['year_published']}")
        st.markdown(f"**Cluster:** {book['cluster']}")

    with colB:
        st.markdown("**Genres:**")
        st.write(book["all_genres"])

    # -------------------------------------------
    # 12. HYBRID RECOMMENDATIONS
    # -------------------------------------------
    st.write("### 3️⃣ Recommended Books")

    sim_title = cos_sim_title[selected_index]
    sim_genre = cos_sim_genre[selected_index]
    sim_rating = cos_sim_rating[selected_index]
    sim_year = cos_sim_year[selected_index]

    hybrid = (
        w_title  * sim_title +
        w_genre  * sim_genre +
        w_rating * sim_rating +
        w_year   * sim_year +
        w_cluster * (df["cluster"] == book["cluster"]).astype(int).values
    )

    # build ranking
    sim_scores = list(enumerate(hybrid))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recs = []
    for idx, score in sim_scores:
        if idx == selected_index:
            continue
        if df.loc[idx, "avg_rating"] < min_rating:
            continue
        if genre_filter:
            if not any(df.loc[idx, g] == 1 for g in genre_filter):
                continue
        recs.append((idx, score))
        if len(recs) >= N:
            break

    # show recommendations
    if len(recs) == 0:
        st.warning("No recommendations found. Try removing filters.")
    else:
        rows = []
        for idx, score in recs:
            row = df.loc[idx, [
                "title", "author", "avg_rating",
                "rating_count", "year_published",
                "all_genres", "cluster"
            ]].copy()
            row["similarity_score"] = round(float(score), 4)
            rows.append(row)

        result_df = pd.DataFrame(rows)
        st.dataframe(result_df)

# ----------------------------------------------------
# END OF APP
# ----------------------------------------------------
