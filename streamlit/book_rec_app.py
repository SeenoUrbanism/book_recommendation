import numpy as np

def recommend(
    query,
    df,
    cosine_sim,      # title/text similarity
    genre_sim,       # genre one-hot similarity
    rating_sim,      # rating similarity
    year_sim,        # year similarity
    n=10,
    w_title=0.45,
    w_genre=0.25,
    w_rating=0.15,
    w_year=0.10,
    w_cluster=0.05   # boost for same-cluster books
):
    """
    Unified hybrid recommendation system:
    - Fuzzy title matching
    - Title (TF-IDF) similarity
    - Genre similarity
    - Rating similarity
    - Published year similarity
    - Cluster-aware boosting
    """

    # ---------------------------------------------------------
    # 1. Fuzzy match the title
    # ---------------------------------------------------------
    q = query.lower().strip()
    matches = df[df["title"].str.lower().str.contains(q)]

    if matches.empty:
        return f"No books found for '{query}'. Try another keyword."

    # take first match
    idx = matches.index[0]
    book_cluster = df.loc[idx, "cluster"]

    # ---------------------------------------------------------
    # 2. Extract all similarity vectors for the book
    # ---------------------------------------------------------
    sim_title_vec  = cosine_sim[idx]
    sim_genre_vec  = genre_sim[idx]
    sim_rating_vec = rating_sim[idx]
    sim_year_vec   = year_sim[idx]

    # ---------------------------------------------------------
    # 3. Normalize weights (optional but improves stability)
    # ---------------------------------------------------------
    w_sum = w_title + w_genre + w_rating + w_year + w_cluster
    w_title  /= w_sum
    w_genre  /= w_sum
    w_rating /= w_sum
    w_year   /= w_sum
    w_cluster /= w_sum

    # ---------------------------------------------------------
    # 4. Compute hybrid score for all books
    # ---------------------------------------------------------
    hybrid_score = (
        w_title  * sim_title_vec +
        w_genre  * sim_genre_vec +
        w_rating * sim_rating_vec +
        w_year   * sim_year_vec
    )

    # ---------------------------------------------------------
    # 5. Apply cluster-aware boosting
    # ---------------------------------------------------------
    same_cluster = (df["cluster"] == book_cluster).astype(int)
    hybrid_score += w_cluster * same_cluster

    # ---------------------------------------------------------
    # 6. Sort by final hybrid score
    # ---------------------------------------------------------
    indices_sorted = np.argsort(hybrid_score)[::-1]

    # remove the input book itself
    indices_sorted = [i for i in indices_sorted if i != idx]

    # take top N
    top_indices = indices_sorted[:n]

    # ---------------------------------------------------------
    # 7. Return final results
    # ---------------------------------------------------------
    return df.iloc[top_indices][[
        "title",
        "author",
        "avg_rating",
        "rating_count",
        "year_published",
        "all_genres",
        "cluster"
    ]]
