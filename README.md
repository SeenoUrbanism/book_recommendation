# Book Data Collection, Cleaning & Recommendation System  
**W10 · Ironhack Data Analytics**

This project builds a real, multi-source book dataset and develops a complete **content-based recommendation pipeline**, ending with an interactive **Streamlit app**.  
Scraped, API-based, and cleaned metadata is consolidated, vectorized, clustered, and then used for similarity-based recommendations.

---

##  Project Overview

The goal is to collect real book metadata, clean and merge heterogeneous sources, engineer meaningful text and genre features, and build a practical recommendation engine capable of suggesting books by:

- title similarity  
- author similarity  
- genre profile  
- rating  
- publication year  
- mixed similarity (combined signal)

The project avoids synthetic datasets and works with real, imperfect data to expose typical cleaning, merging, and modeling challenges.

---

##  1. Data Sources

### **1. Goodreads (Scraped)**
- Book titles  
- Authors  
- Ratings  
- Number of ratings  
- Book URLs  
- Genres (raw)  

Scraping performed using:
- `requests`
- `BeautifulSoup4`

### **2. Open Library API**
- Subjects / genres  
- Publication year  
- Additional metadata


---

##  2. Data Cleaning & Integration

The raw data comes in different formats and quality levels. Key processing steps:

### **Standardization**
- Lowercasing and text normalization  
- Whitespace cleanup  
- Deduplication by title + author  
- Repairing malformed genre strings  
- Normalizing publication years  
- Converting list-like strings into Python lists  

### **Genre Engineering**
- Extract, clean, and unify genre labels  
- Token normalization (`"sci-fi" → "fiction"`)  
- Build top-20 one-hot encoded genre matrix (`genre_fantasy`, `genre_romance`, etc.)

### **Feature Text Creation**
A text field optimized for TF-IDF


### **Final Merged Dataset**
File: `books_merged_cleaned.csv`  
Includes:
- Title  
- Author  
- Genres (raw + cleaned + list form)  
- Ratings  
- Number of ratings  
- Publication year  
- One-hot genre columns  
- `features_text`  
- Cleaned metadata from all sources  

---

##  3. Feature Engineering

### **TF-IDF Vectorization**
- max_features = 5000 (notebook version)  
- max_features = 50 (Streamlit version)  
- Stopwords removed (English)

### **Genre Matrix**
- 20 most frequent genres → binary vectors  
- Combined with TF-IDF using `scipy.sparse.hstack`

### **Additional Similarity Matrices**
Computed separately:
- `genre_sim` → cosine similarity on one-hot matrix  
- `rating_sim` → scaled rating distance  
- `year_sim` → scaled year distance  

Used to build flexible hybrid similarity.

---

##  4. Clustering Analysis

A full elbow + silhouette study was run for k = 2 to 20.

### **Findings**
- High-dimensional TF-IDF (early version) → noisy silhouette curve  
- Reduced feature space (20 genres + TF-IDF 50) → cleaner trend  
- Practical choice: **k = 6 clusters**

Clusters correspond roughly to groups such as:
- fantasy / YA  
- classics/literature  
- romance  
- non-fiction  
- historical  
- sci-fi
- ...

The assigned cluster is saved. 

##  5. Recommendation Engine

A multi-mode content-based recommender using cosine similarity.

### **Supported modes**
- Title-based similarity  
- Genre similarity  
- Rating-weighted similarity  
- Year-weighted similarity  
- A hybrid combined similarity

Returns the top-N relevant books with similarity scores.

---

## 6. Streamlit App

A full web interface for searching and browsing book recommendations.

### **Features**
- Search by **Title / Author / Genre / Keyword**  
- Select number of recommendations  
- Minimum rating filter  
- Optional genre filter  
- Shows similarity scores  
- Displays cluster label  
- Links to book & author pages  
- Uses the cleaned dataset directly

The app auto-builds:
- TF-IDF matrix  
- Genre matrix  
- Combined feature space  
- Cosine similarity  
- K-means clusters  

---

## Motivation

This project demonstrates the full workflow for building a real recommendation system:

- Web scraping  
- Multiple API integrations  
- Data cleaning + merging  
- Genre engineering  
- Text vectorization  
- Clustering diagnostics  
- Multi-criteria recommendation logic  
- Interactive deployment via Streamlit  

The focus is on **practical challenges**, not synthetic model accuracy.

---

## Output Summary

- `books_merged_cleaned.csv` (final unified dataset)  
- Full clustering diagnostics  
- Recommendation engine functions  
- Streamlit app (`book_rec_app.py`)  
- Jupyter notebooks covering all pipeline steps  

---

## Ethical Notes

- Only public Goodreads pages scraped  
- No circumvention of login/rate limits  
- API usage respects terms and quotas  
- Minimal request load, caching where possible  

---


