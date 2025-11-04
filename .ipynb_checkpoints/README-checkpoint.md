# Book Data Collection & Recommendation Project
W10

This project collects real-world book information from Goodreads to build a consolidated dataset for downstream analysis and recommendation tasks. The core goal is to extract structured metadata (e.g., title, author, rating, publication year) from web pages and APIs, then clean and unify it for later modeling.

## Scope
1. Scrape book data from:
   - Goodreads (selected shelf pages)

2. Pull additional structured metadata via APIs:
   - Google Books API
   - Open Library API

3. Combine scraped + API data (~1,000 books) into a single CSV/DB table suitable for:
   - Exploratory analysis
   - Recommender system prototyping
   - Visualizations

## Methods
- **Web scraping** using `requests` + `BeautifulSoup`.  
- **Data cleaning & merging** to standardize fields, deduplicate, and validate entries.

## Motivation
Recommendation systems shape how people discover media. Instead of using synthetic datasets, this project builds a practical dataset from the wild to understand:
- Inconsistencies across sources
- Metadata completeness/quality
- Constraints and biases of real scraping/API workflows

## Output
- Unified dataset of books with key attributes
- Basic statistics and exploratory visualizations
- Optional prototype of a simple recommender (content-based / collaborative)

## Ethical Notes
- Only public pages are accessed.
- Requests respect `robots.txt` and rate limits.
- APIs are preferred when available.

## Status
Data collection + cleaning planned. Further instructions pending.