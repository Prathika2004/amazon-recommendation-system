# Amazon Electronics Recommendation System

## Problem Statement
A large e-commerce catalog has two hard recommendation problems at once: most users have rated only a handful of products (extreme sparsity, so pure collaborative filtering struggles), and most products have no interaction history at all when they're new (the cold-start problem, which pure collaborative filtering can't solve). Neither collaborative filtering nor content-based filtering alone handles both cases well.

## What This Project Solves
A hybrid recommender for Amazon's Electronics category that combines:
- **Collaborative filtering** (SVD++) - learns from the sparse user-product rating matrix to capture "users like you also liked..." patterns.
- **Content-based filtering** - embeds product titles/descriptions into a vector database, so it can recommend semantically similar products even for items with little or no rating history.
- **A weighted hybrid** of both signals, so cold-start products still surface via content similarity while active users get personalized collaborative recommendations.

It also ships a Streamlit UI to generate recommendations per user and compare two users' recommendation lists side by side.

## Approach
- **Data pipeline** (`src/data_loader.py`, `src/preprocessing.py`): loads Amazon Electronics ratings + metadata, filters out users/products below an interaction threshold, encodes IDs.
- **Collaborative filtering** (`src/collaborative.py`): SVD++ via the `scikit-surprise` library, trained on the sparse interaction matrix, evaluated with a held-out train/test split (RMSE/MAE).
- **Content-based filtering** (`src/content_based.py`): product text is embedded with a sentence-transformer model and stored in a ChromaDB vector database for fast semantic similarity search - works over the *full* product catalog, including products with no ratings.
- **Hybrid model** (`src/hybrid_model.py`): combines both signals with configurable weights (default 70% CF / 30% content). Evaluated with a held-out test: a fraction of each user's liked items is hidden, recommendations are generated as if they'd never been rated, and precision/recall/hit-rate measure whether the model actually surfaces them.
- **Visualizations** (`src/visualization.py`): rating distribution, sparsity analysis, user-item interaction heatmap, model performance comparison - saved to `results/`.

## Tech Stack
Python, pandas, NumPy, scikit-surprise (SVD++), ChromaDB, sentence-transformers, scikit-learn, Streamlit, matplotlib/seaborn.

## Dataset
Due to GitHub file size limits, the dataset is hosted separately: [Google Drive link](https://drive.google.com/drive/folders/1UuR6lX9GsYOoktyLROcfUw09RqJeK2y5?usp=sharing). Download `Electronics.jsonl` and `meta_Electronics.jsonl` and place them in a `data/` folder in the project root.

## How to Run
```bash
pip install -r requirements.txt
```
Place the dataset files in `data/` as described above, then run the full pipeline (data loading -> training -> evaluation -> visualizations):
```bash
python main.py
```
Or launch the interactive UI directly:
```bash
streamlit run app.py
```

## Results
See `results/` for generated plots (rating distribution, sparsity analysis, user-item heatmap, model performance) and `evaluation_report.txt` (generated after running `main.py`) for the latest RMSE/MAE and hybrid precision/recall/hit-rate numbers.
