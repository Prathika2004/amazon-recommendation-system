"""
UPGRADED Content-Based Filtering using Vector Database (ChromaDB)
✅ Persistent Storage (No need to re-train every time)
✅ Semantic Search (Understands meaning, not just keywords)
✅ Scalable to millions of products
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import os

class ContentBasedRecommender:
    """
    Content-Based Filtering using ChromaDB Vector Database.
    Uses Sentence Transformers for high-quality semantic embeddings.
    """
    
    def __init__(self, db_path="./chroma_db", collection_name="amazon_electronics"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # 1. Initialize Persistent Client (Saves data to disk)
        print(f"🔌 Connecting to ChromaDB at {self.db_path}...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 2. Setup Embedding Function (all-MiniLM-L6-v2 is fast and accurate)
        # This automatically downloads the model on first run
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 3. Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.ef,
            metadata={"description": "Amazon Electronics Products Embeddings"}
        )
        
    def fit(self, products_df):
        """
        Embeds product descriptions and stores them in the Vector DB.
        Includes a check to skip processing if data already exists.
        """
        # Check if DB is already populated to save time
        existing_count = self.collection.count()
        if existing_count >= len(products_df) * 0.9: # simple threshold check
            print(f"✅ Vector DB already contains {existing_count} items. Skipping embedding process.")
            print("   (To force update, delete the './chroma_db' folder)")
            return

        print(f"🚀 Populating Vector DB with {len(products_df)} products...")
        print("   This might take a while on the first run (using CPU)...")
        
        # Filter valid products
        valid_products = products_df[products_df['combined_features'].str.len() > 0].copy()
        
        # Batch processing settings
        batch_size = 100 
        total_products = len(valid_products)
        
        # Process in batches
        for i in tqdm(range(0, total_products, batch_size), desc="Upserting Vectors"):
            batch = valid_products.iloc[i:i+batch_size]
            
            # Prepare data for Chroma
            ids = batch['product_id'].astype(str).tolist()
            documents = batch['combined_features'].tolist()
            
            # Prepare metadata (useful for debugging/filtering later)
            metadatas = []
            for _, row in batch.iterrows():
                meta = {
                    "title": row['title'][:100],  # Truncate title to save space
                    "category": str(row['categories'])[:50]
                }
                metadatas.append(meta)
            
            # Upsert (Update or Insert) into DB
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                print(f"⚠️ Error batch {i}: {e}")
                continue

        print(f"✅ Vector DB population complete! Total count: {self.collection.count()}")

    def get_similar_products(self, product_id, top_n=10):
        """
        Query the Vector DB for similar products.
        """
        # Ensure product_id is string
        product_id = str(product_id)
        
        # 1. Fetch the source product's data to get its text/embedding
        # We need the text to query, or we can query by ID if we had the embedding.
        # Chroma's 'get' fetches the document text we stored.
        source_item = self.collection.get(ids=[product_id])
        
        if not source_item['documents']:
            print(f"❌ Product {product_id} not found in Vector DB.")
            return []
            
        query_text = source_item['documents'][0]
        
        # 2. Query the database
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_n + 1  # +1 because the result usually includes the query item itself
        )
        
        # 3. Parse results into format [(id, score), ...]
        recommendations = []
        
        if results['ids']:
            found_ids = results['ids'][0]
            distances = results['distances'][0]  # Chroma returns distances (lower is better)
            
            for pid, dist in zip(found_ids, distances):
                if pid == product_id:
                    continue  # Skip the input product itself
                
                # Convert Cosine Distance to Similarity Score (Approximation)
                # Distance ranges roughly 0 to 2. Score needs to be 0 to 1.
                # Score = 1 - (Distance / 2) is a decent approximation for cosine distance
                similarity_score = max(0, 1 - (dist / 2))
                
                recommendations.append((pid, similarity_score))
                
                if len(recommendations) >= top_n:
                    break
                    
        return recommendations

    def get_recommendations(self, user_liked_products, n=10):
        """
        Get recommendations based on a list of product IDs the user liked.
        """
        if not user_liked_products:
            return []

        # Convert to strings
        liked_ids = [str(pid) for pid in user_liked_products]
        
        # Fetch documents for these products
        data = self.collection.get(ids=liked_ids)
        liked_docs = data['documents']
        
        if not liked_docs:
            return []
            
        # Combine texts of liked products to form a "user profile" query
        # (Simple approach: Concatenate texts)
        combined_query = " ".join(liked_docs)[:1000] # Limit length
        
        results = self.collection.query(
            query_texts=[combined_query],
            n_results=n + len(liked_ids) # Fetch extra to filter out liked ones
        )
        
        recommendations = []
        if results['ids']:
            found_ids = results['ids'][0]
            distances = results['distances'][0]
            
            for pid, dist in zip(found_ids, distances):
                if pid in liked_ids:
                    continue
                    
                similarity_score = max(0, 1 - (dist / 2))
                recommendations.append((pid, similarity_score))
                
                if len(recommendations) >= n:
                    break
                    
        return recommendations