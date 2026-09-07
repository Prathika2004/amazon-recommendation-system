"""
Hybrid Recommendation Model - FOR SIMILAR USERS
Combines Collaborative Filtering and Content-Based Filtering
Optimized for showing overlap when users have similar tastes
"""

import numpy as np
from collections import defaultdict

class HybridRecommender:
    def __init__(self, cf_model, content_model, cf_weight=0.7, content_weight=0.3):
        """
        Initialize hybrid recommender
        CHANGED: 70% CF + 30% Content to prioritize similar user patterns
        """
        self.cf_model = cf_model
        self.content_model = content_model
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.interactions = None
        self.products = None
        
    def prepare_data(self, interactions, products):
        """Store data for recommendations"""
        self.interactions = interactions
        self.products = products
    def train(self):
   
        """
        Train the hybrid model (hybrid doesn't need training, 
        but we call it for consistency in main.py)
        """
        print("🔄 Preparing Hybrid Model...")
        if self.interactions is None or self.products is None:
            raise ValueError("Call prepare_data() first!")
        print("✅ Hybrid Model ready (both CF and Content-Based are already trained)")
    def recommend(self, user_id, n=10, unhide_items=None):
        """Generate hybrid recommendations - prioritizes CF for similar users

        unhide_items: items to keep eligible for recommendation even though the
        user rated them, and to exclude from the content-based seed set (used by
        held-out evaluation to test whether a hidden item gets recommended).
        """
        unhide_items = unhide_items or set()
        user_ratings = self.interactions[self.interactions['user_id'] == user_id]

        if user_ratings.empty:
            return []

        # Get CF recommendations (primary signal)
        try:
            cf_recs = self.cf_model.get_top_n_recommendations(user_id, n=n*2, unhide_items=unhide_items)
        except:
            cf_recs = []

        all_candidates = {}
        cf_scores = defaultdict(float)
        content_scores = defaultdict(float)

        # Add CF recommendations with HIGH weight (70%)
        for rank, (item_id, cf_score) in enumerate(cf_recs, 1):
            all_candidates[item_id] = cf_score
            cf_scores[item_id] = cf_score

        # Add content-based recommendations (secondary signal, 30%)
        # Held-out items are excluded from the seed set - using a hidden item as
        # its own seed would trivially "recommend" it back via self-similarity.
        seed_products = user_ratings[
            (user_ratings['rating'] >= 4.0) & (~user_ratings['product_id'].isin(unhide_items))
        ]['product_id'].tolist()

        if len(seed_products) == 0:
            seed_products = user_ratings[~user_ratings['product_id'].isin(unhide_items)]['product_id'].tolist()

        for user_product in seed_products:
            try:
                similar_items = self.content_model.get_similar_products(user_product, top_n=20)
                for rank, (similar_id, similarity) in enumerate(similar_items, 1):
                    if similar_id not in all_candidates:
                        all_candidates[similar_id] = 0.0
                    content_scores[similar_id] += similarity / len(seed_products)
            except:
                continue
        
        # Calculate final scores: 70% CF + 30% Content
        final_scores = {}
        for item_id in all_candidates:
            cf_score = cf_scores.get(item_id, 0.0)
            content_score = content_scores.get(item_id, 0.0)
            
            # 70% CF weight + 30% Content weight
            final_scores[item_id] = (
                self.cf_weight * cf_score +
                self.content_weight * content_score
            )
        
        # Filter out already rated products (except any held-out items being tested)
        if self.interactions is not None:
            user_rated = set(user_ratings['product_id'].tolist()) - unhide_items
            final_scores = {k: v for k, v in final_scores.items() if k not in user_rated}
        
        # Sort by score
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # No aggressive diversity filter - let top CF recommendations appear
        recommendations = []
        category_counts = defaultdict(int)
        
        for product_id, score in sorted_items:
            if len(recommendations) >= n:
                break
            
            product_info = self.products[self.products['product_id'] == product_id]
            if product_info.empty:
                continue
            
            # Allow more products per category (10 instead of 5)
            category = product_info.iloc[0].get('main_category', 'Unknown')
            if category_counts[category] < 10:
                recommendations.append((product_id, score))
                category_counts[category] += 1
        
        return recommendations
    def evaluate(self, n_eval_users=100, k=5, held_out_fraction=0.3, seed=42):
        """Evaluate hybrid model using held-out testing.

        For each sampled user, a fraction of their highly-rated (>=4.0) items
        is hidden from the exclusion filter that normally keeps already-rated
        items out of the recommendation list. Recommendations are then
        generated as if those items were never rated, and we check whether
        the model actually surfaces them - the standard way to test a
        recommender without the answer being trivially excluded by
        construction (which is what the previous version did: it measured
        overlap with already-rated items that recommend() always filters
        out, so precision/recall were guaranteed to be 0).

        Caveat: the CF model itself was fit on the full interaction set
        (including the held-out ratings) to serve predictions, so this is a
        simplified/offline-style evaluation rather than a fully leakage-free
        one - a fully rigorous version would retrain CF per fold.
        """
        print("📊 Evaluating Hybrid Model (held-out)...")
        rng = np.random.RandomState(seed)

        users = self.interactions['user_id'].unique()
        if len(users) > n_eval_users:
            users = rng.choice(users, size=n_eval_users, replace=False)

        precision_sum = 0.0
        recall_sum = 0.0
        hit_rate_sum = 0.0
        count = 0

        for user in users:
            try:
                user_ratings = self.interactions[self.interactions['user_id'] == user]
                highly_rated = user_ratings[user_ratings['rating'] >= 4.0]['product_id'].tolist()

                # Need at least 2 liked items: one to hold out, one left as a
                # real signal for CF/content-based to work from.
                if len(highly_rated) < 2:
                    continue

                n_hold_out = max(1, int(len(highly_rated) * held_out_fraction))
                held_out = set(rng.choice(highly_rated, size=n_hold_out, replace=False))

                recommendations = self.recommend(user, n=k, unhide_items=held_out)
                rec_items = set(r[0] for r in recommendations)
                hits = rec_items & held_out

                if len(rec_items) > 0:
                    precision = len(hits) / len(rec_items)
                    recall = len(hits) / len(held_out)
                    hit_rate = 1.0 if hits else 0.0

                    precision_sum += precision
                    recall_sum += recall
                    hit_rate_sum += hit_rate
                    count += 1
            except:
                continue

        if count == 0:
            return 0.0, 0.0, 0.0

        precision = precision_sum / count
        recall = recall_sum / count
        hit_rate = hit_rate_sum / count

        print(f" Precision@{k}: {precision:.4f}")
        print(f" Recall@{k}: {recall:.4f}")
        print(f" HitRate@{k}: {hit_rate:.4f}  (fraction of users with >=1 held-out item recommended)")
        print(f" Evaluated on {count} users (of {len(users)} sampled)")

        return precision, recall, hit_rate