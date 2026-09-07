
from src.visualization import RecommendationVisualizer
#
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringRecommender
from src.hybrid_model import HybridRecommender
from src.evaluation import ModelEvaluator

warnings.filterwarnings('ignore')

def print_header(title):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def main():
    print_header("PRODUCTION AMAZON ELECTRONICS RECOMMENDATION SYSTEM")
    print(f"Version: 2.0 (Optimized)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. DATA LOADING (More data = Better accuracy)
    # ═══════════════════════════════════════════════════════════════════
    print_header("STEP 1: DATA LOADING")
    
    loader = DataLoader(
        ratings_path='data/Electronics.jsonl',
        metadata_path='data/meta_Electronics.jsonl'
    )
    
    # Load 2M ratings for maximum accuracy
    interactions = loader.load_ratings(sample_size=2000000)
    products = loader.load_metadata()
    
    # CRITICAL: Save FULL catalog before filtering
    products_full = products.copy()
    print(f"📦 Full product catalog: {len(products_full):,} products")
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. PREPROCESSING
    # ═══════════════════════════════════════════════════════════════════
    print_header("STEP 2: DATA PREPROCESSING")
    
    preprocessor = DataPreprocessor()
    products = preprocessor.prepare_metadata(products)
    products_full = preprocessor.prepare_metadata(products_full)  # Prepare full catalog
    
    common_products = set(products['product_id']) & set(interactions['product_id'])
    interactions = interactions[interactions['product_id'].isin(common_products)]
    products = products[products['product_id'].isin(common_products)]
    
    # Lower thresholds for more data
    interactions = preprocessor.filter_sparse_data(
        interactions,
        min_user_interactions=2,
        min_product_interactions=2
    )
    
    interactions = preprocessor.encode_ids(interactions)
    
    common_final = set(products['product_id']) & set(interactions['product_id'])
    products = products[products['product_id'].isin(common_final)]
    interactions = interactions[interactions['product_id'].isin(common_final)]
    
    print(f"\n{'Final Dataset Statistics':^80}")
    print("-" * 80)
    print(f"Full Catalog Products: {len(products_full):,}")
    print(f"Filtered Products (for CF): {len(products):,}")
    print(f"Users: {interactions['user_id'].nunique():,}")
    print(f"Interactions: {len(interactions):,}")
    print(f"Sparsity: {1 - len(interactions)/(interactions['user_id'].nunique()*len(products)):.4%}")
    print("-" * 80)
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. CONTENT-BASED (Enhanced) - FULL CATALOG
    # ═══════════════════════════════════════════════════════════════════
    print_header("STEP 3: CONTENT-BASED FILTERING (FULL CATALOG)")
    
    content_recommender = ContentBasedRecommender()
    
    try:
        content_recommender.fit(products_full)  # ← CORRECTED: Using full catalog
        print(f"✅ Content-based model ready! ({len(products_full):,} products)")
    except Exception as e:
        print(f"⚠️ Content-based failed: {e}")
        content_recommender = None
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. COLLABORATIVE FILTERING (Fixed) - SPARSE DATA
    # ═══════════════════════════════════════════════════════════════════
    print_header("STEP 4: COLLABORATIVE FILTERING (SPARSE DATA)")
    
    cf_recommender = CollaborativeFilteringRecommender(
        n_factors=150,
        n_epochs=30,
        lr_all=0.007,
        reg_all=0.02
    )
    
    cf_recommender.prepare_data(interactions, test_size=0.2)
    cf_recommender.train()
    rmse, mae, predictions = cf_recommender.evaluate()
    
    print(f"\n{'Performance Metrics':^80}")
    print("-" * 80)
    print(f"RMSE: {rmse:.4f} {'✅' if rmse < 1.2 else '⚠️'} (Target: < 1.2)")
    print(f"MAE: {mae:.4f} {'✅' if mae < 1.0 else '⚠️'} (Target: < 1.0)")
    print("-" * 80)
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. HYBRID MODEL (Optimized)
    # ═══════════════════════════════════════════════════════════════════
    print_header("STEP 5: HYBRID MODEL (COMBINING BOTH)")
    
    if content_recommender:
        hybrid_recommender = HybridRecommender(
            cf_recommender,
            content_recommender,
            cf_weight=0.8,
            content_weight=0.2
        )
        hybrid_recommender.prepare_data(interactions, products_full)  # ← CORRECTED: Full catalog
        hybrid_recommender.train()
        precision, recall, hit_rate = hybrid_recommender.evaluate()

        print(f"\n{'Hybrid Performance (held-out evaluation)':^80}")
        print("-" * 80)
        print(f"Precision@5: {precision:.4f} {'✅' if precision > 0.10 else '⚠️'} (Target: > 0.10)")
        print(f"Recall@5: {recall:.4f} {'✅' if recall > 0.15 else '⚠️'} (Target: > 0.15)")
        print(f"HitRate@5: {hit_rate:.4f} {'✅' if hit_rate > 0.30 else '⚠️'} (Target: > 0.30)")
        print("-" * 80)
    else:
        hybrid_recommender = None
        precision = recall = hit_rate = 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # 6. EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    print_header("STEP 6: COMPREHENSIVE EVALUATION")
    
    evaluator = ModelEvaluator()
    cf_metrics = evaluator.evaluate_collaborative_filtering(cf_recommender)
    
    if content_recommender:
        cb_metrics = evaluator.evaluate_content_based(content_recommender, products_full, sample_size=200)
    
    if hybrid_recommender:
        hybrid_metrics = evaluator.evaluate_hybrid(hybrid_recommender)
    
    evaluator.compare_models()
    evaluator.generate_report('evaluation_report.txt')
     
     # ═══════════════════════════════════════════════════════════════════
    # STEP 6: VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════
    print_header("STEP 6: GENERATING VISUALIZATIONS")
    
    visualizer = RecommendationVisualizer(results_dir='results')
    
    print("\n📊 Creating visualizations...")
    visualizer.plot_model_performance(evaluator.results)
    visualizer.plot_rating_distribution(interactions)
    visualizer.plot_user_item_matrix_heatmap(interactions, sample_users=30, sample_items=30)
    visualizer.plot_sparsity_analysis(interactions)
    visualizer.plot_recommendation_quality(hybrid_recommender, interactions)
    visualizer.plot_top_rated_products(interactions, products)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS SAVED TO results/ FOLDER!".center(80))
    print("="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    # ═══════════════════════════════════════════════════════════════════
    # 7. FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print_header("🎉 FINAL RESULTS")
    
    print(f"\n{'Architecture Summary':^80}")
    print("=" * 80)
    print(f"{'Component':<30} {'Data Used':<25} {'Products':<15}")
    print("-" * 80)
    print(f"{'Collaborative Filtering':<30} {'Sparse (filtered)':<25} {len(products):>14,}")
    print(f"{'Content-Based':<30} {'Full catalog':<25} {len(products_full):>14,}")
    print(f"{'Hybrid Model':<30} {'Combined':<25} {len(products_full):>14,}")
    print("=" * 80)
    
    # Note: this run compares against a fixed baseline captured from an
    # earlier CF-only run (data/methodology unchanged for RMSE/MAE, so this
    # comparison is valid). The hybrid Precision/Recall figures from earlier
    # runs are NOT comparable here: the evaluation methodology itself changed
    # (see HybridRecommender.evaluate) from a formula that was guaranteed to
    # return 0 to a real held-out test, so there is no valid "before" to
    # compare the new numbers against.
    baseline_rmse, baseline_mae = 3.56, 3.38
    print(f"\n{'CF Model vs Earlier Baseline Run':^80}")
    print("=" * 80)
    print(f"{'Metric':<20} {'Baseline':>15} {'This Run':>15} {'Change':>15}")
    print("-" * 80)
    print(f"{'RMSE':<20} {baseline_rmse:>15.2f} {rmse:>15.2f} {(baseline_rmse-rmse)/baseline_rmse*100:>14.1f}%")
    print(f"{'MAE':<20} {baseline_mae:>15.2f} {mae:>15.2f} {(baseline_mae-mae)/baseline_mae*100:>14.1f}%")
    print("=" * 80)

    # Honest pass/fail summary - no fallback branch that always reports success
    cf_ok = rmse < 1.2 and mae < 1.0
    hybrid_ok = hybrid_recommender is not None and (precision > 0.10 or hit_rate > 0.30)

    print(f"\n{'Checks':^80}")
    print("=" * 80)
    print(f"{'Collaborative filtering (RMSE < 1.2, MAE < 1.0)':<55} {'✅ PASS' if cf_ok else '❌ FAIL'}")
    print(f"{'Hybrid model (Precision@5 > 0.10 or HitRate@5 > 0.30)':<55} {'✅ PASS' if hybrid_ok else '❌ FAIL'}")
    print("=" * 80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🚀 Run 'streamlit run app.py' to launch web interface!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'ERROR':^80}")
        print("=" * 80)
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
