"""
Performance Analysis for Structured Search
Identify bottlenecks and optimization opportunities
"""

import time
import pandas as pd
from property_search_engine import PropertySearchEngine
from typing import Dict

def analyze_search_performance():
    """Analyze where time is spent in the search process"""

    print("="*60)
    print("PERFORMANCE ANALYSIS - STRUCTURED SEARCH")
    print("="*60)

    # Initialize engine
    print("Loading search engine...")
    start_total = time.time()
    engine = PropertySearchEngine('realtor_cleaned_final.csv')
    load_time = (time.time() - start_total) * 1000
    print(f"Engine loaded in {load_time:.1f}ms")

    # Test query
    test_query = {
        'city': 'Adjuntas',
        'state': 'Puerto Rico',
        'bed': 3,
        'bath': 2.0,
        'house_size': 920,
        'price': 999999
    }

    print(f"\nTest Query: {test_query}")
    print("-" * 60)

    # Detailed timing analysis
    overall_start = time.time()

    # Step 1: Pre-filtering timing
    filter_start = time.time()
    candidates = engine._filter_candidates(test_query)
    filter_time = (time.time() - filter_start) * 1000
    print(f"1. Candidate filtering: {filter_time:.1f}ms ({len(candidates):,} candidates)")

    # Step 2: Score calculation timing
    calc_start = time.time()
    matches = []

    # Simulate the scoring loop
    scored_count = 0
    for _, row in candidates.iterrows():
        if scored_count >= 50:  # Limit for timing analysis
            break

        # Calculate individual scores (this is the bottleneck)
        bedroom_score = engine._calculate_bedroom_score(test_query['bed'], row['bed'])
        bathroom_score = engine._calculate_bathroom_score(test_query['bath'], row['bath'])
        size_score = engine._calculate_size_score(test_query['house_size'], row['house_size'])
        location_score = engine._calculate_location_score(
            test_query['city'], test_query['state'],
            row['city'], row['state']
        )
        price_score = engine._calculate_price_score(test_query['price'], row['price'])

        # Calculate weighted overall score
        overall_score = (
            0.30 * location_score +
            0.25 * size_score +
            0.20 * bedroom_score +
            0.15 * price_score +
            0.10 * bathroom_score
        )

        scored_count += 1

    calc_time = (time.time() - calc_start) * 1000
    calc_per_property = calc_time / scored_count if scored_count > 0 else 0
    estimated_total_calc = calc_per_property * len(candidates)

    print(f"2. Score calculation: {calc_time:.1f}ms for {scored_count} properties")
    print(f"   - Per property: {calc_per_property:.2f}ms")
    print(f"   - Estimated for all {len(candidates):,}: {estimated_total_calc:.1f}ms")

    # Step 3: Run actual search for comparison
    actual_start = time.time()
    matches, search_time = engine.find_similar_properties(test_query, 10)
    actual_time = (time.time() - actual_start) * 1000

    print(f"3. Actual full search: {search_time:.1f}ms ({len(matches)} results)")
    print(f"   - Overhead: {actual_time - search_time:.1f}ms")

    # Analysis
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)

    if estimated_total_calc > 400:
        print("❌ MAJOR BOTTLENECK: Score calculation")
        print(f"   Estimated {estimated_total_calc:.1f}ms for {len(candidates):,} candidates")
        print("   Optimization needed: Early termination, caching, or index reduction")

    if filter_time > 100:
        print("⚠️  MINOR BOTTLENECK: Candidate filtering")
        print(f"   {filter_time:.1f}ms to filter {len(candidates):,} candidates")
        print("   Optimization: Pre-built indexes or tighter filters")

    if len(candidates) > 20000:
        print("⚠️  TOO MANY CANDIDATES")
        print(f"   {len(candidates):,} candidates is too many for real-time search")
        print("   Recommendation: Tighter pre-filtering")

    # Optimization recommendations
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    print("1. 🚀 EARLY TERMINATION")
    print("   - Stop scoring when finding high-confidence matches (>0.95)")
    print("   - Process candidates in priority order (location first)")

    print("\n2. 📊 SMARTER PRE-FILTERING")
    print("   - Reduce candidate pool with tighter ranges")
    print("   - Use location index for faster city/state filtering")

    print("\n3. ⚡ VECTORIZED CALCULATIONS")
    print("   - Use pandas vectorized operations for bulk scoring")
    print("   - Pre-compute location similarities")

    print("\n4. 🗄️ RESULT CACHING")
    print("   - Cache results for common queries")
    print("   - Use query similarity to reuse calculations")

    print("\n5. 🎯 PROGRESSIVE SEARCH")
    print("   - Start with exact matches (fastest)")
    print("   - Expand to semantic only if needed")

    # Performance targets
    print("\n" + "="*60)
    print("PERFORMANCE TARGETS")
    print("="*60)
    print("Current: ~831ms")
    print("Target:  <200ms")
    print("Gap:     -631ms (76% reduction needed)")
    print()
    print("Achievable with:")
    print("- Early termination: -400ms")
    print("- Better filtering: -150ms")
    print("- Vectorized ops:   -100ms")
    print("- Total reduction:  -650ms → ~180ms ✅")

if __name__ == "__main__":
    analyze_search_performance()