"""
Debug script to investigate search accuracy issues
"""
import pandas as pd
import os
from property_search_engine import PropertySearchEngine

def debug_exact_search():
    """Debug the exact search functionality"""
    print("🔍 Debugging exact search functionality...")

    # Check if we have embeddings
    if not os.path.exists('embeddings'):
        print("❌ No embeddings found. Run: python build_embeddings.py --sample 100000")
        return

    # Load the search engine
    try:
        engine = PropertySearchEngine('realtor_cleaned_final.csv')
        print("✅ Search engine loaded successfully")
    except Exception as e:
        print(f"❌ Error loading search engine: {e}")
        return

    # First, let's look at some actual data from our dataset
    print("\n📊 Sample properties in our dataset:")
    print("=" * 60)

    # Show a few sample properties from Adjuntas
    adjuntas_properties = engine.properties_df[
        engine.properties_df['city'].str.lower().str.contains('adjuntas', na=False)
    ].head(5)

    if len(adjuntas_properties) > 0:
        print(f"Found {len(adjuntas_properties)} Adjuntas properties:")
        for idx, row in adjuntas_properties.iterrows():
            print(f"  - {row['bed']}br/{row['bath']}ba, {row['house_size']}sqft, ${row['price']:,} in {row['city']}, {row['state']}")
    else:
        print("❌ No Adjuntas properties found in dataset")

    # Show first few properties overall
    print(f"\nFirst 5 properties in dataset:")
    for idx, row in engine.properties_df.head(5).iterrows():
        print(f"  - {row['bed']}br/{row['bath']}ba, {row['house_size']}sqft, ${row['price']:,} in {row['city']}, {row['state']}")

    # Test exact search with a property that should exist
    print("\n🎯 Testing exact search...")
    print("=" * 40)

    if len(adjuntas_properties) > 0:
        # Use the first Adjuntas property as a test case
        test_property = adjuntas_properties.iloc[0]
        print(f"Testing with: {test_property['bed']}br/{test_property['bath']}ba, {test_property['house_size']}sqft in {test_property['city']}, {test_property['state']}")

        # Create exact test case
        test_listing = {
            'city': test_property['city'],
            'state': test_property['state'],
            'bed': test_property['bed'],
            'bath': test_property['bath'],
            'house_size': test_property['house_size'],
            'price': test_property['price']
        }

        # Test exact search method directly
        exact_matches = engine._exact_search(test_listing)
        print(f"Exact search found: {len(exact_matches)} matches")
        for match in exact_matches:
            print(f"  ✅ {match.bedrooms}br/{match.bathrooms}ba, {match.house_size}sqft, ${match.price:,} in {match.city}, {match.state} (score: {match.similarity_score})")

        # Test full search
        print("\nFull search results:")
        all_matches = engine.find_duplicates(test_listing)
        print(f"Total matches found: {len(all_matches)}")
        for i, match in enumerate(all_matches[:3]):
            print(f"  {i+1}. {match.bedrooms}br/{match.bathrooms}ba, {match.house_size}sqft, ${match.price:,} in {match.city}, {match.state} (score: {match.similarity_score:.3f}, type: {match.match_type})")

    # Test with the problematic case you mentioned
    print("\n🔍 Testing with your problematic case...")
    print("=" * 45)

    problem_listing = {
        'city': 'Adjuntas',
        'state': 'Puerto Rico',
        'bed': 3,
        'bath': 2,
        'house_size': 920,
        'price': 105000
    }

    # Show exact key generation
    exact_key = (
        f"{problem_listing['city'].lower().strip()}_"
        f"{problem_listing['state'].lower().strip()}_"
        f"{problem_listing['bed']}br_"
        f"{problem_listing['bath']}ba_"
        f"{problem_listing['house_size']}sqft"
    )
    print(f"Generated exact key: '{exact_key}'")

    # Check if this key exists in our lookup
    if hasattr(engine, 'exact_lookup'):
        key_exists = exact_key in engine.exact_lookup
        print(f"Key exists in exact_lookup: {key_exists}")
        if key_exists:
            indices = engine.exact_lookup[exact_key]
            print(f"Indices for this key: {indices}")

    # Test exact search
    exact_matches = engine._exact_search(problem_listing)
    print(f"Exact matches found: {len(exact_matches)}")

    # Test full search
    all_matches = engine.find_duplicates(problem_listing, top_k=10)
    print(f"All matches found: {len(all_matches)}")

    for i, match in enumerate(all_matches[:5]):
        print(f"  {i+1}. {match.bedrooms}br/{match.bathrooms}ba, {match.house_size}sqft, ${match.price:,} in {match.city}, {match.state} (score: {match.similarity_score:.3f}, type: {match.match_type})")

def debug_data_alignment():
    """Check if there are data alignment issues"""
    print("\n🔍 Checking data alignment between FAISS and DataFrame...")
    print("=" * 55)

    # Load the search engine
    engine = PropertySearchEngine('realtor_cleaned_final.csv')

    # Check if we have the mapping files
    embedding_files = [
        'embeddings/faiss_index_v1.index',
        'embeddings/property_ids_v1.npy',
        'embeddings/sampled_properties_v1.csv'
    ]

    for file in embedding_files:
        exists = os.path.exists(file)
        print(f"{'✅' if exists else '❌'} {file}")

    if os.path.exists('embeddings/property_ids_v1.npy'):
        import numpy as np
        property_ids = np.load('embeddings/property_ids_v1.npy')
        print(f"Property IDs array shape: {property_ids.shape}")
        print(f"DataFrame shape: {engine.properties_df.shape}")
        print(f"FAISS index total: {engine.faiss_index.ntotal}")

        # Check if they're aligned
        if len(property_ids) == len(engine.properties_df) == engine.faiss_index.ntotal:
            print("✅ Sizes are aligned")
        else:
            print("❌ Size mismatch detected!")

        # Check first few property IDs
        print(f"First 5 property IDs: {property_ids[:5]}")
        print(f"DataFrame index range: {engine.properties_df.index.min()} - {engine.properties_df.index.max()}")

if __name__ == "__main__":
    debug_exact_search()
    debug_data_alignment()