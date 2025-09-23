"""
Test Examples for Structured Property Search
Demonstrates different matching scenarios and edge cases
"""

from property_search_engine import PropertySearchEngine
import time

def run_test_examples():
    """Run comprehensive test examples"""

    print("="*80)
    print("🏠 STRUCTURED PROPERTY SEARCH - TEST EXAMPLES")
    print("="*80)

    # Initialize engine
    engine = PropertySearchEngine('realtor_cleaned_final.csv')

    test_cases = [
        {
            "name": "💰 HIGH-END LUXURY PROPERTY",
            "description": "Test if expensive properties match within reasonable price ranges",
            "query": {
                "city": "Miami",
                "state": "Florida",
                "bed": 4,
                "bath": 3.0,
                "house_size": 2500,
                "price": 1200000
            },
            "expected": "Should find luxury properties in $600K-$2M range, prioritize location & size"
        },

        {
            "name": "🏡 BUDGET-FRIENDLY STARTER HOME",
            "description": "Test affordable property matching with tight price constraints",
            "query": {
                "city": "San Antonio",
                "state": "Texas",
                "bed": 2,
                "bath": 1.0,
                "house_size": 850,
                "price": 95000
            },
            "expected": "Should find similar affordable homes $47K-$142K, focus on size similarity"
        },

        {
            "name": "🏢 EXACT MATCH TEST",
            "description": "Test if system finds exact duplicates with 1.0 scores",
            "query": {
                "city": "Adjuntas",
                "state": "Puerto Rico",
                "bed": 3,
                "bath": 2.0,
                "house_size": 920,
                "price": 105000
            },
            "expected": "Should find exact matches in Puerto Rico with perfect scores"
        },

        {
            "name": "🌊 OCEANFRONT LUXURY",
            "description": "Test high-value coastal property matching",
            "query": {
                "city": "Malibu",
                "state": "California",
                "bed": 5,
                "bath": 4.0,
                "house_size": 3500,
                "price": 2500000
            },
            "expected": "Should find luxury California properties, may expand to similar coastal cities"
        },

        {
            "name": "🏠 SUBURBAN FAMILY HOME",
            "description": "Test typical family home matching - most common use case",
            "query": {
                "city": "Austin",
                "state": "Texas",
                "bed": 3,
                "bath": 2.0,
                "house_size": 1650,
                "price": 425000
            },
            "expected": "Should find similar suburban homes, balance all factors"
        },

        {
            "name": "🏘️ TINY HOME / CONDO",
            "description": "Test small property matching with size constraints",
            "query": {
                "city": "Portland",
                "state": "Oregon",
                "bed": 1,
                "bath": 1.0,
                "house_size": 600,
                "price": 285000
            },
            "expected": "Should find small properties 300-900 sqft, urban pricing"
        },

        {
            "name": "🏰 MANSION / ESTATE",
            "description": "Test large property matching with size priority",
            "query": {
                "city": "Dallas",
                "state": "Texas",
                "bed": 6,
                "bath": 5.0,
                "house_size": 5500,
                "price": 850000
            },
            "expected": "Should find large properties 2750-8250 sqft, focus on size over bed/bath exact match"
        },

        {
            "name": "🏚️ FIXER-UPPER OPPORTUNITY",
            "description": "Test low-price property matching",
            "query": {
                "city": "Detroit",
                "state": "Michigan",
                "bed": 3,
                "bath": 1.0,
                "house_size": 1100,
                "price": 45000
            },
            "expected": "Should find affordable properties needing work, tight price range"
        },

        {
            "name": "🏖️ VACATION RENTAL",
            "description": "Test resort area property matching",
            "query": {
                "city": "Key West",
                "state": "Florida",
                "bed": 2,
                "bath": 2.0,
                "house_size": 1200,
                "price": 675000
            },
            "expected": "Should find vacation properties, may expand to Florida coastal areas"
        },

        {
            "name": "🌆 URBAN PENTHOUSE",
            "description": "Test high-density urban property matching",
            "query": {
                "city": "New York City",
                "state": "New York",
                "bed": 2,
                "bath": 2.0,
                "house_size": 950,
                "price": 1850000
            },
            "expected": "Should find urban properties, price per sqft matters more than total size"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST {i}/10 {'='*20}")
        print(f"📋 {test_case['name']}")
        print(f"📝 {test_case['description']}")
        print(f"🎯 Expected: {test_case['expected']}")

        query = test_case['query']
        print(f"\n🔍 Query: {query['bed']}br/{query['bath']}ba, {query['house_size']:,} sqft")
        print(f"   📍 {query['city']}, {query['state']} - ${query['price']:,}")

        # Perform search
        start_time = time.time()
        matches, search_time = engine.find_duplicates(query, max_results=5)

        print(f"\n⚡ Results: {len(matches)} matches in {search_time:.1f}ms")
        print("-" * 60)

        if not matches:
            print("❌ No matches found - try expanding search criteria")
            continue

        for j, match in enumerate(matches, 1):
            # Calculate differences
            bed_diff = match.bedrooms - query['bed']
            bath_diff = match.bathrooms - query['bath']
            size_diff_pct = ((match.house_size - query['house_size']) / query['house_size']) * 100
            price_diff_pct = ((match.price - query['price']) / query['price']) * 100

            print(f"#{j} 🏠 ID:{match.property_id} | Overall: {match.overall_score:.3f}")
            print(f"    📍 {match.city}, {match.state}")
            print(f"    🏠 {match.bedrooms}br/{match.bathrooms}ba ({bed_diff:+}br/{bath_diff:+.1f}ba)")
            print(f"    📐 {match.house_size:,} sqft ({size_diff_pct:+.1f}%)")
            print(f"    💰 ${match.price:,} ({price_diff_pct:+.1f}%)")
            print(f"    📊 Scores: Bed:{match.bedroom_score:.2f} Bath:{match.bathroom_score:.2f} Size:{match.size_score:.2f} Loc:{match.location_score:.2f} Price:{match.price_score:.2f}")
            print(f"    🎯 Match: {match.match_type}")
            print()

    print("\n" + "="*80)
    print("✅ TEST EXAMPLES COMPLETED")
    print("="*80)
    print("\n💡 Key Insights from Testing:")
    print("• High-price properties should find reasonable price ranges")
    print("• Size similarity prevents unrealistic sqft matches")
    print("• Location scoring prioritizes same city > same state > different state")
    print("• Feature-specific scores explain why each property matched")
    print("• Combined scoring balances all factors appropriately")

    print(f"\n🚀 You can also test these via:")
    print(f"• API: curl -X POST http://localhost:8000/find-duplicates -d '{{query_json}}'")
    print(f"• Streamlit: http://localhost:8501 (use the examples above)")
    print(f"• Quick Demo: python quick_demo.py")


if __name__ == "__main__":
    run_test_examples()