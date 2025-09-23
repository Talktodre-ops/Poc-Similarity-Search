#!/usr/bin/env python3
"""
Test script to verify search logic integrity after performance optimizations
"""
from property_search_engine import PropertySearchEngine

def test_search_integrity():
    print("Testing optimized search logic integrity...")

    # Initialize the optimized engine
    engine = PropertySearchEngine('realtor_cleaned_final.csv')

    # Test case: High-value property in Puerto Rico
    test_listing = {
        'bed': 3,
        'bath': 2.0,
        'city': 'Adjuntas',
        'state': 'Puerto Rico',
        'house_size': 920,
        'price': 999999
    }

    print(f"\nQuery: {test_listing}")

    # Run the search
    matches, search_time = engine.find_duplicates(test_listing, 5)

    # Performance verification
    print(f"\nPerformance Results:")
    print(f"Search time: {search_time:.1f}ms")
    print(f"KPI Status: {'PASS' if search_time < 200 else 'FAIL'} (Target: <200ms)")
    print(f"Results found: {len(matches)}")

    # Logic integrity verification
    print(f"\nSearch Logic Integrity Check:")

    if matches:
        best_match = matches[0]
        print(f"Best match overall score: {best_match.overall_score:.3f}")

        # Verify reasonable results (no extreme mismatches like original problem)
        price_reasonable = True
        location_reasonable = True
        size_reasonable = True

        for match in matches:
            # Check if prices are in reasonable range (not $59K for $999K query)
            price_ratio = min(match.price, test_listing['price']) / max(match.price, test_listing['price'])
            if price_ratio < 0.1:  # More than 10x difference is unreasonable
                price_reasonable = False

            # Check location consistency
            if match.state.lower() != test_listing['state'].lower():
                if match.location_score > 0.5:  # Should not have high location score for different states
                    location_reasonable = False

            # Check size consistency (should not match 920sqft with 5000+sqft)
            size_ratio = min(match.house_size, test_listing['house_size']) / max(match.house_size, test_listing['house_size'])
            if size_ratio < 0.2 and match.size_score > 0.5:  # 5x size difference should not have high size score
                size_reasonable = False

        print(f"Price matching logic: {'PASS' if price_reasonable else 'FAIL'}")
        print(f"Location matching logic: {'PASS' if location_reasonable else 'FAIL'}")
        print(f"Size matching logic: {'PASS' if size_reasonable else 'FAIL'}")

        print(f"\nDetailed Results:")
        for i, match in enumerate(matches, 1):
            print(f"Match {i}:")
            print(f"  Property: {match.bedrooms}br/{match.bathrooms}ba, {match.house_size:,}sqft")
            print(f"  Location: {match.city}, {match.state}")
            print(f"  Price: ${match.price:,}")
            print(f"  Overall Score: {match.overall_score:.3f}")
            print(f"  Feature Scores:")
            print(f"    Location: {match.location_score:.2f}")
            print(f"    Size: {match.size_score:.2f}")
            print(f"    Bedrooms: {match.bedroom_score:.2f}")
            print(f"    Price: {match.price_score:.2f}")
            print(f"    Bathrooms: {match.bathroom_score:.2f}")
            print()

        # Overall assessment
        overall_pass = (search_time < 200 and price_reasonable and
                       location_reasonable and size_reasonable)

        print("="*50)
        print(f"OVERALL ASSESSMENT: {'SUCCESS' if overall_pass else 'ISSUES FOUND'}")
        print("="*50)

        if overall_pass:
            print("Search logic integrity maintained after optimization!")
            print("- Performance KPI met (<200ms)")
            print("- Structured scoring working correctly")
            print("- No extreme mismatches found")
        else:
            print("Issues detected in search logic or performance")

    else:
        print("No matches found - this may indicate an issue")

if __name__ == "__main__":
    test_search_integrity()