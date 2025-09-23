"""
Optimized Property Search Engine
Performance improvements to meet <200ms KPI target
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class PropertyMatch:
    """Enhanced structure for structured search results"""
    property_id: int
    price: int
    bedrooms: int
    bathrooms: float
    city: str
    state: str
    house_size: int

    # Feature-specific match scores
    bedroom_score: float
    bathroom_score: float
    size_score: float
    location_score: float
    price_score: float

    # Overall scores
    overall_score: float
    match_type: str

    # Backward compatibility properties
    @property
    def similarity_score(self) -> float:
        """Backward compatibility: returns overall_score"""
        return self.overall_score


class OptimizedPropertySearchEngine:
    """High-performance property search engine with <200ms target"""

    def __init__(self, csv_path: str, version: str = 'v1'):
        self.csv_path = csv_path
        self.version = version
        self.properties_df: Optional[pd.DataFrame] = None

        print("Starting optimized search engine...")
        start_time = time.time()

        self._load_data()
        self._prepare_optimized_indexes()

        startup_time = (time.time() - start_time) * 1000
        print(f"Optimized search engine ready! Startup time: {startup_time:.1f}ms")

    def _load_data(self):
        """Load property data"""
        # Try to load sampled data first
        try:
            self.properties_df = pd.read_csv("embeddings/sampled_properties_v1.csv")
            print(f"Loaded {len(self.properties_df):,} sampled properties")
        except FileNotFoundError:
            self.properties_df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.properties_df):,} properties")

        # Create normalized columns
        self.properties_df['city_norm'] = self.properties_df['city'].str.lower().str.strip()
        self.properties_df['state_norm'] = self.properties_df['state'].str.lower().str.strip()

    def _prepare_optimized_indexes(self):
        """Prepare optimized indexes for ultra-fast filtering"""
        print("Building optimized indexes...")

        # 1. Location-based index for O(1) city lookups
        self.city_state_index = {}
        for idx, row in self.properties_df.iterrows():
            city_state_key = f"{row['city_norm']}_{row['state_norm']}"
            if city_state_key not in self.city_state_index:
                self.city_state_index[city_state_key] = []
            self.city_state_index[city_state_key].append(idx)

        # 2. Price-sorted index for range queries
        self.properties_df_sorted = self.properties_df.sort_values('price').reset_index(drop=True)
        self.price_to_sorted_idx = {row['price']: idx for idx, row in self.properties_df_sorted.iterrows()}

        # 3. Pre-compute size and bedroom/bathroom ranges for common queries
        self.size_ranges = {}
        self.bed_bath_combinations = {}

        # Group by bedroom/bathroom combinations
        for (bed, bath), group in self.properties_df.groupby(['bed', 'bath']):
            self.bed_bath_combinations[(bed, bath)] = group.index.tolist()

        print(f"Indexes ready: {len(self.city_state_index)} locations, {len(self.bed_bath_combinations)} bed/bath combos")

    @lru_cache(maxsize=1000)
    def _get_price_range_indices(self, min_price: int, max_price: int) -> Tuple[int, int]:
        """Get sorted indices for price range using binary search"""
        prices = self.properties_df_sorted['price'].values
        start_idx = np.searchsorted(prices, min_price, side='left')
        end_idx = np.searchsorted(prices, max_price, side='right')
        return start_idx, end_idx

    def _smart_candidate_filtering(self, listing: Dict) -> pd.DataFrame:
        """Ultra-fast candidate filtering with multiple strategies"""

        # Strategy 1: Start with exact location match if available
        city_state_key = f"{listing['city'].lower().strip()}_{listing['state'].lower().strip()}"
        location_candidates = set()

        if city_state_key in self.city_state_index:
            location_candidates.update(self.city_state_index[city_state_key])
            print(f"Exact location match: {len(location_candidates)} candidates")
        else:
            # Fallback to state-level matching
            state_key = listing['state'].lower().strip()
            for key, indices in self.city_state_index.items():
                if key.endswith(f"_{state_key}"):
                    location_candidates.update(indices)
            print(f"State-level match: {len(location_candidates)} candidates")

        # Strategy 2: Price range filtering (most selective)
        if listing['price'] > 0:
            if listing['price'] < 100000:
                price_min = listing['price'] * 0.6
                price_max = listing['price'] * 1.4
            elif listing['price'] < 500000:
                price_min = listing['price'] * 0.5
                price_max = listing['price'] * 1.5
            else:
                price_min = listing['price'] * 0.3
                price_max = listing['price'] * 1.7

            start_idx, end_idx = self._get_price_range_indices(int(price_min), int(price_max))
            price_candidates = set(self.properties_df_sorted.iloc[start_idx:end_idx].index)
            print(f"Price range {price_min:.0f}-{price_max:.0f}: {len(price_candidates)} candidates")
        else:
            price_candidates = set(self.properties_df.index)

        # Strategy 3: Bedroom/bathroom combination filtering
        bed_bath_candidates = set()
        target_bed = listing['bed']
        target_bath = listing['bath']

        # Include exact and ±1 bedroom, ±1 bathroom combinations
        for bed_offset in [-2, -1, 0, 1, 2]:
            for bath_offset in [-1, -0.5, 0, 0.5, 1]:
                combo_bed = target_bed + bed_offset
                combo_bath = target_bath + bath_offset
                if (combo_bed, combo_bath) in self.bed_bath_combinations:
                    bed_bath_candidates.update(self.bed_bath_combinations[(combo_bed, combo_bath)])

        print(f"Bed/bath combinations: {len(bed_bath_candidates)} candidates")

        # Intersect all filters for maximum selectivity
        if location_candidates:
            final_candidates = location_candidates & price_candidates & bed_bath_candidates
        else:
            final_candidates = price_candidates & bed_bath_candidates

        print(f"Final intersection: {len(final_candidates)} candidates")

        # Limit candidates for performance (prioritize by location if too many)
        if len(final_candidates) > 5000:
            if location_candidates:
                # Prioritize exact location matches
                limited_candidates = list(location_candidates & price_candidates)[:2000]
                final_candidates = set(limited_candidates)
                print(f"Limited to location priority: {len(final_candidates)} candidates")

        return self.properties_df.loc[list(final_candidates)]

    def _vectorized_scoring(self, candidates: pd.DataFrame, listing: Dict) -> pd.DataFrame:
        """Vectorized scoring for multiple properties at once"""

        # Bedroom scoring (vectorized)
        bed_diff = np.abs(candidates['bed'].values - listing['bed'])
        bedroom_scores = np.where(bed_diff == 0, 1.0,
                         np.where(bed_diff == 1, 0.8,
                         np.where(bed_diff == 2, 0.5, 0.0)))

        # Bathroom scoring (vectorized)
        bath_diff = np.abs(candidates['bath'].values - listing['bath'])
        bathroom_scores = np.where(bath_diff == 0, 1.0,
                         np.where(bath_diff <= 0.5, 0.8,
                         np.where(bath_diff <= 1.0, 0.6, 0.0)))

        # Size scoring (vectorized)
        size_ratios = np.minimum(candidates['house_size'].values, listing['house_size']) / \
                     np.maximum(candidates['house_size'].values, listing['house_size'])
        size_scores = np.where(size_ratios >= 0.9, 1.0,
                      np.where(size_ratios >= 0.8, 0.8,
                      np.where(size_ratios >= 0.7, 0.6,
                      np.where(size_ratios >= 0.6, 0.4, 0.0))))

        # Location scoring (vectorized)
        query_city = listing['city'].lower().strip()
        query_state = listing['state'].lower().strip()

        same_city = (candidates['city_norm'] == query_city) & (candidates['state_norm'] == query_state)
        same_state = (candidates['state_norm'] == query_state) & ~same_city
        location_scores = np.where(same_city, 1.0, np.where(same_state, 0.5, 0.0))

        # Price scoring (vectorized)
        price_ratios = np.minimum(candidates['price'].values, listing['price']) / \
                      np.maximum(candidates['price'].values, listing['price'])
        price_scores = np.where(price_ratios >= 0.95, 1.0,
                       np.where(price_ratios >= 0.9, 0.9,
                       np.where(price_ratios >= 0.8, 0.7,
                       np.where(price_ratios >= 0.7, 0.5,
                       np.where(price_ratios >= 0.5, 0.3, 0.1)))))

        # Overall weighted scoring (vectorized)
        overall_scores = (0.30 * location_scores +
                         0.25 * size_scores +
                         0.20 * bedroom_scores +
                         0.15 * price_scores +
                         0.10 * bathroom_scores)

        # Add scores to dataframe
        candidates = candidates.copy()
        candidates['bedroom_score'] = bedroom_scores
        candidates['bathroom_score'] = bathroom_scores
        candidates['size_score'] = size_scores
        candidates['location_score'] = location_scores
        candidates['price_score'] = price_scores
        candidates['overall_score'] = overall_scores

        return candidates

    def _early_termination_search(self, candidates: pd.DataFrame, listing: Dict, max_results: int = 10) -> List[PropertyMatch]:
        """Search with early termination for high-confidence matches"""

        # Score all candidates at once
        scored_candidates = self._vectorized_scoring(candidates, listing)

        # Filter by minimum threshold and sort
        good_matches = scored_candidates[scored_candidates['overall_score'] >= 0.3]
        good_matches = good_matches.sort_values('overall_score', ascending=False)

        # Early termination: if we have enough high-confidence matches, stop
        high_confidence = good_matches[good_matches['overall_score'] >= 0.9]
        if len(high_confidence) >= max_results:
            final_matches = high_confidence.head(max_results)
            print(f"Early termination: Found {len(final_matches)} high-confidence matches")
        else:
            final_matches = good_matches.head(max_results)

        # Convert to PropertyMatch objects
        matches = []
        for _, row in final_matches.iterrows():
            matches.append(PropertyMatch(
                property_id=int(row.name),
                price=int(row['price']),
                bedrooms=int(row['bed']),
                bathrooms=float(row['bath']),
                city=row['city'],
                state=row['state'],
                house_size=int(row['house_size']),
                bedroom_score=float(row['bedroom_score']),
                bathroom_score=float(row['bathroom_score']),
                size_score=float(row['size_score']),
                location_score=float(row['location_score']),
                price_score=float(row['price_score']),
                overall_score=float(row['overall_score']),
                match_type='optimized'
            ))

        return matches

    def find_similar_properties(self, listing: Dict, max_results: int = 10) -> Tuple[List[PropertyMatch], float]:
        """Optimized search with <200ms target"""

        start_time = time.time()

        # Step 1: Smart candidate filtering
        candidates = self._smart_candidate_filtering(listing)

        # Step 2: Early termination search
        matches = self._early_termination_search(candidates, listing, max_results)

        search_time = (time.time() - start_time) * 1000
        return matches, search_time

    # Backward compatibility
    def find_duplicates(self, listing: Dict, max_results: int = 10) -> Tuple[List[PropertyMatch], float]:
        """Backward compatibility method"""
        return self.find_similar_properties(listing, max_results)

    def batch_duplicate_check(self, listings: List[Dict]) -> List[Dict]:
        """Optimized batch processing"""
        results = []
        for listing in listings:
            try:
                matches, search_time = self.find_duplicates(listing, max_results=10)
                results.append({
                    "query": listing,
                    "duplicates_found": len(matches),
                    "matches": [
                        {
                            "property_id": match.property_id,
                            "price": match.price,
                            "bedrooms": match.bedrooms,
                            "bathrooms": match.bathrooms,
                            "city": match.city,
                            "state": match.state,
                            "house_size": match.house_size,
                            "overall_score": match.overall_score,
                            "match_type": match.match_type
                        }
                        for match in matches
                    ],
                    "search_time_ms": search_time,
                    "meets_performance_kpi": search_time < 200
                })
            except Exception as e:
                results.append({
                    "query": listing,
                    "error": str(e),
                    "duplicates_found": 0,
                    "matches": [],
                    "search_time_ms": 0,
                    "meets_performance_kpi": False
                })
        return results


def test_optimized_performance():
    """Test the optimized engine performance"""

    print("="*60)
    print("OPTIMIZED SEARCH ENGINE PERFORMANCE TEST")
    print("="*60)

    engine = OptimizedPropertySearchEngine('realtor_cleaned_final.csv')

    test_cases = [
        {
            'name': 'High-end edge case',
            'query': {'city': 'Adjuntas', 'state': 'Puerto Rico', 'bed': 3, 'bath': 2.0, 'house_size': 920, 'price': 999999}
        },
        {
            'name': 'Budget home',
            'query': {'city': 'San Antonio', 'state': 'Texas', 'bed': 2, 'bath': 1.0, 'house_size': 850, 'price': 95000}
        },
        {
            'name': 'Luxury property',
            'query': {'city': 'Miami', 'state': 'Florida', 'bed': 4, 'bath': 3.0, 'house_size': 2500, 'price': 1200000}
        }
    ]

    search_times = []

    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        query = test_case['query']

        matches, search_time = engine.find_similar_properties(query, 10)
        search_times.append(search_time)

        kpi_status = "PASS" if search_time < 200 else "FAIL"
        best_score = max(match.overall_score for match in matches) if matches else 0

        print(f"Results: {len(matches)} matches in {search_time:.1f}ms ({kpi_status})")
        print(f"Best match: {best_score:.3f}")

    avg_time = sum(search_times) / len(search_times)
    passes = sum(1 for t in search_times if t < 200)

    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Average search time: {avg_time:.1f}ms")
    print(f"KPI compliance: {passes}/{len(search_times)} tests")
    print(f"Target: <200ms")

    if avg_time < 200:
        print("SUCCESS: Performance target achieved!")
    else:
        print(f"Gap: {avg_time - 200:.1f}ms over target")


if __name__ == "__main__":
    test_optimized_performance()