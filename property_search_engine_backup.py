"""
Property Similarity Search Engine
Structured feature-based matching for accurate property duplicate detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass


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


class PropertySearchEngine:
    """Property search engine with structured feature matching"""

    def __init__(self, csv_path: str, version: str = 'v1'):
        self.csv_path = csv_path
        self.version = version
        self.properties_df: Optional[pd.DataFrame] = None

        print("Starting property search engine...")
        start_time = time.time()

        self._load_data()
        self._prepare_indexes()

        startup_time = (time.time() - start_time) * 1000
        print(f"Search engine ready! Startup time: {startup_time:.1f}ms")

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

    def _prepare_indexes(self):
        """Prepare indexes for fast filtering"""
        # Create bedroom index
        self.bedroom_index = {}
        for bed in self.properties_df['bed'].unique():
            if not pd.isna(bed):
                self.bedroom_index[bed] = self.properties_df[self.properties_df['bed'] == bed].index.tolist()

        # Create location index
        self.location_index = {}
        for _, row in self.properties_df.iterrows():
            city_state = f"{row['city_norm']}_{row['state_norm']}"
            if city_state not in self.location_index:
                self.location_index[city_state] = []
            self.location_index[city_state].append(row.name)

        print(f"Indexes prepared: {len(self.bedroom_index)} bedroom groups, {len(self.location_index)} location groups")

    def _calculate_bedroom_score(self, query_bed: float, target_bed: float) -> float:
        """Calculate bedroom similarity: exact=1.0, ±1=0.8, ±2=0.5, else=0"""
        if pd.isna(query_bed) or pd.isna(target_bed):
            return 0.0

        diff = abs(query_bed - target_bed)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.8
        elif diff == 2:
            return 0.5
        else:
            return 0.0

    def _calculate_bathroom_score(self, query_bath: float, target_bath: float) -> float:
        """Calculate bathroom similarity: exact=1.0, ±0.5=0.8, ±1=0.6, else=0"""
        if pd.isna(query_bath) or pd.isna(target_bath):
            return 0.0

        diff = abs(query_bath - target_bath)
        if diff == 0:
            return 1.0
        elif diff <= 0.5:
            return 0.8
        elif diff <= 1:
            return 0.6
        else:
            return 0.0

    def _calculate_size_score(self, query_size: float, target_size: float) -> float:
        """Calculate size similarity with reasonable ranges"""
        if pd.isna(query_size) or pd.isna(target_size) or query_size <= 0 or target_size <= 0:
            return 0.0

        # Calculate percentage difference
        size_ratio = min(query_size, target_size) / max(query_size, target_size)

        # Score based on how close the sizes are
        if size_ratio >= 0.9:  # Within 10%
            return 1.0
        elif size_ratio >= 0.8:  # Within 20%
            return 0.8
        elif size_ratio >= 0.7:  # Within 30%
            return 0.6
        elif size_ratio >= 0.6:  # Within 40%
            return 0.4
        else:
            return 0.0  # Too different

    def _calculate_location_score(self, query_city: str, query_state: str,
                                target_city: str, target_state: str) -> float:
        """Calculate location similarity"""
        query_city = query_city.lower().strip()
        query_state = query_state.lower().strip()
        target_city = target_city.lower().strip()
        target_state = target_state.lower().strip()

        if query_city == target_city and query_state == target_state:
            return 1.0  # Exact city match
        elif query_state == target_state:
            return 0.5  # Same state, different city
        else:
            return 0.0  # Different state

    def _calculate_price_score(self, query_price: float, target_price: float) -> float:
        """Calculate price similarity"""
        if query_price <= 0 or target_price <= 0:
            return 0.0

        price_ratio = min(query_price, target_price) / max(query_price, target_price)

        # Exponential scoring for price similarity
        if price_ratio >= 0.95:  # Within 5%
            return 1.0
        elif price_ratio >= 0.9:  # Within 10%
            return 0.9
        elif price_ratio >= 0.8:  # Within 20%
            return 0.7
        elif price_ratio >= 0.7:  # Within 30%
            return 0.5
        elif price_ratio >= 0.5:  # Within 50%
            return 0.3
        else:
            return 0.1  # Very different prices

    def _filter_candidates(self, listing: Dict) -> pd.DataFrame:
        """Pre-filter candidates based on reasonable ranges"""
        candidates = self.properties_df.copy()

        # Filter by bedroom range (±2)
        if not pd.isna(listing['bed']):
            min_bed = max(1, listing['bed'] - 2)
            max_bed = listing['bed'] + 2
            candidates = candidates[
                (candidates['bed'] >= min_bed) &
                (candidates['bed'] <= max_bed)
            ]

        # Filter by bathroom range (±1)
        if not pd.isna(listing['bath']):
            min_bath = max(0.5, listing['bath'] - 1)
            max_bath = listing['bath'] + 1
            candidates = candidates[
                (candidates['bath'] >= min_bath) &
                (candidates['bath'] <= max_bath)
            ]

        # Filter by size range (±50% but reasonable limits)
        if not pd.isna(listing['house_size']) and listing['house_size'] > 0:
            min_size = listing['house_size'] * 0.5  # 50% smaller
            max_size = listing['house_size'] * 1.5  # 50% larger
            candidates = candidates[
                (candidates['house_size'] >= min_size) &
                (candidates['house_size'] <= max_size)
            ]

        # Filter by price range (±80%)
        if not pd.isna(listing['price']) and listing['price'] > 0:
            min_price = listing['price'] * 0.2  # 80% cheaper
            max_price = listing['price'] * 1.8  # 80% more expensive
            candidates = candidates[
                (candidates['price'] >= min_price) &
                (candidates['price'] <= max_price)
            ]

        print(f"Filtered candidates: {len(self.properties_df):,} -> {len(candidates):,}")
        return candidates

    def _exact_search(self, listing: Dict) -> List[PropertyMatch]:
        """Find exact matches on all features"""
        matches = []

        # Find exact matches
        exact_matches = self.properties_df[
            (self.properties_df['bed'] == listing['bed']) &
            (self.properties_df['bath'] == listing['bath']) &
            (self.properties_df['house_size'] == listing['house_size']) &
            (self.properties_df['city_norm'] == listing['city'].lower().strip()) &
            (self.properties_df['state_norm'] == listing['state'].lower().strip()) &
            (self.properties_df['price'] == listing['price'])
        ]

        for _, row in exact_matches.iterrows():
            matches.append(PropertyMatch(
                property_id=int(row.name),
                price=int(row['price']),
                bedrooms=int(row['bed']),
                bathrooms=float(row['bath']),
                city=row['city'],
                state=row['state'],
                house_size=int(row['house_size']),
                bedroom_score=1.0,
                bathroom_score=1.0,
                size_score=1.0,
                location_score=1.0,
                price_score=1.0,
                overall_score=1.0,
                match_type='exact'
            ))

        return matches

    def _structured_search(self, listing: Dict, max_results: int = 10) -> List[PropertyMatch]:
        """Structured search with feature-specific scoring"""

        # Pre-filter candidates
        candidates = self._filter_candidates(listing)

        if len(candidates) == 0:
            return []

        matches = []

        for _, row in candidates.iterrows():
            # Calculate feature-specific scores
            bedroom_score = self._calculate_bedroom_score(listing['bed'], row['bed'])
            bathroom_score = self._calculate_bathroom_score(listing['bath'], row['bath'])
            size_score = self._calculate_size_score(listing['house_size'], row['house_size'])
            location_score = self._calculate_location_score(
                listing['city'], listing['state'],
                row['city'], row['state']
            )
            price_score = self._calculate_price_score(listing['price'], row['price'])

            # Calculate weighted overall score
            # Weights: location=30%, size=25%, bedrooms=20%, price=15%, bathrooms=10%
            overall_score = (
                0.30 * location_score +
                0.25 * size_score +
                0.20 * bedroom_score +
                0.15 * price_score +
                0.10 * bathroom_score
            )

            # Only include if overall score is reasonable
            if overall_score >= 0.3:
                matches.append(PropertyMatch(
                    property_id=int(row.name),
                    price=int(row['price']),
                    bedrooms=int(row['bed']),
                    bathrooms=float(row['bath']),
                    city=row['city'],
                    state=row['state'],
                    house_size=int(row['house_size']),
                    bedroom_score=bedroom_score,
                    bathroom_score=bathroom_score,
                    size_score=size_score,
                    location_score=location_score,
                    price_score=price_score,
                    overall_score=overall_score,
                    match_type='structured'
                ))

        # Sort by overall score
        matches.sort(key=lambda x: x.overall_score, reverse=True)
        return matches[:max_results]

    # Main API methods for backward compatibility
    def find_duplicates(
        self,
        listing: Dict,
        max_results: int = 10
    ) -> Tuple[List[PropertyMatch], float]:
        """Find similar properties using structured matching (backward compatible)"""
        return self.find_similar_properties(listing, max_results)

    def find_similar_properties(
        self,
        listing: Dict,
        max_results: int = 10
    ) -> Tuple[List[PropertyMatch], float]:
        """Find similar properties using structured matching"""

        start_time = time.time()

        # First try exact search
        exact_matches = self._exact_search(listing)

        # Then do structured search
        structured_matches = self._structured_search(listing, max_results)

        # Combine results (exact matches first)
        all_matches = exact_matches + structured_matches

        # Remove duplicates and limit results
        seen_ids = set()
        unique_matches = []
        for match in all_matches:
            if match.property_id not in seen_ids:
                unique_matches.append(match)
                seen_ids.add(match.property_id)
                if len(unique_matches) >= max_results:
                    break

        search_time = (time.time() - start_time) * 1000
        return unique_matches, search_time

    def batch_duplicate_check(self, listings: List[Dict]) -> List[Dict]:
        """Process multiple listings for duplicate detection"""
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


# Demo function for testing
def quick_demo():
    """Quick demo of the structured search"""
    engine = PropertySearchEngine("realtor_cleaned_final.csv")

    test_listing = {
        'bed': 3,
        'bath': 2.0,
        'city': 'Adjuntas',
        'state': 'Puerto Rico',
        'house_size': 920,
        'price': 999999
    }

    print(f"\nTesting: {test_listing}")
    matches, search_time = engine.find_duplicates(test_listing, 5)

    print(f"Search time: {search_time:.1f}ms")
    print(f"Results: {len(matches)}")

    for i, match in enumerate(matches, 1):
        print(f"#{i}: {match.bedrooms}br/{match.bathrooms}ba, {match.house_size}sqft, ${match.price:,}")
        print(f"    Overall score: {match.overall_score:.3f}")


if __name__ == "__main__":
    quick_demo()