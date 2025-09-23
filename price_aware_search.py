"""
Price-Aware Property Search Engine
Improved version that properly handles price similarity for accurate results
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Optional, Tuple
import time
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PropertyMatch:
    """Enhanced structure for search results with price-aware scoring"""
    property_id: int
    price: int
    bedrooms: int
    bathrooms: int
    city: str
    state: str
    house_size: int
    similarity_score: float
    price_similarity_score: float
    combined_score: float
    match_type: str  # 'exact', 'semantic', 'price_filtered'


class PriceAwareSearchEngine:
    """Enhanced property search with proper price handling"""

    def __init__(self, csv_path: str, version: str = 'v1'):
        self.csv_path = csv_path
        self.version = version
        self.model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.properties_df: Optional[pd.DataFrame] = None
        self.embeddings_dir = Path("embeddings")

        print("Starting price-aware search engine...")
        start_time = time.time()

        self._load_data()
        self._load_prebuilt_system()

        startup_time = (time.time() - start_time) * 1000
        print(f"Price-aware search engine ready! Startup time: {startup_time:.1f}ms")

    def _load_data(self):
        """Load and prepare property data"""
        # Try to load sampled data first (if it exists from build_embeddings.py)
        sampled_data_path = self.embeddings_dir / f"sampled_properties_{self.version}.csv"

        if sampled_data_path.exists():
            print("Loading pre-sampled property data...")
            self.properties_df = pd.read_csv(sampled_data_path)
            print(f"   Loaded {len(self.properties_df):,} sampled properties")
        else:
            print("Loading full property data...")
            self.properties_df = pd.read_csv(self.csv_path)
            print(f"   Loaded {len(self.properties_df):,} properties")

        # Create normalized columns
        self.properties_df['city_norm'] = self.properties_df['city'].str.lower().str.strip()
        self.properties_df['state_norm'] = self.properties_df['state'].str.lower().str.strip()

    def _load_prebuilt_system(self):
        """Load pre-built embeddings and model"""
        # Load metadata
        metadata_path = self.embeddings_dir / f"metadata_{self.version}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"Model: {metadata['model_name']}")

        # Load model
        self.model = SentenceTransformer(metadata['model_name'])

        # Load FAISS index
        index_path = self.embeddings_dir / f"faiss_index_{self.version}.index"
        self.faiss_index = faiss.read_index(str(index_path))
        if self.faiss_index is not None:
            print(f"FAISS index loaded: {self.faiss_index.ntotal:,} vectors")

    def _calculate_price_similarity(self, query_price: float, target_price: float) -> float:
        """Calculate price similarity score (1.0 = identical, 0.0 = very different)"""
        if query_price == 0 or target_price == 0:
            return 0.0

        # Calculate percentage difference
        price_diff = abs(query_price - target_price) / max(query_price, target_price)

        # Convert to similarity score (exponential decay for large differences)
        price_similarity = np.exp(-3 * price_diff)  # Harsh penalty for price differences

        return price_similarity

    def _filter_by_price_range(self, query_price: float, tolerance: float = 0.5) -> pd.DataFrame:
        """Pre-filter properties by reasonable price range"""
        if self.properties_df is None:
            raise RuntimeError("Properties dataframe is not loaded")

        if query_price <= 0:
            return self.properties_df

        # Dynamic price range based on query price
        if query_price < 100000:
            # For lower-priced properties, use tighter range
            price_min = query_price * (1 - tolerance * 0.5)
            price_max = query_price * (1 + tolerance * 0.5)
        elif query_price < 500000:
            # Medium range
            price_min = query_price * (1 - tolerance * 0.7)
            price_max = query_price * (1 + tolerance * 0.7)
        else:
            # For expensive properties, allow wider range
            price_min = query_price * (1 - tolerance)
            price_max = query_price * (1 + tolerance)

        filtered_df = self.properties_df[
            (self.properties_df['price'] >= price_min) &
            (self.properties_df['price'] <= price_max)
        ]

        print(f"Price filter: ${price_min:,.0f} - ${price_max:,.0f} -> {len(filtered_df):,} properties")
        return filtered_df

    def _semantic_search_with_price_awareness(
        self,
        listing: Dict,
        top_k: int = 50,  # Get more candidates for price filtering
        price_tolerance: float = 0.5
    ) -> List[PropertyMatch]:
        """Enhanced semantic search with price awareness"""

        # Step 1: Pre-filter by price range
        price_filtered_df = self._filter_by_price_range(listing['price'], price_tolerance)

        if len(price_filtered_df) == 0:
            print("Warning: No properties in price range")
            return []

        # Step 2: Create price-normalized description
        query_desc = self._create_normalized_description(listing)

        # Step 3: Generate embedding and search
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        if self.faiss_index is None:
            raise RuntimeError("FAISS index is not initialized")

        query_embedding = self.model.encode([query_desc])
        query_embedding_float32 = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding_float32)

        # Step 4: Search in FAISS index (get more results for filtering)
        similarities, indices = self.faiss_index.search(
            query_embedding_float32,
            min(top_k, self.faiss_index.ntotal)
        )

        # Step 5: Filter results to only include price-filtered properties
        price_filtered_indices = set(price_filtered_df.index)

        matches = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx in price_filtered_indices and similarity > 0.7:
                if self.properties_df is None:
                    raise RuntimeError("Properties dataframe is not loaded")
                row = self.properties_df.iloc[idx]

                # Calculate price similarity
                price_sim = self._calculate_price_similarity(listing['price'], row['price'])

                # Combined score: 70% semantic + 30% price
                combined_score = 0.7 * similarity + 0.3 * price_sim

                matches.append(PropertyMatch(
                    property_id=int(idx),
                    price=int(row['price']),
                    bedrooms=int(row['bed']),
                    bathrooms=int(row['bath']),
                    city=row['city'],
                    state=row['state'],
                    house_size=int(row['house_size']),
                    similarity_score=float(similarity),
                    price_similarity_score=float(price_sim),
                    combined_score=float(combined_score),
                    match_type='price_filtered'
                ))

        # Sort by combined score
        matches.sort(key=lambda x: x.combined_score, reverse=True)
        return matches

    def _create_normalized_description(self, listing: Dict) -> str:
        """Create normalized description with price tiers instead of exact prices"""
        # Convert price to tier for better semantic matching
        price = listing['price']
        if price < 100000:
            price_tier = "budget-friendly"
        elif price < 300000:
            price_tier = "moderately-priced"
        elif price < 600000:
            price_tier = "upscale"
        else:
            price_tier = "luxury"

        return (
            f"{listing['bed']} bedroom {listing['bath']} bathroom {price_tier} home "
            f"in {listing['city'].lower().strip()} {listing['state'].lower().strip()} "
            f"with {listing['house_size']} sqft living space"
        )

    def find_similar_properties(
        self,
        listing: Dict,
        max_results: int = 10,
        price_tolerance: float = 0.5
    ) -> Tuple[List[PropertyMatch], float]:
        """Find similar properties with price awareness"""

        start_time = time.time()

        # Use price-aware semantic search
        matches = self._semantic_search_with_price_awareness(
            listing,
            top_k=50,
            price_tolerance=price_tolerance
        )

        # Limit results
        matches = matches[:max_results]

        search_time = (time.time() - start_time) * 1000

        return matches, search_time


def test_price_aware_search():
    """Test the improved search with price awareness"""

    # Initialize engine
    engine = PriceAwareSearchEngine("realtor_cleaned_final.csv")

    # Test with the problematic high-price query
    test_listing = {
        'bed': 3,
        'bath': 2,
        'city': 'Adjuntas',
        'state': 'Puerto Rico',
        'house_size': 920,
        'price': 999999
    }

    print("\n" + "="*60)
    print("TESTING PRICE-AWARE SEARCH")
    print("="*60)
    print(f"Query: {test_listing['bed']}br/{test_listing['bath']}ba in {test_listing['city']}, {test_listing['state']}")
    print(f"Size: {test_listing['house_size']} sqft, Price: ${test_listing['price']:,}")

    # Search with different price tolerances
    for tolerance in [0.3, 0.5, 0.8]:
        print(f"\nPrice tolerance: {tolerance*100}%")
        matches, search_time = engine.find_similar_properties(test_listing, price_tolerance=tolerance)

        print(f"Search time: {search_time:.1f}ms")
        print(f"Results found: {len(matches)}")

        for i, match in enumerate(matches[:5], 1):
            price_diff = match.price - test_listing['price']
            price_diff_pct = (price_diff / test_listing['price']) * 100

            print(f"#{i} - ID:{match.property_id} | ${match.price:,} ({price_diff_pct:+.1f}%) | "
                  f"Semantic:{match.similarity_score:.3f} | Price:{match.price_similarity_score:.3f} | "
                  f"Combined:{match.combined_score:.3f}")

    return engine


if __name__ == "__main__":
    test_price_aware_search()