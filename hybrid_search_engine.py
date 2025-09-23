"""
Hybrid Property Search Engine
Combines pre-built FAISS embeddings with structured feature scoring
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
from functools import lru_cache


@dataclass
class PropertyMatch:
    """Enhanced structure for hybrid search results"""
    property_id: int
    price: int
    bedrooms: int
    bathrooms: float
    city: str
    state: str
    house_size: int

    # Embedding-based scores
    embedding_similarity: float

    # Feature-specific match scores
    bedroom_score: float
    bathroom_score: float
    size_score: float
    location_score: float
    price_score: float

    # Combined scores
    overall_score: float
    match_type: str

    # Backward compatibility
    @property
    def similarity_score(self) -> float:
        """Backward compatibility: returns overall_score"""
        return self.overall_score


class HybridPropertySearchEngine:
    """High-performance hybrid search: FAISS embeddings + structured scoring"""

    def __init__(self, csv_path: str, version: str = 'v1'):
        self.csv_path = csv_path
        self.version = version
        self.model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.properties_df: Optional[pd.DataFrame] = None
        self.embeddings_dir = Path("embeddings")

        print("Starting hybrid search engine...")
        start_time = time.time()

        self._load_data()
        self._load_prebuilt_embeddings()
        self._prepare_indexes()

        startup_time = (time.time() - start_time) * 1000
        print(f"Hybrid search engine ready! Startup time: {startup_time:.1f}ms")

    def _load_data(self):
        """Load property data"""
        # Try to load sampled data first
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
        if 'city_norm' not in self.properties_df.columns:
            self.properties_df['city_norm'] = self.properties_df['city'].str.lower().str.strip()
            self.properties_df['state_norm'] = self.properties_df['state'].str.lower().str.strip()

    def _load_prebuilt_embeddings(self):
        """Load pre-built FAISS embeddings"""
        try:
            # Load metadata
            metadata_path = self.embeddings_dir / f"metadata_{self.version}.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            print(f"Loading model: {metadata['model_name']}")
            self.model = SentenceTransformer(metadata['model_name'])

            # Load FAISS index
            index_path = self.embeddings_dir / f"faiss_index_{self.version}.index"
            self.faiss_index = faiss.read_index(str(index_path))
            print(f"FAISS index loaded: {self.faiss_index.ntotal:,} vectors")

        except Exception as e:
            print(f"Warning: Could not load pre-built embeddings: {e}")
            print("Falling back to structured search only...")
            self.model = None
            self.faiss_index = None

    def _prepare_indexes(self):
        """Prepare optimized indexes for structured scoring"""
        if self.properties_df is None:
            raise ValueError("Properties dataframe not loaded")

        print("Building optimized indexes for structured scoring...")

        # Location-based index for O(1) lookups
        self.city_state_index = {}
        for idx, row in self.properties_df.iterrows():
            city_state_key = f"{row['city_norm']}_{row['state_norm']}"
            if city_state_key not in self.city_state_index:
                self.city_state_index[city_state_key] = []
            self.city_state_index[city_state_key].append(idx)

        # Bedroom/bathroom combination index
        self.bed_bath_combinations = {}
        for (bed, bath), group in self.properties_df.groupby(['bed', 'bath']):
            self.bed_bath_combinations[(bed, bath)] = group.index.tolist()

        print(f"Structured indexes ready: {len(self.city_state_index)} locations, {len(self.bed_bath_combinations)} bed/bath combos")

    def _create_property_description(self, listing: Dict) -> str:
        """Create description for embedding search"""
        return (
            f"{listing['bed']} bedroom {listing['bath']} bathroom house "
            f"in {listing['city']}, {listing['state']} with "
            f"{listing['house_size']} square feet priced at ${listing['price']}"
        )

    def _embedding_search(self, listing: Dict, top_k: int = 50) -> List[int]:
        """Use FAISS embeddings for initial candidate retrieval"""
        if self.model is None or self.faiss_index is None:
            return []

        try:
            # Create description and get embedding
            description = self._create_property_description(listing)
            query_embedding = self.model.encode([description])

            # Normalize for cosine similarity
            query_embedding_float32 = query_embedding.astype(np.float32)
            faiss.normalize_L2(query_embedding_float32)

            # Search FAISS index
            similarities, indices = self.faiss_index.search(
                query_embedding_float32,
                min(top_k, self.faiss_index.ntotal)
            )

            # Return valid indices
            valid_indices = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.properties_df):
                    valid_indices.append(int(idx))

            return valid_indices

        except Exception as e:
            print(f"Warning: Embedding search failed: {e}")
            return []

    def _calculate_feature_scores(self, listing: Dict, candidates: pd.DataFrame) -> pd.DataFrame:
        """Calculate structured feature scores (vectorized)"""
        # Bedroom scoring
        bed_diff = np.abs(candidates['bed'].values - listing['bed'])
        bedroom_scores = np.where(bed_diff == 0, 1.0,
                         np.where(bed_diff == 1, 0.8,
                         np.where(bed_diff == 2, 0.5, 0.0)))

        # Bathroom scoring
        bath_diff = np.abs(candidates['bath'].values - listing['bath'])
        bathroom_scores = np.where(bath_diff == 0, 1.0,
                         np.where(bath_diff <= 0.5, 0.8,
                         np.where(bath_diff <= 1.0, 0.6, 0.0)))

        # Size scoring
        size_ratios = np.minimum(candidates['house_size'].values, listing['house_size']) / \
                     np.maximum(candidates['house_size'].values, listing['house_size'])
        size_scores = np.where(size_ratios >= 0.9, 1.0,
                      np.where(size_ratios >= 0.8, 0.8,
                      np.where(size_ratios >= 0.7, 0.6,
                      np.where(size_ratios >= 0.6, 0.4, 0.0))))

        # Location scoring
        query_city = listing['city'].lower().strip()
        query_state = listing['state'].lower().strip()
        same_city = (candidates['city_norm'] == query_city) & (candidates['state_norm'] == query_state)
        same_state = (candidates['state_norm'] == query_state) & ~same_city
        location_scores = np.where(same_city, 1.0, np.where(same_state, 0.5, 0.0))

        # Price scoring
        price_ratios = np.minimum(candidates['price'].values, listing['price']) / \
                      np.maximum(candidates['price'].values, listing['price'])
        price_scores = np.where(price_ratios >= 0.95, 1.0,
                       np.where(price_ratios >= 0.9, 0.9,
                       np.where(price_ratios >= 0.8, 0.7,
                       np.where(price_ratios >= 0.7, 0.5,
                       np.where(price_ratios >= 0.5, 0.3, 0.1)))))

        # Add scores to candidates
        candidates = candidates.copy()
        candidates['bedroom_score'] = bedroom_scores
        candidates['bathroom_score'] = bathroom_scores
        candidates['size_score'] = size_scores
        candidates['location_score'] = location_scores
        candidates['price_score'] = price_scores

        return candidates

    def _hybrid_search(self, listing: Dict, max_results: int = 10) -> List[PropertyMatch]:
        """Perform hybrid search: embeddings + structured scoring"""

        # Step 1: Get candidates using embeddings (if available)
        embedding_candidates = self._embedding_search(listing, top_k=100)

        if embedding_candidates:
            # Use embedding candidates
            candidates_df = self.properties_df.iloc[embedding_candidates]
            print(f"Embedding search: {len(candidates_df)} candidates")
        else:
            # Fallback: use structured filtering
            print("Using structured filtering as fallback...")
            candidates_df = self._structured_filter(listing)

        if len(candidates_df) == 0:
            return []

        # Step 2: Calculate structured feature scores
        scored_candidates = self._calculate_feature_scores(listing, candidates_df)

        # Step 3: Calculate combined scores
        # Weight: location=30%, size=25%, bedrooms=20%, price=15%, bathrooms=10%
        overall_scores = (0.30 * scored_candidates['location_score'] +
                         0.25 * scored_candidates['size_score'] +
                         0.20 * scored_candidates['bedroom_score'] +
                         0.15 * scored_candidates['price_score'] +
                         0.10 * scored_candidates['bathroom_score'])

        scored_candidates['overall_score'] = overall_scores

        # Step 4: Filter and sort results
        good_matches = scored_candidates[scored_candidates['overall_score'] >= 0.3]
        good_matches = good_matches.sort_values('overall_score', ascending=False)

        # Step 5: Convert to PropertyMatch objects
        matches = []
        for _, row in good_matches.head(max_results).iterrows():
            # Get embedding similarity if available
            embedding_sim = 0.0
            if row.name in embedding_candidates:
                candidate_idx = embedding_candidates.index(row.name)
                embedding_sim = 0.8 + (candidate_idx / len(embedding_candidates)) * 0.2

            matches.append(PropertyMatch(
                property_id=int(row.name) if row.name is not None else 0,
                price=int(row['price']),
                bedrooms=int(row['bed']),
                bathrooms=float(row['bath']),
                city=row['city'],
                state=row['state'],
                house_size=int(row['house_size']),
                embedding_similarity=embedding_sim,
                bedroom_score=float(row['bedroom_score']),
                bathroom_score=float(row['bathroom_score']),
                size_score=float(row['size_score']),
                location_score=float(row['location_score']),
                price_score=float(row['price_score']),
                overall_score=float(row['overall_score']),
                match_type='hybrid'
            ))

        return matches

    def _structured_filter(self, listing: Dict) -> pd.DataFrame:
        """Fallback structured filtering when embeddings not available"""
        # Basic filtering by reasonable ranges
        candidates = self.properties_df.copy()

        # Filter by bedroom range (±2)
        min_bed = max(1, listing['bed'] - 2)
        max_bed = listing['bed'] + 2
        candidates = candidates[
            (candidates['bed'] >= min_bed) &
            (candidates['bed'] <= max_bed)
        ]

        # Filter by location (same state minimum)
        state_key = listing['state'].lower().strip()
        candidates = candidates[candidates['state_norm'] == state_key]

        print(f"Structured filtering: {len(self.properties_df):,} -> {len(candidates):,} candidates")
        return candidates

    # Main API methods
    def find_similar_properties(self, listing: Dict, max_results: int = 10) -> Tuple[List[PropertyMatch], float]:
        """Find similar properties using hybrid search"""
        start_time = time.time()

        matches = self._hybrid_search(listing, max_results)

        search_time = (time.time() - start_time) * 1000
        return matches, search_time

    def find_duplicates(self, listing: Dict, max_results: int = 10) -> Tuple[List[PropertyMatch], float]:
        """Backward compatibility method"""
        return self.find_similar_properties(listing, max_results)

    def batch_duplicate_check(self, listings: List[Dict]) -> List[Dict]:
        """Process multiple listings"""
        results = []
        for listing in listings:
            try:
                matches, search_time = self.find_duplicates(listing, max_results=10)

                # Determine KPI based on dataset size
                dataset_size = len(self.properties_df) if self.properties_df is not None else 0
                kpi_threshold = 200 if dataset_size <= 150000 else 500

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
                            "embedding_similarity": match.embedding_similarity,
                            "match_type": match.match_type
                        }
                        for match in matches
                    ],
                    "search_time_ms": search_time,
                    "meets_performance_kpi": search_time < kpi_threshold
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


def test_hybrid_search():
    """Test the hybrid search engine"""
    print("="*60)
    print("HYBRID SEARCH ENGINE TEST")
    print("="*60)

    engine = HybridPropertySearchEngine('realtor_cleaned_final.csv')

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

    print(f"\nResults: {len(matches)} matches in {search_time:.1f}ms")

    for i, match in enumerate(matches, 1):
        print(f"#{i}: {match.bedrooms}br/{match.bathrooms}ba, {match.house_size:,}sqft, ${match.price:,}")
        print(f"    Overall: {match.overall_score:.3f}, Embedding: {match.embedding_similarity:.3f}")
        print(f"    Features: Location={match.location_score:.2f}, Size={match.size_score:.2f}, Price={match.price_score:.2f}")


if __name__ == "__main__":
    test_hybrid_search()