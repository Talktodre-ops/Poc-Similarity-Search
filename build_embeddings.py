"""
Embedding Builder Script - Production Version
Builds and saves embeddings/FAISS index once for instant engine startup

Run this script whenever:
- You have new property data
- You want to change embedding models
- You need to rebuild the search index

Usage: python build_embeddings.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional


class EmbeddingBuilder:
    """Professional embedding builder for production deployment"""

    def __init__(self, csv_path: str, model_name: str = 'paraphrase-MiniLM-L3-v2', version: str = 'v1', sample_size: Optional[int] = None):
        self.csv_path = csv_path
        self.model_name = model_name
        self.version = version
        self.sample_size = sample_size
        self.embeddings_dir = Path("embeddings")

        # File paths
        self.embeddings_path = self.embeddings_dir / f"property_embeddings_{version}.npy"
        self.index_path = self.embeddings_dir / f"faiss_index_{version}.index"
        self.metadata_path = self.embeddings_dir / f"metadata_{version}.json"
        self.property_ids_path = self.embeddings_dir / f"property_ids_{version}.npy"
        self.sampled_data_path = self.embeddings_dir / f"sampled_properties_{version}.csv"

        # Initialize
        self.model: SentenceTransformer = None
        self.properties_df: pd.DataFrame = None

        # Create embeddings directory
        self.embeddings_dir.mkdir(exist_ok=True)

    def load_data(self) -> None:
        """Load and prepare property data"""
        print(f"📊 Loading data from {self.csv_path}...")
        full_df = pd.read_csv(self.csv_path)

        # Apply sampling if specified
        if self.sample_size and self.sample_size < len(full_df):
            print(f"🎯 Sampling {self.sample_size:,} properties from {len(full_df):,} total...")
            # Add unique identifier column to track original indices
            full_df['original_index'] = full_df.index
            self.properties_df = full_df.sample(n=self.sample_size, random_state=42)
            print(f"✅ Using sample of {len(self.properties_df):,} properties for faster POC")
            print(f"   📌 Original indices: {self.properties_df['original_index'].min()} to {self.properties_df['original_index'].max()}")
        else:
            self.properties_df = full_df
            # Add original index column for consistency
            self.properties_df['original_index'] = self.properties_df.index
            print(f"✅ Using full dataset: {len(self.properties_df):,} properties")

        # Data validation
        required_columns = ['city', 'state', 'bed', 'bath', 'house_size', 'price']
        missing_columns = [col for col in required_columns if col not in self.properties_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Clean and prepare data
        self.properties_df['city_norm'] = self.properties_df['city'].str.lower().str.strip()
        self.properties_df['state_norm'] = self.properties_df['state'].str.lower().str.strip()

    def initialize_model(self) -> None:
        """Initialize the sentence transformer model"""
        print(f"🤖 Loading model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print(f"✅ Model loaded - Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def create_descriptions(self) -> List[str]:
        """Generate optimized property descriptions for embedding"""
        print("📝 Creating property descriptions...")

        descriptions = []
        for _, row in self.properties_df.iterrows():
            desc = (
                f"{row['bed']} bedroom {row['bath']} bathroom home "
                f"in {row['city_norm']} {row['state_norm']} "
                f"with {row['house_size']} sqft living space "
                f"priced at ${row['price']:,}"
            )
            descriptions.append(desc)

        print(f"✅ Created {len(descriptions):,} descriptions")
        return descriptions

    def compute_embeddings(self, descriptions: List[str]) -> np.ndarray:
        """Generate embeddings for all property descriptions"""
        print("🧠 Computing embeddings (this may take a few minutes)...")
        start_time = time.time()

        # Compute embeddings with progress bar
        embeddings = self.model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=32,  # Optimize for memory usage
            convert_to_numpy=True
        )

        compute_time = time.time() - start_time
        print(f"✅ Computed {len(embeddings):,} embeddings in {compute_time:.1f}s")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Memory usage: ~{embeddings.nbytes / 1024 / 1024:.1f}MB")

        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build and optimize FAISS index"""
        print("🚀 Building FAISS index...")

        # Create index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # Normalize for cosine similarity
        embeddings_float32 = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings_float32)

        # Add to index
        index.add(embeddings_float32)

        print(f"✅ FAISS index built:")
        print(f"   Total vectors: {index.ntotal:,}")
        print(f"   Dimension: {index.d}")
        print(f"   Index type: Inner Product (normalized for cosine similarity)")

        return index

    def save_all_artifacts(self, embeddings: np.ndarray, index: faiss.IndexFlatIP) -> None:
        """Save embeddings, index, and metadata"""
        print("💾 Saving all artifacts...")

        # Save raw embeddings (backup)
        np.save(self.embeddings_path, embeddings)
        print(f"   ✅ Embeddings saved: {self.embeddings_path}")

        # Save FAISS index
        faiss.write_index(index, str(self.index_path))
        print(f"   ✅ FAISS index saved: {self.index_path}")

        # Save the exact sampled property data for perfect reproduction
        # Reset index to 0-based for FAISS alignment, but keep original_index column
        sampled_properties = self.properties_df.copy()
        sampled_properties.reset_index(drop=True, inplace=True)
        sampled_properties.to_csv(self.sampled_data_path, index=False)
        print(f"   ✅ Sampled properties saved: {self.sampled_data_path}")

        # Save original indices for mapping back to full dataset
        original_indices = self.properties_df['original_index'].to_numpy()
        np.save(self.property_ids_path, original_indices)
        print(f"   ✅ Original property indices saved: {self.property_ids_path}")

        # Save metadata
        metadata = {
            "version": self.version,
            "model_name": self.model_name,
            "total_properties": len(self.properties_df),
            "sample_size": self.sample_size,
            "is_sample": self.sample_size is not None,
            "embedding_dimension": embeddings.shape[1],
            "build_timestamp": time.time(),
            "build_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "csv_file": self.csv_path,
            "file_sizes": {
                "embeddings_mb": os.path.getsize(self.embeddings_path) / 1024 / 1024,
                "index_mb": os.path.getsize(self.index_path) / 1024 / 1024,
                "sampled_data_mb": os.path.getsize(self.sampled_data_path) / 1024 / 1024,
            }
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadata saved: {self.metadata_path}")

    def verify_build(self) -> None:
        """Verify the build was successful"""
        print("🔍 Verifying build...")

        # Test loading index
        test_index = faiss.read_index(str(self.index_path))
        print(f"   ✅ Index loads successfully: {test_index.ntotal:,} vectors")

        # Test loading embeddings
        test_embeddings = np.load(self.embeddings_path)
        print(f"   ✅ Embeddings load successfully: {test_embeddings.shape}")

        # Test loading sampled data
        test_sampled_df = pd.read_csv(self.sampled_data_path)
        print(f"   ✅ Sampled data loads successfully: {len(test_sampled_df):,} properties")

        # Test loading original indices
        test_original_indices = np.load(self.property_ids_path)
        print(f"   ✅ Original indices load successfully: {len(test_original_indices):,} mappings")

        # Verify alignment
        assert len(test_sampled_df) == test_index.ntotal == len(test_original_indices), "Data alignment mismatch!"
        print(f"   ✅ Data alignment verified: All components match {len(test_sampled_df):,} entries")

        # Test search functionality
        query_vector = test_embeddings[0:1].astype(np.float32)
        faiss.normalize_L2(query_vector)

        similarities, indices = test_index.search(query_vector, 5)
        print(f"   ✅ Search test successful: Found {len(indices[0])} results")
        print(f"   Top similarity score: {similarities[0][0]:.3f}")

        # Test index mapping
        test_idx = indices[0][0]
        original_idx = test_original_indices[test_idx]
        print(f"   ✅ Index mapping test: FAISS[{test_idx}] → Original[{original_idx}]")

    def build_complete_system(self) -> Dict:
        """Main method to build the complete embedding system"""
        total_start = time.time()

        print("="*60)
        print("🏗️  BUILDING PRODUCTION EMBEDDING SYSTEM")
        print("="*60)

        try:
            # Step 1: Load data
            self.load_data()

            # Step 2: Initialize model
            self.initialize_model()

            # Step 3: Create descriptions
            descriptions = self.create_descriptions()

            # Step 4: Compute embeddings
            embeddings = self.compute_embeddings(descriptions)

            # Step 5: Build FAISS index
            index = self.build_faiss_index(embeddings)

            # Step 6: Save everything
            self.save_all_artifacts(embeddings, index)

            # Step 7: Verify
            self.verify_build()

            total_time = time.time() - total_start

            print("\n" + "="*60)
            print("🎉 BUILD COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Total build time: {total_time:.1f}s")
            print(f"Version: {self.version}")
            print(f"Properties processed: {len(self.properties_df):,}")
            print(f"Files created in: {self.embeddings_dir}/")
            print("\nNext steps:")
            print("1. Run your search engine - it will now start in <200ms!")
            print("2. Test with: python quick_demo.py")
            print("3. Run compliance tests: python test_kpi_compliance.py")

            return {
                "success": True,
                "build_time": total_time,
                "version": self.version,
                "properties_count": len(self.properties_df),
                "files_created": [
                    str(self.embeddings_path),
                    str(self.index_path),
                    str(self.metadata_path),
                    str(self.property_ids_path),
                    str(self.sampled_data_path)
                ]
            }

        except Exception as e:
            print(f"\n❌ BUILD FAILED: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """Main function to run the embedding builder"""
    parser = argparse.ArgumentParser(description="Build property embeddings for instant search startup")
    parser.add_argument("--sample", type=int, help="Number of properties to sample (for faster POC testing)")
    parser.add_argument("--model", default="paraphrase-MiniLM-L3-v2", help="SentenceTransformer model to use")
    parser.add_argument("--version", default="v1", help="Version tag for the embeddings")
    parser.add_argument("--csv", default="realtor_cleaned_final.csv", help="CSV file path")

    args = parser.parse_args()

    # Configuration from arguments
    CSV_FILE = args.csv
    MODEL_NAME = args.model
    VERSION = args.version
    SAMPLE_SIZE = args.sample

    # Display configuration
    print("\n🔧 Configuration:")
    print(f"   CSV file: {CSV_FILE}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Version: {VERSION}")
    if SAMPLE_SIZE:
        print(f"   Sample size: {SAMPLE_SIZE:,} properties")
        print(f"   Estimated build time: 2-5 minutes")
        print(f"   Estimated file size: ~{SAMPLE_SIZE * 0.006:.1f}MB")
    else:
        print(f"   Sample size: Full dataset")
        print(f"   ⚠️  This may take 30-60 minutes for large datasets!")

    # Check if CSV exists
    if not os.path.exists(CSV_FILE):
        print(f"\n❌ Error: {CSV_FILE} not found!")
        print("Please ensure your property dataset is in the current directory.")
        return

    # Confirm for large builds
    if not SAMPLE_SIZE:
        response = input(f"\n❓ Build full dataset? This may take a long time. Continue? (y/N): ").lower()
        if response != 'y':
            print("Build cancelled. Use --sample parameter for faster testing.")
            return

    # Build embeddings
    builder = EmbeddingBuilder(CSV_FILE, MODEL_NAME, VERSION, SAMPLE_SIZE)
    result = builder.build_complete_system()

    if result["success"]:
        print(f"\n🚀 Ready for production deployment!")
        if SAMPLE_SIZE:
            print(f"💡 For full dataset, run without --sample parameter")
    else:
        print(f"\n💥 Build failed - check error above")


if __name__ == "__main__":
    main()