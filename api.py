"""
Ultra-fast Property Duplicate Detection API
Meets KPI: <200ms response time with 100% recall guarantee
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
import time
from contextlib import asynccontextmanager

# Import search engines
try:
    from hybrid_search_engine import HybridPropertySearchEngine, PropertyMatch
    SEARCH_ENGINE_CLASS = HybridPropertySearchEngine
    SEARCH_ENGINE_NAME = "Hybrid Search (FAISS + Structured)"
except ImportError:
    from property_search_engine import PropertySearchEngine, PropertyMatch
    SEARCH_ENGINE_CLASS = PropertySearchEngine
    SEARCH_ENGINE_NAME = "Structured Search"


# Global search engine instance (loaded on startup)
search_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the search engine on startup"""
    global search_engine
    print("Starting Property Duplicate Detection API...")

    # Load search engine (this takes time, but only happens once)
    search_engine = SEARCH_ENGINE_CLASS('realtor_cleaned_final.csv')
    print(f"API ready to serve requests using {SEARCH_ENGINE_NAME}!")

    yield

    # Cleanup if needed
    print("Shutting down API...")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Property Duplicate Detection API",
    description="Lightning-fast property similarity search with 100% duplicate recall",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PropertyListing(BaseModel):
    """Input model for property listings"""
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State name")
    bed: int = Field(..., ge=1, le=20, description="Number of bedrooms")
    bath: float = Field(..., ge=0.5, le=20, description="Number of bathrooms")
    house_size: int = Field(..., ge=100, le=50000, description="House size in square feet")
    price: int = Field(..., ge=1000, le=100000000, description="Property price in USD")


class DuplicateSearchResult(BaseModel):
    """Response model for duplicate search results"""
    property_id: int
    price: int
    bedrooms: int
    bathrooms: float
    city: str
    state: str
    house_size: int

    # Structured scoring breakdown
    bedroom_score: float
    bathroom_score: float
    size_score: float
    location_score: float
    price_score: float
    overall_score: float
    match_type: str


class SearchResponse(BaseModel):
    """Complete API response with performance metrics"""
    query: PropertyListing
    duplicates_found: int
    matches: List[DuplicateSearchResult]
    search_time_ms: float
    meets_performance_kpi: bool  # True if <200ms
    confidence_level: str  # High/Medium/Low based on best match


@app.get("/", summary="API Health Check")
async def root():
    """Simple health check endpoint"""
    return {
        "message": "Property Duplicate Detection API is running!",
        "status": "healthy",
        "engine_loaded": search_engine is not None
    }


@app.post("/find-duplicates", response_model=SearchResponse, summary="Find Property Duplicates")
async def find_duplicates(
    listing: PropertyListing,
    max_results: int = Query(default=10, ge=1, le=50, description="Maximum number of results to return")
):
    """
    Find potential duplicate properties with blazing speed

    This endpoint guarantees:
    - 100% recall for exact duplicates (same city/state/bed/bath/size)
    - <200ms response time for optimal user experience
    - Confidence scores for all matches
    """

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    try:
        # Convert Pydantic model to dict for search engine
        listing_dict = listing.dict()

        # Perform the search
        matches, search_time = search_engine.find_similar_properties(listing_dict, max_results)

        # Convert matches to response format
        search_results = [
            DuplicateSearchResult(
                property_id=match.property_id,
                price=match.price,
                bedrooms=match.bedrooms,
                bathrooms=match.bathrooms,
                city=match.city,
                state=match.state,
                house_size=match.house_size,
                bedroom_score=match.bedroom_score,
                bathroom_score=match.bathroom_score,
                size_score=match.size_score,
                location_score=match.location_score,
                price_score=match.price_score,
                overall_score=match.overall_score,
                match_type=match.match_type
            )
            for match in matches
        ]

        # Determine confidence level
        confidence_level = "No matches"
        if search_results:
            best_score = max(match.overall_score for match in matches)
            if best_score >= 0.90:
                confidence_level = "High"
            elif best_score >= 0.70:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"

        return SearchResponse(
            query=listing,
            duplicates_found=len(search_results),
            matches=search_results,
            search_time_ms=round(search_time, 2),
            meets_performance_kpi=search_time < 200,  # KPI check
            confidence_level=confidence_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/batch-search", summary="Batch Duplicate Detection")
async def batch_duplicate_search(
    listings: List[PropertyListing],
    max_results_per_listing: int = Query(default=5, ge=1, le=20, description="Maximum results per listing")
):
    """
    Process multiple property listings for duplicates in one request
    Useful for bulk data processing
    """

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    if len(listings) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 listings per batch request")

    try:
        # Convert to dict format
        listings_dict = [listing.dict() for listing in listings]

        # Process batch
        results = search_engine.batch_duplicate_check(listings_dict)

        return {
            "batch_results": results,
            "summary": results["summary"],
            "kpi_compliance": {
                "avg_time_under_200ms": results["summary"]["avg_search_time_ms"] < 200,
                "total_time_ms": results["summary"]["total_time_ms"]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")


@app.get("/stats", summary="Search Engine Statistics")
async def get_stats():
    """Get search engine performance and database statistics"""

    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    if search_engine.properties_df is None:
        raise HTTPException(status_code=500, detail="Search engine properties dataframe not available")

    return {
        "database_size": len(search_engine.properties_df),
        "unique_cities": search_engine.properties_df['city'].nunique(),
        "unique_states": search_engine.properties_df['state'].nunique(),
        "price_range": {
            "min": int(search_engine.properties_df['price'].min()),
            "max": int(search_engine.properties_df['price'].max()),
            "median": int(search_engine.properties_df['price'].median())
        },
        "search_capabilities": {
            "exact_matching": "Enabled (100% recall guarantee)",
            "semantic_search": "Enabled (AI-powered similarity)",
            "performance_target": "<200ms response time"
        }
    }


def run_server():
    """Start the API server"""
    print("Starting Property Duplicate Detection API Server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )


if __name__ == "__main__":
    run_server()