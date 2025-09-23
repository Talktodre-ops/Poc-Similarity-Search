# Property Similarity Search POC

## POC Objective & Success Metrics

**Objective**: To build a demonstrable API that ingests new property listings, converts them into vector embeddings, and instantly returns a list of potential duplicates or highly similar properties from a stored database.

### Success Metrics for the Demo:

- **Accuracy**: The system should identify obvious duplicates (e.g., same address, slightly different description) with 100% recall.

- **Performance**: API response time for a similarity search should be under 200ms.

- **Clarity**: The output should include the similar listings and a clear confidence score (e.g., 0.95).

## Commands

Run tests: `python test_kpi_compliance.py`
Install dependencies: `pip install -r requirements.txt`
Start API server: `python api.py`
Demo search engine: `python property_search_engine.py`


Step 1: Data Preprocessing & Cleaning
Your data needs to be cleaned and transformed into a format suitable for generating embeddings.

Handle Missing Values:

For numerical columns (bed, bath, acre_lot, house_size), fill missing values (NaN) with the median value of that column. (Using the mean could be skewed by outliers).

For categorical columns (city, state), you can fill missing values with a placeholder like "unknown".

The street column appears to be numeric IDs (e.g., 1962661.0). This is not useful for semantic similarity. We will drop this column. If you had raw text addresses, we would use them.

Create a Unified "Property Description" Text Field:
This is the most important step. We will combine relevant features into a single string that an embedding model can understand. The goal is to create a rich textual description.

python
# Example: Combine features into a descriptive string
df['description'] = (
    df['bed'].astype(str) + " bedroom " +
    df['bath'].astype(str) + " bathroom house " +
    "located in " + df['city'] + ", " + df['state'] + ". " +
    "The lot size is " + df['acre_lot'].astype(str) + " acres " +
    "and the house size is " + df['house_size'].astype(str) + " square feet. " +
    "The property is priced at $" + df['price'].astype(str) + "."
)

# Handle any potential NaN in the new description field
df['description'] = df['description'].fillna("")
Example Output: "3 bedroom 2 bathroom house located in Adjuntas, Puerto Rico. The lot size is 0.12 acres and the house size is 920.0 square feet. The property is priced at $105000.0."

Standardize Text:

Convert the entire description to lowercase.

Remove any special characters or extra whitespace.

python
df['description'] = df['description'].str.lower().str.replace(r'[^\w\s]', ' ', regex=True)
Step 2: Generate Embeddings
We will use a pre-trained model to convert the textual descriptions into numerical vectors (embeddings).

Choose a Model:

OpenAI API (text-embedding-ada-002): Best-in-class accuracy, but costs money per API call.

Free Local Model (sentence-transformers/all-MiniLM-L6-v2): Very good accuracy, runs locally for free. Perfect for a POC. We'll use this.

Install Library:

bash
pip install sentence-transformers
Generate Embeddings:

python
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all descriptions
descriptions = df['description'].tolist()
embeddings = model.encode(descriptions)

# The 'embeddings' variable is now a NumPy array of shape (num_listings, 384)
# Each listing is represented by a 384-dimensional vector.
Step 3: Store Embeddings in a Vector Database
For fast similarity search, we need a dedicated vector database. ChromaDB is ideal for this POC because it's simple and runs in-memory.

Install and Initialize ChromaDB:

bash
pip install chromadb
python
import chromadb

# Create a Chroma client and collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="property_listings")
Add Embeddings to the Database:
We need to add the vectors along with their metadata (like price, bed) so we can return useful information when we get a search result.

python
# Prepare data for ChromaDB
ids = [str(i) for i in df.index.tolist()] # Chroma needs string IDs
metadatas = df[['price', 'bed', 'bath', 'city', 'state', 'zip_code']].to_dict('records')

# Add to collection
collection.add(
    embeddings=embeddings.tolist(), # The vectors
    metadatas=metadatas,            # The associated data
    ids=ids                         # Unique IDs for each vector
)
Step 4: Build the Search API
Now we build the function that takes a new, unseen listing description and finds the most similar ones in the database.

Create a Search Function:

python
def find_similar_properties(query_description, top_k=5):
    # Step 1: Generate embedding for the query text
    query_embedding = model.encode([query_description]).tolist() # Note: wrap in a list

    # Step 2: Query the vector database
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=['metadatas', 'distances'] # Return the metadata and similarity scores
    )

    # Step 3: Format and return the results
    similar_listings = []
    for i in range(top_k):
        listing_data = results['metadatas'][0][i]
        similarity_score = 1 - results['distances'][0][i] # Convert distance to a similarity score
        similar_listings.append({**listing_data, "similarity_score": similarity_score})

    return similar_listings
Test the Search:

python
# Example query: A new listing we want to check for duplicates
new_listing = "3 bedroom 2 bath home in Adjuntas on a small lot"

similar = find_similar_properties(new_listing)
print(similar)
Expected Output:

python
[
  {'price': 105000.0, 'bed': 3.0, 'bath': 2.0, 'city': 'adjuntas', 'state': 'puerto rico', 'zip_code': 601.0, 'similarity_score': 0.92},
  {'price': 80000.0, 'bed': 4.0, 'bath': 2.0, 'city': 'adjuntas', 'state': 'puerto rico', 'zip_code': 601.0, 'similarity_score': 0.87},
  ... # 3 more results
]
Step 5: Evaluate and Refine (Crucial for the Demo)
Create a Test Set: Manually create a few example queries. For example, take a listing from your dataset and slightly reword its description. Your system should return the original listing as the top match with a very high similarity score (~0.95+).

Tune the Description: The quality of the search is entirely dependent on the quality of the description field you create. Experiment with including or excluding features. Maybe zip_code isn't necessary, or maybe you need to add status.

Set a Similarity Threshold: Determine what score constitutes a "duplicate." For example, you might decide that any score above 0.9 requires manual review.