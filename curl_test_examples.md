# API Test Examples

## Single Property Search Examples

### 1. Luxury Miami Property
```bash
curl -X POST "http://localhost:8000/find-duplicates" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Miami",
    "state": "Florida",
    "bed": 4,
    "bath": 3.0,
    "house_size": 2500,
    "price": 1200000
  }'
```

### 2. Budget Starter Home
```bash
curl -X POST "http://localhost:8000/find-duplicates" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "San Antonio",
    "state": "Texas",
    "bed": 2,
    "bath": 1.0,
    "house_size": 850,
    "price": 95000
  }'
```

### 3. High-End Edge Case (Previous Problem)
```bash
curl -X POST "http://localhost:8000/find-duplicates" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Adjuntas",
    "state": "Puerto Rico",
    "bed": 3,
    "bath": 2.0,
    "house_size": 920,
    "price": 999999
  }'
```

### 4. Family Suburban Home
```bash
curl -X POST "http://localhost:8000/find-duplicates" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Austin",
    "state": "Texas",
    "bed": 3,
    "bath": 2.0,
    "house_size": 1650,
    "price": 425000
  }'
```

### 5. Urban Condo/Apartment
```bash
curl -X POST "http://localhost:8000/find-duplicates" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "New York City",
    "state": "New York",
    "bed": 2,
    "bath": 2.0,
    "house_size": 950,
    "price": 1850000
  }'
```

## Batch Search Example

```bash
curl -X POST "http://localhost:8000/batch-search" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "city": "Miami",
      "state": "Florida",
      "bed": 4,
      "bath": 3.0,
      "house_size": 2500,
      "price": 1200000
    },
    {
      "city": "San Antonio",
      "state": "Texas",
      "bed": 2,
      "bath": 1.0,
      "house_size": 850,
      "price": 95000
    },
    {
      "city": "Austin",
      "state": "Texas",
      "bed": 3,
      "bath": 2.0,
      "house_size": 1650,
      "price": 425000
    }
  ]'
```

## Expected Response Format

The API will return structured results with feature-specific scoring:

```json
{
  "query": { ... },
  "duplicates_found": 5,
  "matches": [
    {
      "property_id": 7028,
      "price": 850000,
      "bedrooms": 4,
      "bathrooms": 2.0,
      "city": "Dorado",
      "state": "Puerto Rico",
      "house_size": 999,
      "bedroom_score": 0.8,
      "bathroom_score": 1.0,
      "size_score": 1.0,
      "location_score": 0.5,
      "price_score": 0.7,
      "overall_score": 0.765,
      "match_type": "structured"
    }
  ],
  "search_time_ms": 528.9,
  "meets_performance_kpi": false,
  "confidence_level": "Medium"
}
```

## Key Features to Notice

1. **Feature-Specific Scores**: Each match shows breakdown of bedroom, bathroom, size, location, and price similarity
2. **Overall Score**: Weighted combination of all factors
3. **Price Range Filtering**: No more unrealistic price matches
4. **Size Constraints**: Properties within reasonable size ranges
5. **Location Hierarchy**: Same city > same state > different state
6. **Performance Metrics**: Search time and KPI compliance