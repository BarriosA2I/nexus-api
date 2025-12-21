"""Verify data in Qdrant."""
import os
from dotenv import load_dotenv
load_dotenv('.env')

from qdrant_client import QdrantClient

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print(f"Connecting to: {qdrant_url[:50]}...")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Get collection info
collections = client.get_collections()
print(f"\nCollections: {[c.name for c in collections.collections]}")

# Get nexus_knowledge info
info = client.get_collection("nexus_knowledge")
print(f"\nnexus_knowledge collection:")
print(f"  Points count: {info.points_count}")

# Get some sample points
points = client.scroll(
    collection_name="nexus_knowledge",
    limit=5,
    with_payload=True,
    with_vectors=False
)

print(f"\nSample data (first 5 points):")
for i, point in enumerate(points[0]):
    payload = point.payload
    print(f"\n--- Point {i+1} (ID: {point.id}) ---")
    print(f"  Industry: {payload.get('industry', 'N/A')}")
    print(f"  Type: {payload.get('chunk_type', 'N/A')}")
    content = payload.get('content', '')[:150]
    print(f"  Content: {content}...")
